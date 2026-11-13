# =============================================================================
# MACRO REGIME FROM PUBLISHED FED INDICATORS
# =============================================================================
# Classifies the macro regime (Expansion, Late-cycle, Recession, Recovery)
# from three indicators that Federal Reserve researchers already estimate and
# publish, instead of from hand-picked indicator weights:
#
#   CFNAI-MA3      Chicago Fed National Activity Index, 3-month average
#                  (FRED CFNAIMA3). First principal component of 85 monthly
#                  activity indicators, scaled so 0 = trend growth, 1 = one
#                  standard deviation.
#   Sahm rule      Real-time Sahm rule recession indicator (FRED SAHMREALTIME),
#                  in percentage points, built from unemployment as first
#                  reported.
#   Chauvet-Piger  Smoothed U.S. recession probability from a dynamic-factor
#                  Markov-switching model (FRED RECPROUSM156N), as 0-1.
#
# plus the 10-year minus 3-month Treasury spread (FRED T10Y3M) as an early
# warning flag that does not vote.
#
# Rules (all thresholds are the publishers' own, set in config.yaml under
# METHODOLOGY_PARAMS.MACRO_REGIME_MODEL):
#
#   1. Recession vote. Each indicator fires on its published rule:
#        CFNAI-MA3 <= -0.70, Sahm >= 0.50pp, Chauvet-Piger >= 0.50.
#      Two or more firing -> Recession.
#   2. Otherwise a business-cycle clock on CFNAI-MA3, level versus trend (0)
#      and direction over the last 6 months, where a new phase must hold for
#      2 consecutive months before it replaces the current one:
#        above trend, rising   -> Expansion
#        above trend, falling  -> Late-cycle
#        below trend, rising   -> Recovery
#        below trend, falling  -> Late-cycle (slowdown without a recession call)
#
# Regime probabilities: P(Recession) is the share of the available indicators
# that fire; the rest goes to the clock phase. No weights are estimated here.
#
# Everything is point-in-time in its logic: the regime at month t uses only
# rows dated on or before t. The FRED series themselves are the latest
# vintage, so CFNAI and Chauvet-Piger values for past months include later
# revisions (the Sahm series is real-time). A strict backtest should use
# ALFRED vintages.
# =============================================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

REGIMES: tuple = ("Expansion", "Late-cycle", "Recession", "Recovery")

# Column names in df_macro_raw and the FRED series behind each one
PUBLISHED_INDICATOR_COLUMNS: Dict[str, str] = {
    "cfnai_ma3": "CFNAIMA3",
    "sahm_rule_realtime_pp": "SAHMREALTIME",
    "recession_prob_chauvet_piger": "RECPROUSM156N",
    "term_spread_10y3m": "T10Y3M",
}

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "method": "published_indicators",
    "growth_clock": {
        "indicator": "cfnai_ma3",
        "trend_level": 0.0,
        "direction_lookback_months": 6,
        "confirm_months": 2,
    },
    "recession_vote": {
        "min_votes": 2,
        "cfnai_ma3_at_or_below": -0.70,
        "sahm_at_or_above": 0.50,
        "chauvet_piger_at_or_above": 0.50,
    },
    "early_warning": {
        "term_spread_inverted_below": 0.0,
        "lookback_months": 12,
    },
    "max_staleness_months": 4,
    "zscore_min_periods": 24,
}

# Weights for the three descriptive scores that no longer decide the regime
DEFAULT_SCORE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "inflation": {"cpi_yoy": 0.7, "cpi_mom": 0.3},
    "monetary_policy": {"fed_funds_rate": 0.5, "fed_funds_3m_change": 0.5},
    "financial_conditions": {"financial_conditions_index": 0.6, "financial_conditions_mom": 0.4},
}

_SCORE_CLIP: float = 3.0


def _merge_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_MODEL_PARAMS.items()}
    for key, value in (params or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _check_inputs(df_macro: pd.DataFrame) -> None:
    if not isinstance(df_macro, pd.DataFrame):
        raise TypeError(f"df_macro must be a pd.DataFrame, got {type(df_macro).__name__}.")
    if not isinstance(df_macro.index, pd.DatetimeIndex):
        raise TypeError("df_macro must have a DatetimeIndex.")
    needed = ["cfnai_ma3", "sahm_rule_realtime_pp", "recession_prob_chauvet_piger"]
    missing = [c for c in needed if c not in df_macro.columns]
    if missing:
        raise ValueError(
            f"df_macro is missing published regime indicators {missing}. "
            f"Download them from FRED ({[PUBLISHED_INDICATOR_COLUMNS[c] for c in missing]}); "
            "agent_market_data.build_free_pipeline_data adds them."
        )


# =============================================================================
# Vectorised history (one row per month, each using data up to that month)
# =============================================================================

def compute_regime_frame(
    df_macro: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    score_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Classify every month in df_macro. Row t depends only on rows <= t.

    Returns a DataFrame indexed like df_macro with columns: regime,
    p_<regime> for each regime, confidence, the indicator values, which
    recession rules fired, the clock level and direction, the yield-curve flag,
    and the descriptive scores s_g, s_pi, s_pol, s_fc in [-3, 3].
    Months without enough data (the first direction_lookback_months, or with
    fewer than min_votes indicators available) have regime NaN.
    """
    _check_inputs(df_macro)
    p = _merge_params(params)
    clock, vote, warn = p["growth_clock"], p["recession_vote"], p["early_warning"]
    df = df_macro.sort_index()
    staleness = int(p["max_staleness_months"])

    def latest(col: str) -> pd.Series:
        # Carry the last published value forward, but not beyond the staleness limit
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[col].astype(float).ffill(limit=staleness)

    level = latest(clock["indicator"])
    lookback = int(clock["direction_lookback_months"])
    change = level - level.shift(lookback)
    sahm = latest("sahm_rule_realtime_pp")
    cp = latest("recession_prob_chauvet_piger")
    spread = latest("term_spread_10y3m")

    fires = pd.DataFrame({
        "cfnai": level <= vote["cfnai_ma3_at_or_below"],
        "sahm": sahm >= vote["sahm_at_or_above"],
        "chauvet_piger": cp >= vote["chauvet_piger_at_or_above"],
    })
    available = pd.DataFrame({"cfnai": level.notna(), "sahm": sahm.notna(), "chauvet_piger": cp.notna()})
    fires = fires & available
    n_available = available.sum(axis=1)
    n_votes = fires.sum(axis=1)
    min_votes = int(vote["min_votes"])

    trend = float(clock["trend_level"])
    above = level >= trend
    rising = change >= 0
    clock_reading = pd.Series(np.where(above, np.where(rising, "Expansion", "Late-cycle"),
                                       np.where(rising, "Recovery", "Late-cycle")), index=df.index, dtype=object)
    is_recession = n_votes >= min_votes
    valid = (n_available >= min_votes) & level.notna() & change.notna()
    phase = _confirm_phases(clock_reading, valid, is_recession, int(clock.get("confirm_months", 1)))
    regime = phase.where(~is_recession, "Recession").where(valid)

    out = pd.DataFrame(index=df.index)
    out["regime"] = regime
    p_rec = (n_votes / n_available.replace(0, np.nan)).where(valid)
    for r in REGIMES:
        if r == "Recession":
            out[f"p_{r}"] = p_rec
        else:
            out[f"p_{r}"] = ((1.0 - p_rec) * (phase == r)).where(valid)
    out["confidence"] = pd.Series(
        [out.at[t, f"p_{r}"] if isinstance(r, str) else np.nan for t, r in regime.items()],
        index=df.index, dtype=float,
    )
    out["clock_reading"] = clock_reading.where(valid)
    out["cfnai_ma3"] = level
    out["cfnai_ma3_change"] = change
    out["sahm_rule_realtime_pp"] = sahm
    out["recession_prob_chauvet_piger"] = cp
    out["term_spread_10y3m"] = spread
    out["fires_cfnai"] = fires["cfnai"]
    out["fires_sahm"] = fires["sahm"]
    out["fires_chauvet_piger"] = fires["chauvet_piger"]
    out["recession_votes"] = n_votes
    out["indicators_available"] = n_available
    inverted = spread < warn["term_spread_inverted_below"]
    out["curve_inverted"] = inverted.where(spread.notna())
    out["months_inverted_last_12"] = inverted.astype(float).where(spread.notna()).rolling(
        int(warn["lookback_months"]), min_periods=1).sum()

    scores = compute_dimension_scores(df, level, score_weights, int(p["zscore_min_periods"]))
    return out.join(scores)


def _confirm_phases(reading: pd.Series, valid: pd.Series, is_recession: pd.Series, confirm_months: int) -> pd.Series:
    """
    Hold the current clock phase until a new reading persists for confirm_months
    consecutive months (1 = no confirmation). Recession is never delayed, and the
    first phase after a recession (or after a gap in the data) takes effect
    immediately. Each month depends only on earlier months.
    """
    out = pd.Series(np.nan, index=reading.index, dtype=object)
    current: Optional[str] = None
    candidate: Optional[str] = None
    streak = 0
    for t in reading.index:
        if not valid[t]:
            current, candidate, streak = None, None, 0
            continue
        if is_recession[t]:
            current, candidate, streak = "Recession", None, 0
            out[t] = reading[t]          # unused: the regime is Recession this month
            continue
        new = reading[t]
        if current in (None, "Recession") or new == current:
            current, candidate, streak = new, None, 0
        else:
            streak = streak + 1 if new == candidate else 1
            candidate = new
            if streak >= confirm_months:
                current, candidate, streak = new, None, 0
        out[t] = current
    return out


def compute_dimension_scores(
    df: pd.DataFrame,
    cfnai_ma3: pd.Series,
    score_weights: Optional[Dict[str, Dict[str, float]]] = None,
    min_periods: int = 24,
) -> pd.DataFrame:
    """
    Descriptive dimension scores for macro-view.json, each in [-3, 3].

    s_g is CFNAI-MA3 itself (already in standard-deviation units). s_pi, s_pol
    and s_fc are weighted expanding-window z-scores (only data up to each
    month), using the config weights. They inform the narrative and the TAA
    layer; they do not decide the regime.
    """
    weights = score_weights or DEFAULT_SCORE_WEIGHTS
    out = pd.DataFrame(index=df.index)
    out["s_g"] = cfnai_ma3.clip(-_SCORE_CLIP, _SCORE_CLIP)
    for dim, key in (("inflation", "s_pi"), ("monetary_policy", "s_pol"), ("financial_conditions", "s_fc")):
        total = pd.Series(0.0, index=df.index)
        ok = pd.Series(True, index=df.index)
        for col, w in weights[dim].items():
            if col not in df.columns:
                raise ValueError(f"df_macro is missing '{col}' needed for score {key}.")
            x = df[col].astype(float)
            mean = x.expanding(min_periods=min_periods).mean()
            std = x.expanding(min_periods=min_periods).std(ddof=1)
            z = (x - mean) / std.where(std > 1e-12)
            total = total + w * z.fillna(0.0)
            ok &= z.notna()
        out[key] = total.where(ok).clip(-_SCORE_CLIP, _SCORE_CLIP)
    return out


def build_regime_history(
    df_macro: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    as_of_date: Optional[str] = None,
) -> pd.Series:
    """Monthly regime labels up to as_of_date (NaN months dropped), for CMA Method 2."""
    frame = compute_regime_frame(df_macro, params)
    if as_of_date is not None:
        frame = frame.loc[: pd.Timestamp(as_of_date)]
    history = frame["regime"].dropna().astype(str)
    history.name = "regime"
    return history


# =============================================================================
# Current view (what macro-view.json records)
# =============================================================================

def classify_regime_published(
    df_macro: pd.DataFrame,
    as_of_date: str,
    params: Optional[Dict[str, Any]] = None,
    score_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    The regime as of as_of_date, with probabilities, the indicator readings and
    a plain-language rationale. Uses only rows dated on or before as_of_date.
    """
    _check_inputs(df_macro)
    as_of = pd.Timestamp(as_of_date)
    history = df_macro.loc[df_macro.index <= as_of]
    if history.empty:
        raise ValueError(f"No macro data on or before {as_of_date}.")
    frame = compute_regime_frame(history, params, score_weights)
    row = frame.iloc[-1]
    if not isinstance(row["regime"], str):
        raise ValueError(
            f"Cannot classify the regime at {frame.index[-1].date()}: need CFNAI-MA3 for the last "
            f"{_merge_params(params)['growth_clock']['direction_lookback_months']} months and at least "
            f"{_merge_params(params)['recession_vote']['min_votes']} of the 3 recession indicators."
        )
    scores = {k: float(row[k]) if pd.notna(row[k]) else 0.0 for k in ("s_g", "s_pi", "s_pol", "s_fc")}
    probabilities = {r: float(row[f"p_{r}"]) for r in REGIMES}

    def _num(x: Any) -> Optional[float]:
        return None if pd.isna(x) else float(x)

    indicators = {
        "cfnai_ma3": _num(row["cfnai_ma3"]),
        "cfnai_ma3_change": _num(row["cfnai_ma3_change"]),
        "clock_reading": row["clock_reading"] if isinstance(row["clock_reading"], str) else None,
        "sahm_rule_realtime_pp": _num(row["sahm_rule_realtime_pp"]),
        "recession_prob_chauvet_piger": _num(row["recession_prob_chauvet_piger"]),
        "term_spread_10y3m": _num(row["term_spread_10y3m"]),
        "recession_votes": int(row["recession_votes"]),
        "indicators_available": int(row["indicators_available"]),
        "curve_inverted": None if pd.isna(row["curve_inverted"]) else bool(row["curve_inverted"]),
        "months_inverted_last_12": _num(row["months_inverted_last_12"]),
    }
    view = {
        "regime": str(row["regime"]),
        "confidence": float(row["confidence"]),
        "regime_probabilities": probabilities,
        "scores": scores,
        "indicators": indicators,
        "rationale": _rationale(row, _merge_params(params)),
        "as_of_date": str(as_of_date),
        "data_date": str(frame.index[-1].date()),
        "method": "published_indicators",
    }
    logger.info("classify_regime_published: %s (confidence %.2f) as of %s",
                view["regime"], view["confidence"], as_of_date)
    return view


def _rationale(row: pd.Series, p: Dict[str, Any]) -> str:
    vote, clock = p["recession_vote"], p["growth_clock"]
    fired: List[str] = []
    if row["fires_cfnai"]:
        fired.append(f"CFNAI-MA3 {row['cfnai_ma3']:.2f} <= {vote['cfnai_ma3_at_or_below']:.2f}")
    if row["fires_sahm"]:
        fired.append(f"Sahm rule {row['sahm_rule_realtime_pp']:.2f}pp >= {vote['sahm_at_or_above']:.2f}pp")
    if row["fires_chauvet_piger"]:
        fired.append(f"Chauvet-Piger {row['recession_prob_chauvet_piger']:.0%} >= {vote['chauvet_piger_at_or_above']:.0%}")
    votes = f"{int(row['recession_votes'])} of {int(row['indicators_available'])} recession indicators fired"
    votes += f" ({'; '.join(fired)})." if fired else "."
    level, change = row["cfnai_ma3"], row["cfnai_ma3_change"]
    side = "above" if level >= clock["trend_level"] else "below"
    direction = "rising" if change >= 0 else "falling"
    clock_text = (f"CFNAI-MA3 is {level:.2f}, {side} trend and {direction} "
                  f"({change:+.2f} over {clock['direction_lookback_months']} months).")
    regime = row["regime"]
    if regime == "Recession":
        head = f"Recession: {votes} {clock_text}"
    else:
        head = f"{regime} from the business-cycle clock: {clock_text} {votes}"
        reading = row.get("clock_reading")
        if isinstance(reading, str) and reading != regime:
            head += (f" The clock now reads {reading}; it becomes the regime once it holds for "
                     f"{clock.get('confirm_months', 1)} consecutive months.")
    if pd.notna(row["term_spread_10y3m"]):
        months = int(row["months_inverted_last_12"]) if pd.notna(row["months_inverted_last_12"]) else 0
        curve = (f" Early warning: the 10y-3m curve is inverted at {row['term_spread_10y3m']:.2%}."
                 if row["curve_inverted"] else
                 f" The 10y-3m curve is {row['term_spread_10y3m']:.2%}")
        if not row["curve_inverted"]:
            curve += f" (inverted {months} of the last 12 months)." if months else " (not inverted in the last 12 months)."
        head += curve
    return head

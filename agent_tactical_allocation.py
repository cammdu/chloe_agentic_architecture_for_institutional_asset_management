# =============================================================================
# SELF-DRIVING PORTFOLIO: TACTICAL ASSET ALLOCATION (TAA) LAYER
# =============================================================================
# The rest of the repository answers a strategic question: what long-run policy
# mix of the 18 asset classes fits the Investment Policy Statement, given
# 3-year capital market assumptions. This module answers the tactical one:
# given what we can observe today, how far should the portfolio lean away from
# that policy mix over the next few months, and is the lean worth its cost?
#
# Design (see TAA_PARAMS in config.yaml):
#   1. Anchor      w_s: the strategic policy portfolio (from config, or from the
#                  SAA pipeline's final_weights.json).
#   2. Signals     momentum (time-series and cross-sectional), value, carry and
#                  the macro regime, each a point-in-time z-score per asset.
#   3. Alpha       Grinold's rule, alpha_i = IC * sigma_i * z_i (annualised).
#   4. Optimizer   max  h*alpha'd - (lambda/2)*h*d'Sigma d - c'|w - w_0|
#                  over d = w - w_s, subject to per-asset and per-category bands
#                  around w_s, an ex-ante tracking-error budget versus w_s, a
#                  turnover cap, full investment and long-only weights.
#                  h is the holding horizon in years, so the expected gain is
#                  measured over the same period the trading cost is paid for.
#   5. Backtest    monthly walk-forward: signals use data up to month t only,
#                  trades pay costs, and returns are earned in month t+1. All
#                  results are reported as active versus the policy portfolio.
#
# Everything here is deterministic numpy/pandas/scipy. No LLM calls, no API keys.
# =============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger: logging.Logger = logging.getLogger(__name__)

SIGNAL_FAMILIES: Tuple[str, ...] = ("momentum", "value", "carry", "regime")
CATEGORY_ORDER: Tuple[str, ...] = ("equity", "fixed_income", "real_assets", "cash")
CATEGORY_LABELS: Dict[str, str] = {
    "equity": "Equity",
    "fixed_income": "Fixed Income",
    "real_assets": "Real Assets",
    "cash": "Cash",
}


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class TAAInputs:
    """Monthly, month-end aligned inputs for the TAA layer.

    returns:      T x N simple monthly total returns, columns = asset class names.
    fundamentals: field -> T x N frame (cape_ratio, earnings_yield, dividend_yield).
    rates:        T x k Treasury yields and credit spreads, in decimals.
    macro:        T x m macro indicators (the df_macro_raw columns).
    categories:   asset class -> category key (equity, fixed_income, ...).
    """

    returns: pd.DataFrame
    fundamentals: Dict[str, pd.DataFrame]
    rates: pd.DataFrame
    macro: pd.DataFrame
    categories: Dict[str, str]
    # Optional external regime per month-end (e.g. from a dedicated regime
    # identifier). Labels must match TAA_PARAMS.REGIME_TILTS keys. It must be
    # point in time: the label at month t may only use data published by t.
    # When None, classify_regimes() derives it from the macro data.
    regimes: Optional[pd.Series] = None

    @property
    def assets(self) -> List[str]:
        return list(self.returns.columns)


def canonical_asset_order(config: Mapping[str, Any]) -> List[str]:
    """The 18 asset classes in the IPS order used by every weight vector in the repo."""
    classes = config["IPS_GOVERNANCE"]["ASSET_CLASSES"]
    return [a for cat in CATEGORY_ORDER for a in classes.get(cat, [])]


def asset_categories(config: Mapping[str, Any]) -> Dict[str, str]:
    classes = config["IPS_GOVERNANCE"]["ASSET_CLASSES"]
    return {a: cat for cat in CATEGORY_ORDER for a in classes.get(cat, [])}


def _panel_to_wide(panel: pd.DataFrame, column: str, ticker_to_asset: Mapping[str, str]) -> pd.DataFrame:
    """(date, ticker, investment_universe) panel column -> month-end date x asset class frame."""
    s = panel[column]
    wide = s.reset_index().pivot_table(index="date", columns="ticker", values=column, aggfunc="last")
    wide.index = pd.DatetimeIndex(wide.index) + pd.offsets.MonthEnd(0)
    wide = wide.groupby(level=0).last()
    return wide.rename(columns=ticker_to_asset)


def _to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index) + pd.offsets.MonthEnd(0)
    return out.groupby(level=0).last()


def build_taa_inputs(
    raw_dataframes: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    universe_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> TAAInputs:
    """Turn the pipeline's raw dataframes (synthetic or free real data) into TAAInputs."""
    universe_map = universe_map or config["REGISTRIES"]["universe_map"]
    order = canonical_asset_order(config)
    ticker_to_asset = {m["ticker"]: a for a, m in universe_map.items()}

    tri = _panel_to_wide(raw_dataframes["df_total_return_raw"], "total_return_index", ticker_to_asset)
    tri = tri.reindex(columns=order)
    returns = tri.pct_change(fill_method=None)
    returns = returns.iloc[1:]

    fundamentals: Dict[str, pd.DataFrame] = {}
    df_f = raw_dataframes.get("df_fundamentals_raw")
    if df_f is not None:
        for col in ("cape_ratio", "earnings_yield", "dividend_yield"):
            if col in df_f.columns:
                fundamentals[col] = _panel_to_wide(df_f, col, ticker_to_asset).reindex(
                    index=returns.index, columns=order
                )

    rates = _to_month_end(raw_dataframes["df_fixed_income_curves_spreads_raw"]).reindex(returns.index).ffill(limit=2)
    macro = _to_month_end(raw_dataframes["df_macro_raw"]).reindex(returns.index).ffill(limit=3)
    return TAAInputs(returns=returns, fundamentals=fundamentals, rates=rates, macro=macro,
                     categories=asset_categories(config))


# =============================================================================
# Point-in-time helpers
# =============================================================================

def expanding_zscore(df: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """z-score of each value against its own history up to and including that date."""
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std(ddof=1)
    return (df - mean) / std.where(std > 1e-12)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """z-score across assets at each date (rows), ignoring missing values."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    return df.sub(mean, axis=0).div(std.where(std > 1e-12), axis=0)


def _clip(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    return df.clip(lower=-cap, upper=cap)


def _interp_yield(rates_row: pd.Series, tenor: float) -> float:
    """Linear interpolation on the 3m / 2y / 10y / 30y Treasury curve."""
    pts = [(0.25, "ust_3m_yield"), (2.0, "ust_2y_yield"), (10.0, "ust_10y_yield"), (30.0, "ust_30y_yield")]
    xs = [t for t, c in pts if c in rates_row and pd.notna(rates_row[c])]
    ys = [rates_row[c] for t, c in pts if c in rates_row and pd.notna(rates_row[c])]
    if len(xs) < 2:
        return np.nan
    return float(np.interp(tenor, xs, ys))


# =============================================================================
# Signals (each returns a T x N frame of z-scores; NaN = no view)
# =============================================================================

def momentum_signal(returns: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Time-series and cross-sectional momentum.

    Time-series: the 12-1 month return (skipping the latest month, which tends to
    reverse) and a faster 3-month return, each divided by the asset's own
    volatility over the same window, so a 1.0 reading means a one-standard-
    deviation move. Cross-sectional: the 12-1 return ranked across assets.
    """
    lookback = int(params.get("lookback_months", 12))
    skip = int(params.get("skip_months", 1))
    fast = int(params.get("fast_lookback_months", 3))
    vol_win = int(params.get("vol_window_months", 36))
    w_ts = float(params.get("time_series_weight", 0.5))

    log_r = np.log1p(returns)
    vol_m = returns.rolling(vol_win, min_periods=max(12, vol_win // 2)).std()
    slow = log_r.shift(skip).rolling(lookback - skip, min_periods=lookback - skip).sum()
    slow_t = slow / (vol_m * np.sqrt(lookback - skip))
    fast_ret = log_r.rolling(fast, min_periods=fast).sum()
    fast_t = fast_ret / (vol_m * np.sqrt(fast))
    ts = 0.5 * slow_t + 0.5 * fast_t
    xs = cross_sectional_zscore(slow / vol_m)
    out = w_ts * ts + (1 - w_ts) * xs
    # Cash is the funding asset: its tiny volatility would blow up a vol-scaled trend
    funding = [a for a in out.columns if a in params.get("funding_assets", ["Cash"])]
    out.loc[:, funding] = np.nan
    return out


def value_signal(inputs: TAAInputs, params: Mapping[str, Any]) -> pd.DataFrame:
    """Valuation relative to each asset's own history (higher = cheaper).

    Equities: earnings yield minus the real 10-year Treasury yield (an excess
      earnings yield, a Fed-model style equity risk premium proxy).
    Treasuries: term spread (long/intermediate) and the real yield.
    Credit: option-adjusted spread for the matching index.
    Gold: minus the real 10-year yield (gold has no cash flow, so its opportunity
      cost is the real rate).
    Others: no view.
    """
    min_hist = int(params.get("min_history_months", 60))
    idx, cols = inputs.returns.index, inputs.assets
    out = pd.DataFrame(np.nan, index=idx, columns=cols)
    r = inputs.rates
    cpi_yoy = inputs.macro.get("cpi_yoy", pd.Series(np.nan, index=idx))
    lag = int(params.get("macro_publication_lag_months", 1))
    real_10y = r.get("ust_10y_yield") - cpi_yoy.shift(lag)
    term = r.get("ust_10y_yield") - r.get("ust_3m_yield")
    short_term = r.get("ust_2y_yield") - r.get("ust_3m_yield")

    z = lambda s: expanding_zscore(s.to_frame(), min_hist).iloc[:, 0]  # noqa: E731
    ey = inputs.fundamentals.get("earnings_yield")
    spread_cols = {a: v.get("spread_column") for a, v in params.get("fi_assumptions", {}).items()}

    for a in cols:
        cat = inputs.categories.get(a)
        if cat == "equity" and ey is not None:
            out[a] = z(ey[a] - real_10y)
        elif a in ("Intermediate Treasuries", "Long-Term Treasuries"):
            out[a] = 0.5 * z(term) + 0.5 * z(real_10y)
        elif a == "Short-Term Treasuries":
            out[a] = z(short_term)
        elif a == "International Sovereign Bonds":
            out[a] = z(real_10y)
        elif spread_cols.get(a) and spread_cols[a] in r:
            out[a] = z(r[spread_cols[a]])
        elif a == "Gold":
            out[a] = -z(real_10y)
    return out


def carry_signal(inputs: TAAInputs, params: Mapping[str, Any]) -> pd.DataFrame:
    """Carry per unit of risk: what the asset earns over cash if prices stand still.

    Equities: dividend yield minus the 3-month bill.
    Bonds: yield at the fund's typical maturity, plus credit spread where there is
      one, minus the 3-month bill.
    Divided by each asset's trailing volatility and z-scored across assets, so it
    ranks assets on carry-to-risk. Gold, commodities and REITs: no view.
    """
    vol_win = int(params.get("vol_window_months", 36))
    idx, cols = inputs.returns.index, inputs.assets
    vol_ann = inputs.returns.rolling(vol_win, min_periods=max(12, vol_win // 2)).std() * np.sqrt(12)
    raw = pd.DataFrame(np.nan, index=idx, columns=cols)
    r = inputs.rates
    bill = r.get("ust_3m_yield")
    dy = inputs.fundamentals.get("dividend_yield")
    fi = params.get("fi_assumptions", {})
    for a in cols:
        cat = inputs.categories.get(a)
        if cat == "equity" and dy is not None:
            raw[a] = dy[a] - bill
        elif cat == "fixed_income" and a in fi:
            tenor = float(fi[a].get("target_tenor", 5.0))
            y = r.apply(lambda row: _interp_yield(row, tenor), axis=1)
            spread_col = fi[a].get("spread_column")
            if spread_col and spread_col in r:
                y = y + r[spread_col]
            raw[a] = y - bill
    return cross_sectional_zscore(raw / vol_ann)


def classify_regimes(macro: pd.DataFrame, config: Mapping[str, Any], params: Mapping[str, Any]) -> pd.Series:
    """Point-in-time macro regime per month, using the SAA pipeline's own rules.

    Same scoring weights and thresholds as Task 20 (MACRO_SCORING_WEIGHTS,
    MACRO_REGIME_THRESHOLDS), with three tactical differences:
      * macro data is lagged by its publication delay, so month t only sees what
        was published by t;
      * when no regime's thresholds all hold, the regime with the smallest total
        threshold violation is chosen;
      * a change of regime must persist for `regime_persistence_months` before it
        is adopted, so the tilt does not flip on one noisy print.
    """
    mp = config["METHODOLOGY_PARAMS"]
    weights = mp["MACRO_SCORING_WEIGHTS"]
    keymap = mp["MACRO_DIMENSION_KEY_MAP"]
    thresholds = mp["MACRO_REGIME_THRESHOLDS"]
    regimes = list(mp["MACRO_REGIMES"])
    lag = int(params.get("macro_publication_lag_months", 1))
    persist = int(params.get("regime_persistence_months", 2))
    min_hist = int(params.get("regime_min_history_months", 36))

    cols = sorted({ind for dim in weights.values() for ind in dim})
    avail = [c for c in cols if c in macro.columns]
    zs = expanding_zscore(macro[avail].shift(lag), min_hist)
    scores = pd.DataFrame(index=macro.index)
    for dim, inds in weights.items():
        s = sum(zs[i] * w for i, w in inds.items() if i in zs) if any(i in zs for i in inds) else np.nan
        scores[keymap[dim]] = s

    raw: List[Optional[str]] = []
    for _, row in scores.iterrows():
        if row.isna().any():
            raw.append(None)
            continue
        best, best_v = None, np.inf
        for reg in regimes:
            v = 0.0
            for k, (lo, hi) in thresholds[reg].items():
                x = row[k]
                if lo is not None and x <= lo:
                    v += lo - x
                if hi is not None and x > hi:
                    v += x - hi
            if v < best_v:
                best, best_v = reg, v
        raw.append(best)

    out: List[Optional[str]] = []
    current: Optional[str] = None
    candidate: Optional[str] = None
    streak = 0
    for reg in raw:
        if reg is None:
            out.append(current)
            continue
        if current is None:
            current = reg
        elif reg != current:
            streak = streak + 1 if reg == candidate else 1
            candidate = reg
            if streak >= persist:
                current, candidate, streak = reg, None, 0
        else:
            candidate, streak = None, 0
        out.append(current)
    return pd.Series(out, index=macro.index, name="regime", dtype="object")


def regime_signal(regimes: pd.Series, assets: Sequence[str], tilt_map: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Map each month's regime to a per-asset tilt score (in z units) from config."""
    out = pd.DataFrame(np.nan, index=regimes.index, columns=list(assets))
    for reg, tilts in tilt_map.items():
        mask = regimes == reg
        if mask.any():
            out.loc[mask, :] = 0.0
            for a, v in tilts.items():
                if a in out.columns:
                    out.loc[mask, a] = float(v)
    return out


def compute_signals(inputs: TAAInputs, config: Mapping[str, Any]) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """All signal families as clipped z-score panels, plus the regime series."""
    p = config["TAA_PARAMS"]
    sp = dict(p["SIGNALS"])
    sp["fi_assumptions"] = config["METHODOLOGY_PARAMS"].get("FI_ASSUMPTIONS", {})
    cap = float(sp.get("zscore_cap", 3.0))
    if inputs.regimes is not None:
        regimes = inputs.regimes.reindex(inputs.returns.index, method="ffill").astype("object")
        unknown = set(regimes.dropna()) - set(p["REGIME_TILTS"])
        if unknown:
            raise ValueError(f"External regimes {sorted(unknown)} have no entry in TAA_PARAMS.REGIME_TILTS.")
    else:
        regimes = classify_regimes(inputs.macro, config, sp)
    sigs = {
        "momentum": momentum_signal(inputs.returns, sp),
        "value": value_signal(inputs, sp),
        "carry": carry_signal(inputs, sp),
        "regime": regime_signal(regimes, inputs.assets, p["REGIME_TILTS"]),
    }
    return {k: _clip(v, cap) for k, v in sigs.items()}, regimes


def composite_score(signals: Mapping[str, pd.DataFrame], family_weights: Mapping[str, float]) -> pd.DataFrame:
    """Weighted sum of signal families; a missing view counts as neutral (0)."""
    total = None
    for fam, w in family_weights.items():
        if w == 0 or fam not in signals:
            continue
        part = signals[fam].fillna(0.0) * float(w)
        total = part if total is None else total + part
    if total is None:
        any_sig = next(iter(signals.values()))
        return pd.DataFrame(0.0, index=any_sig.index, columns=any_sig.columns)
    return total


# =============================================================================
# Risk and alpha
# =============================================================================

def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity, annualised (x12)."""
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    mu = np.trace(s) / n
    target = mu * np.eye(n)
    d2 = np.sum((s - target) ** 2)
    # sum_t ||x_t x_t' - S||_F^2 = sum_t ||x_t||^4 - T ||S||_F^2
    b2 = (np.sum(np.sum(x ** 2, axis=1) ** 2) - t * np.sum(s ** 2)) / t ** 2
    shrink = 0.0 if d2 <= 0 else min(1.0, b2 / d2)
    cov = shrink * target + (1 - shrink) * s
    return 12.0 * (cov + cov.T) / 2


def grinold_alpha(score: np.ndarray, vol_ann: np.ndarray, ic: float) -> np.ndarray:
    """alpha_i = IC * sigma_i * z_i: the forecast an analyst with skill IC should make."""
    return ic * vol_ann * score


# =============================================================================
# Policy portfolio
# =============================================================================

def load_policy_weights(
    config: Mapping[str, Any],
    final_weights_path: Optional[str | Path] = None,
) -> pd.Series:
    """The strategic anchor: the SAA pipeline's final weights if given, else config."""
    order = canonical_asset_order(config)
    p = config["TAA_PARAMS"]
    source = p.get("POLICY_SOURCE", "config")
    path = final_weights_path or p.get("POLICY_WEIGHTS_PATH")
    if source == "saa_run" or final_weights_path is not None:
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text())
            w = np.asarray(data["weights"], dtype=float)
            if len(w) != len(order):
                raise ValueError(f"{path} has {len(w)} weights; expected {len(order)}.")
            logger.info("Policy portfolio loaded from the SAA run at %s.", path)
            return pd.Series(w / w.sum(), index=order, name="policy")
        logger.warning("No SAA final_weights.json found; using the policy weights in config.yaml.")
    w = pd.Series({a: float(p["POLICY_WEIGHTS"].get(a, 0.0)) for a in order}, name="policy")
    if abs(w.sum() - 1.0) > 1e-6:
        raise ValueError(f"TAA_PARAMS.POLICY_WEIGHTS sum to {w.sum():.4f}, not 1.")
    return w


# =============================================================================
# Optimizer
# =============================================================================

@dataclass
class TiltResult:
    weights: np.ndarray
    status: str
    expected_active_return: float
    ex_ante_te: float
    turnover: float
    cost: float


def optimise_tilt(
    alpha: np.ndarray,
    cov: np.ndarray,
    w_policy: np.ndarray,
    w_current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cost_rate: np.ndarray,
    category_index: Mapping[str, np.ndarray],
    category_band: Mapping[str, float],
    te_budget: float,
    turnover_cap: float,
    risk_aversion: float,
    horizon_years: float,
) -> TiltResult:
    """Cost-aware tilt around the policy portfolio.

    Variables are the buys u >= 0 and sells v >= 0 relative to current holdings,
    so w = w_current + u - v and the L1 trading cost c'(u + v) is linear and the
    problem is smooth. Solved with SLSQP. If the turnover cap leaves no feasible
    point (for example after a big drift), it is lifted for that month.
    """
    n = len(w_policy)
    h = horizon_years

    def unpack(x: np.ndarray) -> np.ndarray:
        return w_current + x[:n] - x[n:]

    def objective(x: np.ndarray) -> float:
        d = unpack(x) - w_policy
        gain = h * alpha @ d - 0.5 * risk_aversion * h * d @ cov @ d
        return -(gain - cost_rate @ (x[:n] + x[n:]))

    def grad(x: np.ndarray) -> np.ndarray:
        d = unpack(x) - w_policy
        gw = h * alpha - risk_aversion * h * cov @ d
        return -np.concatenate([gw - cost_rate, -gw - cost_rate])

    base = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n] - x[n:]) + w_current.sum() - 1.0,
         "jac": lambda x: np.concatenate([np.ones(n), -np.ones(n)])},
        {"type": "ineq", "fun": lambda x: unpack(x) - lower,
         "jac": lambda x: np.hstack([np.eye(n), -np.eye(n)])},
        {"type": "ineq", "fun": lambda x: upper - unpack(x),
         "jac": lambda x: np.hstack([-np.eye(n), np.eye(n)])},
        {"type": "ineq", "fun": lambda x: te_budget ** 2 - (unpack(x) - w_policy) @ cov @ (unpack(x) - w_policy),
         "jac": lambda x: (lambda g: np.concatenate([g, -g]))(-2 * cov @ (unpack(x) - w_policy))},
    ]
    for cat, idx in category_index.items():
        band = category_band.get(cat)
        if band is None or len(idx) == 0:
            continue
        m = np.zeros(n)
        m[idx] = 1.0
        pol = float(w_policy[idx].sum())
        jm = np.concatenate([m, -m])
        base.append({"type": "ineq", "fun": lambda x, m=m, pol=pol, b=band: b - (m @ unpack(x) - pol), "jac": lambda x, jm=jm: -jm})
        base.append({"type": "ineq", "fun": lambda x, m=m, pol=pol, b=band: b + (m @ unpack(x) - pol), "jac": lambda x, jm=jm: jm})
    turnover_con = {"type": "ineq", "fun": lambda x: 2 * turnover_cap - np.sum(x),
                    "jac": lambda x: -np.ones(2 * n)}

    # Start from the policy portfolio (always inside the bands and TE budget)
    move = w_policy - w_current
    x0 = np.concatenate([np.maximum(move, 0), np.maximum(-move, 0)])
    bounds = [(0.0, 1.0)] * (2 * n)

    status = "optimal"
    attempts = [base + [turnover_con], base]
    res = None
    for i, cons in enumerate(attempts):
        res = minimize(objective, x0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success and _feasible(unpack(res.x), w_policy, lower, upper, cov, te_budget, category_index, category_band):
            if i == 1:
                status = "turnover_cap_lifted"
            break
    else:
        status = "fallback_to_policy"

    w = w_policy.copy() if status == "fallback_to_policy" else _clean(unpack(res.x), lower, upper)
    d = w - w_policy
    trade = np.abs(w - w_current)
    return TiltResult(
        weights=w,
        status=status,
        expected_active_return=float(alpha @ d),
        ex_ante_te=float(np.sqrt(max(d @ cov @ d, 0.0))),
        turnover=float(0.5 * trade.sum()),
        cost=float(cost_rate @ trade),
    )


def _clean(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    w = np.clip(w, lower, upper)
    w[np.abs(w) < 1e-10] = 0.0
    return w / w.sum()


def _feasible(w, w_policy, lower, upper, cov, te, category_index, category_band, tol=1e-6) -> bool:
    d = w - w_policy
    if np.any(w < lower - tol) or np.any(w > upper + tol) or abs(w.sum() - 1) > 1e-6:
        return False
    if d @ cov @ d > te ** 2 * (1 + 1e-4) + 1e-10:
        return False
    for cat, idx in category_index.items():
        b = category_band.get(cat)
        if b is not None and abs(d[idx].sum()) > b + tol:
            return False
    return True


# =============================================================================
# Walk-forward backtest
# =============================================================================

@dataclass
class BacktestResult:
    """Monthly walk-forward record of the TAA strategy versus the policy portfolio."""

    weights: pd.DataFrame              # TAA weights held during month t+1 (indexed by t)
    policy_weights: pd.DataFrame
    taa_returns: pd.Series             # net of costs, earned in month t+1 (indexed by t+1)
    policy_returns: pd.Series          # net of costs
    turnover: pd.Series
    costs: pd.Series
    ex_ante_te: pd.Series
    regimes: pd.Series
    status: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def active_returns(self) -> pd.Series:
        return (self.taa_returns - self.policy_returns).rename("active")


def _bands(config: Mapping[str, Any], assets: Sequence[str], w_policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b = config["TAA_PARAMS"]["BANDS"]
    per = b.get("per_asset_overrides", {}) or {}
    width = np.array([float(per.get(a, b["per_asset"])) for a in assets])
    lower = np.maximum(0.0, w_policy - width)
    upper = w_policy + width
    ips_max = config["IPS_GOVERNANCE"]["IMPLEMENTATION_CONSTRAINTS"].get("max_weight_per_asset")
    if b.get("respect_ips_max_weight", True) and ips_max is not None:
        upper = np.minimum(upper, np.maximum(float(ips_max), w_policy))
    return lower, upper


def run_taa_backtest(
    inputs: TAAInputs,
    config: Mapping[str, Any],
    policy: Optional[pd.Series] = None,
    signal_weights: Optional[Mapping[str, float]] = None,
    signals: Optional[Tuple[Dict[str, pd.DataFrame], pd.Series]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    """Monthly walk-forward TAA backtest, net of trading costs.

    At each month-end t: the covariance uses the last `cov_window_months` returns
    up to t, signals use data up to t, the optimizer trades from the drifted
    holdings to the new weights paying costs, and the portfolio earns month t+1.
    The policy portfolio is rebalanced to its weights every month and pays its
    own (smaller) rebalancing costs, so the comparison is like for like.
    Assets with too little history are held at zero and the policy weight is
    spread over the rest.
    """
    p = config["TAA_PARAMS"]
    rp = p["RISK"]
    assets = inputs.assets
    n = len(assets)
    policy = (policy if policy is not None else load_policy_weights(config)).reindex(assets).fillna(0.0)
    sig, regimes = signals if signals is not None else compute_signals(inputs, config)
    fam_w = signal_weights or p["SIGNAL_WEIGHTS"]
    score = composite_score(sig, fam_w)

    cov_win = int(rp["cov_window_months"])
    min_hist = int(rp.get("min_history_months", cov_win))
    ic = float(rp["information_coefficient"])
    lam = float(rp["risk_aversion"])
    te = float(rp["tracking_error_budget"])
    tcap = float(p["TURNOVER"]["max_one_way_per_rebalance"])
    h = float(p["HORIZON_MONTHS"]) / 12.0
    costs_bps = p["TRANSACTION_COSTS_BPS"]
    cost_rate = np.array([float(costs_bps.get(a, costs_bps.get("default", 10.0))) / 1e4 for a in assets])
    cat_band = {k: float(v) for k, v in p["BANDS"]["per_category"].items()}

    R = inputs.returns
    dates = R.index
    first = max(cov_win, min_hist)
    t_idx = list(range(first, len(dates) - 1))
    if start:
        t_idx = [i for i in t_idx if dates[i] >= pd.Timestamp(start)]
    if end:
        t_idx = [i for i in t_idx if dates[i + 1] <= pd.Timestamp(end)]
    if not t_idx:
        raise ValueError("Not enough history for the TAA backtest; need more months than cov_window_months.")

    w_hold = None
    pol_hold = None
    rec_w, rec_p, rec_ret, rec_pret, rec_to, rec_cost, rec_te, rec_status = [], [], [], [], [], [], [], []
    for i in t_idx:
        t, t1 = dates[i], dates[i + 1]
        hist = R.iloc[i - min_hist + 1: i + 1]
        avail = hist.notna().all().to_numpy()
        ws = policy.to_numpy() * avail
        ws = ws / ws.sum() if ws.sum() > 0 else avail / avail.sum()

        win = R.iloc[i - cov_win + 1: i + 1].to_numpy()[:, avail]
        cov_a = ledoit_wolf_covariance(win)
        cov = np.zeros((n, n))
        cov[np.ix_(avail, avail)] = cov_a
        vol = np.sqrt(np.diag(cov))
        z = score.loc[t].reindex(assets).fillna(0.0).to_numpy() * avail
        alpha = grinold_alpha(z, vol, ic)

        lower, upper = _bands(config, assets, ws)
        lower, upper = lower * avail, upper * avail
        cat_idx = {CATEGORY_LABELS.get(c, c): np.array([k for k, a in enumerate(assets) if inputs.categories.get(a) == c and avail[k]], dtype=int)
                   for c in CATEGORY_ORDER}
        cat_idx = {c: idx for c, idx in cat_idx.items() if len(idx)}
        w0 = ws.copy() if w_hold is None else np.where(avail, w_hold, 0.0)
        w0 = w0 / w0.sum() if w0.sum() > 0 else ws.copy()

        res = optimise_tilt(alpha, cov, ws, w0, lower, upper, cost_rate, cat_idx, cat_band, te, tcap, lam, h)
        w = res.weights

        p0 = ws.copy() if pol_hold is None else np.where(avail, pol_hold, 0.0)
        p0 = p0 / p0.sum() if p0.sum() > 0 else ws.copy()
        pol_cost = float(cost_rate @ np.abs(ws - p0))

        r1 = R.loc[t1].reindex(assets).fillna(0.0).to_numpy()
        gross = float(w @ r1)
        pgross = float(ws @ r1)
        rec_ret.append(gross - res.cost)
        rec_pret.append(pgross - pol_cost)
        rec_w.append(w)
        rec_p.append(ws)
        rec_to.append(res.turnover)
        rec_cost.append(res.cost)
        rec_te.append(res.ex_ante_te)
        rec_status.append(res.status)

        # Drift the holdings through month t+1
        w_hold = w * (1 + r1) / (1 + gross)
        pol_hold = ws * (1 + r1) / (1 + pgross)

    t_dates = dates[t_idx]
    r_dates = dates[[i + 1 for i in t_idx]]
    result = BacktestResult(
        weights=pd.DataFrame(rec_w, index=t_dates, columns=assets),
        policy_weights=pd.DataFrame(rec_p, index=t_dates, columns=assets),
        taa_returns=pd.Series(rec_ret, index=r_dates, name="taa"),
        policy_returns=pd.Series(rec_pret, index=r_dates, name="policy"),
        turnover=pd.Series(rec_to, index=t_dates, name="turnover"),
        costs=pd.Series(rec_cost, index=t_dates, name="cost"),
        ex_ante_te=pd.Series(rec_te, index=t_dates, name="ex_ante_te"),
        regimes=regimes.reindex(t_dates),
        status=pd.Series(rec_status, index=t_dates, name="status"),
    )
    result.metrics = backtest_metrics(result, inputs)
    return result


def _max_drawdown(r: pd.Series) -> float:
    v = (1 + r).cumprod()
    return float((v / v.cummax() - 1).min())


def backtest_metrics(res: BacktestResult, inputs: Optional[TAAInputs] = None) -> Dict[str, float]:
    a = res.active_returns
    n = len(a)
    te = float(a.std(ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    ann_active = float(a.mean() * 12)
    act_w = (res.weights - res.policy_weights).abs().sum(axis=1) / 2
    out = {
        "months": n,
        "start": str(res.taa_returns.index[0].date()),
        "end": str(res.taa_returns.index[-1].date()),
        "taa_return_ann": float((1 + res.taa_returns).prod() ** (12 / n) - 1),
        "policy_return_ann": float((1 + res.policy_returns).prod() ** (12 / n) - 1),
        "taa_vol_ann": float(res.taa_returns.std(ddof=1) * np.sqrt(12)),
        "policy_vol_ann": float(res.policy_returns.std(ddof=1) * np.sqrt(12)),
        "active_return_ann": ann_active,
        "tracking_error_realised": te,
        "information_ratio": ann_active / te if te and te > 0 else float("nan"),
        "hit_rate": float((a > 0).mean()),
        "turnover_one_way_ann": float(res.turnover.sum() * 12 / n),
        "cost_drag_ann": float(res.costs.sum() * 12 / n),
        "avg_active_share": float(act_w.mean()),
        "avg_ex_ante_te": float(res.ex_ante_te.mean()),
        "max_active_drawdown": _max_drawdown(a),
        "taa_max_drawdown": _max_drawdown(res.taa_returns),
        "policy_max_drawdown": _max_drawdown(res.policy_returns),
        "months_turnover_cap_lifted": int((res.status == "turnover_cap_lifted").sum()),
        "months_fallback_to_policy": int((res.status == "fallback_to_policy").sum()),
    }
    if inputs is not None and "ust_3m_yield" in inputs.rates:
        rf = inputs.rates["ust_3m_yield"].shift(1).reindex(res.taa_returns.index).fillna(0.0) / 12
        for k, s in (("taa", res.taa_returns), ("policy", res.policy_returns)):
            ex = s - rf
            sd = ex.std(ddof=1)
            out[f"{k}_sharpe"] = float(ex.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan")
    return out


def signal_attribution(
    inputs: TAAInputs,
    config: Mapping[str, Any],
    policy: Optional[pd.Series] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run the backtest with each signal family on its own, and with all of them.

    The rows show which signal earns its keep after costs. Reuses one signal
    computation for every run.
    """
    sig = compute_signals(inputs, config)
    rows = {}
    base = config["TAA_PARAMS"]["SIGNAL_WEIGHTS"]
    runs = {"all signals": dict(base)}
    runs.update({f"{f} only": {f: 1.0} for f in SIGNAL_FAMILIES if base.get(f, 0) > 0})
    for name, fw in runs.items():
        res = run_taa_backtest(inputs, config, policy=policy, signal_weights=fw, signals=sig, **kwargs)
        m = res.metrics
        rows[name] = {k: m[k] for k in ("active_return_ann", "tracking_error_realised", "information_ratio",
                                        "hit_rate", "turnover_one_way_ann", "cost_drag_ann")}
    return pd.DataFrame(rows).T


# =============================================================================
# Current recommendation (what to do at AS_OF_DATE)
# =============================================================================

def current_recommendation(
    inputs: TAAInputs,
    config: Mapping[str, Any],
    policy: Optional[pd.Series] = None,
    current_weights: Optional[pd.Series] = None,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """The tactical weights for the next period, with the signals behind each tilt.

    current_weights are the holdings today; if not given, the portfolio is
    assumed to sit at the policy weights.
    """
    p = config["TAA_PARAMS"]
    rp = p["RISK"]
    assets = inputs.assets
    policy = (policy if policy is not None else load_policy_weights(config)).reindex(assets).fillna(0.0)
    sig, regimes = compute_signals(inputs, config)
    t = pd.Timestamp(as_of or config.get("AS_OF_DATE") or inputs.returns.index[-1]) + pd.offsets.MonthEnd(0)
    t = inputs.returns.index[inputs.returns.index <= t][-1]
    i = inputs.returns.index.get_loc(t)
    cov_win = int(rp["cov_window_months"])
    min_hist = int(rp.get("min_history_months", cov_win))
    if i + 1 < max(cov_win, min_hist):
        raise ValueError("Not enough history before the as-of date.")

    hist = inputs.returns.iloc[i - min_hist + 1: i + 1]
    avail = hist.notna().all().to_numpy()
    ws = policy.to_numpy() * avail
    ws = ws / ws.sum()
    win = inputs.returns.iloc[i - cov_win + 1: i + 1].to_numpy()[:, avail]
    n = len(assets)
    cov = np.zeros((n, n))
    cov[np.ix_(avail, avail)] = ledoit_wolf_covariance(win)
    vol = np.sqrt(np.diag(cov))
    fam_w = p["SIGNAL_WEIGHTS"]
    z_fam = {f: sig[f].loc[t].reindex(assets) for f in sig}
    z = composite_score({f: s.loc[[t]] for f, s in sig.items()}, fam_w).iloc[0].reindex(assets).fillna(0).to_numpy() * avail
    alpha = grinold_alpha(z, vol, float(rp["information_coefficient"]))
    lower, upper = _bands(config, assets, ws)
    lower, upper = lower * avail, upper * avail
    cat_idx = {CATEGORY_LABELS[c]: np.array([k for k, a in enumerate(assets) if inputs.categories.get(a) == c and avail[k]], dtype=int)
               for c in CATEGORY_ORDER}
    cat_idx = {c: idx for c, idx in cat_idx.items() if len(idx)}
    w0 = ws if current_weights is None else current_weights.reindex(assets).fillna(0.0).to_numpy()
    costs_bps = p["TRANSACTION_COSTS_BPS"]
    cost_rate = np.array([float(costs_bps.get(a, costs_bps.get("default", 10.0))) / 1e4 for a in assets])
    res = optimise_tilt(alpha, cov, ws, w0 / w0.sum(), lower, upper, cost_rate, cat_idx,
                        {k: float(v) for k, v in p["BANDS"]["per_category"].items()},
                        float(rp["tracking_error_budget"]), float(p["TURNOVER"]["max_one_way_per_rebalance"]),
                        float(rp["risk_aversion"]), float(p["HORIZON_MONTHS"]) / 12.0)

    rows = []
    for k, a in enumerate(assets):
        active = float(res.weights[k] - ws[k])
        rows.append({
            "asset_class": a,
            "category": CATEGORY_LABELS.get(inputs.categories.get(a, ""), ""),
            "policy_weight": float(ws[k]),
            "tactical_weight": float(res.weights[k]),
            "active_weight": active,
            "view": "overweight" if active > 0.0025 else "underweight" if active < -0.0025 else "neutral",
            "composite_z": float(z[k]),
            "signals_z": {f: (None if pd.isna(z_fam[f][a]) else float(z_fam[f][a])) for f in z_fam},
            "alpha_ann": float(alpha[k]),
            "band": [float(lower[k]), float(upper[k])],
            "available": bool(avail[k]),
        })
    return {
        "as_of_date": str(t.date()),
        "horizon_months": int(p["HORIZON_MONTHS"]),
        "regime": None if pd.isna(regimes.loc[t]) else regimes.loc[t],
        "optimizer_status": res.status,
        "expected_active_return_ann": res.expected_active_return,
        "ex_ante_tracking_error": res.ex_ante_te,
        "tracking_error_budget": float(rp["tracking_error_budget"]),
        "one_way_turnover": res.turnover,
        "trading_cost": res.cost,
        "signal_weights": dict(fam_w),
        "assets": rows,
    }


# =============================================================================
# Reporting
# =============================================================================

def _pct(x: float, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def write_taa_report(
    output_dir: str | Path,
    recommendation: Dict[str, Any],
    backtest: Optional[BacktestResult] = None,
    attribution: Optional[pd.DataFrame] = None,
    data_label: str = "",
) -> Dict[str, Path]:
    """Write taa_recommendation.json, taa_backtest.json and taa_report.md."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    paths["recommendation"] = out / "taa_recommendation.json"
    paths["recommendation"].write_text(json.dumps(recommendation, indent=2, default=float))

    lines = [f"# Tactical asset allocation report", ""]
    if data_label:
        lines += [f"*Data: {data_label}*", ""]
    rec = recommendation
    lines += [
        f"## Recommendation at {rec['as_of_date']} (horizon {rec['horizon_months']} months)", "",
        f"Macro regime: **{rec['regime'] or 'unclassified'}**. "
        f"Expected active return {_pct(rec['expected_active_return_ann'])} a year at an ex-ante tracking error of "
        f"{_pct(rec['ex_ante_tracking_error'])} (budget {_pct(rec['tracking_error_budget'])}). "
        f"Trading to these weights turns over {_pct(rec['one_way_turnover'], 1)} of the portfolio and costs "
        f"{rec['trading_cost'] * 1e4:.1f} bps.", "",
        "| Asset class | Policy | Tactical | Active | View | Momentum | Value | Carry | Regime |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rec["assets"]:
        s = r["signals_z"]
        f = lambda v: "" if v is None else f"{v:+.2f}"  # noqa: E731
        lines.append(
            f"| {r['asset_class']} | {_pct(r['policy_weight'], 1)} | {_pct(r['tactical_weight'], 1)} | "
            f"{100 * r['active_weight']:+.1f}pp | {r['view']} | {f(s.get('momentum'))} | {f(s.get('value'))} | "
            f"{f(s.get('carry'))} | {f(s.get('regime'))} |"
        )
    lines += ["", "Signal columns are z-scores: positive favours an overweight.", ""]

    if backtest is not None:
        m = backtest.metrics
        paths["backtest"] = out / "taa_backtest.json"
        paths["backtest"].write_text(json.dumps({
            "metrics": m,
            "monthly": pd.DataFrame({
                "taa_return": backtest.taa_returns,
                "policy_return": backtest.policy_returns,
                "active_return": backtest.active_returns,
            }).reset_index(names="date").assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        }, indent=2, default=float))
        lines += [
            f"## Walk-forward backtest, {m['start']} to {m['end']} ({m['months']} months, net of costs)", "",
            "| | TAA | Policy |", "|---|---:|---:|",
            f"| Return (annualised) | {_pct(m['taa_return_ann'])} | {_pct(m['policy_return_ann'])} |",
            f"| Volatility | {_pct(m['taa_vol_ann'])} | {_pct(m['policy_vol_ann'])} |",
            f"| Sharpe ratio | {m.get('taa_sharpe', float('nan')):.2f} | {m.get('policy_sharpe', float('nan')):.2f} |",
            f"| Max drawdown | {_pct(m['taa_max_drawdown'])} | {_pct(m['policy_max_drawdown'])} |",
            "",
            "| Active metric | Value |", "|---|---:|",
            f"| Active return (annualised) | {_pct(m['active_return_ann'])} |",
            f"| Realised tracking error | {_pct(m['tracking_error_realised'])} |",
            f"| Information ratio | {m['information_ratio']:.2f} |",
            f"| Hit rate (months beating policy) | {_pct(m['hit_rate'], 0)} |",
            f"| One-way turnover a year | {_pct(m['turnover_one_way_ann'], 0)} |",
            f"| Cost drag a year | {_pct(m['cost_drag_ann'])} |",
            f"| Average active share | {_pct(m['avg_active_share'], 1)} |",
            f"| Worst active drawdown | {_pct(m['max_active_drawdown'])} |",
            "",
        ]
        if m["months_turnover_cap_lifted"] or m["months_fallback_to_policy"]:
            lines += [f"The turnover cap had to be lifted in {m['months_turnover_cap_lifted']} months and the "
                      f"optimizer fell back to policy in {m['months_fallback_to_policy']}.", ""]
    if attribution is not None:
        lines += ["## Which signals earn their keep (each run on its own, net of costs)", "",
                  "| Run | Active return | Tracking error | IR | Hit rate | Turnover | Cost drag |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, r in attribution.iterrows():
            lines.append(f"| {name} | {_pct(r['active_return_ann'])} | {_pct(r['tracking_error_realised'])} | "
                         f"{r['information_ratio']:.2f} | {_pct(r['hit_rate'], 0)} | "
                         f"{_pct(r['turnover_one_way_ann'], 0)} | {_pct(r['cost_drag_ann'])} |")
        lines.append("")
    paths["report"] = out / "taa_report.md"
    paths["report"].write_text("\n".join(lines))
    return paths


# =============================================================================
# Synthetic data for trying the layer without downloads
# =============================================================================

def make_synthetic_taa_inputs(config: Mapping[str, Any], n_months: int = 360, seed: int = 7,
                              end: str = "2026-03-31") -> TAAInputs:
    """Plausible synthetic monthly data for checking the mechanics.

    Returns come from a one-factor model with a slow-moving expected-return
    component (so trends exist, by construction), yields and spreads follow mean-
    reverting walks, and macro series are noise around sensible levels. Results on
    this data show that the plumbing works, not that any signal has an edge.
    """
    rng = np.random.default_rng(seed)
    assets = canonical_asset_order(config)
    cats = asset_categories(config)
    idx = pd.date_range(end=end, periods=n_months, freq="ME")
    base = {"equity": (0.07, 0.16, 1.0), "fixed_income": (0.035, 0.06, 0.15),
            "real_assets": (0.05, 0.17, 0.5), "cash": (0.025, 0.005, 0.0)}
    mkt = rng.normal(0, 0.15 / np.sqrt(12), n_months)
    rets = {}
    for a in assets:
        mu, vol, beta = base[cats[a]]
        if a == "Cash":
            rets[a] = np.full(n_months, mu / 12) + rng.normal(0, 0.0005, n_months)
            continue
        drift = np.zeros(n_months)
        for t in range(1, n_months):
            drift[t] = 0.95 * drift[t - 1] + rng.normal(0, 0.0006 * vol / 0.16)
        idio = np.sqrt(max(vol ** 2 - (beta * 0.15) ** 2, (0.3 * vol) ** 2)) / np.sqrt(12)
        rets[a] = mu / 12 + drift + beta * mkt + rng.normal(0, idio, n_months)
    returns = pd.DataFrame(rets, index=idx)

    def ou(level, sd, n=n_months, k=0.05):
        x = np.empty(n)
        x[0] = level
        for t in range(1, n):
            x[t] = x[t - 1] + k * (level - x[t - 1]) + rng.normal(0, sd)
        return x

    y3m = np.clip(ou(0.03, 0.002), 0.0, None)
    rates = pd.DataFrame({
        "ust_3m_yield": y3m,
        "ust_2y_yield": y3m + ou(0.004, 0.001),
        "ust_10y_yield": y3m + ou(0.012, 0.0015),
        "ust_30y_yield": y3m + ou(0.016, 0.0015),
        "ig_oas": np.clip(ou(0.013, 0.0008), 0.004, None),
        "hy_oas": np.clip(ou(0.045, 0.003), 0.02, None),
        "em_spread": np.clip(ou(0.035, 0.002), 0.01, None),
    }, index=idx)
    macro = pd.DataFrame({
        "real_gdp_growth_rev": ou(0.022, 0.003),
        "nonfarm_payrolls_mom": ou(150000, 30000),
        "cpi_yoy": ou(0.025, 0.002),
        "cpi_mom": ou(0.002, 0.0008),
        "fed_funds_rate": np.clip(y3m + rng.normal(0, 0.001, n_months), 0, None),
        "financial_conditions_index": ou(0.0, 0.08),
    }, index=idx)
    macro["fed_funds_3m_change"] = macro["fed_funds_rate"].diff(3).fillna(0.0)
    macro["financial_conditions_mom"] = macro["financial_conditions_index"].diff().fillna(0.0)
    cape = pd.DataFrame({a: ou(25.0, 0.6) for a in assets}, index=idx)
    fundamentals = {
        "cape_ratio": cape,
        "earnings_yield": 1 / cape,
        "dividend_yield": pd.DataFrame({a: np.clip(ou(0.02, 0.0007), 0.005, None) for a in assets}, index=idx),
    }
    for k in fundamentals:
        fundamentals[k].loc[:, [a for a in assets if cats[a] != "equity"]] = np.nan
    return TAAInputs(returns=returns, fundamentals=fundamentals, rates=rates, macro=macro, categories=cats)


# =============================================================================
# One-call entry point
# =============================================================================

def run_tactical_allocation(
    config: Mapping[str, Any],
    inputs: TAAInputs,
    output_dir: str | Path,
    final_weights_path: Optional[str | Path] = None,
    current_weights: Optional[pd.Series] = None,
    with_attribution: bool = True,
    data_label: str = "",
) -> Dict[str, Any]:
    """Recommendation, walk-forward backtest, signal attribution and report in one call."""
    policy = load_policy_weights(config, final_weights_path)
    rec = current_recommendation(inputs, config, policy=policy, current_weights=current_weights)
    bt = run_taa_backtest(inputs, config, policy=policy)
    attr = signal_attribution(inputs, config, policy=policy) if with_attribution else None
    paths = write_taa_report(output_dir, rec, bt, attr, data_label=data_label)
    return {"recommendation": rec, "backtest": bt, "attribution": attr, "paths": paths}

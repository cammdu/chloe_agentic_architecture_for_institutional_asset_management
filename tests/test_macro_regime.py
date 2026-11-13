"""Tests for agent_macro_regime: published-indicator regime model (no network, no API calls)."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_macro_regime import (  # noqa: E402
    REGIMES,
    build_regime_history,
    classify_regime_published,
    compute_regime_frame,
)


def _macro(cfnai, sahm, cp, spread=None, seed=0):
    n = len(cfnai)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame({
        "cfnai_ma3": cfnai,
        "sahm_rule_realtime_pp": sahm,
        "recession_prob_chauvet_piger": cp,
        "term_spread_10y3m": spread if spread is not None else np.full(n, 0.01),
        "cpi_yoy": rng.normal(0.02, 0.01, n),
        "cpi_mom": rng.normal(0.002, 0.001, n),
        "fed_funds_rate": rng.uniform(0.0, 0.05, n),
        "fed_funds_3m_change": rng.normal(0.0, 0.002, n),
        "financial_conditions_index": rng.normal(0.0, 1.0, n),
        "financial_conditions_mom": rng.normal(0.0, 0.3, n),
    }, index=idx)


def _cycle():
    """60 months: slowdown, recession (months 24-35), recovery, expansion."""
    t = np.arange(60)
    cfnai = np.concatenate([np.linspace(0.5, -0.2, 20), np.linspace(-0.2, -1.5, 10), np.linspace(-1.5, 0.4, 30)])
    sahm = np.where((t >= 24) & (t < 36), 0.8, 0.1)
    cp = np.where((t >= 25) & (t < 34), 0.9, 0.05)
    return _macro(cfnai, sahm, cp)


def test_clock_phases():
    flat = np.zeros(10)
    rising_above = _macro(np.linspace(0.1, 1.0, 10), flat, flat)
    falling_above = _macro(np.linspace(1.0, 0.1, 10), flat, flat)
    rising_below = _macro(np.linspace(-0.6, -0.1, 10), flat, flat)
    falling_below = _macro(np.linspace(-0.1, -0.6, 10), flat, flat)
    assert compute_regime_frame(rising_above)["regime"].iloc[-1] == "Expansion"
    assert compute_regime_frame(falling_above)["regime"].iloc[-1] == "Late-cycle"
    assert compute_regime_frame(rising_below)["regime"].iloc[-1] == "Recovery"
    assert compute_regime_frame(falling_below)["regime"].iloc[-1] == "Late-cycle"


def test_confirmation_holds_phase_until_it_persists():
    flat = np.zeros(14)
    # Above trend: rising for 10 months, then one falling month, then rising again
    level = np.concatenate([np.linspace(0.1, 1.0, 10), [0.2], np.linspace(1.1, 1.4, 3)])
    frame = compute_regime_frame(_macro(level, flat, flat), params={"growth_clock": {"direction_lookback_months": 1}})
    assert frame["clock_reading"].iloc[10] == "Late-cycle"
    assert frame["regime"].iloc[10] == "Expansion"            # one month is not enough
    # Two falling months in a row do switch
    level2 = np.concatenate([np.linspace(0.1, 1.0, 10), [0.8, 0.6]])
    frame2 = compute_regime_frame(_macro(level2, flat[:12], flat[:12]), params={"growth_clock": {"direction_lookback_months": 1}})
    assert list(frame2["regime"].iloc[-2:]) == ["Expansion", "Late-cycle"]
    # confirm_months=1 switches at once
    frame3 = compute_regime_frame(_macro(level, flat, flat),
                                  params={"growth_clock": {"direction_lookback_months": 1, "confirm_months": 1}})
    assert frame3["regime"].iloc[10] == "Late-cycle"


def test_recession_is_not_delayed():
    n = 12
    sahm = np.r_[np.zeros(n - 1), 0.9]
    cp = np.r_[np.zeros(n - 1), 0.9]
    frame = compute_regime_frame(_macro(np.linspace(0.1, 1.0, n), sahm, cp))
    assert frame["regime"].iloc[-1] == "Recession"


def test_recession_needs_two_of_three():
    n = 10
    one = _macro(np.full(n, 0.2), np.full(n, 0.9), np.full(n, 0.1))      # only Sahm fires
    two = _macro(np.full(n, 0.2), np.full(n, 0.9), np.full(n, 0.6))      # Sahm + Chauvet-Piger
    f1, f2 = compute_regime_frame(one).iloc[-1], compute_regime_frame(two).iloc[-1]
    assert f1["regime"] != "Recession" and f1["p_Recession"] == pytest.approx(1 / 3)
    assert f2["regime"] == "Recession" and f2["p_Recession"] == pytest.approx(2 / 3)


def test_published_thresholds_inclusive():
    n = 10
    at = _macro(np.full(n, -0.70), np.full(n, 0.50), np.full(n, 0.50))
    row = compute_regime_frame(at).iloc[-1]
    assert row["recession_votes"] == 3 and row["regime"] == "Recession"


def test_probabilities_sum_to_one_and_confidence_matches():
    frame = compute_regime_frame(_cycle()).dropna(subset=["regime"])
    probs = frame[[f"p_{r}" for r in REGIMES]]
    assert np.allclose(probs.sum(axis=1), 1.0)
    for t, row in frame.iterrows():
        assert row["confidence"] == pytest.approx(row[f"p_{row['regime']}"])


def test_full_cycle_sequence():
    hist = build_regime_history(_cycle())
    assert set(hist.unique()) == {"Late-cycle", "Recession", "Recovery", "Expansion"}
    assert (hist.loc["2002-03":"2002-09"] == "Recession").all()
    assert hist.iloc[-1] == "Expansion"


def test_no_look_ahead():
    """The regime at month t must not change when later data is appended or altered."""
    df = _cycle()
    full = compute_regime_frame(df)
    truncated = compute_regime_frame(df.iloc[:40])
    pd.testing.assert_frame_equal(full.iloc[:40], truncated)
    shocked = df.copy()
    shocked.iloc[40:, :] = shocked.iloc[40:, :] * -5.0
    pd.testing.assert_frame_equal(compute_regime_frame(shocked).iloc[:40], truncated)


def test_staleness_limit():
    df = _cycle()
    df.loc[df.index[-6:], "recession_prob_chauvet_piger"] = np.nan   # 6 months unpublished
    row = compute_regime_frame(df, params={"max_staleness_months": 4}).iloc[-1]
    assert row["indicators_available"] == 2


def test_classify_view_is_json_safe_and_consistent():
    df = _cycle()
    view = classify_regime_published(df, "2002-06-30")
    json.dumps(view)
    assert view["regime"] == "Recession"
    assert view["data_date"] == "2002-06-30"
    assert set(view["scores"]) == {"s_g", "s_pi", "s_pol", "s_fc"}
    assert all(-3 <= v <= 3 for v in view["scores"].values())
    assert "recession indicators fired" in view["rationale"]
    # Ignores rows after as_of_date
    assert classify_regime_published(df.iloc[:30], "2002-06-30") == view


def test_curve_warning():
    df = _cycle()
    df["term_spread_10y3m"] = -0.002
    view = classify_regime_published(df, df.index[-1])
    assert view["indicators"]["curve_inverted"] is True
    assert "Early warning" in view["rationale"]


def test_missing_indicator_columns_raise():
    df = _cycle().drop(columns=["sahm_rule_realtime_pp"])
    with pytest.raises(ValueError, match="SAHMREALTIME"):
        compute_regime_frame(df)


def test_method_2_uses_regime_history():
    pytest.importorskip("scipy")
    from agent_tools import run_cma_method_2

    df = _cycle()
    hist = build_regime_history(df)
    rets = pd.Series(np.random.default_rng(1).normal(0.007, 0.04, len(hist)), index=hist.index)
    view = classify_regime_published(df, df.index[15])       # Late-cycle: enough months in history
    assert view["regime"] == "Late-cycle"
    mv = {"regime": view["regime"], "confidence": view["confidence"], "scores": view["scores"]}
    out = run_cma_method_2({"monthly_returns_series": rets}, mv, 0.03, regime_history=hist)
    assert "Regime-filtered ERP" in out["rationale"]
    without = run_cma_method_2({"monthly_returns_series": rets}, mv, 0.03)
    assert "Fallback" in without["rationale"]

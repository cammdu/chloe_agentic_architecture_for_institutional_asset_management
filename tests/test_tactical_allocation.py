"""Tests for the tactical asset allocation layer (agent_tactical_allocation.py).

Run with:  pytest tests/test_tactical_allocation.py -q
No network access or API keys are needed.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent_tactical_allocation as taa  # noqa: E402


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def inputs(config):
    return taa.make_synthetic_taa_inputs(config, n_months=200, seed=3)


@pytest.fixture(scope="module")
def backtest(inputs, config):
    return taa.run_taa_backtest(inputs, config)


def test_policy_weights_sum_to_one(config):
    w = taa.load_policy_weights(config)
    assert len(w) == 18
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_policy_from_saa_run(tmp_path, config):
    weights = np.full(18, 1 / 18)
    path = tmp_path / "final_weights.json"
    path.write_text(json.dumps({"weights": weights.tolist()}))
    w = taa.load_policy_weights(config, final_weights_path=path)
    assert np.allclose(w.to_numpy(), weights)


def test_signals_are_point_in_time(inputs, config):
    """A signal at month t must not change when data after t is removed."""
    full, reg_full = taa.compute_signals(inputs, config)
    t = inputs.returns.index[150]
    cut = taa.TAAInputs(
        returns=inputs.returns.loc[:t],
        fundamentals={k: v.loc[:t] for k, v in inputs.fundamentals.items()},
        rates=inputs.rates.loc[:t],
        macro=inputs.macro.loc[:t],
        categories=inputs.categories,
    )
    part, reg_part = taa.compute_signals(cut, config)
    for fam in full:
        a = full[fam].loc[t].fillna(-99)
        b = part[fam].loc[t].fillna(-99)
        assert np.allclose(a.to_numpy(), b.to_numpy()), fam
    assert reg_full.loc[t] == reg_part.loc[t]


def test_backtest_respects_constraints(backtest, config):
    p = config["TAA_PARAMS"]
    w, ws = backtest.weights, backtest.policy_weights
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w.to_numpy() >= -1e-9).all()
    band = p["BANDS"]["per_asset"]
    assert ((w - ws).abs().to_numpy() <= band + 1e-6).all()
    te_budget = p["RISK"]["tracking_error_budget"]
    assert (backtest.ex_ante_te <= te_budget * 1.001 + 1e-9).all()
    capped = backtest.status == "optimal"
    assert (backtest.turnover[capped] <= p["TURNOVER"]["max_one_way_per_rebalance"] + 1e-6).all()


def test_category_bands(backtest, inputs, config):
    bands = config["TAA_PARAMS"]["BANDS"]["per_category"]
    active = backtest.weights - backtest.policy_weights
    for key, label in taa.CATEGORY_LABELS.items():
        cols = [a for a in inputs.assets if inputs.categories[a] == key]
        assert (active[cols].sum(axis=1).abs() <= bands[label] + 1e-6).all(), label


def test_zero_signals_and_costs_means_no_tilt(inputs, config):
    """With no signals and free trading, the strategy should hold the policy portfolio."""
    cfg = copy.deepcopy(config)
    cfg["TAA_PARAMS"]["TRANSACTION_COSTS_BPS"] = {"default": 0}
    res = taa.run_taa_backtest(inputs, cfg, signal_weights={"momentum": 0.0})
    assert np.allclose(res.weights.to_numpy(), res.policy_weights.to_numpy(), atol=1e-5)
    assert res.metrics["active_return_ann"] == pytest.approx(0.0, abs=1e-6)


def test_zero_signals_with_costs_lets_drift_run(inputs, config):
    """With no signals but real costs, drift is only traded back at the band edges."""
    res = taa.run_taa_backtest(inputs, config, signal_weights={"momentum": 0.0})
    band = config["TAA_PARAMS"]["BANDS"]["per_asset"]
    assert ((res.weights - res.policy_weights).abs().to_numpy() <= band + 1e-6).all()
    assert res.turnover.mean() < 0.01


def test_costs_reduce_trading(inputs, config):
    """Higher trading costs should lead to less turnover."""
    cheap = copy.deepcopy(config)
    dear = copy.deepcopy(config)
    cheap["TAA_PARAMS"]["TRANSACTION_COSTS_BPS"] = {"default": 0}
    dear["TAA_PARAMS"]["TRANSACTION_COSTS_BPS"] = {"default": 100}
    sig = taa.compute_signals(inputs, config)
    to_cheap = taa.run_taa_backtest(inputs, cheap, signals=sig).turnover.mean()
    to_dear = taa.run_taa_backtest(inputs, dear, signals=sig).turnover.mean()
    assert to_dear < to_cheap


def test_optimizer_moves_toward_positive_alpha():
    n = 3
    cov = np.diag([0.04, 0.04, 0.04])
    ws = np.array([0.4, 0.4, 0.2])
    res = taa.optimise_tilt(
        alpha=np.array([0.02, -0.02, 0.0]), cov=cov, w_policy=ws, w_current=ws,
        lower=np.zeros(n), upper=np.ones(n), cost_rate=np.zeros(n),
        category_index={}, category_band={}, te_budget=0.05, turnover_cap=1.0,
        risk_aversion=2.0, horizon_years=0.25,
    )
    assert res.weights[0] > ws[0] and res.weights[1] < ws[1]
    assert res.weights.sum() == pytest.approx(1.0)
    assert res.ex_ante_te <= 0.05 + 1e-6


def test_ledoit_wolf_is_psd():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.03, (60, 18))
    cov = taa.ledoit_wolf_covariance(x)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_recommendation_and_report(tmp_path, inputs, config, backtest):
    rec = taa.current_recommendation(inputs, config, as_of=str(inputs.returns.index[-1].date()))
    assert len(rec["assets"]) == 18
    total = sum(r["tactical_weight"] for r in rec["assets"])
    assert total == pytest.approx(1.0)
    paths = taa.write_taa_report(tmp_path, rec, backtest)
    assert paths["report"].exists() and paths["recommendation"].exists()
    assert "Information ratio" in paths["report"].read_text()


def test_build_inputs_from_pipeline_frames(config):
    """build_taa_inputs reads the same raw dataframes as the SAA pipeline."""
    universe_map = {a: {"ticker": f"T{i}", "category": taa.CATEGORY_LABELS[c]}
                    for i, (a, c) in enumerate(taa.asset_categories(config).items())}
    dates = pd.date_range("2015-01-31", periods=30, freq="ME")
    idx = pd.MultiIndex.from_tuples(
        [(d, m["ticker"], m["category"]) for d in dates for m in universe_map.values()],
        names=["date", "ticker", "investment_universe"])
    rng = np.random.default_rng(1)
    tri = pd.DataFrame({"total_return_index": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (30, 18)), axis=0)).reshape(-1)}, index=idx)
    fund = pd.DataFrame({"cape_ratio": 25.0, "earnings_yield": 0.04, "dividend_yield": 0.02}, index=idx)
    rates = pd.DataFrame({"ust_3m_yield": 0.03, "ust_10y_yield": 0.04}, index=dates)
    macro = pd.DataFrame({"cpi_yoy": 0.02}, index=dates)
    cfg = copy.deepcopy(config)
    out = taa.build_taa_inputs({"df_total_return_raw": tri, "df_fundamentals_raw": fund,
                                "df_fixed_income_curves_spreads_raw": rates, "df_macro_raw": macro},
                               cfg, universe_map)
    assert list(out.returns.columns) == taa.canonical_asset_order(config)
    assert len(out.returns) == 29
    assert out.returns.notna().all().all()

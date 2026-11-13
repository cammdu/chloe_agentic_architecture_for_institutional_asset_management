# From SAA to TAA: plan for the agentic portfolio repo

*13 Nov 2026*

## Short answer

The repo today answers a strategic question: "what should the long-run policy mix of 18 asset classes be, given 3-year capital market assumptions?" It rebuilds the whole portfolio from scratch every cycle, with only a 0–25% box per asset and a loose 6% tracking-error cap against a 60/40.

Tactical asset allocation answers a different question: "given where we are *now*, how far should we lean away from the policy mix over the next 1–12 months, and is the lean worth the trading cost?" So the change is not a rewrite. The existing SAA pipeline becomes the **anchor** (the policy portfolio), and a new **TAA layer** produces bounded, cost-aware tilts around it from short-horizon signals, judged by a walk-forward backtest against that anchor.

## Diagnosis: where the code is strategic, not tactical

| Area | What the code does now | Why it is SAA, not TAA |
|---|---|---|
| Horizon | `cma_horizon_years: 3`, covariance window 120 months, macro z-scores on an expanding window (`config.yaml` §2, §7.4; Task 20) | TAA forecasts 1–12 months ahead; risk should use a shorter, more responsive window (e.g. 36–60m, or exponentially weighted). |
| Expected returns | Seven equity CMA methods (historical ERP, BL equilibrium, Gordon, CAPE ERP, survey…) and a yield-plus-roll FI builder (Tasks 21–23) | These are long-run anchors. They say little about the next quarter. TAA needs signal-based *active* return forecasts. |
| Signals | `momentum_12m`, `rsi_14d`, breadth are only used in `check_signal_alignment` (`agent_tools.py:6548`) as a sanity check on the 3-year CMA's sign | The signals that matter for TAA are present in the data but never become forecasts or weights. |
| Portfolio construction | 20 optimizers each build a full portfolio from zero, long-only, 0–25% per asset (Task 25) | No strategic anchor, so there are no *deviations* to size or bound. TAA works in active-weight space: w = w_SAA + tilt. |
| Benchmark | 60/40 projected onto just two of 18 assets (US Large Cap, Intermediate Treasuries) (Task 5) | The right TAA benchmark is the 18-asset policy portfolio itself, so active risk measures the tactical bet, not the strategic mismatch. |
| Active risk budget | Ex-ante TE ≤ 6% vs that 60/40 | 6% vs a crude 60/40 is mostly strategic difference. A TAA risk budget is usually 1–3% TE vs the policy portfolio. |
| Trading | No current holdings, no turnover, no costs | A tilt that does not beat its round-trip cost destroys value. TAA needs costs, a no-trade band and a turnover cap. |
| Rebalancing | Quarterly calendar plus a 5% L1 drift trigger, but not used in construction | TAA rebalances monthly on signal changes, only when the improvement clears costs. |
| Backtest | Final weights applied statically to the full history (`compute_realized_portfolio_returns`, Task 33) | That is in-sample: today's weights scored on the past. TAA must be judged walk-forward, point-in-time, with costs, on *active* return: information ratio, hit rate, turnover. |
| Regime | Four discrete regimes from level z-scores; used to pick CMA methods and blend votes | Useful, but for TAA the regime should map into tilt directions (risk-on / risk-off, duration, credit), and should not flip-flop month to month. |

## Target TAA design

**1. Anchor.** The policy portfolio w_SAA over the 18 assets. Default: the existing pipeline's `final_weights.json` if a run exists, otherwise a policy mix written into `config.yaml` (a diversified 60/40-like split spread across the 18 classes). Its fixed weights are the benchmark for every TAA metric.

**2. Signals (all point-in-time, monthly, from data the repo already downloads).**

- *Time-series momentum*: 12-1 month return (skip the last month to avoid short-term reversal), plus a faster 3-month trend, vol-scaled.
- *Cross-sectional momentum*: rank of 12-1 momentum across the 18 classes.
- *Value*: equities from CAPE earnings yield minus the real 10y yield, versus its own history; credit from OAS z-scores (IG, HY, EM); duration from the term spread (10y − 3m) and the real yield.
- *Carry*: dividend yield for equities, yield to maturity minus cash for bonds, spread carry for credit.
- *Macro regime*: a fixed map from regime to tilt (e.g. Late-cycle: underweight HY and small cap, overweight short duration and gold; Recession: overweight long Treasuries; Recovery: overweight small cap, EM, HY).

Each signal is turned into a z-score, winsorised at ±3, and blended with config weights (default 35% momentum, 25% value, 20% carry, 20% regime).

**3. Alpha model.** Convert the composite score to an active return forecast with the Grinold rule: α_i = IC × σ_i × z_i, with a conservative IC (default 0.05). This keeps forecasts small and comparable across assets, which is what stops the optimizer from going to the corners.

**4. Optimizer.** Choose the tilt d = w − w_SAA to maximise

α'd − (λ/2) d'Σd − c'|w − w_current|

subject to: fully invested, long-only, per-asset bands (default ±5pp around policy), per-category bands (±10pp for Equity, Fixed Income, Real Assets, Cash), ex-ante TE vs w_SAA ≤ 2%, and one-way turnover ≤ 20% per month. Σ from a 60-month Ledoit-Wolf estimate (reusing the repo's estimator). Costs c in basis points by asset (e.g. 2 bps Treasuries, 5 bps large cap, 15 bps EM/HY). A no-trade zone falls out naturally from the L1 cost term.

**5. Walk-forward backtest.** Each month-end t: compute signals with data up to t only, solve, trade, earn month t+1 returns minus costs. Report versus the policy portfolio rebalanced monthly: active return, tracking error, information ratio, hit rate, turnover, cost drag, max active drawdown, and the contribution of each signal family (run each signal alone to see which one earns its keep).

**6. Agent layer.** The LLM agents keep their role of judgment and narrative, reframed: the asset-class agents give a 3–12 month view (underweight / neutral / overweight with conviction 1–3 and a short rationale) instead of a 3-year CMA; their views can override a signal's direction only within the bands. The CIO memo explains the tilts against policy, what they cost, and what would make the committee reverse them. The voting and CIO scores use information ratio and turnover instead of total-portfolio Sharpe.

## Implementation plan

**Phase 1: deterministic TAA engine (no API calls, costs nothing).**
- `config.yaml`: new `TAA_PARAMS` block (horizon, policy weights, bands, TE budget, costs, turnover cap, signal weights, IC, regime tilt map).
- New module `agent_tactical_allocation.py`: signal builders, alpha model, cost-aware optimizer, walk-forward backtest, report writer. Pure numpy/pandas/scipy, no new dependencies.
- Unit tests that check the constraints hold (bands, TE, turnover, sum to 1) and that there is no look-ahead (signals at t never see t+1).
- A new notebook section "Tactical Asset Allocation" that runs on the synthetic data out of the box and on the real free data if `USE_REAL_DATA = True`.

**Phase 2: reframe the agents (changes prompts; testing it costs a little).**
- New prompt templates for the tactical views and the TAA CIO memo, wired into the existing ReAct and AutoGen plumbing.
- Swap the static backtest metric used by the CRO, voting and CIO scoring for the walk-forward active metrics.
- Testing this needs Claude calls; I will ask before running anything, with a spend cap.

**Phase 3: docs.** README section on the TAA framing and how to read the outputs.

Phase 1 is fully additive: nothing in the existing SAA pipeline changes, so it cannot break current runs.

## Decisions for Chloë

1. **The anchor.** I recommend using the SAA pipeline's own output as the policy portfolio when one exists, with a fixed policy mix in `config.yaml` as the fallback. The alternative is a fixed policy mix only, which is simpler and independent of any LLM run.
2. **Risk appetite.** Defaults are ±5pp per asset, 2% TE vs policy, 20% monthly turnover cap. Say if you want a more aggressive (e.g. ±10pp, 4% TE) or more conservative profile.
3. **Phase 2 testing.** It needs a small paid Claude run; I will ask before doing it.

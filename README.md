# Portfolio Risk, Stress Testing & Macro Scenario Engine

**A production-style market-risk engine that measures portfolio risk, backtests it honestly, and stress-tests it against real historical shocks built in Python, deployed as a live interactive dashboard.**

🔗 **[Live dashboard →](https://dynamic-var-stress-testing-engine-rk3qwtdff5nyyftrtqsfla.streamlit.app/)**

*Boopesh Mohanraj*

---

## What this is

A risk officer's core job is answering one question every morning: *"If the market moves against us this week, how much do we lose — and is our risk model even trustworthy right now?"* Most student risk projects compute a single Value-at-Risk number and stop there. The problem is that a VaR model calibrated on calm markets silently fails the moment volatility regime-shifts — which is exactly when you need it most.

This engine addresses that gap end to end. It runs an 18-asset multi-sector portfolio (15 equities across 9 GICS sectors + TLT, GLD, UUP as cross-asset diversifiers) over 2017–2024, and does four things a real risk desk does:

1. **Measures** risk three static ways (Historical, Parametric, Monte Carlo VaR) plus CVaR / Expected Shortfall at the Basel III FRTB standard.
2. **Adapts** to volatility regimes using Filtered Historical Simulation on three conditional-volatility models (EWMA, GARCH, GJR-GARCH).
3. **Validates** every model with formal Kupiec and Christoffersen backtests on genuinely out-of-sample data — and reports where models *fail*, not just where they pass.
4. **Stress-tests** the portfolio against forward scenarios and replays of the March 2020 COVID crash and the 2022 rate shock.

Everything is wired into a live 9-panel dashboard so the output is explorable, not just a static notebook.

---

## Key results

All figures below are computed directly from the code in this repo and are reproduced live in the dashboard.

**The central finding — static VaR fails through regime shifts; dynamic VaR survives.** Using a walk-forward design (calibrate on 2017–2020, test 2021; recalibrate on 2017–2021, test 2022), static Historical-Simulation VaR failed the Kupiec backtest in *both* out-of-sample years, while all three Filtered-Historical-Simulation models passed in both:

| Model | 2021 breach rate | 2022 breach rate | Kupiec result |
|---|---|---|---|
| Static HS-VaR | 1.6% | 9.2% | ❌ Fails both years |
| FHS-EWMA | 6.0% | 5.6% | ✅ Passes both |
| FHS-GARCH | 5.6% | 6.4% | ✅ Passes both |
| FHS-GJR-GARCH | 6.0% | 7.2% | ✅ Passes both |

*(Target breach rate for a 95% VaR is 5%.)* The honest takeaway is not that one dynamic model dominates — EWMA happened to track the 5% target most closely in 2022, and GJR-GARCH ran slightly hot — but that **volatility-adaptive VaR stays statistically valid through a regime shift where static VaR breaks down.** That distinction is the entire point of the engine.

![Walk-forward Kupiec backtest across all four models](plots/phase8_01_kupiec_walkforward.png)

**The models work day by day, not just on average.** Plotting 2022's actual daily losses against all four VaR thresholds shows the dynamic models widening ahead of the drawdown while static VaR stays flat and gets breached repeatedly:

![2022 daily loss vs all four VaR models](plots/phase2b_02_var_timeseries_2022.png)

**Diversification assumptions broke in 2022 — and the engine captures it.** The equity–bond (TLT) correlation, reliably negative for decades, flipped *positive* during the 2022 rate shock as stocks and bonds fell together. A risk model using a stable historical correlation would have understated portfolio risk precisely when it mattered:

![Cross-asset correlation regime shift](plots/phase3_05_correlation_regimes.png)

**Conditional volatility models track the regimes.** EWMA, GARCH, and GJR-GARCH all spike through the COVID and 2022 stress windows, which is what lets the filtered VaR adapt:

![Conditional volatility models](plots/phase2b_01_cond_vol_training.png)

**Mean-CVaR optimization cut tail risk sharply — with an honest caveat.** Implementing the Rockafellar–Uryasev linear program in `cvxpy` reduced in-sample 95% CVaR from **19.0% to 3.0%**. But the optimizer did this by concentrating into low-CVaR defensive assets (≈20% each into TLT, GLD, and the AAPL/UUP block), and that concentration *hurt* in the specific 2022 scenario: the CVaR-optimal portfolio returned **−15.4%** vs the equal-weight **−11.2%**, because long-duration Treasuries (TLT) were themselves a primary casualty of the rate shock. This is a textbook illustration of why a corner solution that minimizes a historical risk metric is not the same as a portfolio that survives the next specific shock — and why production mandates impose concentration limits.

![CVaR-optimal vs equal weights](plots/phase6_01_weights_comparison.png)

---

## Methodology & academic references

Each component implements a specific paper. Listed below is the paper, what was implemented, and — where measurable — what it produced on this portfolio, including limitations.

**Static VaR / CVaR / Expected Shortfall — Basel III / FRTB (2019).** Computed 95%/99% VaR three ways (Historical, Parametric via the cross-asset covariance matrix, and Monte Carlo via Cholesky decomposition), plus Expected Shortfall at the 97.5% FRTB standard. The three methods diverge exactly where their assumptions do — Parametric understates tails, Historical assumes the past distribution recurs.

**Filtered Historical Simulation — Barone-Adesi, Giannopoulos & Vosper (1998).** Standardized historical returns by their conditional volatility, then rescaled by today's volatility forecast. This is the mechanism that lets VaR keep empirical fat tails *and* react to the current regime — the reason the dynamic models pass Kupiec where static fails.

**Conditional volatility — Engle (1982) [ARCH] and Glosten, Jagannathan & Runkle (1993) [GJR-GARCH].** Implemented EWMA (RiskMetrics λ=0.94), symmetric GARCH(1,1), and the asymmetric GJR-GARCH via the `arch` library. On this equity portfolio the GJR leverage term (γ) was statistically significant (likelihood-ratio test p ≈ 0.0008), confirming that negative shocks raise volatility more than positive ones. *Note on naming:* the dashboard labels the asymmetric model "T-GARCH" as shorthand; the model actually implemented (`arch_model(..., o=1)`) is GJR-GARCH, which is the correct attribution.

**VaR backtesting — Kupiec (1995) and Christoffersen (1998).** Kupiec's proportion-of-failures test checks whether the breach *count* is consistent with the VaR level; Christoffersen's test checks whether breaches are independent (not clustered). Running both across all four models turned validation into a model-comparison exercise — the central result above.

**Mean-CVaR optimization — Rockafellar & Uryasev (2000).** Implemented their CVaR-minimization linear program from scratch in `cvxpy` rather than calling a black-box optimizer. Result and limitation stated in Key Results — a large in-sample CVaR reduction that concentrated into defensives and underperformed equal-weight in the 2022 duration shock.

**Macro linkage — Ang & Piazzesi (2003).** Regressed portfolio returns on changes in the Fed Funds rate, CPI, the 10Y yield, and the IG credit spread (R² ≈ 0.49). The resulting sensitivity table ("a +100bp move in the 10Y costs the portfolio ≈4.7%") is the kind of one-line answer a CIO actually asks for. Out-of-sample on 2022, the regression predicted the direction of the loss correctly.

**CVaR optimization scale note — NVIDIA quantitative-finance blueprint (2024).** NVIDIA's reference implementation uses GPU-accelerated `cuOpt` at ~400-stock institutional scale; this project implements the identical Rockafellar–Uryasev formulation in `cvxpy` at 18-stock scale — same mathematics, CPU-appropriate tooling for the size. Stated plainly rather than overclaimed.

---

## Tech stack

**Language:** Python
**Risk & stats:** NumPy, pandas, SciPy, statsmodels, `arch` (GARCH family)
**Optimization:** cvxpy (Rockafellar–Uryasev LP)
**Data:** Tiingo (prices), FRED API (macro series)
**Storage / reporting:** SQLite (window-function risk queries)
**Visualization:** Plotly (interactive), Matplotlib (static)
**App:** Streamlit, deployed on Streamlit Community Cloud

---

## Repository structure

```
streamlit_app.py     Live dashboard (9 panels)
data/                Prices, returns, covariance matrices, macro series
results/             VaR, backtest, stress, optimization, and macro outputs + SQLite DB
plots/               Selected result visualizations
requirements.txt     Dependencies
```

---

## Data & limitations

Honest limitations, because a risk professional who oversells a model is a liability:

- **Survivorship bias** — the equity universe uses currently-listed tickers, excluding delisted names.
- **CVaR corner solution** — the unconstrained optimizer concentrates into defensives; realistic mandates would impose position/sector limits (partially modeled here at 20%/35%).
- **Single-scenario stress** — historical replays (COVID, 2022) are informative but are not a substitute for a full forward scenario distribution.
- **GARCH parameter drift** — the leverage parameter is estimated from history and shifts over time; rolling re-estimation only partially addresses this.
- **EVT excluded** — Extreme Value Theory tail-fitting was scoped out as beyond the level of this project; noted as a natural extension.

---
 Data sourced from public APIs (Tiingo, FRED). This is a research and educational project, not investment advice.*

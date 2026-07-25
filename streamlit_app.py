"""
=============================================================================
PROJECT 1 — Portfolio Risk, Stress Testing & Macro Scenario Engine
PHASE 9  — Streamlit Dashboard
=============================================================================
Author  : Boopesh Mohanraj
School  : Northeastern University

HOW TO RUN:
  1. Install: pip install streamlit --quiet
  2. Save this file as app.py in the same folder as data/ and results/
  3. Run: streamlit run app.py
  4. Opens at http://localhost:8501

PANELS:
  1 — Live P&L & Portfolio Summary
  2 — VaR Comparison (Static vs Dynamic)
  3 — Dynamic Volatility (EWMA / GARCH / T-GARCH)
  4 — Stress Test Scenarios (Bear / Base / Bull + Historical)
  5 — Macro Linkage (OLS sensitivity, yield curve)
  6 — CVaR-Optimal Weights
  7 — Liquidity Risk Dashboard
  8 — Walk-Forward Kupiec Status
  9 — SQL Risk Report (live queries)
=============================================================================
"""

import os, json, sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False
    print("streamlit not installed — run: pip install streamlit --quiet")

# =============================================================================
# PAGE CONFIG
# =============================================================================

if STREAMLIT:
    st.set_page_config(
        page_title="Portfolio Risk & Stress Testing Engine",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ── Color palette ─────────────────────────────────────────────────────────────
C_STATIC = "#566573"; C_EWMA   = "#2980B9"
C_GARCH  = "#229954"; C_TGARCH = "#C0392B"
C_BULL   = "#229954"; C_BEAR   = "#C0392B"
C_STEEL  = "#1C2833"; C_GREY   = "#7F8C8D"
C_ORANGE = "#E67E22"; C_OK     = "#229954"

# ── Plotly layout helper ──────────────────────────────────────────────────────
def playout(title, sub="", height=400):
    return dict(
        title=dict(
            text=f"<b>{title}</b>" + (f"<br><sup style='color:#7F8C8D'>{sub}</sup>" if sub else ""),
            font=dict(size=14, family="Arial, sans-serif", color=C_STEEL),
            x=0.02, xanchor="left"),
        font=dict(family="Arial, sans-serif", size=11, color=C_STEEL),
        template="plotly_white", height=height,
        margin=dict(l=55, r=30, t=75, b=50),
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
    )

# =============================================================================
# DATA LOADERS  (cached so Streamlit doesn't reload on every interaction)
# =============================================================================

@st.cache_data if STREAMLIT else lambda f: f
def load_all():
    data = {}

    # Phase 2b
    data["var_ts"] = pd.read_csv(
        "results/phase2b_var_timeseries_2022.csv",
        index_col=0, parse_dates=True)

    # Phase 2b Kupiec
    data["kupiec"] = pd.read_csv("results/phase2b_kupiec_results.csv", index_col=0)

    # Phase 3
    with open("results/phase3_scenario_results.json") as f:
        data["ph3"] = json.load(f)

    # Phase 4
    with open("results/phase4_ols_results.json") as f:
        data["ph4"] = json.load(f)

    # Phase 5
    data["liq_df"]   = pd.read_csv("results/phase5_liquidity_metrics.csv", index_col=0)
    data["stress_df"] = pd.read_csv("results/phase5_stress_exit_costs.csv", index_col=0)
    with open("results/phase5_liquidity_summary.json") as f:
        data["ph5"] = json.load(f)

    # Phase 6
    data["weights_df"] = pd.read_csv("results/phase6_optimal_weights.csv", index_col=0)
    with open("results/phase6_summary.json") as f:
        data["ph6"] = json.load(f)

    # Phase 8
    data["walkfwd"] = pd.read_csv("results/phase8_kupiec_walkforward.csv")
    data["stability"] = pd.read_csv("results/phase8_garch_stability.csv")

    # Portfolio daily
    data["port_daily"] = pd.read_csv(
        "data/portfolio_daily_returns.csv", index_col=0, parse_dates=True
    )["portfolio_return"]

    # Metadata
    with open("data/metadata.json") as f:
        data["meta"] = json.load(f)

    # EWMA vol full history
    r = data["port_daily"].values
    lam = data["meta"]["lambda_ewma"]
    ev = np.zeros(len(r)); ev[0] = r[0]**2
    for t in range(1, len(r)):
        ev[t] = lam*ev[t-1] + (1-lam)*r[t-1]**2
    data["ewma_vol"] = pd.Series(np.sqrt(ev)*100, index=data["port_daily"].index)

    # SQLite
    if os.path.exists("results/portfolio_risk.db"):
        conn = sqlite3.connect("results/portfolio_risk.db")
        data["daily_db"] = pd.read_sql(
            "SELECT * FROM daily_risk_enriched ORDER BY date",
            conn, index_col="date", parse_dates=["date"])
        data["regime_db"] = pd.read_sql(
            """SELECT month,
                      ROUND(AVG(ewma_vol),3)           AS avg_ewma_vol,
                      ROUND(AVG(var95_tgarch)*100,3)   AS avg_tgarch_var_pct,
                      SUM(breach_tgarch)               AS tgarch_breaches,
                      MAX(vol_regime)                  AS vol_regime
               FROM daily_risk_enriched
               GROUP BY month ORDER BY month""", conn)
        conn.close()
    else:
        data["daily_db"] = None

    return data

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    if not STREAMLIT:
        print("Run: streamlit run P1_Phase9_Streamlit_Dashboard.py")
        return

    D = load_all()
    var_ts    = D["var_ts"]
    port      = D["port_daily"]
    meta      = D["meta"]
    ewma_vol  = D["ewma_vol"]
    ph3       = D["ph3"]
    ph4       = D["ph4"]
    weights   = D["weights_df"]
    liq_df    = D["liq_df"]
    stress_df = D["stress_df"]
    walkfwd   = D["walkfwd"]

    ALL_TICKERS = meta["all_tickers"]

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    st.sidebar.title("📊 Risk Engine")
    st.sidebar.markdown("**Portfolio Risk, Stress Testing & Macro Scenario Engine**")
    st.sidebar.markdown("*Boopesh Mohanraj · Northeastern University*")
    st.sidebar.divider()

    panel = st.sidebar.radio("Navigation", [
        "1 · Portfolio Summary",
        "2 · VaR Comparison",
        "3 · Dynamic Volatility",
        "4 · Stress Scenarios",
        "5 · Macro Linkage",
        "6 · CVaR-Optimal Weights",
        "7 · Liquidity Risk",
        "8 · Walk-Forward Kupiec",
        "9 · SQL Risk Report",
    ])

    st.sidebar.divider()
    st.sidebar.markdown("**Key Stats  (2022 backtest)**")

    # These four stats are computed live from the loaded results —
    # never hardcoded — so they can't drift out of sync with the underlying data.
    kup = D["kupiec"]

    # Static model = first row, T-GARCH = last row (matches Panel 2's fixed
    # display order: Static, EWMA, GARCH, T-GARCH).
    static_row  = kup.iloc[0]
    tgarch_row  = kup.iloc[-1]

    static_br = float(static_row["breach_rate"]) * 100
    static_ok = bool(static_row["pass"])
    st.sidebar.metric(
        "Static VaR breach rate", f"{static_br:.1f}%",
        delta=("✓ PASS" if static_ok else "↑ above 5% target — FAIL"),
        delta_color=("normal" if static_ok else "inverse")
    )

    tgarch_br = float(tgarch_row["breach_rate"]) * 100
    tgarch_ok = bool(tgarch_row["pass"])
    st.sidebar.metric(
        "T-GARCH breach rate", f"{tgarch_br:.1f}%",
        delta=("✓ PASS" if tgarch_ok else "↑ above 5% target — FAIL"),
        delta_color=("normal" if tgarch_ok else "inverse")
    )

    cvar_improve = D["ph6"].get("cvar_improvement_pct")
    if cvar_improve is not None:
        st.sidebar.metric("CVaR improvement", f"{float(cvar_improve):.1f}%",
                          delta="vs equal-weight", delta_color="normal")

    # Worst liquidity offender — whichever asset has the highest % of ADV,
    # rather than a hardcoded ticker/number.
    try:
        worst = liq_df["pct_of_adv"].astype(float).idxmax()
        worst_pct = float(liq_df.loc[worst, "pct_of_adv"])
        over_limit = worst_pct > 10.0
        st.sidebar.metric(
            f"{worst} ADV usage", f"{worst_pct:.2f}%",
            delta=("exceeds 10% limit" if over_limit else "within limit"),
            delta_color=("inverse" if over_limit else "normal")
        )
    except Exception:
        pass  # liquidity file format differs — skip rather than show a stale number

    # =========================================================================
    # PANEL 1 — PORTFOLIO SUMMARY
    # =========================================================================
    if panel == "1 · Portfolio Summary":
        st.title("📈 Portfolio Summary")
        st.caption("Equal-weight 18-asset portfolio  ·  $10M AUM (assumed)  ·  2017–2024")

        # KPI row
        cum_2022 = float((1 + port.loc["2022"]).prod() - 1)
        max_dd   = float(((1+port).cumprod() / (1+port).cumprod().cummax() - 1).min())
        vol_ann  = float(port.std() * np.sqrt(252) * 100)
        sharpe   = float(port.mean() / port.std() * np.sqrt(252))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("2022 Total Return",   f"{cum_2022*100:+.2f}%",  delta_color="inverse" if cum_2022<0 else "normal")
        c2.metric("Max Drawdown (all)",  f"{max_dd*100:.2f}%",     delta_color="inverse")
        c3.metric("Annual Volatility",   f"{vol_ann:.2f}%")
        c4.metric("Full-Period Sharpe",  f"{sharpe:.3f}")

        st.divider()

        # Cumulative return chart
        cum_full = (1 + port).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_full.index, y=cum_full.values,
            mode="lines", name="Portfolio", line=dict(color=C_EWMA, width=2),
            fill="tozeroy", fillcolor="rgba(41,128,185,0.08)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Value: %{y:.4f}<extra></extra>"
        ))
        for x0, x1, lbl, col in [
            ("2020-02-20","2020-04-07","COVID",     C_BEAR),
            ("2022-01-01","2022-12-31","Rate Shock",C_ORANGE),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=col, opacity=0.08,
                          annotation_text=lbl, annotation_font=dict(size=10, color=col))
        fig.update_layout(**playout("Cumulative Portfolio Value  (Base = 1.0, Jan 2017)",
                                     "COVID and 2022 rate shock highlighted", height=380))
        st.plotly_chart(fig)

        # EWMA vol
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol.values,
            mode="lines", name="EWMA Vol", line=dict(color=C_TGARCH, width=1.5),
            hovertemplate="<b>%{x|%b %Y}</b> EWMA Vol: %{y:.2f}%<extra></extra>"
        ))
        fig2.update_layout(**playout("EWMA Daily Volatility  (λ=0.94)", height=250))
        st.plotly_chart(fig2)

    # =========================================================================
    # PANEL 2 — VAR COMPARISON
    # =========================================================================
    elif panel == "2 · VaR Comparison":
        st.title("🛡️ VaR Model Comparison  (2022 Backtest)")

        col1, col2, col3, col4 = st.columns(4)
        kup = D["kupiec"]
        for col_st, model, c in zip([col1,col2,col3,col4],
                                     kup.index, [C_STATIC,C_EWMA,C_GARCH,C_TGARCH]):
            br   = float(kup.loc[model,"breach_rate"])*100
            pf   = "✅ PASS" if kup.loc[model,"pass"] else "❌ FAIL"
            col_st.metric(model, f"{br:.1f}%", delta=pf,
                          delta_color="normal" if kup.loc[model,"pass"] else "inverse")

        st.divider()

        # VaR time series
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=var_ts.index, y=(var_ts["actual_loss"]*100).round(3),
            name="Actual Loss", marker_color="#AED6F1", opacity=0.7,
            hovertemplate="<b>%{x|%b %d}</b> Loss: %{y:.3f}%<extra></extra>"
        ))
        for key, col, dash, nm in [
            ("var95_static", C_STATIC, "dash",   "Static HS-VaR 95%"),
            ("var95_ewma",   C_EWMA,   "dot",    "FHS-EWMA 95%"),
            ("var95_garch",  C_GARCH,  "dashdot","FHS-GARCH 95%"),
            ("var95_tgarch", C_TGARCH, "solid",  "FHS-T-GARCH 95%"),
        ]:
            fig.add_trace(go.Scatter(
                x=var_ts.index, y=(var_ts[key]*100).round(3),
                mode="lines", name=nm,
                line=dict(color=col, width=1.8, dash=dash),
                hovertemplate=f"<b>{nm}</b>: %{{y:.3f}}%<extra></extra>"
            ))
        fig.update_layout(**playout("2022 Daily Loss vs All Four VaR Thresholds",
                                     "Dark = T-GARCH breach · Light blue = no breach",
                                     height=420))
        st.plotly_chart(fig)

        # Kupiec table
        st.subheader("Kupiec POF Test Results")
        kup_display = kup[["n_breaches","breach_rate","LR_stat","p_value","pass"]].copy()
        kup_display["breach_rate"] = (kup_display["breach_rate"]*100).round(1).astype(str) + "%"
        kup_display["pass"] = kup_display["pass"].map({True:"✅ PASS", False:"❌ FAIL"})
        st.dataframe(kup_display)

    # =========================================================================
    # PANEL 3 — DYNAMIC VOLATILITY
    # =========================================================================
    elif panel == "3 · Dynamic Volatility":
        st.title("📉 Dynamic Volatility")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol.values,
            mode="lines", name="EWMA (full history)",
            line=dict(color=C_EWMA, width=1.5),
            hovertemplate="<b>%{x|%b %Y}</b> EWMA: %{y:.2f}%<extra></extra>"
        ))
        for x0, x1, lbl, col in [
            ("2020-02-20","2020-04-07","COVID",     C_BEAR),
            ("2022-01-01","2022-12-31","Rate Shock",C_ORANGE),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=col, opacity=0.08,
                          annotation_text=lbl, annotation_font=dict(size=10))
        fig.update_layout(**playout("EWMA Portfolio Volatility  (2017–2024)", height=350))
        st.plotly_chart(fig)

        st.subheader("2022 Monthly Vol Regime")
        if D["daily_db"] is not None:
            db = D["daily_db"]
            monthly = (db.groupby("month")
                         .agg(avg_vol=("ewma_vol","mean"),
                              avg_tgarch_var=("var95_tgarch", lambda x: x.mean()*100),
                              regime=("vol_regime","max"))
                         .reset_index())
            col_map = {"HIGH": C_BEAR, "MEDIUM": C_ORANGE, "LOW": C_OK}
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=monthly["month"],
                y=monthly["avg_vol"],
                marker_color=[col_map.get(r, C_GREY) for r in monthly["regime"]],
                name="Avg EWMA Vol",
                hovertemplate="<b>%{x}</b><br>Avg Vol: %{y:.3f}%<extra></extra>"
            ))
            fig2.update_layout(**playout("Monthly Avg EWMA Vol  (Red=HIGH, Orange=MEDIUM, Green=LOW)",
                                          height=300))
            st.plotly_chart(fig2)

    # =========================================================================
    # PANEL 4 — STRESS SCENARIOS
    # =========================================================================
    elif panel == "4 · Stress Scenarios":
        st.title("⚠️ Stress Test Scenarios")

        scen = ph3["scenarios"]
        cols = st.columns(3)
        for col_st, (name, col) in zip(cols, [("Bear",C_BEAR),("Base",C_EWMA),("Bull",C_BULL)]):
            pnl = scen[name]["pnl_pct"]
            col_st.metric(
                f"{name} Scenario",
                f"{pnl:+.2f}%",
                delta=f"{scen[name]['vs_var_95']:+.1f}× daily VaR",
                delta_color="inverse" if pnl < 0 else "normal"
            )

        st.divider()

        # Scenario P&L bar
        scenarios = ["Bear","Base","Bull"]
        pnl_vals  = [scen[s]["pnl_pct"] for s in scenarios]
        fig = go.Figure(go.Bar(
            x=scenarios, y=pnl_vals,
            marker_color=[C_BEAR, C_EWMA, C_BULL],
            text=[f"{v:+.2f}%" for v in pnl_vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b>: %{y:+.2f}%<extra></extra>"
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=C_STEEL)
        fig.update_layout(**playout("Forward Scenario P&L", height=350))
        st.plotly_chart(fig)

        st.subheader("Historical Replays")
        col1, col2 = st.columns(2)
        covid = ph3["covid_replay"]
        rate  = ph3["rate_shock_replay"]
        col1.metric("COVID Crash  (Feb–Apr 2020)",
                    f"{covid['peak_to_trough_dd']*100:.2f}%",
                    delta=f"Trough: {covid['trough_date']}", delta_color="inverse")
        col2.metric("2022 Rate Shock  (Full Year)",
                    f"{rate['peak_to_trough_dd']*100:.2f}%",
                    delta=f"Trough: {rate['trough_date']}", delta_color="inverse")

    # =========================================================================
    # PANEL 5 — MACRO LINKAGE
    # =========================================================================
    elif panel == "5 · Macro Linkage":
        st.title("🌐 Macro Linkage  (OLS Regression)")

        r2   = ph4["r_squared"]
        adjr2= ph4["adj_r_squared"]
        fp   = ph4["f_pvalue"]
        col1, col2, col3 = st.columns(3)
        col1.metric("R²",           f"{r2:.4f}")
        col2.metric("Adj. R²",      f"{adjr2:.4f}")
        col3.metric("F-stat p-val", f"{fp:.4f}")

        st.divider()
        st.subheader("Macro Sensitivity Table  (CIO Deliverable)")
        sens = ph4["sensitivity_table"]
        sens_df = pd.DataFrame([
            {"Macro Shock": k,
             "Portfolio Impact (%)": f"{v:+.3f}%",
             "Direction": "📉 LOSS" if v < 0 else "📈 GAIN"}
            for k, v in sens.items()
        ])
        st.dataframe(sens_df, hide_index=True, width="stretch")

        st.divider()
        # OOS validation
        st.subheader("2022 Out-of-Sample Validation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Actual 2022 Return",    f"{ph4['oos_cum_actual']:+.2f}%")
        col2.metric("Predicted (OLS)",       f"{ph4['oos_cum_pred']:+.2f}%")
        col3.metric("Direction Correct",
                    "✅ YES" if ph4["oos_direction_correct"] else "❌ NO")

    # =========================================================================
    # PANEL 6 — CVaR-OPTIMAL WEIGHTS
    # =========================================================================
    elif panel == "6 · CVaR-Optimal Weights":
        st.title("⚖️ CVaR-Optimal Portfolio Weights")

        c1, c2, c3 = st.columns(3)
        c1.metric("EW CVaR 95%",          f"{D['ph6']['ew_cvar_95']:.2f}%")
        c2.metric("CVaR-Optimal CVaR 95%", f"{D['ph6']['opt_cvar_95']:.2f}%",
                  delta=f"−{D['ph6']['cvar_improvement_pct']:.1f}% improvement",
                  delta_color="normal")
        c3.metric("2022 Return (Opt vs EW)",
                  f"{D['ph6']['opt_2022_return']:+.2f}%  vs  {D['ph6']['ew_2022_return']:+.2f}%")

        st.divider()
        fig = go.Figure()
        for col_key, col_c, nm in [
            ("ew_weight",     C_STATIC, "Equal-Weight"),
            ("opt_weight",    C_EWMA,   "CVaR-Optimal"),
            ("tgarch_weight", C_TGARCH, "T-GARCH Filtered"),
        ]:
            fig.add_trace(go.Bar(
                x=weights.index.tolist(),
                y=(weights[col_key]*100).tolist(),
                name=nm, opacity=0.85,
                hovertemplate=f"<b>%{{x}}</b> ({nm}): %{{y:.2f}}%<extra></extra>"
            ))
        fig.add_hline(y=20, line_dash="dash", line_color=C_STEEL, line_width=1,
                      annotation_text="20% limit")
        fig.update_layout(**playout("Portfolio Weights: Equal-Weight vs CVaR-Optimal",
                                     "Optimizer concentrates in TLT/GLD/UUP — low-CVaR defensive assets",
                                     height=420),
                           barmode="group")
        st.plotly_chart(fig)

        st.info("**Note:** 84% CVaR reduction achieved by concentrating in low-vol defensive "
                "assets (TLT/GLD/UUP). In practice, sector concentration limits would moderate "
                "this corner solution.")

    # =========================================================================
    # PANEL 7 — LIQUIDITY RISK
    # =========================================================================
    elif panel == "7 · Liquidity Risk":
        st.title("💧 Liquidity Risk Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Roll Spread",    f"{D['ph5']['avg_cs_spread_bps']:.1f} bps")
        c2.metric("Normal Exit Cost",   f"${D['ph5']['normal_exit_cost']:,.0f}")
        c3.metric("Stress Exit Cost 3×",f"${D['ph5']['stress_exit_cost_3x']:,.0f}",
                  delta=f"{D['ph5']['stress_cost_pct_portfolio']:.4f}% of AUM",
                  delta_color="inverse")
        c4.metric("ADV Violations",     str(D["ph5"]["n_sizing_violations"]),
                  delta="UUP: 10.91% of ADV", delta_color="inverse")

        st.divider()

        # Liquidity table
        st.subheader("Per-Asset Liquidity Metrics")
        liq_display = liq_df[["position_$M","adv_proxy_$M","days_to_liq",
                                "pct_of_adv","flag"]].copy()
        liq_display.columns = ["Position $M","ADV $M","Days to Liq","% of ADV","Status"]

        def highlight_flag(row):
            if "FLAGGED" in str(row["Status"]):
                return ["background-color: #FADBD8"]*len(row)
            return [""]*len(row)

        st.dataframe(
            liq_display.style.apply(highlight_flag, axis=1),
            width="stretch"
        )

        # Amihud correlation
        st.metric("Amihud–EWMA Vol Correlation",
                  f"{D['ph5']['amihud_ewma_corr']:+.4f}",
                  delta="p=0.0000 — significant",
                  delta_color="normal")
        st.caption("When T-GARCH signals elevated risk, liquidity also deteriorates. "
                   "Exit costs are highest exactly when you most want to reduce exposure.")

    # =========================================================================
    # PANEL 8 — WALK-FORWARD KUPIEC
    # =========================================================================
    elif panel == "8 · Walk-Forward Kupiec":
        st.title("🔄 Walk-Forward Kupiec Validation")
        st.caption("Expanding window: W1 calibrate 2017–2020 → test 2021  |  "
                   "W2 calibrate 2017–2021 → test 2022")

        # Breach rate heatmap
        pivot = walkfwd.pivot(index="model", columns="test_period",
                               values="breach_rate") * 100
        deviation = np.abs(pivot.values - 5.0)

        fig = go.Figure(go.Heatmap(
            z=deviation,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#EAFAF1"],[0.4,"#F9E79F"],[1.0,"#FADBD8"]],
            zmin=0, zmax=5,
            text=pivot.values.round(1),
            texttemplate="%{text:.1f}%",
            textfont=dict(size=13),
            colorbar=dict(title="Deviation<br>from 5%", thickness=12)
        ))
        layout_p8 = playout("Walk-Forward Breach Rates  (Deviation from 5% Target)",
                            "Green = on target · Red = too high or too low",
                            height=360)
        layout_p8["margin"] = dict(l=160, r=60, t=90, b=60)
        fig.update_layout(**layout_p8)
        st.plotly_chart(fig)

        # Full results table
        st.subheader("Kupiec POF Results")
        wf_display = walkfwd[["window","test_period","model","breach_rate",
                                "lr_stat","p_value","pass"]].copy()
        wf_display["breach_rate"] = (wf_display["breach_rate"]*100).round(1).astype(str)+"%"
        wf_display["pass"] = wf_display["pass"].map({True:"✅ PASS", False:"❌ FAIL"})
        st.dataframe(wf_display, hide_index=True, width="stretch")

        # GARCH stability
        st.subheader("T-GARCH Parameter Stability")
        stab = D["stability"][D["stability"]["model"]=="T-GARCH"]
        col1, col2 = st.columns(2)
        for idx, (_, row) in enumerate(stab.iterrows()):
            col = col1 if idx == 0 else col2
            col.markdown(f"**{row['window']}**")
            col.markdown(f"α={row['alpha']:.4f}  γ={row['gamma']:.4f}  "
                         f"β={row['beta']:.4f}  pers={row['persistence']:.4f}")

        st.success("γ (leverage effect) significant in both windows — p=0.0000 both. "
                   "Asymmetric vol response is a structural feature of this portfolio.")

    # =========================================================================
    # PANEL 9 — SQL RISK REPORT
    # =========================================================================
    elif panel == "9 · SQL Risk Report":
        st.title("🗄️ SQL Risk Report")
        st.caption(f"Live queries against SQLite database: results/portfolio_risk.db")

        if not os.path.exists("results/portfolio_risk.db"):
            st.error("Database not found. Run Phase 7 first.")
            return

        conn = sqlite3.connect("results/portfolio_risk.db")

        tab1, tab2, tab3, tab4 = st.tabs([
            "VaR Backtest", "Stress Scenarios", "Liquidity Alerts", "Vol Regime"
        ])

        with tab1:
            st.subheader("Kupiec Backtest Summary")
            df = pd.read_sql("""
                SELECT model, n_breaches,
                       ROUND(breach_rate*100,1)||'%' AS breach_rate,
                       ROUND(kupiec_lr,3) AS LR_stat,
                       ROUND(kupiec_pvalue,4) AS p_value,
                       pass_fail, vol_dynamics
                FROM var_backtest ORDER BY kupiec_lr
            """, conn)
            st.dataframe(df, hide_index=True, width="stretch")

        with tab2:
            st.subheader("Stress Scenario Ranking")
            df = pd.read_sql("""
                SELECT scenario_name,
                       ROUND(portfolio_pnl,2)||'%' AS portfolio_pnl,
                       ROUND(pnl_vs_var,1)||'× VaR' AS vs_var,
                       ROUND(tlt_contribution,3)||'%' AS tlt_contrib,
                       scenario_type
                FROM stress_scenarios ORDER BY portfolio_pnl
            """, conn)
            st.dataframe(df, hide_index=True, width="stretch")

        with tab3:
            st.subheader("Liquidity Violation Alerts")
            df = pd.read_sql("""
                SELECT ticker,
                       ROUND(position_usd/1e6,4) AS pos_M,
                       ROUND(adv_proxy_usd/1e6,2) AS adv_M,
                       pct_of_adv,
                       roll_spread_bps,
                       ROUND(stress_exit_cost,0) AS stress_cost_usd,
                       adv_violation
                FROM liquidity_metrics
                ORDER BY adv_violation DESC, pct_of_adv DESC
                LIMIT 8
            """, conn)
            st.dataframe(df, hide_index=True, width="stretch")

        with tab4:
            st.subheader("Monthly Vol Regime  (2022)")
            df = pd.read_sql("""
                SELECT month,
                       ROUND(AVG(ewma_vol),3) AS avg_ewma_vol,
                       ROUND(AVG(var95_tgarch)*100,3) AS avg_tgarch_var,
                       SUM(breach_tgarch) AS tgarch_breaches,
                       MAX(vol_regime) AS vol_regime
                FROM daily_risk_enriched
                GROUP BY month ORDER BY month
            """, conn)
            st.dataframe(df, hide_index=True, width="stretch")

        conn.close()

        st.divider()
        st.subheader("Dynamic VaR Premium Query")
        if D["daily_db"] is not None:
            db = D["daily_db"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=db.index, y=db["var_premium_pct"].round(1),
                mode="lines", name="VaR Premium (%)",
                line=dict(color=C_TGARCH, width=1.5),
                fill="tozeroy", fillcolor="rgba(192,57,43,0.10)",
                hovertemplate="<b>%{x|%b %d}</b> Premium: %{y:.1f}%<extra></extra>"
            ))
            fig.add_hline(y=30, line_dash="dash", line_color=C_ORANGE,
                          annotation_text="30% alert")
            fig.add_hline(y=0, line_dash="dot", line_color=C_STEEL, line_width=1)
            fig.update_layout(**playout("T-GARCH vs Static VaR Premium  (from SQLite)",
                                         height=300))
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()

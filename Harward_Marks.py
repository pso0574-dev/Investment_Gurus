# streamlit_app.py
# ============================================================
# Howard Marks Style Investment Dashboard
# - Macro cycle / credit / fear / valuation proxy dashboard
# - Streamlit deploy-ready
#
# Install:
#   pip install streamlit pandas numpy plotly requests yfinance
#
# Run:
#   streamlit run streamlit_app.py
#
# Optional:
#   Set FRED_API_KEY as environment variable
# ============================================================

from __future__ import annotations

import os
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Howard Marks Style Dashboard",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Howard Marks Style Investment Dashboard")
st.caption("Cycle / Risk / Fear / Opportunity / Defense")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

lookback_years = st.sidebar.slider("Lookback Years", 3, 20, 10)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=False)
refresh_button = st.sidebar.button("Refresh Data")

if auto_refresh:
    st.sidebar.info("Auto refresh enabled (refreshes when app reruns).")

# ============================================================
# Helpers
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

TODAY = pd.Timestamp.today().normalize()
START_DATE = TODAY - pd.DateOffset(years=lookback_years)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch a FRED series as DataFrame with columns: date, value
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date.strftime("%Y-%m-%d"),
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_prices(tickers: List[str], start_date: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    close_df = pd.DataFrame()
    for t in tickers:
        try:
            if len(tickers) == 1:
                s = data["Close"].rename(t)
            else:
                s = data[t]["Close"].rename(t)
            close_df[t] = s
        except Exception:
            pass

    close_df = close_df.dropna(how="all")
    return close_df


def normalize_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series
    return series / s.iloc[0] * 100.0


def compute_drawdown(price: pd.Series) -> pd.Series:
    rolling_peak = price.cummax()
    dd = (price / rolling_peak - 1.0) * 100.0
    return dd


def zscore_last(series: pd.Series, window: int = 252) -> float:
    s = series.dropna()
    if len(s) < max(20, window // 4):
        return np.nan
    s = s.iloc[-window:] if len(s) >= window else s
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float((s.iloc[-1] - mean) / std)


def percentile_rank_last(series: pd.Series, window: int = 252) -> float:
    s = series.dropna()
    if len(s) < max(20, window // 4):
        return np.nan
    s = s.iloc[-window:] if len(s) >= window else s
    last = s.iloc[-1]
    rank = (s <= last).mean() * 100.0
    return float(rank)


def latest_value(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def format_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def risk_bucket(score: float) -> str:
    if score >= 70:
        return "Defensive"
    elif score >= 40:
        return "Neutral"
    else:
        return "Opportunistic"


def risk_color(score: float) -> str:
    if score >= 70:
        return "#d62728"  # red
    elif score >= 40:
        return "#ff7f0e"  # orange
    else:
        return "#2ca02c"  # green


def make_line_chart(
    df: pd.DataFrame,
    title: str,
    y_title: str = "",
    height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
    )
    return fig


def make_area_chart(
    series: pd.Series,
    title: str,
    y_title: str = "",
    height: int = 350,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            fill="tozeroy",
            mode="lines",
            name=series.name if series.name else "Series"
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
    )
    return fig


# ============================================================
# Data Definitions
# ============================================================
fred_map = {
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "BAMLH0A0HYM2": "US High Yield Spread",
    "VIXCLS": "VIX",
    "NFCI": "Chicago Fed NFCI",
    "FEDFUNDS": "Fed Funds Rate",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
    "GDPC1": "Real GDP",
}

asset_tickers = ["SPY", "QQQ", "TLT", "GLD", "UUP"]

# ============================================================
# Force refresh trigger
# ============================================================
if refresh_button:
    st.cache_data.clear()
    st.success("Cache cleared and data refreshed.")

# ============================================================
# Load FRED Data
# ============================================================
with st.spinner("Loading FRED macro data..."):
    fred_data: Dict[str, pd.DataFrame] = {}
    for sid in fred_map.keys():
        try:
            fred_data[sid] = fetch_fred_series(sid, START_DATE)
        except Exception as e:
            fred_data[sid] = pd.DataFrame(columns=["date", "value"])

# Convert to aligned series
fred_series = {}
for sid, df in fred_data.items():
    if not df.empty:
        s = df.set_index("date")["value"].sort_index()
        fred_series[sid] = s
    else:
        fred_series[sid] = pd.Series(dtype=float)

# ============================================================
# Load Market Prices
# ============================================================
with st.spinner("Loading market prices..."):
    try:
        prices = fetch_yfinance_prices(asset_tickers, START_DATE)
    except Exception:
        prices = pd.DataFrame()

# ============================================================
# Derived Metrics
# ============================================================
spy = prices["SPY"].dropna() if "SPY" in prices.columns else pd.Series(dtype=float)
qqq = prices["QQQ"].dropna() if "QQQ" in prices.columns else pd.Series(dtype=float)
tlt = prices["TLT"].dropna() if "TLT" in prices.columns else pd.Series(dtype=float)
gld = prices["GLD"].dropna() if "GLD" in prices.columns else pd.Series(dtype=float)
uup = prices["UUP"].dropna() if "UUP" in prices.columns else pd.Series(dtype=float)

spy_dd = compute_drawdown(spy) if not spy.empty else pd.Series(dtype=float, name="SPY Drawdown")
qqq_dd = compute_drawdown(qqq) if not qqq.empty else pd.Series(dtype=float, name="QQQ Drawdown")

# FRED series
yc = fred_series.get("T10Y2Y", pd.Series(dtype=float))
hy = fred_series.get("BAMLH0A0HYM2", pd.Series(dtype=float))
vix = fred_series.get("VIXCLS", pd.Series(dtype=float))
nfci = fred_series.get("NFCI", pd.Series(dtype=float))
fedfunds = fred_series.get("FEDFUNDS", pd.Series(dtype=float))
unrate = fred_series.get("UNRATE", pd.Series(dtype=float))
cpi = fred_series.get("CPIAUCSL", pd.Series(dtype=float))

# Inflation YoY
if not cpi.empty:
    cpi_yoy = cpi.pct_change(12) * 100.0
    cpi_yoy.name = "CPI YoY"
else:
    cpi_yoy = pd.Series(dtype=float)

# ============================================================
# Risk Scoring
# ============================================================
risk_components = []

# 1) Yield curve inversion risk
yc_last = latest_value(yc)
if pd.notna(yc_last):
    yc_risk = 100 if yc_last < 0 else max(0, min(100, 50 - yc_last * 20))
    risk_components.append(("Yield Curve", yc_risk, yc_last))

# 2) High yield spread risk
hy_pct = percentile_rank_last(hy, 252 * 3) if not hy.empty else np.nan
if pd.notna(hy_pct):
    hy_risk = hy_pct
    risk_components.append(("HY Spread", hy_risk, latest_value(hy)))

# 3) VIX risk
vix_pct = percentile_rank_last(vix, 252 * 3) if not vix.empty else np.nan
if pd.notna(vix_pct):
    vix_risk = vix_pct
    risk_components.append(("VIX", vix_risk, latest_value(vix)))

# 4) Financial conditions
nfci_pct = percentile_rank_last(nfci, 252 * 3) if not nfci.empty else np.nan
if pd.notna(nfci_pct):
    nfci_risk = nfci_pct
    risk_components.append(("Financial Conditions", nfci_risk, latest_value(nfci)))

# 5) Equity drawdown opportunity / fear
if not spy_dd.empty:
    spy_dd_last = abs(latest_value(spy_dd))
    # drawdown itself is fear: bigger DD can mean more opportunity than risk
    # for total risk score, still include as stress signal
    dd_risk = min(100, spy_dd_last * 3.0)
    risk_components.append(("SPY Drawdown", dd_risk, latest_value(spy_dd)))

if risk_components:
    risk_score = float(np.mean([x[1] for x in risk_components]))
else:
    risk_score = np.nan

current_regime = risk_bucket(risk_score) if pd.notna(risk_score) else "Unknown"

# Opportunity score: high fear but not catastrophic fundamentals
opportunity_elements = []

if not spy_dd.empty:
    opportunity_elements.append(min(100, abs(latest_value(spy_dd)) * 4))

if not vix.empty:
    opportunity_elements.append(percentile_rank_last(vix, 252 * 3))

if not hy.empty:
    opportunity_elements.append(percentile_rank_last(hy, 252 * 3))

if not yc.empty and pd.notna(yc_last):
    # deeply inverted curve reduces opportunity score a bit
    curve_penalty = 30 if yc_last < -0.5 else 10 if yc_last < 0 else 0
else:
    curve_penalty = 0

if opportunity_elements:
    opportunity_score = max(0.0, min(100.0, float(np.mean(opportunity_elements)) - curve_penalty))
else:
    opportunity_score = np.nan

# ============================================================
# Interpretation
# ============================================================
def generate_commentary() -> str:
    if pd.isna(risk_score):
        return "Insufficient data to classify the current market regime."

    comments = []

    if current_regime == "Defensive":
        comments.append(
            "The market environment is elevated-risk. Howard Marks style interpretation: prioritize survival, reduce aggression, and avoid overpaying for optimism."
        )
    elif current_regime == "Neutral":
        comments.append(
            "The market environment is balanced. Howard Marks style interpretation: stay selective, keep quality high, and avoid extreme positioning."
        )
    else:
        comments.append(
            "The market environment is fear-heavy but potentially opportunity-rich. Howard Marks style interpretation: if fundamentals remain intact, dislocation may create attractive entry points."
        )

    if pd.notna(yc_last):
        if yc_last < 0:
            comments.append("Yield curve is inverted, which historically signals tighter future growth conditions.")
        else:
            comments.append("Yield curve is positive, indicating less recession stress than during inversion periods.")

    hy_last = latest_value(hy)
    if pd.notna(hy_last):
        if hy_last > 5.5:
            comments.append("High-yield spreads are wide, meaning credit markets are demanding more risk premium.")
        elif hy_last < 3.5:
            comments.append("Credit spreads are tight, suggesting investor complacency or confidence.")
        else:
            comments.append("Credit spreads are in a middle zone, not yet signaling extreme stress.")

    vix_last = latest_value(vix)
    if pd.notna(vix_last):
        if vix_last > 30:
            comments.append("VIX is high, indicating strong fear and higher short-term uncertainty.")
        elif vix_last < 18:
            comments.append("VIX is relatively calm, which can coincide with complacent market behavior.")
        else:
            comments.append("VIX is elevated but not extreme.")

    if not spy_dd.empty:
        dd_last = latest_value(spy_dd)
        if dd_last < -15:
            comments.append("Equities are meaningfully below prior peaks, often where long-term investors begin screening more aggressively.")
        elif dd_last < -8:
            comments.append("Equities have corrected moderately, which may improve future return potential.")
        else:
            comments.append("Equities remain relatively close to previous highs.")

    return " ".join(comments)


commentary = generate_commentary()

# ============================================================
# Top Summary
# ============================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Market Risk Score", format_num(risk_score, 1))

with c2:
    st.metric("Opportunity Score", format_num(opportunity_score, 1))

with c3:
    st.metric("Current Regime", current_regime)

with c4:
    st.metric("Last Update", datetime.now().strftime("%Y-%m-%d %H:%M"))

st.markdown(
    f"""
<div style="padding:14px;border-radius:12px;background-color:{risk_color(risk_score) if pd.notna(risk_score) else '#999'}22;border:1px solid {risk_color(risk_score) if pd.notna(risk_score) else '#999'};">
<b>Howard Marks Interpretation:</b><br>
{commentary}
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Risk Component Table
# ============================================================
st.subheader("1. Risk Components")

if risk_components:
    risk_df = pd.DataFrame(risk_components, columns=["Component", "Risk Score", "Latest Value"])
    risk_df["Risk Score"] = risk_df["Risk Score"].round(1)
    risk_df["Latest Value"] = risk_df["Latest Value"].round(2)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
else:
    st.warning("No risk component data available.")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Cycle", "Credit & Fear", "Equity Stress", "Defense Assets", "Howard Marks Playbook"]
)

# ============================================================
# Tab 1: Cycle
# ============================================================
with tab1:
    st.markdown("### Market Cycle Indicators")

    col1, col2 = st.columns(2)

    with col1:
        if not yc.empty:
            fig_yc = make_line_chart(
                yc.to_frame(name="10Y-2Y Spread"),
                title="Yield Curve (10Y - 2Y)",
                y_title="Spread (%)"
            )
            fig_yc.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_yc, use_container_width=True)
        else:
            st.info("Yield curve data unavailable.")

    with col2:
        if not fedfunds.empty:
            fig_ff = make_line_chart(
                fedfunds.to_frame(name="Fed Funds"),
                title="Fed Funds Rate",
                y_title="Rate (%)"
            )
            st.plotly_chart(fig_ff, use_container_width=True)
        else:
            st.info("Fed Funds data unavailable.")

    col3, col4 = st.columns(2)

    with col3:
        if not unrate.empty:
            fig_un = make_line_chart(
                unrate.to_frame(name="Unemployment Rate"),
                title="Unemployment Rate",
                y_title="Rate (%)"
            )
            st.plotly_chart(fig_un, use_container_width=True)
        else:
            st.info("Unemployment data unavailable.")

    with col4:
        if not cpi_yoy.empty:
            fig_cpi = make_line_chart(
                cpi_yoy.to_frame(name="CPI YoY"),
                title="Inflation (CPI YoY)",
                y_title="YoY (%)"
            )
            st.plotly_chart(fig_cpi, use_container_width=True)
        else:
            st.info("Inflation data unavailable.")

# ============================================================
# Tab 2: Credit & Fear
# ============================================================
with tab2:
    st.markdown("### Credit Stress and Fear Indicators")

    col1, col2 = st.columns(2)

    with col1:
        if not hy.empty:
            fig_hy = make_line_chart(
                hy.to_frame(name="HY Spread"),
                title="US High Yield Spread",
                y_title="Spread (%)"
            )
            st.plotly_chart(fig_hy, use_container_width=True)
        else:
            st.info("HY spread data unavailable.")

    with col2:
        if not vix.empty:
            fig_vix = make_line_chart(
                vix.to_frame(name="VIX"),
                title="VIX",
                y_title="Level"
            )
            st.plotly_chart(fig_vix, use_container_width=True)
        else:
            st.info("VIX data unavailable.")

    if not nfci.empty:
        fig_nfci = make_line_chart(
            nfci.to_frame(name="NFCI"),
            title="National Financial Conditions Index",
            y_title="Index"
        )
        st.plotly_chart(fig_nfci, use_container_width=True)
    else:
        st.info("Financial conditions data unavailable.")

# ============================================================
# Tab 3: Equity Stress
# ============================================================
with tab3:
    st.markdown("### Equity Market Stress")

    col1, col2 = st.columns(2)

    with col1:
        if not spy.empty or not qqq.empty:
            perf_df = pd.DataFrame(index=prices.index)
            if not spy.empty:
                perf_df["SPY"] = normalize_to_100(prices["SPY"])
            if not qqq.empty:
                perf_df["QQQ"] = normalize_to_100(prices["QQQ"])

            fig_perf = make_line_chart(
                perf_df.dropna(how="all"),
                title="Relative Performance (Base = 100)",
                y_title="Normalized"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Equity price data unavailable.")

    with col2:
        dd_df = pd.DataFrame(index=prices.index)
        if not spy_dd.empty:
            dd_df["SPY Drawdown"] = spy_dd
        if not qqq_dd.empty:
            dd_df["QQQ Drawdown"] = qqq_dd

        if not dd_df.empty:
            fig_dd = make_line_chart(
                dd_df.dropna(how="all"),
                title="Drawdown vs Previous Peak",
                y_title="Drawdown (%)"
            )
            fig_dd.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("Drawdown data unavailable.")

# ============================================================
# Tab 4: Defense Assets
# ============================================================
with tab4:
    st.markdown("### Defense and Safe-Haven Proxies")

    defense_df = pd.DataFrame(index=prices.index)
    if "TLT" in prices.columns:
        defense_df["TLT"] = normalize_to_100(prices["TLT"])
    if "GLD" in prices.columns:
        defense_df["GLD"] = normalize_to_100(prices["GLD"])
    if "UUP" in prices.columns:
        defense_df["UUP"] = normalize_to_100(prices["UUP"])
    if "SPY" in prices.columns:
        defense_df["SPY"] = normalize_to_100(prices["SPY"])

    if not defense_df.empty:
        fig_def = make_line_chart(
            defense_df.dropna(how="all"),
            title="Defense Assets vs Equity (Base = 100)",
            y_title="Normalized"
        )
        st.plotly_chart(fig_def, use_container_width=True)
    else:
        st.info("Defense asset data unavailable.")

    st.markdown(
        """
**Reading guide**
- **TLT strength**: bond rally / growth slowdown / disinflation fear
- **GLD strength**: macro uncertainty / inflation hedge / policy distrust
- **UUP strength**: dollar strength / liquidity stress / global risk-off
- **SPY weakness + GLD/TLT resilience**: classic defensive rotation
"""
    )

# ============================================================
# Tab 5: Howard Marks Playbook
# ============================================================
with tab5:
    st.markdown("### Howard Marks Style Playbook")

    regime = current_regime
    st.write(f"**Current regime:** {regime}")

    if regime == "Defensive":
        st.markdown(
            """
#### Suggested stance
- Reduce aggressive beta exposure
- Prefer quality balance sheets
- Avoid overpaying for stories
- Keep some dry powder
- Watch for credit accidents

#### Typical interpretation
This is not the time to be a hero blindly. Focus on survival first, then selectivity.
"""
        )
    elif regime == "Neutral":
        st.markdown(
            """
#### Suggested stance
- Stay balanced
- Add selectively on weakness
- Prefer profitable businesses
- Maintain diversification
- Track credit and liquidity carefully

#### Typical interpretation
No extreme signal yet. Discipline and valuation awareness matter more than bold macro bets.
"""
        )
    else:
        st.markdown(
            """
#### Suggested stance
- Screen for forced-selling situations
- Focus on quality names trading below intrinsic value
- Scale in gradually instead of all-in
- Use fear to build long-term positions
- Confirm balance-sheet resilience

#### Typical interpretation
Great opportunities often appear when headlines feel worst. Fear alone is not enough; price relative to value matters.
"""
        )

    st.markdown("### Practical Checklist")
    checklist = pd.DataFrame({
        "Question": [
            "Is fear high?",
            "Are credit spreads widening?",
            "Is valuation improving after drawdown?",
            "Is the balance sheet strong?",
            "Is the market pricing in too much optimism or too much fear?"
        ],
        "Why It Matters": [
            "Fear creates dislocations.",
            "Credit often signals real stress earlier than equity headlines.",
            "Lower prices can improve forward returns.",
            "Weak balance sheets fail first in tough cycles.",
            "Howard Marks emphasizes price vs value, not story vs story."
        ]
    })
    st.dataframe(checklist, use_container_width=True, hide_index=True)

# ============================================================
# Bottom Notes
# ============================================================
st.markdown("---")
st.markdown(
    """
### Notes
- This dashboard is a **market regime / risk-awareness tool**, not a buy-sell oracle.
- Howard Marks style investing is less about precise forecasting and more about:
  - understanding cycles,
  - controlling risk,
  - and acting rationally when others become emotional.
- Best used together with your own **FRED macro dashboard**, valuation work, and MDD-based screening process.
"""
)

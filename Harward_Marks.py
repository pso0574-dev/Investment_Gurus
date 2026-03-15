# streamlit_app.py
# ============================================================
# Howard Marks Style Pro Dashboard
# - Macro regime / risk / opportunity / action signal
# - Asset allocation hint
# - Watchlist screener for dislocation opportunities
#
# Install:
#   pip install streamlit pandas numpy plotly requests yfinance
#
# Run:
#   streamlit run streamlit_app.py
#
# Optional:
#   export FRED_API_KEY="YOUR_KEY"
# ============================================================

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Tuple

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
    page_title="Howard Marks Pro Dashboard",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Howard Marks Pro Dashboard")
st.caption("Cycle / Risk / Opportunity / Allocation / Watchlist")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

lookback_years = st.sidebar.slider("Lookback Years", 3, 20, 10)
refresh_button = st.sidebar.button("Refresh Data")

watchlist_default = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "NFLX", "AMD", "ADBE", "CRM", "PLTR", "VRSK",
    "COST", "QQQ", "SPY", "IWM", "SMH", "SOXX"
]

watchlist_text = st.sidebar.text_area(
    "Watchlist Tickers (comma separated)",
    value=", ".join(watchlist_default),
    height=140
)

show_top_n = st.sidebar.slider("Top Watchlist Rows", 5, 30, 12)

# ============================================================
# Constants
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
TODAY = pd.Timestamp.today().normalize()
START_DATE = TODAY - pd.DateOffset(years=lookback_years)

FRED_SERIES = {
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "BAMLH0A0HYM2": "US High Yield Spread",
    "VIXCLS": "VIX",
    "NFCI": "Chicago Fed NFCI",
    "FEDFUNDS": "Fed Funds Rate",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
}

ASSET_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "UUP"]
USER_WATCHLIST = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]

# ============================================================
# Cache Refresh
# ============================================================
if refresh_button:
    st.cache_data.clear()
    st.success("Cache cleared.")

# ============================================================
# Helpers
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, start_date: pd.Timestamp) -> pd.DataFrame:
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
def fetch_prices(tickers: List[str], start_date: pd.Timestamp) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    close_df = pd.DataFrame()

    if len(tickers) == 1:
        try:
            close_df[tickers[0]] = data["Close"]
        except Exception:
            return pd.DataFrame()
    else:
        for t in tickers:
            try:
                close_df[t] = data[t]["Close"]
            except Exception:
                pass

    close_df = close_df.dropna(how="all")
    return close_df


def latest_value(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def normalize_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series
    return series / s.iloc[0] * 100.0


def compute_drawdown(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    peak = s.cummax()
    return (s / peak - 1.0) * 100.0


def percentile_rank_last(series: pd.Series, window: int = 756) -> float:
    s = series.dropna()
    if len(s) < 30:
        return np.nan
    s = s.iloc[-window:] if len(s) >= window else s
    last = s.iloc[-1]
    return float((s <= last).mean() * 100.0)


def format_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def risk_bucket(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 70:
        return "Defensive"
    elif score >= 45:
        return "Neutral"
    return "Opportunistic"


def regime_color(regime: str) -> str:
    if regime == "Defensive":
        return "#d62728"
    if regime == "Neutral":
        return "#ff7f0e"
    if regime == "Opportunistic":
        return "#2ca02c"
    return "#999999"


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def annualized_volatility(series: pd.Series, window: int = 63) -> float:
    s = series.dropna()
    if len(s) < window + 1:
        return np.nan
    r = s.pct_change().dropna().iloc[-window:]
    return float(r.std() * np.sqrt(252) * 100.0)


def total_return(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-days - 1] - 1.0) * 100.0)


def distance_from_ma(series: pd.Series, window: int) -> float:
    s = series.dropna()
    if len(s) < window:
        return np.nan
    ma = s.rolling(window).mean().iloc[-1]
    if pd.isna(ma) or ma == 0:
        return np.nan
    return float((s.iloc[-1] / ma - 1.0) * 100.0)


def line_chart(df: pd.DataFrame, title: str, y_title: str = "", height: int = 380) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines",
            name=col
        ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Date",
        yaxis_title=y_title,
    )
    return fig


def bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        text=df[y_col].round(1),
        textposition="outside",
        name=y_col
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="",
        yaxis_title=y_col,
    )
    return fig


# ============================================================
# Load FRED
# ============================================================
with st.spinner("Loading FRED macro data..."):
    fred_raw: Dict[str, pd.DataFrame] = {}
    for sid in FRED_SERIES.keys():
        try:
            fred_raw[sid] = fetch_fred_series(sid, START_DATE)
        except Exception:
            fred_raw[sid] = pd.DataFrame(columns=["date", "value"])

fred = {}
for sid, df in fred_raw.items():
    if not df.empty:
        fred[sid] = df.set_index("date")["value"].sort_index()
    else:
        fred[sid] = pd.Series(dtype=float)

# ============================================================
# Load Prices
# ============================================================
all_tickers = sorted(list(set(ASSET_TICKERS + USER_WATCHLIST)))

with st.spinner("Loading market prices..."):
    prices = fetch_prices(all_tickers, START_DATE)

# ============================================================
# Core Series
# ============================================================
yc = fred.get("T10Y2Y", pd.Series(dtype=float))
hy = fred.get("BAMLH0A0HYM2", pd.Series(dtype=float))
vix = fred.get("VIXCLS", pd.Series(dtype=float))
nfci = fred.get("NFCI", pd.Series(dtype=float))
fedfunds = fred.get("FEDFUNDS", pd.Series(dtype=float))
unrate = fred.get("UNRATE", pd.Series(dtype=float))
cpi = fred.get("CPIAUCSL", pd.Series(dtype=float))

if not cpi.empty:
    cpi_yoy = cpi.pct_change(12) * 100.0
    cpi_yoy.name = "CPI YoY"
else:
    cpi_yoy = pd.Series(dtype=float)

# ============================================================
# Risk Score
# ============================================================
risk_components: List[Tuple[str, float, float]] = []

yc_last = latest_value(yc)
if pd.notna(yc_last):
    yc_risk = 100 if yc_last < 0 else max(0, min(100, 50 - yc_last * 20))
    risk_components.append(("Yield Curve", yc_risk, yc_last))

hy_last = latest_value(hy)
hy_pct = percentile_rank_last(hy, 756) if not hy.empty else np.nan
if pd.notna(hy_pct):
    risk_components.append(("HY Spread", hy_pct, hy_last))

vix_last = latest_value(vix)
vix_pct = percentile_rank_last(vix, 756) if not vix.empty else np.nan
if pd.notna(vix_pct):
    risk_components.append(("VIX", vix_pct, vix_last))

nfci_last = latest_value(nfci)
nfci_pct = percentile_rank_last(nfci, 756) if not nfci.empty else np.nan
if pd.notna(nfci_pct):
    risk_components.append(("Financial Conditions", nfci_pct, nfci_last))

if "SPY" in prices.columns:
    spy_dd = compute_drawdown(prices["SPY"])
    spy_dd_last = abs(latest_value(spy_dd))
    dd_risk = min(100, spy_dd_last * 3.0)
    risk_components.append(("SPY Drawdown", dd_risk, latest_value(spy_dd)))
else:
    spy_dd = pd.Series(dtype=float)

if risk_components:
    risk_score = float(np.mean([x[1] for x in risk_components]))
else:
    risk_score = np.nan

regime = risk_bucket(risk_score)

# ============================================================
# Opportunity Score
# ============================================================
opportunity_parts = []

if not spy_dd.empty:
    opportunity_parts.append(min(100, abs(latest_value(spy_dd)) * 4.0))

if pd.notna(vix_pct):
    opportunity_parts.append(vix_pct)

if pd.notna(hy_pct):
    opportunity_parts.append(hy_pct)

curve_penalty = 0
if pd.notna(yc_last):
    if yc_last < -0.5:
        curve_penalty = 30
    elif yc_last < 0:
        curve_penalty = 10

if opportunity_parts:
    opportunity_score = max(0.0, min(100.0, float(np.mean(opportunity_parts)) - curve_penalty))
else:
    opportunity_score = np.nan

# ============================================================
# Action Signal
# ============================================================
def get_action_signal(regime_name: str, opp_score: float) -> str:
    if regime_name == "Defensive":
        if pd.notna(opp_score) and opp_score >= 70:
            return "Selective Accumulation"
        return "Capital Preservation"
    elif regime_name == "Neutral":
        if pd.notna(opp_score) and opp_score >= 65:
            return "Build Watchlist / Scale In"
        return "Balanced Positioning"
    else:
        if pd.notna(opp_score) and opp_score >= 60:
            return "Aggressive Watchlist Accumulation"
        return "Selective Opportunity Hunting"

action_signal = get_action_signal(regime, opportunity_score)

# ============================================================
# Allocation Hint
# ============================================================
def get_allocation_hint(regime_name: str, opp_score: float) -> pd.DataFrame:
    if regime_name == "Defensive":
        data = {
            "Asset": ["SPY/QQQ", "TLT", "GLD", "Cash"],
            "Suggested Weight (%)": [20, 30, 20, 30]
        }
    elif regime_name == "Neutral":
        data = {
            "Asset": ["SPY/QQQ", "TLT", "GLD", "Cash"],
            "Suggested Weight (%)": [45, 20, 15, 20]
        }
    else:
        if pd.notna(opp_score) and opp_score >= 60:
            data = {
                "Asset": ["SPY/QQQ", "TLT", "GLD", "Cash"],
                "Suggested Weight (%)": [60, 15, 10, 15]
            }
        else:
            data = {
                "Asset": ["SPY/QQQ", "TLT", "GLD", "Cash"],
                "Suggested Weight (%)": [50, 20, 10, 20]
            }
    return pd.DataFrame(data)

allocation_df = get_allocation_hint(regime, opportunity_score)

# ============================================================
# Commentary
# ============================================================
def make_commentary() -> str:
    parts = []

    if regime == "Defensive":
        parts.append("시장 스트레스가 높은 구간입니다. 하워드 막스 관점에서는 수익 극대화보다 생존과 선택성이 우선입니다.")
    elif regime == "Neutral":
        parts.append("시장은 극단적이지 않은 중립 구간입니다. 강한 확신 베팅보다 균형과 선별이 중요합니다.")
    elif regime == "Opportunistic":
        parts.append("공포와 변동성이 기회로 전환될 수 있는 구간입니다. 단, 가격 대비 가치 확인이 필요합니다.")
    else:
        parts.append("데이터가 충분하지 않아 현재 시장 국면 판별이 제한됩니다.")

    if pd.notna(yc_last):
        if yc_last < 0:
            parts.append("장단기 금리차 역전은 향후 성장 둔화 신호로 해석될 수 있습니다.")
        else:
            parts.append("장단기 금리차는 역전 상태가 아니어서 극단적 침체 신호는 완화된 상태입니다.")

    if pd.notna(hy_last):
        if hy_last > 5.5:
            parts.append("하이일드 스프레드 확대는 크레딧 시장이 리스크 프리미엄을 더 요구하고 있음을 의미합니다.")
        elif hy_last < 3.5:
            parts.append("하이일드 스프레드가 낮아 시장 안도감이나 안일함이 반영되었을 수 있습니다.")

    if pd.notna(vix_last):
        if vix_last > 30:
            parts.append("VIX가 높아 단기 공포와 불확실성이 큰 구간입니다.")
        elif vix_last < 18:
            parts.append("VIX가 낮아 시장이 비교적 안도하는 구간일 수 있습니다.")

    if not spy_dd.empty:
        dd_last = latest_value(spy_dd)
        if dd_last < -15:
            parts.append("주식시장이 이전 고점 대비 크게 밀려 장기 투자자에게는 관찰 가치가 높아진 구간입니다.")
        elif dd_last < -8:
            parts.append("주식시장이 중간 수준 조정을 거치며 미래 기대수익률이 다소 개선될 수 있습니다.")

    return " ".join(parts)

commentary = make_commentary()

# ============================================================
# Watchlist Screener
# ============================================================
def build_watchlist_table(price_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        if ticker not in price_df.columns:
            continue

        s = price_df[ticker].dropna()
        if len(s) < 210:
            continue

        last_price = s.iloc[-1]
        dd = compute_drawdown(s)
        dd_last = latest_value(dd)

        ma50 = moving_average(s, 50).iloc[-1]
        ma200 = moving_average(s, 200).iloc[-1]
        ret_1m = total_return(s, 21)
        ret_3m = total_return(s, 63)
        ret_6m = total_return(s, 126)
        vol_3m = annualized_volatility(s, 63)
        dist_50 = distance_from_ma(s, 50)
        dist_200 = distance_from_ma(s, 200)

        # Opportunity score per ticker
        ticker_score = 0.0

        # Larger drawdown => more interesting
        if pd.notna(dd_last):
            ticker_score += min(40, abs(min(0, dd_last)) * 1.2)

        # Below 200DMA => more dislocated
        if pd.notna(dist_200) and dist_200 < 0:
            ticker_score += min(20, abs(dist_200) * 0.8)

        # Short-term weakness
        if pd.notna(ret_3m) and ret_3m < 0:
            ticker_score += min(15, abs(ret_3m) * 0.4)

        # Avoid too-hot momentum
        if pd.notna(ret_6m) and ret_6m > 30:
            ticker_score -= 8

        # Higher vol = more risk but potentially more opportunity
        if pd.notna(vol_3m):
            ticker_score += min(10, vol_3m * 0.15)

        # Bonus if above 50DMA improving from weak long-term base
        if pd.notna(dist_50) and pd.notna(dist_200):
            if dist_50 > 0 and dist_200 < 0:
                ticker_score += 8

        # Penalty if way above both averages
        if pd.notna(dist_50) and pd.notna(dist_200):
            if dist_50 > 10 and dist_200 > 15:
                ticker_score -= 10

        ticker_score = max(0, min(100, ticker_score))

        rows.append({
            "Ticker": ticker,
            "Price": last_price,
            "Drawdown %": dd_last,
            "1M Return %": ret_1m,
            "3M Return %": ret_3m,
            "6M Return %": ret_6m,
            "Dist vs 50DMA %": dist_50,
            "Dist vs 200DMA %": dist_200,
            "3M Vol %": vol_3m,
            "Opportunity Score": ticker_score,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Opportunity Score", ascending=False).reset_index(drop=True)
    return df

watchlist_df = build_watchlist_table(prices, USER_WATCHLIST)

# ============================================================
# Top Summary
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Risk Score", format_num(risk_score, 1))
with c2:
    st.metric("Opportunity Score", format_num(opportunity_score, 1))
with c3:
    st.metric("Regime", regime)
with c4:
    st.metric("Action Signal", action_signal)
with c5:
    st.metric("Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

st.markdown(
    f"""
<div style="padding:14px;border-radius:12px;background-color:{regime_color(regime)}22;border:1px solid {regime_color(regime)};">
<b>Howard Marks Interpretation</b><br>
{commentary}
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Risk Components Table
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Cycle",
    "Credit & Fear",
    "Equity Stress",
    "Allocation",
    "Watchlist Screener",
    "Playbook"
])

# ============================================================
# Tab 1 - Cycle
# ============================================================
with tab1:
    st.markdown("### Market Cycle Indicators")

    col1, col2 = st.columns(2)

    with col1:
        if not yc.empty:
            fig = line_chart(yc.to_frame("10Y-2Y Spread"), "Yield Curve (10Y - 2Y)", "Spread (%)")
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Yield curve data unavailable.")

    with col2:
        if not fedfunds.empty:
            fig = line_chart(fedfunds.to_frame("Fed Funds Rate"), "Fed Funds Rate", "Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fed funds data unavailable.")

    col3, col4 = st.columns(2)

    with col3:
        if not unrate.empty:
            fig = line_chart(unrate.to_frame("Unemployment Rate"), "Unemployment Rate", "Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unemployment data unavailable.")

    with col4:
        if not cpi_yoy.empty:
            fig = line_chart(cpi_yoy.to_frame("CPI YoY"), "Inflation (CPI YoY)", "YoY (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CPI data unavailable.")

# ============================================================
# Tab 2 - Credit & Fear
# ============================================================
with tab2:
    st.markdown("### Credit and Fear Indicators")

    col1, col2 = st.columns(2)

    with col1:
        if not hy.empty:
            fig = line_chart(hy.to_frame("HY Spread"), "US High Yield Spread", "Spread (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("HY spread unavailable.")

    with col2:
        if not vix.empty:
            fig = line_chart(vix.to_frame("VIX"), "VIX", "Level")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("VIX unavailable.")

    if not nfci.empty:
        fig = line_chart(nfci.to_frame("NFCI"), "National Financial Conditions Index", "Index")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("NFCI unavailable.")

# ============================================================
# Tab 3 - Equity Stress
# ============================================================
with tab3:
    st.markdown("### Equity Stress and Relative Performance")

    perf_df = pd.DataFrame(index=prices.index)
    for t in ["SPY", "QQQ", "TLT", "GLD"]:
        if t in prices.columns:
            perf_df[t] = normalize_to_100(prices[t])

    if not perf_df.empty:
        fig = line_chart(perf_df, "Relative Performance (Base=100)", "Normalized")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough price data.")

    dd_df = pd.DataFrame(index=prices.index)
    for t in ["SPY", "QQQ"]:
        if t in prices.columns:
            dd_df[f"{t} Drawdown"] = compute_drawdown(prices[t])

    if not dd_df.empty:
        fig = line_chart(dd_df, "Drawdown vs Previous Peak", "Drawdown (%)")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Drawdown data unavailable.")

# ============================================================
# Tab 4 - Allocation
# ============================================================
with tab4:
    st.markdown("### Allocation Hint")

    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    alloc_fig = bar_chart(allocation_df, "Asset", "Suggested Weight (%)", "Suggested Allocation")
    st.plotly_chart(alloc_fig, use_container_width=True)

    st.markdown(
        """
**Reading guide**
- **Defensive**: quality, patience, liquidity 확보
- **Neutral**: 균형 유지, 무리한 추격 금지
- **Opportunistic**: 공포 확대 시 분할매수 우선
- 이 비중은 **교육용 힌트**이며 개인 자산 규모/세금/통화 노출은 별도로 조정해야 합니다.
"""
    )

# ============================================================
# Tab 5 - Watchlist Screener
# ============================================================
with tab5:
    st.markdown("### Top Opportunity Watchlist")

    if watchlist_df.empty:
        st.warning("Watchlist results unavailable.")
    else:
        show_df = watchlist_df.head(show_top_n).copy()
        num_cols = [c for c in show_df.columns if c != "Ticker"]
        show_df[num_cols] = show_df[num_cols].round(2)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        fig = bar_chart(show_df, "Ticker", "Opportunity Score", "Watchlist Opportunity Score")
        st.plotly_chart(fig, use_container_width=True)

        selected_ticker = st.selectbox("Select ticker for detailed chart", show_df["Ticker"].tolist())

        if selected_ticker in prices.columns:
            s = prices[selected_ticker].dropna()
            ma50 = moving_average(s, 50)
            ma200 = moving_average(s, 200)

            chart_df = pd.DataFrame({
                selected_ticker: s,
                "50DMA": ma50,
                "200DMA": ma200,
            }).dropna(how="all")

            fig = line_chart(chart_df, f"{selected_ticker} Price with 50DMA / 200DMA", "Price")
            st.plotly_chart(fig, use_container_width=True)

            dd = compute_drawdown(s)
            if not dd.empty:
                fig = line_chart(dd.to_frame(f"{selected_ticker} Drawdown"), f"{selected_ticker} Drawdown", "Drawdown (%)")
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 6 - Playbook
# ============================================================
with tab6:
    st.markdown("### Howard Marks Style Playbook")

    st.write(f"**Current Regime:** {regime}")
    st.write(f"**Action Signal:** {action_signal}")

    if regime == "Defensive":
        st.markdown(
            """
#### Suggested stance
- Reduce aggressive beta exposure
- Prefer quality balance sheets
- Avoid expensive narratives
- Maintain dry powder
- Watch credit stress carefully

#### Practical reading
이 구간에서는 공격보다 방어가 먼저입니다. 다만 급락이 심한 우량주가 나타나면 감시 목록을 강화할 시기입니다.
"""
        )
    elif regime == "Neutral":
        st.markdown(
            """
#### Suggested stance
- Stay balanced
- Add selectively on weakness
- Focus on profitable and durable businesses
- Avoid emotional market timing
- Let valuation guide position sizing

#### Practical reading
강한 확신 베팅보다는 포트폴리오의 질을 유지하면서 약세 구간에 천천히 비중을 늘리는 접근이 좋습니다.
"""
        )
    elif regime == "Opportunistic":
        st.markdown(
            """
#### Suggested stance
- Screen forced-selling opportunities
- Scale in gradually, not all-in
- Prefer quality companies with strong cash flows
- Buy fear, but only with valuation discipline
- Monitor whether macro stress is stabilizing

#### Practical reading
공포는 기회가 될 수 있지만, 싼 이유가 영구 훼손인지 일시적 과민반응인지 구분해야 합니다.
"""
        )
    else:
        st.info("Insufficient regime data.")

    checklist_df = pd.DataFrame({
        "Question": [
            "Is fear high?",
            "Are credit spreads widening?",
            "Is the market already pricing bad news?",
            "Is the business quality still intact?",
            "Can I scale in instead of betting all at once?"
        ],
        "Why It Matters": [
            "Fear creates dislocations.",
            "Credit often leads equity stress.",
            "Price vs value is the core question.",
            "A weak balance sheet breaks first.",
            "Gradual entries reduce timing risk."
        ]
    })
    st.dataframe(checklist_df, use_container_width=True, hide_index=True)

# ============================================================
# Bottom Notes
# ============================================================
st.markdown("---")
st.markdown(
    """
### Notes
- This is a **market regime / risk awareness dashboard**, not a guaranteed timing tool.
- Howard Marks style investing is about:
  - understanding cycles,
  - controlling risk,
  - comparing price vs value,
  - and acting rationally when others are emotional.
- Best used together with your own:
  - macro dashboard,
  - MDD screen,
  - quality / ROE / FCF analysis,
  - and staged buying rules.
"""
)

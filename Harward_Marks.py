# streamlit_app.py
# ============================================================
# Howard Marks Hybrid Quant Dashboard
# - FRED Macro + Quant Screener + Portfolio Builder
# - Concept:
#   Howard Marks = Cycle + Risk Control + Contrarian + Value
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy yfinance plotly requests
# ============================================================

from __future__ import annotations

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
# Page config
# ============================================================
st.set_page_config(
    page_title="Howard Marks Hybrid Quant Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Howard Marks Hybrid Quant Dashboard")
st.caption(
    "FRED Macro + Value + Quality + Risk + Contrarian + Momentum"
)

# ============================================================
# Constants
# ============================================================
FRED_SERIES = {
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "FEDFUNDS": "Fed Funds Rate",
    "BAMLH0A0HYM2": "US High Yield Spread",
    "VIXCLS": "VIX Index",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
}

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "COST", "NFLX", "AMD", "ADBE", "INTU", "QCOM", "AMGN", "TXN",
    "ISRG", "BKNG", "PANW", "CRWD", "PLTR", "APP", "VRSK", "LRCX",
    "KLAC", "MU", "ANET", "MELI", "ASML", "CSCO", "ADP", "CMCSA",
    "PEP", "LIN", "TMUS", "VRTX", "MDLZ", "GILD", "ADI", "ABNB",
]

MACRO_SERIES_ORDER = ["DGS10", "DGS2", "FEDFUNDS", "BAMLH0A0HYM2", "VIXCLS", "UNRATE", "CPIAUCSL"]

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

universe_mode = st.sidebar.selectbox(
    "Universe mode",
    ["Default large-cap universe", "Manual tickers"],
    index=0,
)

manual_tickers = st.sidebar.text_area(
    "Manual tickers (comma-separated)",
    value="AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,TSLA,COST,NFLX,PLTR,APP,VRSK",
    height=100,
)

top_n = st.sidebar.slider("Top N portfolio candidates", 5, 20, 10)
price_period = st.sidebar.selectbox("Price history period", ["2y", "3y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Factor weights")

w_value = st.sidebar.slider("Value", 0.0, 1.0, 0.25, 0.05)
w_quality = st.sidebar.slider("Quality", 0.0, 1.0, 0.20, 0.05)
w_risk = st.sidebar.slider("Risk", 0.0, 1.0, 0.20, 0.05)
w_contrarian = st.sidebar.slider("Contrarian", 0.0, 1.0, 0.15, 0.05)
w_momentum = st.sidebar.slider("Momentum", 0.0, 1.0, 0.15, 0.05)
w_macro = st.sidebar.slider("Macro", 0.0, 1.0, 0.05, 0.05)

weight_sum = w_value + w_quality + w_risk + w_contrarian + w_momentum + w_macro
if weight_sum <= 0:
    weight_sum = 1.0

weights = {
    "value": w_value / weight_sum,
    "quality": w_quality / weight_sum,
    "risk": w_risk / weight_sum,
    "contrarian": w_contrarian / weight_sum,
    "momentum": w_momentum / weight_sum,
    "macro": w_macro / weight_sum,
}

st.sidebar.markdown("---")
st.sidebar.subheader("Risk controls")

max_allowed_vol = st.sidebar.slider("Max annualized volatility", 0.10, 1.00, 0.45, 0.01)
min_roe = st.sidebar.slider("Min ROE (if available)", -0.20, 0.50, 0.05, 0.01)
min_market_cap_b = st.sidebar.slider("Min market cap (B USD, if available)", 0, 1000, 10, 5)

run_button = st.sidebar.button("Run analysis", type="primary")

# ============================================================
# Helpers
# ============================================================
def parse_tickers(mode: str, manual_text: str) -> List[str]:
    if mode == "Default large-cap universe":
        tickers = DEFAULT_UNIVERSE
    else:
        tickers = [x.strip().upper() for x in manual_text.split(",") if x.strip()]
    # deduplicate while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@st.cache_data(ttl=60 * 60)
def fetch_fred_series(series_id: str) -> pd.Series:
    url = fred_csv_url(series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd, "compat") and hasattr(pd.compat, "StringIO") else pd.read_csv(pd.io.common.StringIO(r.text))
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df.set_index("DATE")[series_id].dropna()
    return s


@st.cache_data(ttl=60 * 60)
def fetch_all_fred() -> pd.DataFrame:
    data = {}
    for sid in MACRO_SERIES_ORDER:
        try:
            data[sid] = fetch_fred_series(sid)
        except Exception:
            data[sid] = pd.Series(dtype=float)
    df = pd.concat(data, axis=1).sort_index()
    return df


@st.cache_data(ttl=60 * 30)
def fetch_prices(tickers: List[str], period: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if len(tickers) == 1:
        # yfinance returns flat columns for single ticker
        close = data[["Close"]].copy()
        close.columns = tickers
        return close.dropna(how="all")

    close_map = {}
    for t in tickers:
        try:
            close_map[t] = data[t]["Close"]
        except Exception:
            pass

    if not close_map:
        return pd.DataFrame()

    close = pd.DataFrame(close_map).dropna(how="all")
    return close


@st.cache_data(ttl=60 * 60)
def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, start=1):
        try:
            tk = yf.Ticker(t)
            info = tk.info if tk.info is not None else {}
        except Exception:
            info = {}

        row = {
            "ticker": t,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "freeCashflow": info.get("freeCashflow"),
            "currentPrice": info.get("currentPrice"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "recommendationKey": info.get("recommendationKey"),
            "beta": info.get("beta"),
        }

        price = row["currentPrice"]
        shares = row["sharesOutstanding"]
        fcf = row["freeCashflow"]

        if price and shares and fcf:
            mcap = price * shares
            row["fcfYield"] = fcf / mcap if mcap else np.nan
        else:
            row["fcfYield"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def annualized_volatility(price_series: pd.Series) -> float:
    rets = price_series.pct_change().dropna()
    if len(rets) < 20:
        return np.nan
    return rets.std() * np.sqrt(252)


def max_drawdown(price_series: pd.Series) -> float:
    if price_series.dropna().empty:
        return np.nan
    wealth = price_series / price_series.dropna().iloc[0]
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min()


def current_drawdown(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.empty:
        return np.nan
    peak = s.cummax().iloc[-1]
    # wrong if using cummax last; use full running peak current comparison
    running_peak = s.cummax()
    return s.iloc[-1] / running_peak.iloc[-1] - 1.0


def current_drawdown_from_peak(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.empty:
        return np.nan
    peak = s.max()
    return s.iloc[-1] / peak - 1.0


def rsi(series: pd.Series, window: int = 14) -> float:
    s = series.dropna()
    if len(s) < window + 5:
        return np.nan
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(window).mean()
    avg_down = down.rolling(window).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])


def total_return(price_series: pd.Series, days: int) -> float:
    s = price_series.dropna()
    if len(s) < days + 1:
        return np.nan
    return s.iloc[-1] / s.iloc[-days - 1] - 1.0


def ma_gap(price_series: pd.Series, window: int) -> float:
    s = price_series.dropna()
    if len(s) < window + 1:
        return np.nan
    ma = s.rolling(window).mean().iloc[-1]
    if pd.isna(ma) or ma == 0:
        return np.nan
    return s.iloc[-1] / ma - 1.0


def safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = series.copy()
    rank = s.rank(pct=True, ascending=ascending)
    return rank.fillna(0.5)


def clip_series(s: pd.Series, low: float = None, high: float = None) -> pd.Series:
    out = s.copy()
    if low is not None:
        out = out.clip(lower=low)
    if high is not None:
        out = out.clip(upper=high)
    return out


def compute_macro_state(fred_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    df = fred_df.copy().sort_index()

    # Forward-fill for alignment
    df = df.ffill()

    latest = {}
    for col in df.columns:
        latest[col] = float(df[col].dropna().iloc[-1]) if not df[col].dropna().empty else np.nan

    yc = latest.get("DGS10", np.nan) - latest.get("DGS2", np.nan)
    hy = latest.get("BAMLH0A0HYM2", np.nan)
    vix = latest.get("VIXCLS", np.nan)
    fed = latest.get("FEDFUNDS", np.nan)
    unrate = latest.get("UNRATE", np.nan)

    score = 0

    if not np.isnan(yc):
        if yc < 0:
            score += 3
        elif yc < 0.5:
            score += 2
        elif yc < 1.0:
            score += 1

    if not np.isnan(hy):
        if hy > 6:
            score += 3
        elif hy > 4.5:
            score += 2
        elif hy > 3.5:
            score += 1

    if not np.isnan(vix):
        if vix > 35:
            score += 3
        elif vix > 25:
            score += 2
        elif vix > 20:
            score += 1

    if not np.isnan(fed):
        if fed > 5:
            score += 2
        elif fed > 3:
            score += 1

    if not np.isnan(unrate):
        if unrate > 5:
            score += 2
        elif unrate > 4.2:
            score += 1

    # Regime
    if score >= 9:
        regime = "Risk-Off"
    elif score >= 5:
        regime = "Cautious"
    else:
        regime = "Risk-On"

    macro = {
        "yield_curve_spread": yc,
        "high_yield_spread": hy,
        "vix": vix,
        "fed_funds": fed,
        "unemployment": unrate,
        "macro_risk_score": score,
    }
    return df, macro, regime


def compute_quant_table(prices: pd.DataFrame, fundamentals: pd.DataFrame, macro_score: float) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < 120:
            continue

        vol = annualized_volatility(s)
        mdd = max_drawdown(s)
        cdd = current_drawdown_from_peak(s)
        rsi14 = rsi(s, 14)
        ret_6m = total_return(s, 126)
        ret_12m = total_return(s, 252)
        gap_50 = ma_gap(s, 50)
        gap_200 = ma_gap(s, 200)

        rows.append({
            "ticker": t,
            "last_price": s.iloc[-1],
            "ann_vol": vol,
            "max_drawdown": mdd,
            "current_drawdown": cdd,
            "rsi14": rsi14,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "gap_50ma": gap_50,
            "gap_200ma": gap_200,
        })

    qdf = pd.DataFrame(rows)
    if qdf.empty:
        return qdf

    df = qdf.merge(fundamentals, on="ticker", how="left")

    # Clean / clip outliers
    df["trailingPE"] = clip_series(pd.to_numeric(df["trailingPE"], errors="coerce"), 0, 200)
    df["forwardPE"] = clip_series(pd.to_numeric(df["forwardPE"], errors="coerce"), 0, 200)
    df["priceToBook"] = clip_series(pd.to_numeric(df["priceToBook"], errors="coerce"), 0, 50)
    df["returnOnEquity"] = clip_series(pd.to_numeric(df["returnOnEquity"], errors="coerce"), -1, 2)
    df["debtToEquity"] = clip_series(pd.to_numeric(df["debtToEquity"], errors="coerce"), 0, 1000)
    df["enterpriseToEbitda"] = clip_series(pd.to_numeric(df["enterpriseToEbitda"], errors="coerce"), -50, 100)
    df["fcfYield"] = clip_series(pd.to_numeric(df["fcfYield"], errors="coerce"), -1, 1)
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")

    # Factor scores
    # Lower valuation multiples = better
    pe_combo = df[["trailingPE", "forwardPE"]].mean(axis=1, skipna=True)
    pb = df["priceToBook"]
    ev_ebitda = df["enterpriseToEbitda"]
    fcfy = df["fcfYield"]

    value_score = (
        0.30 * safe_rank(pe_combo, ascending=True) +
        0.20 * safe_rank(pb, ascending=True) +
        0.20 * safe_rank(ev_ebitda, ascending=True) +
        0.30 * safe_rank(fcfy, ascending=False)
    )

    quality_score = (
        0.55 * safe_rank(df["returnOnEquity"], ascending=False) +
        0.25 * safe_rank(df["debtToEquity"], ascending=True) +
        0.20 * safe_rank(df["marketCap"], ascending=False)
    )

    risk_score = (
        0.45 * safe_rank(df["ann_vol"], ascending=True) +
        0.35 * safe_rank(df["max_drawdown"], ascending=False) +  # less negative is better
        0.20 * safe_rank(df["beta"], ascending=True)
    )

    # Contrarian: Howard Marks style
    # Better when drawdown is meaningful but not collapsing forever
    # Prefer current drawdown and lower RSI
    drawdown_abs = -df["current_drawdown"]
    contrarian_score = (
        0.60 * safe_rank(drawdown_abs, ascending=False) +
        0.40 * safe_rank(df["rsi14"], ascending=True)
    )

    momentum_score = (
        0.30 * safe_rank(df["ret_6m"], ascending=False) +
        0.25 * safe_rank(df["ret_12m"], ascending=False) +
        0.20 * safe_rank(df["gap_50ma"], ascending=False) +
        0.25 * safe_rank(df["gap_200ma"], ascending=False)
    )

    # Macro score is universe-wide: lower macro risk is better for gross exposure
    macro_component = max(0.0, min(1.0, 1 - (macro_score / 12.0)))
    df["macro_score_component"] = macro_component

    df["value_score"] = value_score
    df["quality_score"] = quality_score
    df["risk_score"] = risk_score
    df["contrarian_score"] = contrarian_score
    df["momentum_score"] = momentum_score

    df["hybrid_score"] = (
        weights["value"] * df["value_score"] +
        weights["quality"] * df["quality_score"] +
        weights["risk"] * df["risk_score"] +
        weights["contrarian"] * df["contrarian_score"] +
        weights["momentum"] * df["momentum_score"] +
        weights["macro"] * df["macro_score_component"]
    )

    # Filter flags
    df["marketCapB"] = df["marketCap"] / 1e9
    df["pass_vol"] = df["ann_vol"] <= max_allowed_vol
    df["pass_roe"] = df["returnOnEquity"].isna() | (df["returnOnEquity"] >= min_roe)
    df["pass_mcap"] = df["marketCapB"].isna() | (df["marketCapB"] >= min_market_cap_b)
    df["pass_filters"] = df["pass_vol"] & df["pass_roe"] & df["pass_mcap"]

    # Ranking
    df = df.sort_values(["pass_filters", "hybrid_score"], ascending=[False, False]).reset_index(drop=True)
    return df


def decide_positioning(regime: str, macro_score: float) -> Dict[str, float]:
    if regime == "Risk-On":
        return {
            "equity": 0.90,
            "cash": 0.10,
        }
    elif regime == "Cautious":
        return {
            "equity": 0.65,
            "cash": 0.35,
        }
    else:
        return {
            "equity": 0.40,
            "cash": 0.60,
        }


def build_portfolio(df: pd.DataFrame, top_n: int, regime: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    eligible = df[df["pass_filters"]].copy()
    if eligible.empty:
        eligible = df.copy()

    portfolio = eligible.head(top_n).copy()

    # Base weights proportional to score
    scores = portfolio["hybrid_score"].clip(lower=0.0001)
    w = scores / scores.sum()

    alloc = decide_positioning(regime, float(df["macro_score_component"].iloc[0] if "macro_score_component" in df.columns else 0.5))
    equity_alloc = alloc["equity"]

    portfolio["target_weight"] = w * equity_alloc
    portfolio["cash_buffer_note"] = f"Cash buffer = {1 - equity_alloc:.0%}"
    return portfolio


def format_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"


def format_num(x, digits=2):
    return f"{x:,.{digits}f}" if pd.notna(x) else "-"


def make_price_chart(prices: pd.DataFrame, tickers: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        if t in prices.columns:
            s = prices[t].dropna()
            if s.empty:
                continue
            norm = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=t))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Indexed Price (Start=100)",
        height=520,
        legend_title="Ticker",
    )
    return fig


def make_drawdown_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    s = prices[ticker].dropna()
    running_peak = s.cummax()
    dd = s / running_peak - 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{ticker} Price"))
    fig.add_trace(go.Scatter(x=running_peak.index, y=running_peak.values, mode="lines", name="Running Peak"))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=f"{ticker} Drawdown"))
    fig2.update_layout(
        title=f"{ticker} Drawdown vs Previous Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=420,
    )
    return fig, fig2


def make_mva_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    s = prices[ticker].dropna()
    ma50 = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Price"))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, mode="lines", name="MA200"))
    fig.update_layout(
        title=f"{ticker} Price with MA50 / MA200",
        xaxis_title="Date",
        yaxis_title="Price",
        height=420,
    )
    return fig


def factor_interpretation(row: pd.Series, regime: str) -> str:
    msgs = []

    if pd.notna(row.get("current_drawdown")) and row["current_drawdown"] < -0.25:
        msgs.append("deep pullback")
    if pd.notna(row.get("rsi14")) and row["rsi14"] < 40:
        msgs.append("oversold/contrarian")
    if pd.notna(row.get("gap_200ma")) and row["gap_200ma"] > 0:
        msgs.append("above 200MA")
    if pd.notna(row.get("returnOnEquity")) and row["returnOnEquity"] > 0.15:
        msgs.append("strong ROE")
    if pd.notna(row.get("trailingPE")) and row["trailingPE"] < 25:
        msgs.append("reasonable PE")
    if pd.notna(row.get("ann_vol")) and row["ann_vol"] < 0.30:
        msgs.append("controlled volatility")

    if not msgs:
        msgs.append("mixed factor profile")

    regime_msg = {
        "Risk-On": "macro backdrop supports broader equity exposure",
        "Cautious": "macro backdrop suggests selective exposure",
        "Risk-Off": "macro backdrop favors defense and cash discipline",
    }.get(regime, "macro backdrop neutral")

    return f"{', '.join(msgs)}; {regime_msg}."


# ============================================================
# Main execution
# ============================================================
tickers = parse_tickers(universe_mode, manual_tickers)

if run_button:
    st.cache_data.clear()

with st.spinner("Loading macro, price, and fundamental data..."):
    fred_df = fetch_all_fred()
    macro_hist, macro_now, regime = compute_macro_state(fred_df)
    prices = fetch_prices(tickers, price_period)
    fundamentals = fetch_fundamentals(tickers)
    quant_df = compute_quant_table(prices, fundamentals, macro_now["macro_risk_score"])
    portfolio_df = build_portfolio(quant_df, top_n=top_n, regime=regime)

# ============================================================
# Top summary
# ============================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Regime", regime)
c2.metric("Macro Risk Score", f'{macro_now["macro_risk_score"]:.0f}/12')
c3.metric("10Y-2Y", format_num(macro_now["yield_curve_spread"], 2))
c4.metric("HY Spread", format_num(macro_now["high_yield_spread"], 2))
c5.metric("VIX", format_num(macro_now["vix"], 2))
c6.metric("Fed Funds", format_num(macro_now["fed_funds"], 2))

positioning = decide_positioning(regime, macro_now["macro_risk_score"])
st.info(
    f"Suggested positioning: Equity {positioning['equity']:.0%} / Cash {positioning['cash']:.0%} "
    f"based on current macro regime = {regime}"
)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Macro Dashboard", "Quant Screener", "Portfolio", "Chart Lab", "Raw Data"]
)

# ============================================================
# Tab 1: Macro Dashboard
# ============================================================
with tab1:
    st.subheader("Howard Marks Macro Cycle Dashboard")

    macro_display = pd.DataFrame(
        {
            "Series": [
                "10Y Treasury",
                "2Y Treasury",
                "10Y-2Y Spread",
                "Fed Funds",
                "High Yield Spread",
                "VIX",
                "Unemployment",
            ],
            "Latest": [
                macro_hist["DGS10"].dropna().iloc[-1] if not macro_hist["DGS10"].dropna().empty else np.nan,
                macro_hist["DGS2"].dropna().iloc[-1] if not macro_hist["DGS2"].dropna().empty else np.nan,
                macro_now["yield_curve_spread"],
                macro_now["fed_funds"],
                macro_now["high_yield_spread"],
                macro_now["vix"],
                macro_now["unemployment"],
            ],
            "Interpretation": [
                "Long-term growth / inflation expectation",
                "Policy-sensitive short rate",
                "Inversion = recession warning",
                "Liquidity tightness proxy",
                "Credit stress proxy",
                "Fear / volatility proxy",
                "Labor market stress",
            ],
        }
    )
    st.dataframe(macro_display, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_yc = go.Figure()
        if "DGS10" in macro_hist and "DGS2" in macro_hist:
            spread = macro_hist["DGS10"] - macro_hist["DGS2"]
            fig_yc.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines", name="10Y-2Y Spread"))
            fig_yc.update_layout(
                title="Yield Curve Spread (10Y - 2Y)",
                xaxis_title="Date",
                yaxis_title="Spread",
                height=420,
            )
        st.plotly_chart(fig_yc, use_container_width=True)

    with col_b:
        fig_hy = go.Figure()
        if "BAMLH0A0HYM2" in macro_hist:
            s = macro_hist["BAMLH0A0HYM2"].dropna()
            fig_hy.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="HY Spread"))
            fig_hy.update_layout(
                title="US High Yield Spread",
                xaxis_title="Date",
                yaxis_title="Spread",
                height=420,
            )
        st.plotly_chart(fig_hy, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig_vix = go.Figure()
        if "VIXCLS" in macro_hist:
            s = macro_hist["VIXCLS"].dropna()
            fig_vix.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="VIX"))
            fig_vix.update_layout(
                title="VIX",
                xaxis_title="Date",
                yaxis_title="Index",
                height=420,
            )
        st.plotly_chart(fig_vix, use_container_width=True)

    with col_d:
        fig_ff = go.Figure()
        if "FEDFUNDS" in macro_hist:
            s = macro_hist["FEDFUNDS"].dropna()
            fig_ff.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Fed Funds"))
            fig_ff.update_layout(
                title="Fed Funds Rate",
                xaxis_title="Date",
                yaxis_title="Rate",
                height=420,
            )
        st.plotly_chart(fig_ff, use_container_width=True)

    st.markdown("### Macro interpretation")
    if regime == "Risk-On":
        st.success(
            "Macro regime is Risk-On. Howard Marks 관점에서는 공격 가능 구간이지만, "
            "여전히 리스크 관리와 분산이 중요합니다."
        )
    elif regime == "Cautious":
        st.warning(
            "Macro regime is Cautious. 선택적 매수, 높은 품질, 낮은 레버리지, "
            "적절한 현금 비중이 유리합니다."
        )
    else:
        st.error(
            "Macro regime is Risk-Off. Howard Marks 관점에서는 현금, 방어, "
            "손실 회피, 그리고 진짜 기회가 생길 때까지 기다리는 자세가 중요합니다."
        )

# ============================================================
# Tab 2: Quant Screener
# ============================================================
with tab2:
    st.subheader("Hybrid Quant Screener")

    if quant_df.empty:
        st.warning("No screening results available.")
    else:
        screen_df = quant_df.copy()

        sector_options = ["All"] + sorted([x for x in screen_df["sector"].dropna().unique()])
        selected_sector = st.selectbox("Sector filter", sector_options, index=0)

        pass_only = st.checkbox("Show pass-filters only", value=True)

        if selected_sector != "All":
            screen_df = screen_df[screen_df["sector"] == selected_sector]

        if pass_only:
            tmp = screen_df[screen_df["pass_filters"]]
            if not tmp.empty:
                screen_df = tmp

        show_cols = [
            "ticker", "name", "sector", "industry",
            "hybrid_score",
            "value_score", "quality_score", "risk_score", "contrarian_score", "momentum_score",
            "last_price",
            "trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "fcfYield",
            "ann_vol", "max_drawdown", "current_drawdown", "rsi14",
            "ret_6m", "ret_12m", "gap_50ma", "gap_200ma",
            "marketCapB", "pass_filters",
        ]
        show_cols = [c for c in show_cols if c in screen_df.columns]

        st.dataframe(
            screen_df[show_cols].sort_values("hybrid_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Top candidates")
        top_preview = screen_df.sort_values("hybrid_score", ascending=False).head(10)
        for _, row in top_preview.iterrows():
            st.markdown(
                f"**{row['ticker']} — {row.get('name', '')}**  \n"
                f"Score: {row['hybrid_score']:.3f} | Sector: {row.get('sector', '-') or '-'} | "
                f"ROE: {format_pct(row.get('returnOnEquity'))} | "
                f"Current DD: {format_pct(row.get('current_drawdown'))} | "
                f"6M: {format_pct(row.get('ret_6m'))}  \n"
                f"{factor_interpretation(row, regime)}"
            )

# ============================================================
# Tab 3: Portfolio
# ============================================================
with tab3:
    st.subheader("Suggested Portfolio")

    if portfolio_df.empty:
        st.warning("No portfolio candidates available.")
    else:
        port_show = portfolio_df[
            [
                "ticker", "name", "sector", "hybrid_score",
                "target_weight", "value_score", "quality_score",
                "risk_score", "contrarian_score", "momentum_score",
                "current_drawdown", "ret_6m", "ret_12m"
            ]
        ].copy()

        st.dataframe(port_show, use_container_width=True, hide_index=True)

        fig_w = go.Figure(
            data=[
                go.Bar(
                    x=portfolio_df["ticker"],
                    y=portfolio_df["target_weight"],
                    text=[f"{w:.1%}" for w in portfolio_df["target_weight"]],
                    textposition="auto",
                    name="Target Weight",
                )
            ]
        )
        fig_w.update_layout(
            title="Target Portfolio Weights",
            xaxis_title="Ticker",
            yaxis_title="Weight",
            height=450,
        )
        st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("### Howard Marks style portfolio logic")
        st.write(
            f"- Current macro regime: **{regime}**\n"
            f"- Suggested equity allocation: **{positioning['equity']:.0%}**\n"
            f"- Suggested cash allocation: **{positioning['cash']:.0%}**\n"
            f"- Selection logic: **Value + Quality + Risk Control + Contrarian + Momentum**"
        )

# ============================================================
# Tab 4: Chart Lab
# ============================================================
with tab4:
    st.subheader("Chart Lab")

    if quant_df.empty or prices.empty:
        st.warning("No chart data available.")
    else:
        candidate_list = quant_df["ticker"].head(20).tolist()
        selected_chart_ticker = st.selectbox("Select ticker", candidate_list, index=0)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_mva_chart(prices, selected_chart_ticker), use_container_width=True)

        with c2:
            pfig, dfig = make_drawdown_chart(prices, selected_chart_ticker)
            st.plotly_chart(dfig, use_container_width=True)

        st.markdown("### Relative performance: Top portfolio candidates")
        if not portfolio_df.empty:
            top_tickers = portfolio_df["ticker"].tolist()
            st.plotly_chart(
                make_price_chart(prices, top_tickers, "Relative Price Performance (Indexed to 100)"),
                use_container_width=True,
            )

# ============================================================
# Tab 5: Raw Data
# ============================================================
with tab5:
    st.subheader("Raw data")

    with st.expander("FRED macro data"):
        st.dataframe(macro_hist.tail(50), use_container_width=True)

    with st.expander("Price data"):
        st.dataframe(prices.tail(50), use_container_width=True)

    with st.expander("Fundamentals"):
        st.dataframe(fundamentals, use_container_width=True)

    with st.expander("Quant table"):
        st.dataframe(quant_df, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
**Notes**
- This app is an educational/research tool, not investment advice.
- `yfinance` fundamentals can be incomplete or delayed for some tickers.
- Howard Marks style interpretation here is translated into a quant framework:
  **Cycle awareness + risk control + value discipline + contrarian behavior + trend confirmation**.
"""
)

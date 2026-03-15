# streamlit_app.py
# ============================================================
# Guru Investor Strategy Dashboard
# ------------------------------------------------------------
# A Streamlit dashboard that analyzes the current market
# through the lens of 10 top guru investors.
#
# Gurus:
# 1. Warren Buffett
# 2. Ray Dalio
# 3. Howard Marks
# 4. Stanley Druckenmiller
# 5. George Soros
# 6. Peter Lynch
# 7. Jim Simons
# 8. Cathie Wood
# 9. Michael Burry
# 10. Paul Tudor Jones
#
# Features:
# - Live market snapshot using yfinance
# - 10 guru tabs
# - Each tab shows:
#   * strategy overview
#   * current market interpretation
#   * key indicators
#   * suggested positioning
#   * candidate stocks / ETFs
#   * charts
# - Simple regime scoring
# - Plotly key collision fixed
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# Run:
#   streamlit run streamlit_app.py
# ============================================================

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Guru Investor Strategy Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Guru Investor Strategy Dashboard")
st.caption("Analyze the current market through the strategies of 10 top guru investors")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Dashboard Settings")

period_map = {
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}
selected_period_label = st.sidebar.selectbox(
    "Price History Period",
    list(period_map.keys()),
    index=2,
)
selected_period = period_map[selected_period_label]

benchmark = st.sidebar.selectbox(
    "Benchmark",
    ["SPY", "QQQ", "^NDX", "^GSPC"],
    index=0,
)

risk_free_proxy = st.sidebar.selectbox(
    "Cash / Short-Term Proxy",
    ["BIL", "SHY", "SGOV"],
    index=0,
)

show_raw_data = st.sidebar.checkbox("Show raw downloaded data", value=False)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

# ============================================================
# Universe / proxies
# ============================================================
MARKET_TICKERS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF",
    "TLT": "20+Y Treasury ETF",
    "IEF": "7-10Y Treasury ETF",
    "GLD": "Gold ETF",
    "DBC": "Broad Commodities ETF",
    "USO": "Oil ETF",
    "XLE": "Energy Select Sector ETF",
    "XLK": "Technology Select Sector ETF",
    "XLF": "Financials Select Sector ETF",
    "XLV": "Healthcare Select Sector ETF",
    "SOXX": "Semiconductor ETF",
    "ARKK": "ARK Innovation ETF",
    "VIXY": "VIX Short-Term Futures ETF",
    "XLP": "Consumer Staples ETF",
    "LQD": "Investment Grade Corporate Bond ETF",
}

GURU_CANDIDATES = {
    "Buffett": ["BRK-B", "AAPL", "V", "KO", "AMZN", "MCO", "AXP"],
    "Dalio": ["GLD", "TLT", "DBC", "SPY", "IEF", "USO"],
    "Marks": ["SPY", "XLV", "XLF", "BIL", "GLD", "LQD"],
    "Druckenmiller": ["NVDA", "AVGO", "MSFT", "TSM", "AMZN", "XLE", "SOXX"],
    "Soros": ["QQQ", "SOXX", "GLD", "USO", "VIXY"],
    "Lynch": ["PLTR", "COST", "VRSK", "APP", "LLY", "AMZN", "META"],
    "Simons": ["SPY", "QQQ", "IWM", "SOXX", "XLE", "GLD"],
    "Cathie": ["ARKK", "TSLA", "ROKU", "PLTR", "CRWD", "COIN", "NVDA"],
    "Burry": ["GLD", "TLT", "XLV", "XLP", "BIL", "VIXY"],
    "Tudor": ["GLD", "DBC", "USO", "XLE", "TLT", "SPY"],
}

ALL_TICKERS = sorted(
    set(
        list(MARKET_TICKERS.keys())
        + sum(GURU_CANDIDATES.values(), [])
        + [benchmark, risk_free_proxy]
    )
)

# ============================================================
# Utility functions
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def download_price_data(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def download_info(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def extract_close_matrix(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            try:
                closes[t] = raw[t]["Close"]
            except Exception:
                pass
        return pd.DataFrame(closes)

    if "Close" in raw.columns and len(tickers) == 1:
        return pd.DataFrame({tickers[0]: raw["Close"]})

    return pd.DataFrame()


def extract_volume_matrix(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        vols = {}
        for t in tickers:
            try:
                vols[t] = raw[t]["Volume"]
            except Exception:
                pass
        return pd.DataFrame(vols)

    if "Volume" in raw.columns and len(tickers) == 1:
        return pd.DataFrame({tickers[0]: raw["Volume"]})

    return pd.DataFrame()


def compute_return_metrics(price: pd.Series) -> Dict[str, float]:
    s = price.dropna().copy()

    if len(s) < 30:
        return {
            "last": np.nan,
            "mva50": np.nan,
            "mva200": np.nan,
            "distance_50": np.nan,
            "distance_200": np.nan,
            "ytd_return": np.nan,
            "sixm_return": np.nan,
            "oney_return": np.nan,
            "vol_20d": np.nan,
            "mdd": np.nan,
            "momentum_3m": np.nan,
            "momentum_6m": np.nan,
        }

    out = {}
    out["last"] = s.iloc[-1]
    out["mva50"] = s.rolling(50).mean().iloc[-1] if len(s) >= 50 else np.nan
    out["mva200"] = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
    out["distance_50"] = (out["last"] / out["mva50"] - 1.0) * 100 if pd.notna(out["mva50"]) else np.nan
    out["distance_200"] = (out["last"] / out["mva200"] - 1.0) * 100 if pd.notna(out["mva200"]) else np.nan

    daily_ret = s.pct_change().dropna()
    out["vol_20d"] = daily_ret.tail(20).std() * np.sqrt(252) * 100 if len(daily_ret) >= 20 else np.nan

    peak = s.cummax()
    dd = s / peak - 1.0
    out["mdd"] = dd.min() * 100

    today = s.index[-1]
    current_year = today.year
    year_start = s[s.index.year == current_year]
    out["ytd_return"] = (s.iloc[-1] / year_start.iloc[0] - 1.0) * 100 if len(year_start) > 1 else np.nan

    out["sixm_return"] = (s.iloc[-1] / s.iloc[-126] - 1.0) * 100 if len(s) >= 126 else np.nan
    out["oney_return"] = (s.iloc[-1] / s.iloc[-252] - 1.0) * 100 if len(s) >= 252 else np.nan
    out["momentum_3m"] = (s.iloc[-1] / s.iloc[-63] - 1.0) * 100 if len(s) >= 63 else np.nan
    out["momentum_6m"] = (s.iloc[-1] / s.iloc[-126] - 1.0) * 100 if len(s) >= 126 else np.nan

    return out


def relative_performance(price: pd.Series, bench: pd.Series) -> pd.Series:
    s1 = price.dropna()
    s2 = bench.dropna()
    df = pd.concat([s1, s2], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    rel = (df.iloc[:, 0] / df.iloc[:, 0].iloc[0]) / (df.iloc[:, 1] / df.iloc[:, 1].iloc[0])
    return rel


def drawdown_series(price: pd.Series) -> pd.Series:
    s = price.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s / s.cummax() - 1.0


def normalize_series(price: pd.Series) -> pd.Series:
    s = price.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s / s.iloc[0] * 100.0


def score_market_regime(metrics: Dict[str, Dict[str, float]]) -> Tuple[int, str, Dict[str, int]]:
    score_details = {
        "Trend": 0,
        "Breadth Proxy": 0,
        "Risk Appetite": 0,
        "Defensive Preference": 0,
        "Commodity Pressure": 0,
    }

    try:
        if metrics["SPY"]["distance_200"] > 0:
            score_details["Trend"] += 2
        elif metrics["SPY"]["distance_200"] > -5:
            score_details["Trend"] += 1

        if metrics["QQQ"]["distance_200"] > 0:
            score_details["Breadth Proxy"] += 2
        elif metrics["QQQ"]["distance_200"] > -5:
            score_details["Breadth Proxy"] += 1

        if metrics["SOXX"]["momentum_6m"] > metrics["TLT"]["momentum_6m"]:
            score_details["Risk Appetite"] += 2
        elif metrics["SOXX"]["momentum_6m"] > 0:
            score_details["Risk Appetite"] += 1

        if metrics["XLV"]["momentum_6m"] > metrics["XLK"]["momentum_6m"]:
            score_details["Defensive Preference"] -= 1
        else:
            score_details["Defensive Preference"] += 1

        if metrics["DBC"]["momentum_6m"] > 10 or metrics["USO"]["momentum_6m"] > 10:
            score_details["Commodity Pressure"] -= 1
        else:
            score_details["Commodity Pressure"] += 1
    except Exception:
        pass

    total = sum(score_details.values())

    if total >= 5:
        label = "Bull / Risk-On"
    elif total >= 2:
        label = "Cautious Bull"
    elif total >= 0:
        label = "Neutral"
    elif total >= -2:
        label = "Cautious / Late Cycle"
    else:
        label = "Risk-Off"

    return total, label, score_details


def make_price_chart(
    prices: pd.DataFrame,
    ticker: str,
    title: str | None = None,
    show_ma: bool = True,
) -> go.Figure:
    s = prices[ticker].dropna()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=ticker,
            line=dict(width=2),
        )
    )

    if show_ma and len(s) >= 50:
        ma50 = s.rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=ma50.index,
                y=ma50.values,
                mode="lines",
                name="MA50",
                line=dict(width=1.5, dash="dash"),
            )
        )

    if show_ma and len(s) >= 200:
        ma200 = s.rolling(200).mean()
        fig.add_trace(
            go.Scatter(
                x=ma200.index,
                y=ma200.values,
                mode="lines",
                name="MA200",
                line=dict(width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        title=title or f"{ticker} Price",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_relative_chart(prices: pd.DataFrame, ticker: str, benchmark_ticker: str) -> go.Figure:
    rel = relative_performance(prices[ticker], prices[benchmark_ticker])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rel.index,
            y=rel.values,
            mode="lines",
            name=f"{ticker} / {benchmark_ticker}",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title=f"Relative Performance vs {benchmark_ticker}",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_drawdown_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    dd = drawdown_series(prices[ticker]) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title=f"{ticker} Drawdown from Previous Peak",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_compare_chart(prices: pd.DataFrame, tickers: List[str], title: str) -> go.Figure:
    fig = go.Figure()

    for t in tickers:
        if t in prices.columns:
            s = normalize_series(prices[t])
            if not s.empty:
                fig.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        mode="lines",
                        name=t,
                    )
                )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def format_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.2f}%"


def format_num(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.2f}"


def get_fundamental_snapshot(ticker: str) -> Dict[str, float | str]:
    info = download_info(ticker)
    return {
        "shortName": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "marketCap": info.get("marketCap", np.nan),
        "trailingPE": info.get("trailingPE", np.nan),
        "forwardPE": info.get("forwardPE", np.nan),
        "pegRatio": info.get("pegRatio", np.nan),
        "priceToBook": info.get("priceToBook", np.nan),
        "returnOnEquity": info.get("returnOnEquity", np.nan),
        "debtToEquity": info.get("debtToEquity", np.nan),
        "freeCashflow": info.get("freeCashflow", np.nan),
        "operatingMargins": info.get("operatingMargins", np.nan),
        "revenueGrowth": info.get("revenueGrowth", np.nan),
        "earningsGrowth": info.get("earningsGrowth", np.nan),
        "currentRatio": info.get("currentRatio", np.nan),
        "quickRatio": info.get("quickRatio", np.nan),
        "beta": info.get("beta", np.nan),
        "dividendYield": info.get("dividendYield", np.nan),
        "longBusinessSummary": info.get("longBusinessSummary", ""),
    }


def metrics_table_for_tickers(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t not in prices.columns:
            continue

        m = compute_return_metrics(prices[t])
        f = get_fundamental_snapshot(t)

        rows.append(
            {
                "Ticker": t,
                "Name": f["shortName"],
                "Sector": f["sector"],
                "Price": m["last"],
                "YTD %": m["ytd_return"],
                "6M %": m["sixm_return"],
                "1Y %": m["oney_return"],
                "Dist vs MA50 %": m["distance_50"],
                "Dist vs MA200 %": m["distance_200"],
                "MDD %": m["mdd"],
                "Vol 20D Ann. %": m["vol_20d"],
                "PE": f["trailingPE"],
                "Forward PE": f["forwardPE"],
                "PEG": f["pegRatio"],
                "P/B": f["priceToBook"],
                "ROE %": f["returnOnEquity"] * 100 if pd.notna(f["returnOnEquity"]) else np.nan,
                "Revenue Growth %": f["revenueGrowth"] * 100 if pd.notna(f["revenueGrowth"]) else np.nan,
                "Debt/Equity": f["debtToEquity"],
            }
        )

    return pd.DataFrame(rows)


def render_fundamental_card(ticker: str):
    f = get_fundamental_snapshot(ticker)
    st.markdown(f"### {ticker} — {f['shortName']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sector", f"{f['sector']}")
    c2.metric("Industry", f"{f['industry']}")
    c3.metric("Trailing PE", format_num(f["trailingPE"]))
    c4.metric("Forward PE", format_num(f["forwardPE"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("PEG", format_num(f["pegRatio"]))
    c6.metric("P/B", format_num(f["priceToBook"]))
    c7.metric("ROE", format_pct(f["returnOnEquity"] * 100 if pd.notna(f["returnOnEquity"]) else np.nan))
    c8.metric("Debt/Equity", format_num(f["debtToEquity"]))

    summary = f.get("longBusinessSummary", "")
    if summary:
        st.caption(summary[:700] + ("..." if len(summary) > 700 else ""))


# ============================================================
# Download data
# ============================================================
with st.spinner("Downloading market data..."):
    raw_data = download_price_data(ALL_TICKERS, period=selected_period)
    prices = extract_close_matrix(raw_data, ALL_TICKERS)
    volumes = extract_volume_matrix(raw_data, ALL_TICKERS)

if prices.empty:
    st.error("No price data could be loaded. Please try again.")
    st.stop()

if show_raw_data:
    st.subheader("Raw Price Data")
    st.dataframe(prices.tail(30), use_container_width=True)

# ============================================================
# Core market snapshot
# ============================================================
snapshot_tickers = ["SPY", "QQQ", "TLT", "GLD", "DBC", "USO", "XLK", "XLV", "SOXX"]
market_metrics = {t: compute_return_metrics(prices[t]) for t in snapshot_tickers if t in prices.columns}

regime_score, regime_label, regime_detail = score_market_regime(market_metrics)

st.subheader("Market Snapshot")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Market Regime Score", f"{regime_score}")
c2.metric("Regime Label", regime_label)
c3.metric("SPY 6M", format_pct(market_metrics["SPY"]["sixm_return"]))
c4.metric("QQQ 6M", format_pct(market_metrics["QQQ"]["sixm_return"]))
c5.metric("TLT 6M", format_pct(market_metrics["TLT"]["sixm_return"]))

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("GLD 6M", format_pct(market_metrics["GLD"]["sixm_return"]))
c7.metric("USO 6M", format_pct(market_metrics["USO"]["sixm_return"]))
c8.metric("SOXX 6M", format_pct(market_metrics["SOXX"]["sixm_return"]))
c9.metric("SPY vs MA200", format_pct(market_metrics["SPY"]["distance_200"]))
c10.metric("QQQ vs MA200", format_pct(market_metrics["QQQ"]["distance_200"]))

with st.expander("Regime Score Details", expanded=False):
    st.json(regime_detail)

st.plotly_chart(
    make_compare_chart(
        prices,
        ["SPY", "QQQ", "TLT", "GLD", "DBC"],
        "Normalized Performance: Equities vs Bonds vs Gold vs Commodities",
    ),
    use_container_width=True,
    key="top_market_snapshot_compare_chart",
)

# ============================================================
# Global interpretation
# ============================================================
def global_market_commentary(regime_label: str, metrics: Dict[str, Dict[str, float]]) -> str:
    spy_200 = metrics["SPY"]["distance_200"]
    qqq_200 = metrics["QQQ"]["distance_200"]
    soxx_6m = metrics["SOXX"]["sixm_return"]
    tlt_6m = metrics["TLT"]["sixm_return"]
    gld_6m = metrics["GLD"]["sixm_return"]
    uso_6m = metrics["USO"]["sixm_return"]

    comments = []

    if regime_label in ["Bull / Risk-On", "Cautious Bull"]:
        comments.append("Equity trend remains constructive, with broad risk appetite still present.")
    elif regime_label == "Neutral":
        comments.append("The market looks mixed: trend is positive in some areas, but leadership is not fully broad.")
    else:
        comments.append("The market is showing late-cycle or defensive behavior, with higher fragility beneath the surface.")

    if pd.notna(qqq_200) and pd.notna(spy_200) and qqq_200 > spy_200:
        comments.append("Growth / technology leadership is stronger than the broad market.")
    if pd.notna(soxx_6m) and soxx_6m > 0:
        comments.append("Semiconductor momentum still supports the AI / compute narrative.")
    if pd.notna(gld_6m) and gld_6m > 0:
        comments.append("Gold strength suggests ongoing demand for macro hedges.")
    if pd.notna(uso_6m) and uso_6m > 10:
        comments.append("Energy strength may indicate inflation or geopolitical pressure.")
    if pd.notna(tlt_6m) and tlt_6m < 0:
        comments.append("Long-duration bonds remain under pressure, which can challenge valuation-heavy assets.")

    return " ".join(comments)


st.info(global_market_commentary(regime_label, market_metrics))

# ============================================================
# Guru configuration
# ============================================================
GURU_META = {
    "Warren Buffett": {
        "alias": "Buffett",
        "strategy": "Value Investing / Quality / Moat",
        "focus": "Durable moat, high ROE, low leverage, cash generation, reasonable valuation",
        "style_interpretation": "Favors strong businesses with stable economics. More cautious when speculative enthusiasm dominates the market.",
        "indicators": [
            "ROE > 15%",
            "Positive free cash flow",
            "Moderate debt",
            "Reasonable PE / P/B",
            "Stable margins",
        ],
        "portfolio": {
            "Quality Value Equities": "60%",
            "Cash / Short-Term": "15%",
            "Defensive Equities": "10%",
            "Gold / Hedge": "5%",
            "Special Situations": "10%",
        },
    },
    "Ray Dalio": {
        "alias": "Dalio",
        "strategy": "Global Macro / All Weather / Economic Cycles",
        "focus": "Diversification across growth, inflation, disinflation, recession regimes",
        "style_interpretation": "Focuses on macro balance rather than one-sided equity exposure.",
        "indicators": [
            "Stocks vs bonds trend",
            "Gold trend",
            "Commodity trend",
            "Dollar / rates pressure",
            "Cross-asset diversification",
        ],
        "portfolio": {
            "Equities": "30%",
            "Bonds": "25%",
            "Gold": "20%",
            "Commodities": "15%",
            "Cash": "10%",
        },
    },
    "Howard Marks": {
        "alias": "Marks",
        "strategy": "Market Cycle / Psychology / Risk Control",
        "focus": "Understand where we are in the cycle and avoid excess optimism",
        "style_interpretation": "Tends to get more defensive when the crowd becomes complacent.",
        "indicators": [
            "Valuation stretch",
            "Price vs MA200",
            "Drawdown risk",
            "Credit-sensitive behavior proxy",
            "Sentiment / risk appetite proxy",
        ],
        "portfolio": {
            "Core Equities": "40%",
            "Defensive Sectors": "20%",
            "Cash": "20%",
            "Gold": "10%",
            "Opportunistic": "10%",
        },
    },
    "Stanley Druckenmiller": {
        "alias": "Druckenmiller",
        "strategy": "Macro Trend / Concentrated Conviction",
        "focus": "Big themes, strong price action, asymmetric opportunities",
        "style_interpretation": "Willing to concentrate heavily behind major trends such as AI, semiconductors, or energy.",
        "indicators": [
            "6M and 1Y momentum",
            "Relative strength vs benchmark",
            "Theme leadership",
            "Trend persistence",
            "Macro confirmation",
        ],
        "portfolio": {
            "High Conviction Growth": "50%",
            "Theme / Cyclical": "20%",
            "Macro Hedge": "10%",
            "Cash": "10%",
            "Trading Sleeve": "10%",
        },
    },
    "George Soros": {
        "alias": "Soros",
        "strategy": "Reflexivity / Narrative + Price Feedback Loop",
        "focus": "Narrative drives capital flows; price confirms narrative until it breaks",
        "style_interpretation": "Looks for unstable equilibrium where consensus can reverse sharply.",
        "indicators": [
            "Narrative leadership (AI, inflation, geopolitical hedges)",
            "Momentum extremes",
            "Volatility hedge demand",
            "Cross-asset stress",
            "Crowding",
        ],
        "portfolio": {
            "Trend Trades": "35%",
            "Hedge / Optionality": "20%",
            "Global Macro": "20%",
            "Cash": "15%",
            "Tactical Reversal Trades": "10%",
        },
    },
    "Peter Lynch": {
        "alias": "Lynch",
        "strategy": "GARP / Growth at a Reasonable Price",
        "focus": "Understandable businesses, growth, not paying absurd valuations",
        "style_interpretation": "Seeks practical growth opportunities where fundamentals justify the price.",
        "indicators": [
            "Revenue growth",
            "PEG ratio",
            "ROE",
            "Sector opportunity",
            "Business simplicity",
        ],
        "portfolio": {
            "Growth at Reasonable Price": "55%",
            "Steady Compounders": "20%",
            "Turnaround / Special Names": "10%",
            "Cash": "10%",
            "Hedge": "5%",
        },
    },
    "Jim Simons": {
        "alias": "Simons",
        "strategy": "Quant / Statistical / Momentum + Mean Reversion",
        "focus": "Systematic signals, market behavior, diversification of edges",
        "style_interpretation": "Less interested in stories, more interested in robust signal behavior.",
        "indicators": [
            "3M / 6M momentum",
            "Volatility",
            "Relative strength",
            "Drawdown",
            "Cross-sectional ranking",
        ],
        "portfolio": {
            "Systematic Trend": "35%",
            "Mean Reversion": "20%",
            "Low Vol / Defensive": "15%",
            "Diversifiers": "15%",
            "Cash": "15%",
        },
    },
    "Cathie Wood": {
        "alias": "Cathie",
        "strategy": "Disruptive Innovation / Exponential Growth",
        "focus": "AI, robotics, genomics, software platforms, crypto infrastructure",
        "style_interpretation": "Accepts high volatility in exchange for long-duration innovation upside.",
        "indicators": [
            "Top-line growth",
            "Innovation leadership",
            "High-beta momentum",
            "Long-duration sensitivity",
            "Narrative persistence",
        ],
        "portfolio": {
            "Disruptive Tech": "60%",
            "AI Infrastructure": "15%",
            "Speculative Innovation": "10%",
            "Cash": "10%",
            "Hedge": "5%",
        },
    },
    "Michael Burry": {
        "alias": "Burry",
        "strategy": "Bubble Detection / Contrarian / Deep Risk",
        "focus": "Overvaluation, leverage, weak market structure, crowded trades",
        "style_interpretation": "Looks for what can go wrong when consensus gets too comfortable.",
        "indicators": [
            "Price far above MA200",
            "Valuation stretch",
            "Crowded momentum",
            "Fragile breadth",
            "Defensive hedges outperforming",
        ],
        "portfolio": {
            "Defensive Equities": "25%",
            "Treasuries / Duration": "20%",
            "Gold": "15%",
            "Cash": "25%",
            "Tail Hedge / Volatility": "15%",
        },
    },
    "Paul Tudor Jones": {
        "alias": "Tudor",
        "strategy": "Macro Trading / Inflation / Risk Management",
        "focus": "Protect capital first, exploit macro trends second",
        "style_interpretation": "Focuses on inflation-sensitive assets, trend, and position sizing discipline.",
        "indicators": [
            "Gold trend",
            "Commodity / oil trend",
            "Bond trend",
            "Equity trend",
            "Volatility / risk control",
        ],
        "portfolio": {
            "Macro Trend Equities": "30%",
            "Gold": "20%",
            "Commodities / Energy": "20%",
            "Bonds": "10%",
            "Cash": "20%",
        },
    },
}

# ============================================================
# Interpretation helpers
# ============================================================
def guru_interpretation(guru: str, market_metrics: Dict[str, Dict[str, float]]) -> str:
    spy = market_metrics["SPY"]
    qqq = market_metrics["QQQ"]
    tlt = market_metrics["TLT"]
    gld = market_metrics["GLD"]
    uso = market_metrics["USO"]
    soxx = market_metrics["SOXX"]

    if guru == "Warren Buffett":
        if pd.notna(qqq["distance_200"]) and qqq["distance_200"] > 10:
            return (
                "A Buffett-style view would likely say the market contains strong businesses, "
                "but parts of large-cap growth may be priced for very optimistic outcomes. "
                "The focus should stay on durable quality and disciplined valuation."
            )
        return (
            "A Buffett-style view would see opportunities in high-quality businesses with stable cash flow, "
            "especially outside the most crowded momentum names."
        )

    if guru == "Ray Dalio":
        if pd.notna(gld["sixm_return"]) and pd.notna(uso["sixm_return"]) and gld["sixm_return"] > 0 and uso["sixm_return"] > 0:
            return (
                "A Dalio-style view sees persistent macro uncertainty: inflation hedges are still relevant, "
                "so diversification across equities, bonds, gold, and commodities remains important."
            )
        return (
            "A Dalio-style view would emphasize balance. The market may still reward equities, "
            "but a one-asset portfolio is vulnerable to macro regime shifts."
        )

    if guru == "Howard Marks":
        if pd.notna(qqq["distance_200"]) and pd.notna(soxx["sixm_return"]) and qqq["distance_200"] > 10 and soxx["sixm_return"] > 20:
            return (
                "A Marks-style reading suggests the market is in an optimistic phase. "
                "That does not automatically mean an imminent crash, but it argues for greater selectivity and risk control."
            )
        return (
            "A Marks-style reading suggests the cycle is not at panic levels. "
            "Investors should stay rational and avoid paying peak multiples without a margin of safety."
        )

    if guru == "Stanley Druckenmiller":
        if pd.notna(soxx["sixm_return"]) and pd.notna(qqq["distance_200"]) and soxx["sixm_return"] > 0 and qqq["distance_200"] > 0:
            return (
                "A Druckenmiller-style view would probably stay constructive on the dominant trend: "
                "AI infrastructure, semiconductor leadership, and strong growth franchises."
            )
        return "A Druckenmiller-style view would wait for clearer trend confirmation before building aggressive positions."

    if guru == "George Soros":
        if pd.notna(qqq["distance_200"]) and pd.notna(gld["sixm_return"]) and qqq["distance_200"] > 10 and gld["sixm_return"] > 0:
            return (
                "A Soros-style lens sees a powerful narrative driving prices higher, but also watches for instability. "
                "When both risk assets and hedges rise together, the system may be more fragile than it looks."
            )
        return (
            "A Soros-style lens would watch how narrative and price reinforce one another, "
            "then look for the moment that reflexive confidence begins to break."
        )

    if guru == "Peter Lynch":
        return (
            "A Lynch-style approach would still look for companies with understandable business models, "
            "solid growth, and valuations that are not completely disconnected from fundamentals."
        )

    if guru == "Jim Simons":
        if pd.notna(qqq["momentum_6m"]) and pd.notna(spy["momentum_6m"]) and qqq["momentum_6m"] > 0 and spy["momentum_6m"] > 0:
            return (
                "A Simons-style system would likely classify the market as trend-friendly, "
                "while also ranking assets by momentum, volatility, and relative strength rather than stories."
            )
        return "A Simons-style system would probably reduce trend exposure and rely more on diversified signal blending."

    if guru == "Cathie Wood":
        if pd.notna(qqq["distance_200"]) and pd.notna(soxx["sixm_return"]) and qqq["distance_200"] > 0 and soxx["sixm_return"] > 0:
            return (
                "A Cathie Wood-style framework remains positive on innovation leadership, "
                "especially where AI and software platforms can scale rapidly over several years."
            )
        return (
            "A Cathie Wood-style framework would still seek innovation names, "
            "but market conditions are less supportive when rates or long-duration pressure rise."
        )

    if guru == "Michael Burry":
        if pd.notna(qqq["distance_200"]) and qqq["distance_200"] > 10:
            return (
                "A Burry-style interpretation would warn that enthusiasm, valuation stretch, and crowded positioning "
                "can create downside asymmetry even while prices still look strong."
            )
        return "A Burry-style interpretation would stay skeptical and continue screening for hidden fragility under the surface."

    if guru == "Paul Tudor Jones":
        if (pd.notna(gld["sixm_return"]) and gld["sixm_return"] > 0) or (pd.notna(uso["sixm_return"]) and uso["sixm_return"] > 10):
            return (
                "A Paul Tudor Jones-style approach sees inflation and macro hedging as essential. "
                "Trend matters, but capital protection and optionality matter more."
            )
        return (
            "A Paul Tudor Jones-style approach would remain tactical: participate in trends, "
            "but stay ready to pivot quickly when macro conditions change."
        )

    return "No interpretation available."


def suggested_action_by_guru(guru: str, market_metrics: Dict[str, Dict[str, float]]) -> str:
    if guru == "Warren Buffett":
        return "Prefer quality compounders, hold some cash, and avoid chasing the most extended speculative names."
    if guru == "Ray Dalio":
        return "Use balanced diversification across equities, bonds, gold, and commodities rather than pure equity concentration."
    if guru == "Howard Marks":
        return "Stay selective, lean slightly defensive, and keep dry powder for better risk/reward entries."
    if guru == "Stanley Druckenmiller":
        return "Follow the strongest macro trend leaders while cutting losers quickly if trend breaks."
    if guru == "George Soros":
        return "Participate tactically in strong narratives but monitor instability and reversal risk closely."
    if guru == "Peter Lynch":
        return "Focus on understandable growth businesses with strong fundamentals and acceptable valuation."
    if guru == "Jim Simons":
        return "Rank opportunities systematically using momentum, volatility, and drawdown rather than opinion."
    if guru == "Cathie Wood":
        return "Hold high-conviction innovation names, but accept that volatility can remain very high."
    if guru == "Michael Burry":
        return "Raise defense, keep hedges, and avoid assuming that strong performance means low risk."
    if guru == "Paul Tudor Jones":
        return "Combine trend participation with gold, commodities, and disciplined risk limits."
    return "No action defined."


# ============================================================
# Render helper for each guru tab
# ============================================================
def render_guru_tab(guru_name: str, prices: pd.DataFrame, benchmark_ticker: str):
    meta = GURU_META[guru_name]
    alias = meta["alias"]
    candidates = GURU_CANDIDATES[alias]

    st.markdown(f"## {guru_name}")
    st.markdown(f"**Strategy:** {meta['strategy']}")
    st.markdown(f"**Core Focus:** {meta['focus']}")
    st.markdown(f"**Style Interpretation:** {meta['style_interpretation']}")

    st.info(guru_interpretation(guru_name, market_metrics))

    st.markdown("### Key Indicators")
    indicator_cols = st.columns(len(meta["indicators"]))
    for i, ind in enumerate(meta["indicators"]):
        indicator_cols[i].metric(f"Indicator {i + 1}", ind)

    st.markdown("### Suggested Positioning")
    portfolio_df = pd.DataFrame(
        [{"Bucket": k, "Allocation": v} for k, v in meta["portfolio"].items()]
    )
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

    st.markdown("### Candidate Assets")
    available_candidates = [t for t in candidates if t in prices.columns]

    if not available_candidates:
        st.warning(f"No available price data for {guru_name} candidate assets.")
        return

    metrics_df = metrics_table_for_tickers(prices, available_candidates)

    if guru_name in ["Warren Buffett", "Peter Lynch"]:
        sort_col = "ROE %"
        ascending = False
    elif guru_name in ["Stanley Druckenmiller", "Cathie Wood", "Jim Simons"]:
        sort_col = "6M %"
        ascending = False
    elif guru_name in ["Michael Burry", "Howard Marks"]:
        sort_col = "Dist vs MA200 %"
        ascending = False
    else:
        sort_col = "6M %"
        ascending = False

    if not metrics_df.empty and sort_col in metrics_df.columns:
        metrics_df = metrics_df.sort_values(sort_col, ascending=ascending)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.markdown("### Suggested Action")
    st.success(suggested_action_by_guru(guru_name, market_metrics))

    st.markdown("### Visual Analysis")
    selected_ticker = st.selectbox(
        f"Select a ticker for {guru_name}",
        available_candidates,
        index=0,
        key=f"{guru_name}_ticker_select",
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_price = make_price_chart(
            prices,
            selected_ticker,
            title=f"{selected_ticker} Price Trend",
            show_ma=True,
        )
        st.plotly_chart(
            fig_price,
            use_container_width=True,
            key=f"{guru_name}_{selected_ticker}_price_chart",
        )

    with c2:
        fig_relative = make_relative_chart(prices, selected_ticker, benchmark_ticker)
        st.plotly_chart(
            fig_relative,
            use_container_width=True,
            key=f"{guru_name}_{selected_ticker}_{benchmark_ticker}_relative_chart",
        )

    fig_drawdown = make_drawdown_chart(prices, selected_ticker)
    st.plotly_chart(
        fig_drawdown,
        use_container_width=True,
        key=f"{guru_name}_{selected_ticker}_drawdown_chart",
    )

    st.markdown("### Fundamentals / Business Snapshot")
    render_fundamental_card(selected_ticker)

    st.markdown("### Group Comparison")
    top_compare = available_candidates[: min(5, len(available_candidates))]
    fig_compare = make_compare_chart(
        prices,
        top_compare,
        f"{guru_name} Candidate Basket Comparison",
    )
    st.plotly_chart(
        fig_compare,
        use_container_width=True,
        key=f"{guru_name}_group_compare_chart",
    )


# ============================================================
# Tabs
# ============================================================
tab_names = [
    "Overview",
    "Warren Buffett",
    "Ray Dalio",
    "Howard Marks",
    "Stanley Druckenmiller",
    "George Soros",
    "Peter Lynch",
    "Jim Simons",
    "Cathie Wood",
    "Michael Burry",
    "Paul Tudor Jones",
]
tabs = st.tabs(tab_names)

# ============================================================
# Overview tab
# ============================================================
with tabs[0]:
    st.subheader("Overview: 10 Guru Strategies on the Current Market")

    overview_rows = []
    for guru_name, meta in GURU_META.items():
        overview_rows.append(
            {
                "Guru": guru_name,
                "Strategy": meta["strategy"],
                "Core Focus": meta["focus"],
                "Current Interpretation": guru_interpretation(guru_name, market_metrics),
                "Suggested Action": suggested_action_by_guru(guru_name, market_metrics),
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    st.markdown("### Cross-Asset Scoreboard")
    cross_rows = []
    for t in ["SPY", "QQQ", "TLT", "GLD", "DBC", "USO", "XLK", "XLV", "SOXX"]:
        m = compute_return_metrics(prices[t])
        cross_rows.append(
            {
                "Ticker": t,
                "Name": MARKET_TICKERS.get(t, t),
                "Price": m["last"],
                "YTD %": m["ytd_return"],
                "6M %": m["sixm_return"],
                "1Y %": m["oney_return"],
                "Dist vs MA50 %": m["distance_50"],
                "Dist vs MA200 %": m["distance_200"],
                "Max Drawdown %": m["mdd"],
                "Volatility %": m["vol_20d"],
            }
        )

    cross_df = pd.DataFrame(cross_rows)
    st.dataframe(cross_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            make_compare_chart(
                prices,
                ["SPY", "QQQ", "TLT", "GLD", "USO"],
                "Cross-Asset Leadership",
            ),
            use_container_width=True,
            key="overview_cross_asset_chart",
        )

    with c2:
        st.plotly_chart(
            make_compare_chart(
                prices,
                ["XLK", "XLV", "XLE", "XLF", "SOXX"],
                "Sector / Theme Leadership",
            ),
            use_container_width=True,
            key="overview_sector_theme_chart",
        )

    st.markdown("### Strategic Summary")
    st.write(
        f"""
- **Current regime:** {regime_label}  
- **Broad market trend:** SPY vs MA200 = {format_pct(market_metrics['SPY']['distance_200'])}  
- **Growth leadership:** QQQ vs MA200 = {format_pct(market_metrics['QQQ']['distance_200'])}  
- **AI / semiconductor momentum:** SOXX 6M = {format_pct(market_metrics['SOXX']['sixm_return'])}  
- **Macro hedge behavior:** GLD 6M = {format_pct(market_metrics['GLD']['sixm_return'])}  
- **Inflation / energy pressure:** USO 6M = {format_pct(market_metrics['USO']['sixm_return'])}
"""
    )

# ============================================================
# Individual guru tabs
# ============================================================
with tabs[1]:
    render_guru_tab("Warren Buffett", prices, benchmark)

with tabs[2]:
    render_guru_tab("Ray Dalio", prices, benchmark)

with tabs[3]:
    render_guru_tab("Howard Marks", prices, benchmark)

with tabs[4]:
    render_guru_tab("Stanley Druckenmiller", prices, benchmark)

with tabs[5]:
    render_guru_tab("George Soros", prices, benchmark)

with tabs[6]:
    render_guru_tab("Peter Lynch", prices, benchmark)

with tabs[7]:
    render_guru_tab("Jim Simons", prices, benchmark)

with tabs[8]:
    render_guru_tab("Cathie Wood", prices, benchmark)

with tabs[9]:
    render_guru_tab("Michael Burry", prices, benchmark)

with tabs[10]:
    render_guru_tab("Paul Tudor Jones", prices, benchmark)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Data source: Yahoo Finance via yfinance. "
    "This dashboard is for research / educational use only and is not investment advice."
)

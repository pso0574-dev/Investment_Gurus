# streamlit_app.py
# ============================================================
# Guru Investor Strategy Dashboard
# ------------------------------------------------------------
# Full version with:
# - 10 Guru tabs
# - Market parameters by guru
# - Guru stance / score
# - Likes / avoids / risk triggers
# - Market cycle analysis included
# - Overview tab with cross-guru comparison
# - Unique Plotly keys to avoid StreamlitDuplicateElementId
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
st.caption(
    "Analyze the current market through the strategies of 10 top guru investors, including market-cycle interpretation."
)

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

show_raw_data = st.sidebar.checkbox("Show raw downloaded data", value=False)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

# ============================================================
# Universe
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
    "XLE": "Energy ETF",
    "XLK": "Technology ETF",
    "XLF": "Financials ETF",
    "XLV": "Healthcare ETF",
    "XLP": "Consumer Staples ETF",
    "SOXX": "Semiconductor ETF",
    "ARKK": "ARK Innovation ETF",
    "VIXY": "VIX ETF",
    "BIL": "1-3 Month T-Bill ETF",
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
        + [benchmark]
    )
)

# ============================================================
# Utilities
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

    current_year = s.index[-1].year
    year_start = s[s.index.year == current_year]
    out["ytd_return"] = (s.iloc[-1] / year_start.iloc[0] - 1.0) * 100 if len(year_start) > 1 else np.nan
    out["sixm_return"] = (s.iloc[-1] / s.iloc[-126] - 1.0) * 100 if len(s) >= 126 else np.nan
    out["oney_return"] = (s.iloc[-1] / s.iloc[-252] - 1.0) * 100 if len(s) >= 252 else np.nan
    out["momentum_3m"] = (s.iloc[-1] / s.iloc[-63] - 1.0) * 100 if len(s) >= 63 else np.nan
    out["momentum_6m"] = (s.iloc[-1] / s.iloc[-126] - 1.0) * 100 if len(s) >= 126 else np.nan

    return out


def relative_performance(price: pd.Series, bench: pd.Series) -> pd.Series:
    df = pd.concat([price.dropna(), bench.dropna()], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df.iloc[:, 0] / df.iloc[:, 0].iloc[0]) / (df.iloc[:, 1] / df.iloc[:, 1].iloc[0])


def drawdown_series(price: pd.Series) -> pd.Series:
    s = price.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s / s.cummax() - 1.0


def normalize_series(price: pd.Series) -> pd.Series:
    s = price.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s / s.iloc[0] * 100


def format_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.2f}%"


def format_num(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.2f}"


def format_big_num(x) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= 1e12:
        return f"{x/1e12:,.2f}T"
    if x >= 1e9:
        return f"{x/1e9:,.2f}B"
    if x >= 1e6:
        return f"{x/1e6:,.2f}M"
    return f"{x:,.0f}"


def make_price_chart(prices: pd.DataFrame, ticker: str, title: str | None = None, show_ma: bool = True) -> go.Figure:
    s = prices[ticker].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=ticker, line=dict(width=2)))

    if show_ma and len(s) >= 50:
        ma50 = s.rolling(50).mean()
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode="lines", name="MA50", line=dict(width=1.4, dash="dash")))
    if show_ma and len(s) >= 200:
        ma200 = s.rolling(200).mean()
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, mode="lines", name="MA200", line=dict(width=1.4, dash="dot")))

    fig.update_layout(
        title=title or f"{ticker} Price",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def make_relative_chart(prices: pd.DataFrame, ticker: str, benchmark_ticker: str) -> go.Figure:
    rel = relative_performance(prices[ticker], prices[benchmark_ticker])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rel.index, y=rel.values, mode="lines", name=f"{ticker}/{benchmark_ticker}", line=dict(width=2)))
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
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown %", fill="tozeroy"))
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
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=t))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig


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
        rows.append({
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
        })
    return pd.DataFrame(rows)


def render_fundamental_card(ticker: str):
    f = get_fundamental_snapshot(ticker)
    st.markdown(f"### {ticker} — {f['shortName']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sector", f"{f['sector']}")
    c2.metric("Industry", f"{f['industry']}")
    c3.metric("Market Cap", format_big_num(f["marketCap"]))
    c4.metric("Trailing PE", format_num(f["trailingPE"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Forward PE", format_num(f["forwardPE"]))
    c6.metric("PEG", format_num(f["pegRatio"]))
    c7.metric("P/B", format_num(f["priceToBook"]))
    c8.metric("ROE", format_pct(f["returnOnEquity"] * 100 if pd.notna(f["returnOnEquity"]) else np.nan))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Debt/Equity", format_num(f["debtToEquity"]))
    c10.metric("Revenue Growth", format_pct(f["revenueGrowth"] * 100 if pd.notna(f["revenueGrowth"]) else np.nan))
    c11.metric("Earnings Growth", format_pct(f["earningsGrowth"] * 100 if pd.notna(f["earningsGrowth"]) else np.nan))
    c12.metric("Dividend Yield", format_pct(f["dividendYield"] * 100 if pd.notna(f["dividendYield"]) else np.nan))

    summary = f.get("longBusinessSummary", "")
    if summary:
        st.caption(summary[:800] + ("..." if len(summary) > 800 else ""))


# ============================================================
# Market regime and cycle
# ============================================================
def score_market_regime(metrics: Dict[str, Dict[str, float]]) -> Tuple[int, str, Dict[str, int]]:
    score_details = {
        "Trend": 0,
        "Growth Leadership": 0,
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
            score_details["Growth Leadership"] += 2
        elif metrics["QQQ"]["distance_200"] > -5:
            score_details["Growth Leadership"] += 1

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
        label = "Late Cycle / Cautious"
    else:
        label = "Risk-Off"

    return total, label, score_details


def compute_market_cycle(metrics: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    spy = metrics["SPY"]
    qqq = metrics["QQQ"]
    tlt = metrics["TLT"]
    gld = metrics["GLD"]
    uso = metrics["USO"]
    soxx = metrics["SOXX"]
    xlk = metrics["XLK"]
    xlv = metrics["XLV"]

    detail = {}

    detail["Broad Equity Trend"] = 2 if spy["distance_200"] > 0 else 1 if spy["distance_200"] > -5 else -1
    detail["Growth Leadership"] = 2 if qqq["distance_200"] > spy["distance_200"] else 0
    detail["AI / Semiconductor Momentum"] = 2 if soxx["sixm_return"] > 15 else 1 if soxx["sixm_return"] > 0 else -1
    detail["Bond Confirmation"] = 1 if tlt["distance_200"] > 0 else 0 if tlt["distance_200"] > -5 else -1
    detail["Inflation / Oil Pressure"] = -2 if uso["sixm_return"] > 15 else -1 if uso["sixm_return"] > 5 else 0
    detail["Gold Hedge Demand"] = -1 if gld["sixm_return"] > 8 else 0
    detail["Defensive Rotation"] = -1 if xlv["sixm_return"] > xlk["sixm_return"] else 1

    score = int(sum(detail.values()))

    if score >= 6:
        phase = "Expansion / Bull"
        phase_desc = "Broad uptrend, growth leadership, strong risk appetite."
    elif score >= 3:
        phase = "Mid Cycle / Constructive"
        phase_desc = "Positive market tone, but with emerging valuation or macro constraints."
    elif score >= 0:
        phase = "Late Cycle / Selective"
        phase_desc = "Trend still exists, but inflation, hedging demand, or leadership concentration raise fragility."
    elif score >= -3:
        phase = "Slowdown / Defensive"
        phase_desc = "Risk appetite is weakening, leadership narrows, and defense becomes more relevant."
    else:
        phase = "Risk-Off / Contraction"
        phase_desc = "Weak trend and high fragility; preservation of capital becomes primary."

    if phase in ["Expansion / Bull", "Mid Cycle / Constructive"] and uso["sixm_return"] > 10 and gld["sixm_return"] > 0:
        sub_view = "Bull market with macro stress"
    elif phase == "Late Cycle / Selective" and qqq["distance_200"] > 0 and soxx["sixm_return"] > 0:
        sub_view = "Late-cycle growth-led market"
    elif phase in ["Slowdown / Defensive", "Risk-Off / Contraction"]:
        sub_view = "Defensive transition"
    else:
        sub_view = "Balanced cycle"

    return {
        "score": score,
        "phase": phase,
        "phase_desc": phase_desc,
        "sub_view": sub_view,
        "detail": detail,
    }


# ============================================================
# Guru metadata
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
        "market_parameters": [
            "Valuation discipline",
            "Quality of business",
            "Free cash flow stability",
            "Return on equity",
            "Debt burden",
            "Margin durability",
        ],
        "likes_now": [
            "High-quality cash-generating large caps",
            "Defensive compounders",
            "Businesses with pricing power",
        ],
        "avoids_now": [
            "Highly speculative AI names without profits",
            "Extreme valuation expansion",
            "Weak balance sheets",
        ],
        "risk_triggers": [
            "Market multiple expansion without earnings support",
            "Consumer weakness",
            "Margin compression",
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
        "market_parameters": [
            "Growth regime",
            "Inflation regime",
            "Real rates",
            "Debt burden",
            "Liquidity conditions",
            "Cross-asset correlation",
        ],
        "likes_now": [
            "Balanced portfolios",
            "Gold and commodities when inflation risk is present",
            "Diversified cross-asset exposure",
        ],
        "avoids_now": [
            "Single-factor concentration",
            "Pure tech-only portfolios",
            "Ignoring bond / inflation interaction",
        ],
        "risk_triggers": [
            "Rates shock",
            "Debt refinancing stress",
            "Inflation re-acceleration",
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
        "market_parameters": [
            "Cycle position",
            "Investor sentiment",
            "Valuation stretch",
            "Risk appetite",
            "Margin of safety",
        ],
        "likes_now": [
            "Selective value",
            "Defensive sectors",
            "Cash optionality",
        ],
        "avoids_now": [
            "Crowded momentum trades",
            "Euphoria-driven entries",
            "Buying without margin of safety",
        ],
        "risk_triggers": [
            "Narrative excess",
            "Complacency",
            "Sharp multiple compression",
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
        "market_parameters": [
            "Trend strength",
            "Theme leadership",
            "Liquidity backdrop",
            "Macro acceleration",
            "Price confirmation",
        ],
        "likes_now": [
            "AI infrastructure leaders",
            "Semiconductors",
            "Strong trend-following names",
        ],
        "avoids_now": [
            "Weak relative strength names",
            "Falling narratives",
            "Macro-inconsistent trades",
        ],
        "risk_triggers": [
            "Trend break below key moving averages",
            "Leadership rotation",
            "Liquidity tightening",
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
            "Narrative leadership",
            "Momentum extremes",
            "Volatility hedge demand",
            "Cross-asset stress",
            "Crowding",
        ],
        "market_parameters": [
            "Narrative intensity",
            "Crowding",
            "Reflexive price acceleration",
            "Volatility regime",
            "Capital flow fragility",
        ],
        "likes_now": [
            "Strong but still accelerating narratives",
            "Tactical macro dislocations",
            "Reversal setups after excess",
        ],
        "avoids_now": [
            "Late crowded entries",
            "Low-vol complacency",
            "Consensus certainty",
        ],
        "risk_triggers": [
            "Narrative failure",
            "Volatility spike",
            "Capital flow reversal",
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
        "market_parameters": [
            "Revenue growth",
            "PEG reasonableness",
            "Business simplicity",
            "Scalable earnings",
            "Sector opportunity",
        ],
        "likes_now": [
            "Profitable growth companies",
            "Reasonably priced compounders",
            "Leaders with understandable products",
        ],
        "avoids_now": [
            "Story stocks without earnings support",
            "Overhyped sectors with no valuation discipline",
            "Complex businesses hard to understand",
        ],
        "risk_triggers": [
            "Growth slowdown",
            "PEG expansion",
            "Execution miss",
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
        "market_parameters": [
            "Momentum rank",
            "Volatility regime",
            "Cross-sectional dispersion",
            "Drawdown behavior",
            "Signal persistence",
        ],
        "likes_now": [
            "Strong momentum assets",
            "Liquid ETFs and trend leaders",
            "Systematically ranked baskets",
        ],
        "avoids_now": [
            "Emotion-driven discretionary entries",
            "Low-liquidity names",
            "Narrative-only investing",
        ],
        "risk_triggers": [
            "Momentum crash",
            "Volatility shock",
            "Factor reversal",
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
        "market_parameters": [
            "Innovation adoption curve",
            "Addressable market size",
            "Growth persistence",
            "Rate sensitivity",
            "Disruptive potential",
        ],
        "likes_now": [
            "AI platforms",
            "Software disruptors",
            "High-conviction innovation leaders",
        ],
        "avoids_now": [
            "Low-growth value traps",
            "Businesses with limited TAM",
            "Capital-intensive weak innovators",
        ],
        "risk_triggers": [
            "Rates rising sharply",
            "Duration compression",
            "Narrative fatigue",
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
        "market_parameters": [
            "Bubble conditions",
            "Crowding",
            "Valuation excess",
            "Liquidity fragility",
            "Downside asymmetry",
        ],
        "likes_now": [
            "Defensive assets",
            "Cash",
            "Hedged structures",
        ],
        "avoids_now": [
            "Parabolic names",
            "Consensus darlings",
            "Overvalued momentum extremes",
        ],
        "risk_triggers": [
            "Valuation collapse",
            "Liquidity event",
            "Retail capitulation",
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
        "market_parameters": [
            "Inflation pressure",
            "Commodity trend",
            "Bond weakness",
            "Equity trend",
            "Risk control regime",
        ],
        "likes_now": [
            "Gold",
            "Energy / commodities",
            "Trend-following macro trades",
        ],
        "avoids_now": [
            "Oversized single bets",
            "Ignoring macro volatility",
            "Unhedged exposure in unstable environments",
        ],
        "risk_triggers": [
            "Inflation shock",
            "Oil spike",
            "Policy surprise",
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
# Data download
# ============================================================
with st.spinner("Downloading market data..."):
    raw_data = download_price_data(ALL_TICKERS, period=selected_period)
    prices = extract_close_matrix(raw_data, ALL_TICKERS)

if prices.empty:
    st.error("No price data could be loaded.")
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
market_cycle = compute_market_cycle(market_metrics)

st.subheader("Market Snapshot")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Regime Score", f"{regime_score}")
c2.metric("Regime Label", regime_label)
c3.metric("Market Cycle Score", f"{market_cycle['score']}")
c4.metric("Market Cycle Phase", market_cycle["phase"])
c5.metric("Cycle Sub-View", market_cycle["sub_view"])

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("SPY 6M", format_pct(market_metrics["SPY"]["sixm_return"]))
c7.metric("QQQ 6M", format_pct(market_metrics["QQQ"]["sixm_return"]))
c8.metric("SOXX 6M", format_pct(market_metrics["SOXX"]["sixm_return"]))
c9.metric("TLT 6M", format_pct(market_metrics["TLT"]["sixm_return"]))
c10.metric("GLD 6M", format_pct(market_metrics["GLD"]["sixm_return"]))

st.info(
    f"**Current Market Cycle:** {market_cycle['phase']} — {market_cycle['phase_desc']} "
    f"Sub-view: **{market_cycle['sub_view']}**."
)

with st.expander("Regime / Cycle Detail", expanded=False):
    left, right = st.columns(2)
    with left:
        st.markdown("#### Regime Detail")
        st.dataframe(
            pd.DataFrame({
                "Factor": list(regime_detail.keys()),
                "Score": list(regime_detail.values()),
            }),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("#### Market Cycle Detail")
        st.dataframe(
            pd.DataFrame({
                "Factor": list(market_cycle["detail"].keys()),
                "Score": list(market_cycle["detail"].values()),
            }),
            use_container_width=True,
            hide_index=True,
        )

st.plotly_chart(
    make_compare_chart(
        prices,
        ["SPY", "QQQ", "TLT", "GLD", "DBC"],
        "Normalized Performance: Equities vs Bonds vs Gold vs Commodities",
    ),
    use_container_width=True,
    key="top_market_compare_chart",
)

# ============================================================
# Commentary helpers
# ============================================================
def global_market_commentary(regime_label: str, metrics: Dict[str, Dict[str, float]], cycle: Dict[str, object]) -> str:
    spy_200 = metrics["SPY"]["distance_200"]
    qqq_200 = metrics["QQQ"]["distance_200"]
    soxx_6m = metrics["SOXX"]["sixm_return"]
    tlt_6m = metrics["TLT"]["sixm_return"]
    gld_6m = metrics["GLD"]["sixm_return"]
    uso_6m = metrics["USO"]["sixm_return"]

    comments = []

    if regime_label in ["Bull / Risk-On", "Cautious Bull"]:
        comments.append("Equity trend remains constructive.")
    elif regime_label == "Neutral":
        comments.append("The market is mixed rather than broadly strong.")
    else:
        comments.append("The market shows late-cycle or defensive characteristics.")

    comments.append(f"Cycle phase currently looks like {cycle['phase']}.")

    if pd.notna(qqq_200) and pd.notna(spy_200) and qqq_200 > spy_200:
        comments.append("Growth / technology leadership remains stronger than the broad market.")
    if pd.notna(soxx_6m) and soxx_6m > 0:
        comments.append("Semiconductor momentum still supports the AI narrative.")
    if pd.notna(gld_6m) and gld_6m > 0:
        comments.append("Gold strength signals ongoing macro hedge demand.")
    if pd.notna(uso_6m) and uso_6m > 10:
        comments.append("Energy strength raises inflation and geopolitical risk.")
    if pd.notna(tlt_6m) and tlt_6m < 0:
        comments.append("Weak long-duration bonds can pressure valuation-heavy assets.")

    return " ".join(comments)


st.info(global_market_commentary(regime_label, market_metrics, market_cycle))

# ============================================================
# Guru analysis
# ============================================================
def compute_guru_market_view(guru_name: str, market_metrics: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    spy = market_metrics["SPY"]
    qqq = market_metrics["QQQ"]
    tlt = market_metrics["TLT"]
    gld = market_metrics["GLD"]
    uso = market_metrics["USO"]
    soxx = market_metrics["SOXX"]

    detail = {}
    score = 0

    if guru_name == "Warren Buffett":
        detail["Valuation Discipline"] = -1 if qqq["distance_200"] > 10 else 1
        detail["Business Quality Environment"] = 2 if spy["distance_200"] > 0 else 0
        detail["Speculation Risk"] = -2 if soxx["sixm_return"] > 20 else -1
        detail["Cycle Comfort"] = 1 if market_cycle["phase"] in ["Mid Cycle / Constructive", "Late Cycle / Selective"] else 0
        score = sum(detail.values())

    elif guru_name == "Ray Dalio":
        detail["Equity Trend"] = 1 if spy["distance_200"] > 0 else -1
        detail["Gold Confirmation"] = 1 if gld["sixm_return"] > 0 else 0
        detail["Commodity Pressure"] = -1 if uso["sixm_return"] > 10 else 0
        detail["Bond Diversification"] = 1 if tlt["distance_200"] > -5 else 0
        detail["Cycle Balance"] = 1 if market_cycle["phase"] in ["Late Cycle / Selective", "Mid Cycle / Constructive"] else 0
        score = sum(detail.values())

    elif guru_name == "Howard Marks":
        detail["Cycle Optimism"] = -2 if qqq["distance_200"] > 10 else -1
        detail["Risk Appetite"] = -1 if soxx["sixm_return"] > 20 else 0
        detail["Margin of Safety"] = -1 if spy["distance_200"] > 8 else 1
        detail["Market Cycle"] = -1 if market_cycle["phase"] in ["Expansion / Bull", "Mid Cycle / Constructive"] else 1
        score = sum(detail.values())

    elif guru_name == "Stanley Druckenmiller":
        detail["Trend Strength"] = 2 if qqq["distance_200"] > 0 else -1
        detail["AI Leadership"] = 2 if soxx["sixm_return"] > 10 else 0
        detail["Macro Confirmation"] = 1 if spy["sixm_return"] > 0 else -1
        detail["Cycle Tailwind"] = 1 if market_cycle["phase"] in ["Expansion / Bull", "Mid Cycle / Constructive"] else 0
        score = sum(detail.values())

    elif guru_name == "George Soros":
        detail["Narrative Power"] = 2 if qqq["distance_200"] > 0 else 0
        detail["Crowding Risk"] = -2 if soxx["sixm_return"] > 20 else -1
        detail["Hedge Demand"] = -1 if gld["sixm_return"] > 0 else 0
        detail["Cycle Instability"] = -1 if market_cycle["phase"] == "Late Cycle / Selective" else 0
        score = sum(detail.values())

    elif guru_name == "Peter Lynch":
        detail["Growth Environment"] = 2 if qqq["distance_200"] > 0 else 0
        detail["Reasonable Pricing"] = -1 if qqq["distance_200"] > 10 else 1
        detail["Business Opportunity"] = 1 if spy["sixm_return"] > 0 else 0
        detail["Cycle Suitability"] = 1 if market_cycle["phase"] in ["Mid Cycle / Constructive", "Late Cycle / Selective"] else 0
        score = sum(detail.values())

    elif guru_name == "Jim Simons":
        detail["Momentum Regime"] = 2 if qqq["momentum_6m"] > 0 else -1
        detail["Relative Strength"] = 2 if soxx["sixm_return"] > tlt["sixm_return"] else 0
        detail["Volatility Stability"] = 1 if spy["vol_20d"] < 25 else -1
        detail["Cycle Persistence"] = 1 if market_cycle["phase"] in ["Expansion / Bull", "Mid Cycle / Constructive"] else 0
        score = sum(detail.values())

    elif guru_name == "Cathie Wood":
        detail["Innovation Tailwind"] = 2 if qqq["distance_200"] > 0 else -1
        detail["AI Narrative"] = 2 if soxx["sixm_return"] > 10 else 0
        detail["Rate Sensitivity"] = -1 if tlt["sixm_return"] < 0 else 1
        detail["Cycle Suitability"] = 1 if market_cycle["phase"] in ["Expansion / Bull", "Mid Cycle / Constructive"] else -1
        score = sum(detail.values())

    elif guru_name == "Michael Burry":
        detail["Bubble Risk"] = -2 if qqq["distance_200"] > 10 else -1
        detail["Crowding"] = -2 if soxx["sixm_return"] > 20 else -1
        detail["Defensive Need"] = 1 if gld["sixm_return"] > 0 else 0
        detail["Late Cycle Warning"] = -1 if market_cycle["phase"] in ["Expansion / Bull", "Mid Cycle / Constructive", "Late Cycle / Selective"] else 1
        score = sum(detail.values())

    elif guru_name == "Paul Tudor Jones":
        detail["Trend Opportunity"] = 1 if spy["distance_200"] > 0 else -1
        detail["Inflation Hedge Need"] = 2 if uso["sixm_return"] > 10 else 1 if gld["sixm_return"] > 0 else 0
        detail["Risk Management Pressure"] = -1 if tlt["sixm_return"] < 0 else 0
        detail["Cycle Read"] = 1 if market_cycle["phase"] in ["Late Cycle / Selective", "Mid Cycle / Constructive"] else 0
        score = sum(detail.values())

    else:
        detail["Generic"] = 0
        score = 0

    if score >= 4:
        stance = "Bullish"
    elif score >= 1:
        stance = "Constructive / Selective"
    elif score >= -1:
        stance = "Neutral / Mixed"
    elif score >= -3:
        stance = "Cautious"
    else:
        stance = "Defensive / Bearish"

    return {"score": score, "stance": stance, "detail": detail}


def guru_interpretation(guru: str, market_metrics: Dict[str, Dict[str, float]]) -> str:
    qqq = market_metrics["QQQ"]
    gld = market_metrics["GLD"]
    uso = market_metrics["USO"]
    soxx = market_metrics["SOXX"]
    spy = market_metrics["SPY"]
    tlt = market_metrics["TLT"]

    if guru == "Warren Buffett":
        return (
            "Buffett would likely see quality businesses still worth owning, but he would be cautious if growth-heavy segments are pricing in too much perfection."
        )
    if guru == "Ray Dalio":
        return (
            "Dalio would likely focus on the interaction between equities, bonds, gold, and commodities rather than making a one-directional equity bet."
        )
    if guru == "Howard Marks":
        return (
            "Marks would likely ask where we are in the cycle, whether optimism is excessive, and whether the current market still offers a sufficient margin of safety."
        )
    if guru == "Stanley Druckenmiller":
        return (
            "Druckenmiller would likely stay with the strongest trend leaders as long as price, macro, and narrative continue to align."
        )
    if guru == "George Soros":
        return (
            "Soros would focus on whether a strong narrative is reflexively feeding price gains and how fragile that loop may become if sentiment shifts."
        )
    if guru == "Peter Lynch":
        return (
            "Lynch would look for understandable companies still delivering real growth without paying extreme valuations simply because a sector is popular."
        )
    if guru == "Jim Simons":
        return (
            "Simons would likely treat the market as a signal environment: trends, volatility, relative strength, and cross-sectional behavior matter more than stories."
        )
    if guru == "Cathie Wood":
        return (
            "Cathie Wood would remain focused on innovation leadership, but long-duration sensitivity and rate pressure still matter greatly for her style."
        )
    if guru == "Michael Burry":
        return (
            "Burry would look for crowding, overvaluation, and hidden fragility beneath strong index performance, especially in narrative-driven leaders."
        )
    if guru == "Paul Tudor Jones":
        return (
            "Paul Tudor Jones would emphasize trend participation with disciplined risk control, especially where inflation or commodity pressure remains active."
        )
    return "No interpretation available."


def guru_market_viewpoint(guru_name: str, market_metrics: Dict[str, Dict[str, float]]) -> str:
    view = compute_guru_market_view(guru_name, market_metrics)
    stance = view["stance"]

    custom = {
        "Warren Buffett": {
            "Bullish": "Buffett would likely see selective opportunities, though he still would not chase euphoric pricing.",
            "Constructive / Selective": "Buffett would prefer disciplined buying in quality businesses rather than joining speculative enthusiasm.",
            "Neutral / Mixed": "Buffett would likely remain patient and wait for stronger value opportunities.",
            "Cautious": "Buffett would probably hold extra cash and avoid aggressively priced growth stocks.",
            "Defensive / Bearish": "Buffett would likely become highly selective and emphasize downside protection.",
        },
        "Ray Dalio": {
            "Bullish": "Dalio would see a workable macro backdrop, but still prefer broad diversification.",
            "Constructive / Selective": "Dalio would likely keep a balanced All-Weather style allocation.",
            "Neutral / Mixed": "Dalio would emphasize diversification because macro signals are conflicting.",
            "Cautious": "Dalio would likely shift more weight toward inflation hedges and balance-sheet resilience.",
            "Defensive / Bearish": "Dalio would reduce concentration risk and prioritize resilient cross-asset structure.",
        },
        "Howard Marks": {
            "Bullish": "Marks would still ask whether the optimism is already priced in.",
            "Constructive / Selective": "Marks would likely be selective rather than broadly aggressive.",
            "Neutral / Mixed": "Marks would likely stay cautious about paying up late in the cycle.",
            "Cautious": "Marks would likely prefer cash optionality and more defensive positioning.",
            "Defensive / Bearish": "Marks would likely avoid crowded assets and wait for better bargains.",
        },
    }

    if guru_name in custom and stance in custom[guru_name]:
        return custom[guru_name][stance]

    return f"{guru_name} current stance: {stance}."


def suggested_action_by_guru(guru: str, market_metrics: Dict[str, Dict[str, float]]) -> str:
    if guru == "Warren Buffett":
        return "Prefer quality compounders, keep some cash, and avoid chasing the most extended speculative names."
    if guru == "Ray Dalio":
        return "Use balanced diversification across equities, bonds, gold, and commodities."
    if guru == "Howard Marks":
        return "Stay selective, lean slightly defensive, and keep dry powder for better entries."
    if guru == "Stanley Druckenmiller":
        return "Follow the strongest macro trend leaders while cutting losers quickly if the trend breaks."
    if guru == "George Soros":
        return "Participate tactically in strong narratives but monitor instability and reversal risk closely."
    if guru == "Peter Lynch":
        return "Focus on understandable growth businesses with strong fundamentals and acceptable valuation."
    if guru == "Jim Simons":
        return "Rank opportunities systematically using momentum, volatility, and drawdown rather than opinion."
    if guru == "Cathie Wood":
        return "Hold high-conviction innovation names, but accept high volatility and duration sensitivity."
    if guru == "Michael Burry":
        return "Raise defense, keep hedges, and do not confuse strong price action with low risk."
    if guru == "Paul Tudor Jones":
        return "Combine trend participation with gold, commodities, and disciplined risk limits."
    return "No action defined."


# ============================================================
# Tab rendering
# ============================================================
def render_guru_tab(guru_name: str, prices: pd.DataFrame, benchmark_ticker: str):
    meta = GURU_META[guru_name]
    alias = meta["alias"]
    candidates = GURU_CANDIDATES[alias]
    guru_view = compute_guru_market_view(guru_name, market_metrics)

    st.markdown(f"## {guru_name}")
    st.markdown(f"**Strategy:** {meta['strategy']}")
    st.markdown(f"**Core Focus:** {meta['focus']}")
    st.markdown(f"**Style Interpretation:** {meta['style_interpretation']}")
    st.info(guru_interpretation(guru_name, market_metrics))

    st.markdown("### Market Cycle Interpretation")
    cycle_alignment = pd.DataFrame({
        "Item": ["Current Market Cycle", "Cycle Sub-View", "Guru Market Score", "Guru Stance"],
        "Value": [
            market_cycle["phase"],
            market_cycle["sub_view"],
            guru_view["score"],
            guru_view["stance"],
        ],
    })
    st.dataframe(cycle_alignment, use_container_width=True, hide_index=True)

    st.markdown("### Market Parameters")
    param_df = pd.DataFrame({
        "Parameter": list(guru_view["detail"].keys()),
        "Score": list(guru_view["detail"].values()),
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Guru Market Score", guru_view["score"])
    c2.metric("Guru Stance", guru_view["stance"])
    c3.metric("Market Cycle Phase", market_cycle["phase"])

    st.markdown("### Guru View on Current Market")
    st.warning(guru_market_viewpoint(guru_name, market_metrics))

    st.markdown("### What This Guru Likes Now")
    st.dataframe(pd.DataFrame({"Likes": meta["likes_now"]}), use_container_width=True, hide_index=True)

    st.markdown("### What This Guru Avoids Now")
    st.dataframe(pd.DataFrame({"Avoids": meta["avoids_now"]}), use_container_width=True, hide_index=True)

    st.markdown("### Current Risk Triggers")
    st.dataframe(pd.DataFrame({"Risk Trigger": meta["risk_triggers"]}), use_container_width=True, hide_index=True)

    st.markdown("### Key Indicators")
    indicator_cols = st.columns(len(meta["indicators"]))
    for i, ind in enumerate(meta["indicators"]):
        indicator_cols[i].metric(f"Indicator {i+1}", ind)

    st.markdown("### Suggested Positioning")
    portfolio_df = pd.DataFrame([{"Bucket": k, "Allocation": v} for k, v in meta["portfolio"].items()])
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

    st.markdown("### Suggested Action")
    st.success(suggested_action_by_guru(guru_name, market_metrics))

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

    st.markdown("### Visual Analysis")
    selected_ticker = st.selectbox(
        f"Select a ticker for {guru_name}",
        available_candidates,
        index=0,
        key=f"{guru_name}_ticker_select",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_price = make_price_chart(prices, selected_ticker, title=f"{selected_ticker} Price Trend", show_ma=True)
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

    fig_dd = make_drawdown_chart(prices, selected_ticker)
    st.plotly_chart(
        fig_dd,
        use_container_width=True,
        key=f"{guru_name}_{selected_ticker}_drawdown_chart",
    )

    st.markdown("### Fundamentals / Business Snapshot")
    render_fundamental_card(selected_ticker)

    st.markdown("### Group Comparison")
    top_compare = available_candidates[: min(5, len(available_candidates))]
    fig_compare = make_compare_chart(prices, top_compare, f"{guru_name} Candidate Basket Comparison")
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
    "Market Cycle",
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
        guru_view = compute_guru_market_view(guru_name, market_metrics)
        overview_rows.append({
            "Guru": guru_name,
            "Strategy": meta["strategy"],
            "Market Score": guru_view["score"],
            "Current Stance": guru_view["stance"],
            "Core Focus": meta["focus"],
            "Current Interpretation": guru_interpretation(guru_name, market_metrics),
            "Suggested Action": suggested_action_by_guru(guru_name, market_metrics),
        })

    overview_df = pd.DataFrame(overview_rows).sort_values("Market Score", ascending=False)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    st.markdown("### Cross-Asset Scoreboard")
    cross_rows = []
    for t in ["SPY", "QQQ", "TLT", "GLD", "DBC", "USO", "XLK", "XLV", "SOXX"]:
        m = compute_return_metrics(prices[t])
        cross_rows.append({
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
        })

    st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            make_compare_chart(prices, ["SPY", "QQQ", "TLT", "GLD", "USO"], "Cross-Asset Leadership"),
            use_container_width=True,
            key="overview_cross_asset_chart",
        )
    with c2:
        st.plotly_chart(
            make_compare_chart(prices, ["XLK", "XLV", "XLE", "XLF", "SOXX"], "Sector / Theme Leadership"),
            use_container_width=True,
            key="overview_sector_chart",
        )

    st.markdown("### Strategic Summary")
    st.write(
        f"""
- **Current regime:** {regime_label}  
- **Current market cycle:** {market_cycle['phase']}  
- **Cycle sub-view:** {market_cycle['sub_view']}  
- **Broad market trend:** SPY vs MA200 = {format_pct(market_metrics['SPY']['distance_200'])}  
- **Growth leadership:** QQQ vs MA200 = {format_pct(market_metrics['QQQ']['distance_200'])}  
- **AI / semiconductor momentum:** SOXX 6M = {format_pct(market_metrics['SOXX']['sixm_return'])}  
- **Macro hedge behavior:** GLD 6M = {format_pct(market_metrics['GLD']['sixm_return'])}  
- **Inflation / energy pressure:** USO 6M = {format_pct(market_metrics['USO']['sixm_return'])}
"""
    )

# ============================================================
# Market Cycle tab
# ============================================================
with tabs[1]:
    st.subheader("Market Cycle Analysis")

    cycle_df = pd.DataFrame({
        "Factor": list(market_cycle["detail"].keys()),
        "Score": list(market_cycle["detail"].values()),
    }).sort_values("Score", ascending=False)
    st.dataframe(cycle_df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Market Cycle Score", market_cycle["score"])
    c2.metric("Cycle Phase", market_cycle["phase"])
    c3.metric("Cycle Sub-View", market_cycle["sub_view"])

    st.info(market_cycle["phase_desc"])

    st.markdown("### Cycle Interpretation")
    cycle_text = {
        "Expansion / Bull": "Broad equity trend is strong, growth leadership is healthy, and risk appetite remains supportive.",
        "Mid Cycle / Constructive": "The market is still constructive, but macro and valuation signals deserve closer monitoring.",
        "Late Cycle / Selective": "Leadership is narrower, macro hedges are relevant, and selectivity becomes more important than broad beta exposure.",
        "Slowdown / Defensive": "Investors should increasingly prioritize quality, defense, and preserving optionality.",
        "Risk-Off / Contraction": "Capital preservation matters most; broad risk exposure should be treated carefully.",
    }
    st.warning(cycle_text.get(market_cycle["phase"], market_cycle["phase_desc"]))

    st.markdown("### Visual Market Cycle Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            make_compare_chart(prices, ["SPY", "QQQ", "SOXX"], "Risk / Growth Leadership"),
            use_container_width=True,
            key="market_cycle_risk_growth_chart",
        )
    with c2:
        st.plotly_chart(
            make_compare_chart(prices, ["TLT", "GLD", "USO", "DBC"], "Macro / Hedge Inputs"),
            use_container_width=True,
            key="market_cycle_macro_chart",
        )

    st.markdown("### Cycle-Aware Positioning")
    cycle_positioning = pd.DataFrame({
        "Cycle Phase": [
            "Expansion / Bull",
            "Mid Cycle / Constructive",
            "Late Cycle / Selective",
            "Slowdown / Defensive",
            "Risk-Off / Contraction",
        ],
        "Preferred Style": [
            "Growth + trend",
            "Balanced growth + quality",
            "Selective growth + quality + hedges",
            "Defensive + cash + hedges",
            "Capital preservation + hedge focus",
        ],
    })
    st.dataframe(cycle_positioning, use_container_width=True, hide_index=True)

# ============================================================
# Guru tabs
# ============================================================
with tabs[2]:
    render_guru_tab("Warren Buffett", prices, benchmark)

with tabs[3]:
    render_guru_tab("Ray Dalio", prices, benchmark)

with tabs[4]:
    render_guru_tab("Howard Marks", prices, benchmark)

with tabs[5]:
    render_guru_tab("Stanley Druckenmiller", prices, benchmark)

with tabs[6]:
    render_guru_tab("George Soros", prices, benchmark)

with tabs[7]:
    render_guru_tab("Peter Lynch", prices, benchmark)

with tabs[8]:
    render_guru_tab("Jim Simons", prices, benchmark)

with tabs[9]:
    render_guru_tab("Cathie Wood", prices, benchmark)

with tabs[10]:
    render_guru_tab("Michael Burry", prices, benchmark)

with tabs[11]:
    render_guru_tab("Paul Tudor Jones", prices, benchmark)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Data source: Yahoo Finance via yfinance. "
    "For research / educational use only. Not investment advice."
)

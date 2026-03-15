# streamlit_app.py
# ============================================================
# Nasdaq-100 Howard Marks Hybrid Quant Dashboard
# English-only version
#
# Features
# - Default universe: live Nasdaq-100 components from Wikipedia
# - Detailed macro risk analysis by item + total weighted risk
# - Quant screener: value + quality + risk + contrarian + momentum
# - Portfolio suggestion with macro-aware equity/cash posture
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy yfinance plotly requests lxml html5lib
# ============================================================

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# Page
# ============================================================
st.set_page_config(
    page_title="Nasdaq-100 Howard Marks Hybrid Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Nasdaq-100 Howard Marks Hybrid Dashboard")
st.caption("Live Nasdaq-100 universe + detailed macro risk + quant screener + portfolio builder")

# ============================================================
# Constants
# ============================================================
WIKI_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Fallback only if live scrape fails
FALLBACK_NDX = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO",
    "COST", "NFLX", "TSLA", "AMD", "QCOM", "ADBE", "PEP", "TMUS",
    "CSCO", "INTU", "AMGN", "TXN", "ISRG", "BKNG", "VRTX", "GILD",
    "ADI", "MU", "MDLZ", "CMCSA", "ADP", "KLAC", "LRCX", "PANW",
    "CRWD", "PLTR", "APP", "MSTR", "AMAT", "ASML", "CDNS", "SNPS",
]

FRED_SERIES = {
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "FEDFUNDS": "Fed Funds Rate",
    "BAMLH0A0HYM2": "US High Yield OAS",
    "VIXCLS": "VIX",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
}

MACRO_WEIGHTS = {
    "yield_curve": 0.20,
    "high_yield_spread": 0.20,
    "vix": 0.20,
    "fed_funds": 0.15,
    "labor_stress": 0.15,
    "inflation": 0.10,
}

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

price_period = st.sidebar.selectbox("Price history period", ["2y", "3y", "5y"], index=1)
top_n = st.sidebar.slider("Top N candidates", 5, 25, 12)
benchmark = st.sidebar.selectbox("Benchmark proxy", ["QQQ", "^NDX"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid factor weights")

w_value = st.sidebar.slider("Value", 0.0, 1.0, 0.25, 0.05)
w_quality = st.sidebar.slider("Quality", 0.0, 1.0, 0.20, 0.05)
w_risk = st.sidebar.slider("Risk Control", 0.0, 1.0, 0.20, 0.05)
w_contrarian = st.sidebar.slider("Contrarian", 0.0, 1.0, 0.15, 0.05)
w_momentum = st.sidebar.slider("Momentum", 0.0, 1.0, 0.15, 0.05)
w_macro = st.sidebar.slider("Macro", 0.0, 1.0, 0.05, 0.05)

weight_sum = w_value + w_quality + w_risk + w_contrarian + w_momentum + w_macro
if weight_sum <= 0:
    weight_sum = 1.0

FACTOR_WEIGHTS = {
    "value": w_value / weight_sum,
    "quality": w_quality / weight_sum,
    "risk": w_risk / weight_sum,
    "contrarian": w_contrarian / weight_sum,
    "momentum": w_momentum / weight_sum,
    "macro": w_macro / weight_sum,
}

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

max_allowed_vol = st.sidebar.slider("Max annualized volatility", 0.10, 1.00, 0.50, 0.01)
min_roe = st.sidebar.slider("Min ROE", -0.20, 0.50, 0.05, 0.01)
min_market_cap_b = st.sidebar.slider("Min market cap (USD bn)", 0, 1000, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Universe")
live_universe = st.sidebar.checkbox("Use live Nasdaq-100 components", value=True)
show_pass_only_default = st.sidebar.checkbox("Default to pass-filter results", value=True)

refresh = st.sidebar.button("Refresh data", type="primary")

# ============================================================
# Utility
# ============================================================
def format_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"

def format_num(x, digits=2):
    return f"{x:,.{digits}f}" if pd.notna(x) else "-"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def clip_series(s: pd.Series, low=None, high=None) -> pd.Series:
    out = s.copy()
    if low is not None:
        out = out.clip(lower=low)
    if high is not None:
        out = out.clip(upper=high)
    return out

def safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    r = series.rank(pct=True, ascending=ascending)
    return r.fillna(0.5)

def annualized_volatility(price_series: pd.Series) -> float:
    rets = price_series.pct_change().dropna()
    if len(rets) < 20:
        return np.nan
    return float(rets.std() * np.sqrt(252))

def max_drawdown(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.empty:
        return np.nan
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())

def current_drawdown_from_peak(price_series: pd.Series) -> float:
    s = price_series.dropna()
    if s.empty:
        return np.nan
    peak = s.max()
    return float(s.iloc[-1] / peak - 1.0)

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
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1.0)

def ma_gap(price_series: pd.Series, window: int) -> float:
    s = price_series.dropna()
    if len(s) < window + 1:
        return np.nan
    ma = s.rolling(window).mean().iloc[-1]
    if pd.isna(ma) or ma == 0:
        return np.nan
    return float(s.iloc[-1] / ma - 1.0)

# ============================================================
# Live Nasdaq-100 universe
# ============================================================
@st.cache_data(ttl=60 * 60)
def get_nasdaq100_from_wikipedia() -> pd.DataFrame:
    tables = pd.read_html(WIKI_NDX_URL)
    best = None

    for tbl in tables:
        cols = [str(c).strip() for c in tbl.columns]
        if "Ticker" in cols and ("Company" in cols or "Security" in cols):
            best = tbl.copy()
            break

    if best is None:
        raise ValueError("Could not find Nasdaq-100 component table.")

    best.columns = [str(c).strip() for c in best.columns]
    company_col = "Company" if "Company" in best.columns else "Security"

    out = best[["Ticker", company_col]].copy()
    out.columns = ["ticker", "company"]
    out["ticker"] = (
        out["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    out = out.drop_duplicates("ticker").reset_index(drop=True)
    return out

def get_default_universe(use_live: bool = True) -> pd.DataFrame:
    if use_live:
        try:
            return get_nasdaq100_from_wikipedia()
        except Exception:
            pass
    return pd.DataFrame({"ticker": FALLBACK_NDX, "company": FALLBACK_NDX})

# ============================================================
# FRED
# ============================================================
@st.cache_data(ttl=60 * 60)
def fetch_fred_series(series_id: str) -> pd.Series:
    url = fred_csv_url(series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df.set_index("DATE")[series_id].dropna()
    return s

@st.cache_data(ttl=60 * 60)
def fetch_all_fred() -> pd.DataFrame:
    data = {}
    for sid in FRED_SERIES:
        try:
            data[sid] = fetch_fred_series(sid)
        except Exception:
            data[sid] = pd.Series(dtype=float)
    return pd.concat(data, axis=1).sort_index().ffill()

# ============================================================
# yfinance
# ============================================================
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

    if data.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        close = data[["Close"]].copy()
        close.columns = tickers
        return close.dropna(how="all")

    close_map = {}
    for t in tickers:
        try:
            close_map[t] = data[t]["Close"]
        except Exception:
            continue

    if not close_map:
        return pd.DataFrame()

    return pd.DataFrame(close_map).dropna(how="all")

@st.cache_data(ttl=60 * 60)
def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        info = {}
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}

        row = {
            "ticker": t,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": safe_float(info.get("marketCap")),
            "trailingPE": safe_float(info.get("trailingPE")),
            "forwardPE": safe_float(info.get("forwardPE")),
            "priceToBook": safe_float(info.get("priceToBook")),
            "returnOnEquity": safe_float(info.get("returnOnEquity")),
            "debtToEquity": safe_float(info.get("debtToEquity")),
            "enterpriseToEbitda": safe_float(info.get("enterpriseToEbitda")),
            "freeCashflow": safe_float(info.get("freeCashflow")),
            "currentPrice": safe_float(info.get("currentPrice")),
            "sharesOutstanding": safe_float(info.get("sharesOutstanding")),
            "beta": safe_float(info.get("beta")),
        }

        if pd.notna(row["currentPrice"]) and pd.notna(row["sharesOutstanding"]) and pd.notna(row["freeCashflow"]):
            mcap = row["currentPrice"] * row["sharesOutstanding"]
            row["fcfYield"] = row["freeCashflow"] / mcap if mcap else np.nan
        else:
            row["fcfYield"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# Macro risk engine
# ============================================================
@dataclass
class MacroItem:
    name: str
    value: float
    score: float
    level: str
    weight: float
    contribution: float
    signal: str
    description: str

def risk_bucket(score: float) -> str:
    if score < 20:
        return "Low"
    if score < 40:
        return "Moderate"
    if score < 60:
        return "Elevated"
    if score < 80:
        return "High"
    return "Extreme"

def score_yield_curve(spread: float) -> Tuple[float, str]:
    if pd.isna(spread):
        return 50.0, "Unknown"
    if spread > 1.0:
        return 10.0, "Healthy steep curve"
    if spread > 0.5:
        return 25.0, "Normal but flattening"
    if spread > 0.0:
        return 45.0, "Late-cycle flattening"
    if spread > -0.5:
        return 75.0, "Inverted curve"
    return 95.0, "Deep inversion"

def score_high_yield(spread: float) -> Tuple[float, str]:
    if pd.isna(spread):
        return 50.0, "Unknown"
    if spread < 3.0:
        return 10.0, "Benign credit stress"
    if spread < 4.0:
        return 30.0, "Normal credit stress"
    if spread < 5.0:
        return 55.0, "Credit stress rising"
    if spread < 6.0:
        return 75.0, "High credit stress"
    return 95.0, "Severe credit stress"

def score_vix(vix: float) -> Tuple[float, str]:
    if pd.isna(vix):
        return 50.0, "Unknown"
    if vix < 16:
        return 10.0, "Calm volatility"
    if vix < 20:
        return 25.0, "Normal volatility"
    if vix < 25:
        return 50.0, "Elevated volatility"
    if vix < 35:
        return 75.0, "High volatility"
    return 95.0, "Stress volatility"

def score_fed_funds(rate: float) -> Tuple[float, str]:
    if pd.isna(rate):
        return 50.0, "Unknown"
    if rate < 2.0:
        return 10.0, "Easy policy"
    if rate < 3.0:
        return 25.0, "Mild restraint"
    if rate < 4.0:
        return 45.0, "Restrictive"
    if rate < 5.0:
        return 70.0, "Highly restrictive"
    return 90.0, "Very highly restrictive"

def score_labor_stress(sahm_gap: float) -> Tuple[float, str]:
    if pd.isna(sahm_gap):
        return 50.0, "Unknown"
    if sahm_gap < 0.30:
        return 10.0, "Stable labor market"
    if sahm_gap < 0.50:
        return 35.0, "Softening labor market"
    if sahm_gap < 0.70:
        return 60.0, "Recession signal area"
    if sahm_gap < 1.00:
        return 80.0, "Strong labor deterioration"
    return 95.0, "Severe labor deterioration"

def score_inflation(cpi_yoy: float) -> Tuple[float, str]:
    if pd.isna(cpi_yoy):
        return 50.0, "Unknown"
    if cpi_yoy < 2.5:
        return 15.0, "Contained inflation"
    if cpi_yoy < 3.0:
        return 30.0, "Mild inflation pressure"
    if cpi_yoy < 4.0:
        return 50.0, "Elevated inflation pressure"
    if cpi_yoy < 5.0:
        return 75.0, "High inflation pressure"
    return 95.0, "Severe inflation pressure"

def compute_macro_risk(fred_df: pd.DataFrame) -> Tuple[pd.DataFrame, float, str, Dict[str, float]]:
    df = fred_df.copy().ffill()

    latest_10y = df["DGS10"].dropna().iloc[-1] if not df["DGS10"].dropna().empty else np.nan
    latest_2y = df["DGS2"].dropna().iloc[-1] if not df["DGS2"].dropna().empty else np.nan
    latest_ff = df["FEDFUNDS"].dropna().iloc[-1] if not df["FEDFUNDS"].dropna().empty else np.nan
    latest_hy = df["BAMLH0A0HYM2"].dropna().iloc[-1] if not df["BAMLH0A0HYM2"].dropna().empty else np.nan
    latest_vix = df["VIXCLS"].dropna().iloc[-1] if not df["VIXCLS"].dropna().empty else np.nan
    latest_unrate = df["UNRATE"].dropna().iloc[-1] if not df["UNRATE"].dropna().empty else np.nan

    cpi = df["CPIAUCSL"].dropna()
    cpi_yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1.0) if len(cpi) >= 13 else np.nan

    unrate = df["UNRATE"].dropna()
    if len(unrate) >= 15:
        recent_3m_avg = unrate.iloc[-3:].mean()
        trailing_12m_low = unrate.iloc[-15:-3].min()
        sahm_gap = recent_3m_avg - trailing_12m_low
    else:
        sahm_gap = np.nan

    yield_curve = latest_10y - latest_2y if pd.notna(latest_10y) and pd.notna(latest_2y) else np.nan

    items = []

    yc_score, yc_signal = score_yield_curve(yield_curve)
    items.append(MacroItem(
        name="Yield Curve (10Y - 2Y)",
        value=yield_curve,
        score=yc_score,
        level=risk_bucket(yc_score),
        weight=MACRO_WEIGHTS["yield_curve"],
        contribution=yc_score * MACRO_WEIGHTS["yield_curve"],
        signal=yc_signal,
        description="Cycle / recession risk. A flatter or inverted curve usually means higher late-cycle risk."
    ))

    hy_score, hy_signal = score_high_yield(latest_hy)
    items.append(MacroItem(
        name="High Yield OAS",
        value=latest_hy,
        score=hy_score,
        level=risk_bucket(hy_score),
        weight=MACRO_WEIGHTS["high_yield_spread"],
        contribution=hy_score * MACRO_WEIGHTS["high_yield_spread"],
        signal=hy_signal,
        description="Credit risk. Wider spreads usually mean tighter financing conditions and greater default stress."
    ))

    vix_score, vix_signal = score_vix(latest_vix)
    items.append(MacroItem(
        name="VIX",
        value=latest_vix,
        score=vix_score,
        level=risk_bucket(vix_score),
        weight=MACRO_WEIGHTS["vix"],
        contribution=vix_score * MACRO_WEIGHTS["vix"],
        signal=vix_signal,
        description="Equity volatility risk. Higher VIX usually means more fragile risk appetite."
    ))

    ff_score, ff_signal = score_fed_funds(latest_ff)
    items.append(MacroItem(
        name="Fed Funds Rate",
        value=latest_ff,
        score=ff_score,
        level=risk_bucket(ff_score),
        weight=MACRO_WEIGHTS["fed_funds"],
        contribution=ff_score * MACRO_WEIGHTS["fed_funds"],
        signal=ff_signal,
        description="Liquidity / policy pressure. Higher short rates usually tighten valuation and financing conditions."
    ))

    labor_score, labor_signal = score_labor_stress(sahm_gap)
    items.append(MacroItem(
        name="Labor Stress (3M avg UNRATE - prior 12M low)",
        value=sahm_gap,
        score=labor_score,
        level=risk_bucket(labor_score),
        weight=MACRO_WEIGHTS["labor_stress"],
        contribution=labor_score * MACRO_WEIGHTS["labor_stress"],
        signal=labor_signal,
        description="Labor deterioration risk. A rising unemployment gap often confirms macro slowdown."
    ))

    infl_score, infl_signal = score_inflation(cpi_yoy)
    items.append(MacroItem(
        name="CPI YoY",
        value=cpi_yoy,
        score=infl_score,
        level=risk_bucket(infl_score),
        weight=MACRO_WEIGHTS["inflation"],
        contribution=infl_score * MACRO_WEIGHTS["inflation"],
        signal=infl_signal,
        description="Inflation pressure. Sticky inflation can keep policy restrictive for longer."
    ))

    risk_df = pd.DataFrame([{
        "Item": i.name,
        "Latest": i.value,
        "Signal": i.signal,
        "Risk Score (0-100)": i.score,
        "Risk Level": i.level,
        "Weight": i.weight,
        "Weighted Contribution": i.contribution,
        "Why It Matters": i.description,
    } for i in items])

    total_risk = float(risk_df["Weighted Contribution"].sum())

    if total_risk < 25:
        regime = "Low Risk / Risk-On"
    elif total_risk < 45:
        regime = "Moderate Risk"
    elif total_risk < 65:
        regime = "Elevated Risk"
    elif total_risk < 80:
        regime = "High Risk / Cautious"
    else:
        regime = "Extreme Risk / Defensive"

    details = {
        "yield_curve": yield_curve,
        "high_yield_oas": latest_hy,
        "vix": latest_vix,
        "fed_funds": latest_ff,
        "unemployment": latest_unrate,
        "labor_stress_gap": sahm_gap,
        "cpi_yoy": cpi_yoy,
    }

    return risk_df, total_risk, regime, details

# ============================================================
# Quant engine
# ============================================================
def compute_quant_table(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    total_macro_risk: float,
    benchmark_prices: pd.Series,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < 220:
            continue

        vol = annualized_volatility(s)
        mdd = max_drawdown(s)
        cdd = current_drawdown_from_peak(s)
        rsi14 = rsi(s, 14)
        ret_6m = total_return(s, 126)
        ret_12m = total_return(s, 252)
        gap_50 = ma_gap(s, 50)
        gap_200 = ma_gap(s, 200)

        rel_6m = np.nan
        rel_12m = np.nan
        if benchmark_prices is not None and not benchmark_prices.empty:
            b = benchmark_prices.reindex(s.index).dropna()
            common = pd.concat([s, b], axis=1).dropna()
            if len(common) > 252:
                stock = common.iloc[:, 0]
                bench = common.iloc[:, 1]
                rel_6m = (stock.iloc[-1] / stock.iloc[-127]) - (bench.iloc[-1] / bench.iloc[-127]) if len(common) > 127 else np.nan
                rel_12m = (stock.iloc[-1] / stock.iloc[-253]) - (bench.iloc[-1] / bench.iloc[-253]) if len(common) > 253 else np.nan

        rows.append({
            "ticker": t,
            "last_price": s.iloc[-1],
            "ann_vol": vol,
            "max_drawdown": mdd,
            "current_drawdown": cdd,
            "rsi14": rsi14,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "rel_6m": rel_6m,
            "rel_12m": rel_12m,
            "gap_50ma": gap_50,
            "gap_200ma": gap_200,
        })

    qdf = pd.DataFrame(rows)
    if qdf.empty:
        return qdf

    df = qdf.merge(fundamentals, on="ticker", how="left")

    df["trailingPE"] = clip_series(pd.to_numeric(df["trailingPE"], errors="coerce"), 0, 200)
    df["forwardPE"] = clip_series(pd.to_numeric(df["forwardPE"], errors="coerce"), 0, 200)
    df["priceToBook"] = clip_series(pd.to_numeric(df["priceToBook"], errors="coerce"), 0, 50)
    df["returnOnEquity"] = clip_series(pd.to_numeric(df["returnOnEquity"], errors="coerce"), -1, 2)
    df["debtToEquity"] = clip_series(pd.to_numeric(df["debtToEquity"], errors="coerce"), 0, 1000)
    df["enterpriseToEbitda"] = clip_series(pd.to_numeric(df["enterpriseToEbitda"], errors="coerce"), -50, 100)
    df["fcfYield"] = clip_series(pd.to_numeric(df["fcfYield"], errors="coerce"), -1, 1)
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")

    pe_combo = df[["trailingPE", "forwardPE"]].mean(axis=1, skipna=True)

    df["value_score"] = (
        0.30 * safe_rank(pe_combo, ascending=True) +
        0.20 * safe_rank(df["priceToBook"], ascending=True) +
        0.20 * safe_rank(df["enterpriseToEbitda"], ascending=True) +
        0.30 * safe_rank(df["fcfYield"], ascending=False)
    )

    df["quality_score"] = (
        0.55 * safe_rank(df["returnOnEquity"], ascending=False) +
        0.25 * safe_rank(df["debtToEquity"], ascending=True) +
        0.20 * safe_rank(df["marketCap"], ascending=False)
    )

    df["risk_score"] = (
        0.45 * safe_rank(df["ann_vol"], ascending=True) +
        0.35 * safe_rank(df["max_drawdown"], ascending=False) +
        0.20 * safe_rank(df["beta"], ascending=True)
    )

    drawdown_abs = -df["current_drawdown"]
    df["contrarian_score"] = (
        0.60 * safe_rank(drawdown_abs, ascending=False) +
        0.40 * safe_rank(df["rsi14"], ascending=True)
    )

    df["momentum_score"] = (
        0.20 * safe_rank(df["ret_6m"], ascending=False) +
        0.20 * safe_rank(df["ret_12m"], ascending=False) +
        0.15 * safe_rank(df["rel_6m"], ascending=False) +
        0.15 * safe_rank(df["rel_12m"], ascending=False) +
        0.15 * safe_rank(df["gap_50ma"], ascending=False) +
        0.15 * safe_rank(df["gap_200ma"], ascending=False)
    )

    macro_component = max(0.0, min(1.0, 1 - total_macro_risk / 100.0))
    df["macro_score_component"] = macro_component

    df["hybrid_score"] = (
        FACTOR_WEIGHTS["value"] * df["value_score"] +
        FACTOR_WEIGHTS["quality"] * df["quality_score"] +
        FACTOR_WEIGHTS["risk"] * df["risk_score"] +
        FACTOR_WEIGHTS["contrarian"] * df["contrarian_score"] +
        FACTOR_WEIGHTS["momentum"] * df["momentum_score"] +
        FACTOR_WEIGHTS["macro"] * df["macro_score_component"]
    )

    df["marketCapB"] = df["marketCap"] / 1e9
    df["pass_vol"] = df["ann_vol"] <= max_allowed_vol
    df["pass_roe"] = df["returnOnEquity"].isna() | (df["returnOnEquity"] >= min_roe)
    df["pass_mcap"] = df["marketCapB"].isna() | (df["marketCapB"] >= min_market_cap_b)
    df["pass_filters"] = df["pass_vol"] & df["pass_roe"] & df["pass_mcap"]

    df = df.sort_values(["pass_filters", "hybrid_score"], ascending=[False, False]).reset_index(drop=True)
    return df

def suggested_positioning(total_macro_risk: float) -> Dict[str, float]:
    if total_macro_risk < 25:
        return {"equity": 0.95, "cash": 0.05}
    if total_macro_risk < 45:
        return {"equity": 0.80, "cash": 0.20}
    if total_macro_risk < 65:
        return {"equity": 0.65, "cash": 0.35}
    if total_macro_risk < 80:
        return {"equity": 0.50, "cash": 0.50}
    return {"equity": 0.35, "cash": 0.65}

def build_portfolio(df: pd.DataFrame, total_macro_risk: float, top_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    eligible = df[df["pass_filters"]].copy()
    if eligible.empty:
        eligible = df.copy()

    port = eligible.head(top_n).copy()
    scores = port["hybrid_score"].clip(lower=0.0001)
    raw = scores / scores.sum()

    pos = suggested_positioning(total_macro_risk)
    port["target_weight"] = raw * pos["equity"]
    port["cash_buffer"] = pos["cash"]
    return port

def interpretation(row: pd.Series, total_macro_risk: float) -> str:
    tags = []

    if pd.notna(row.get("returnOnEquity")) and row["returnOnEquity"] > 0.15:
        tags.append("strong ROE")
    if pd.notna(row.get("current_drawdown")) and row["current_drawdown"] < -0.25:
        tags.append("deep pullback")
    if pd.notna(row.get("rsi14")) and row["rsi14"] < 40:
        tags.append("oversold")
    if pd.notna(row.get("gap_200ma")) and row["gap_200ma"] > 0:
        tags.append("above 200DMA")
    if pd.notna(row.get("ann_vol")) and row["ann_vol"] < 0.30:
        tags.append("controlled volatility")
    if pd.notna(row.get("trailingPE")) and row["trailingPE"] < 25:
        tags.append("reasonable PE")
    if pd.notna(row.get("rel_12m")) and row["rel_12m"] > 0:
        tags.append("benchmark outperformance")

    if not tags:
        tags = ["mixed profile"]

    macro_text = (
        "macro backdrop supportive" if total_macro_risk < 45
        else "macro backdrop selective" if total_macro_risk < 65
        else "macro backdrop defensive"
    )
    return ", ".join(tags) + f"; {macro_text}."

# ============================================================
# Charts
# ============================================================
def make_line_chart(series_dict: Dict[str, pd.Series], title: str, ytitle: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    for name, s in series_dict.items():
        s = s.dropna()
        if not s.empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=ytitle, height=height)
    return fig

def make_relative_price_chart(prices: pd.DataFrame, tickers: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        if s.empty:
            continue
        base = s.iloc[0]
        norm = s / base * 100
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode="lines", name=t))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Indexed Price (Start=100)", height=500)
    return fig

def make_mva_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    s = prices[ticker].dropna()
    ma50 = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Price"))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, mode="lines", name="MA200"))
    fig.update_layout(title=f"{ticker} Price / MA50 / MA200", xaxis_title="Date", yaxis_title="Price", height=420)
    return fig

def make_drawdown_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    s = prices[ticker].dropna()
    peak = s.cummax()
    dd = s / peak - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(title=f"{ticker} Drawdown vs Previous Peak", xaxis_title="Date", yaxis_title="Drawdown", height=420)
    return fig

def make_risk_bar(risk_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=risk_df["Item"],
            y=risk_df["Risk Score (0-100)"],
            text=[f"{x:.0f}" for x in risk_df["Risk Score (0-100)"]],
            textposition="auto",
            name="Risk Score",
        )
    )
    fig.update_layout(title="Macro Risk by Item", xaxis_title="", yaxis_title="Risk Score", height=450)
    return fig

# ============================================================
# Load data
# ============================================================
if refresh:
    st.cache_data.clear()

with st.spinner("Loading universe, macro data, prices, and fundamentals..."):
    universe_df = get_default_universe(use_live=live_universe)
    tickers = universe_df["ticker"].dropna().astype(str).str.upper().tolist()

    fred_df = fetch_all_fred()
    macro_risk_df, total_macro_risk, macro_regime, macro_details = compute_macro_risk(fred_df)

    all_price_tickers = tickers + [benchmark]
    prices_all = fetch_prices(all_price_tickers, price_period)

    if benchmark in prices_all.columns:
        benchmark_prices = prices_all[benchmark].dropna()
    else:
        benchmark_prices = pd.Series(dtype=float)

    prices = prices_all[[c for c in prices_all.columns if c in tickers]].copy()
    fundamentals = fetch_fundamentals(tickers)

    quant_df = compute_quant_table(prices, fundamentals, total_macro_risk, benchmark_prices)
    portfolio_df = build_portfolio(quant_df, total_macro_risk, top_n)

# ============================================================
# Summary
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Universe Size", f"{len(tickers)}")
c2.metric("Macro Regime", macro_regime)
c3.metric("Total Macro Risk", f"{total_macro_risk:.1f} / 100")
c4.metric("10Y - 2Y", format_num(macro_details["yield_curve"], 2))
c5.metric("VIX", format_num(macro_details["vix"], 2))

pos = suggested_positioning(total_macro_risk)
st.info(
    f"Suggested top-down posture: Equity {pos['equity']:.0%} / Cash {pos['cash']:.0%}. "
    f"This is driven by the weighted Total Macro Risk score."
)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Macro Risk",
    "Quant Screener",
    "Portfolio",
    "Charts",
    "Raw Data",
])

# ============================================================
# Tab 1: Macro Risk
# ============================================================
with tab1:
    st.subheader("Detailed Macro Risk Analysis")

    macro_show = macro_risk_df.copy()
    macro_show["Latest"] = macro_show.apply(
        lambda r: format_pct(r["Latest"]) if "CPI YoY" in r["Item"] else format_num(r["Latest"], 2),
        axis=1,
    )
    macro_show["Weight"] = macro_show["Weight"].map(lambda x: f"{x:.0%}")
    macro_show["Weighted Contribution"] = macro_show["Weighted Contribution"].map(lambda x: f"{x:.1f}")

    st.dataframe(macro_show, use_container_width=True, hide_index=True)

    st.plotly_chart(make_risk_bar(macro_risk_df), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        yc_series = fred_df["DGS10"] - fred_df["DGS2"]
        st.plotly_chart(
            make_line_chart({"10Y - 2Y": yc_series}, "Yield Curve Spread", "Spread"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_line_chart({"High Yield OAS": fred_df["BAMLH0A0HYM2"]}, "US High Yield OAS", "Spread"),
            use_container_width=True,
        )

    with col_b:
        st.plotly_chart(
            make_line_chart({"VIX": fred_df["VIXCLS"]}, "VIX", "Index"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_line_chart({"Fed Funds": fred_df["FEDFUNDS"]}, "Fed Funds Rate", "Rate"),
            use_container_width=True,
        )

    st.markdown("### Total Risk Interpretation")
    if total_macro_risk < 25:
        st.success("Total Macro Risk is low. The backdrop is broadly supportive for risk assets.")
    elif total_macro_risk < 45:
        st.info("Total Macro Risk is moderate. Risk assets can work, but selectivity still matters.")
    elif total_macro_risk < 65:
        st.warning("Total Macro Risk is elevated. Favor higher-quality names and keep a cash buffer.")
    elif total_macro_risk < 80:
        st.warning("Total Macro Risk is high. Defense, cash discipline, and tighter filters are appropriate.")
    else:
        st.error("Total Macro Risk is extreme. Capital preservation matters more than chasing upside.")

# ============================================================
# Tab 2: Quant Screener
# ============================================================
with tab2:
    st.subheader("Nasdaq-100 Hybrid Quant Screener")

    if quant_df.empty:
        st.warning("No screening results available.")
    else:
        screen_df = quant_df.copy()

        sector_options = ["All"] + sorted([x for x in screen_df["sector"].dropna().unique()])
        selected_sector = st.selectbox("Sector filter", sector_options, index=0)
        pass_only = st.checkbox("Show pass-filter results only", value=show_pass_only_default)

        if selected_sector != "All":
            screen_df = screen_df[screen_df["sector"] == selected_sector]

        if pass_only:
            filtered = screen_df[screen_df["pass_filters"]]
            if not filtered.empty:
                screen_df = filtered

        show_cols = [
            "ticker", "name", "sector", "industry", "hybrid_score",
            "value_score", "quality_score", "risk_score", "contrarian_score", "momentum_score",
            "last_price", "trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "fcfYield",
            "ann_vol", "max_drawdown", "current_drawdown", "rsi14",
            "ret_6m", "ret_12m", "rel_6m", "rel_12m", "gap_50ma", "gap_200ma",
            "marketCapB", "pass_filters",
        ]
        show_cols = [c for c in show_cols if c in screen_df.columns]

        st.dataframe(
            screen_df[show_cols].sort_values("hybrid_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Top Candidates")
        for _, row in screen_df.sort_values("hybrid_score", ascending=False).head(10).iterrows():
            st.markdown(
                f"**{row['ticker']} — {row.get('name', '') or ''}**  \n"
                f"Hybrid Score: {row['hybrid_score']:.3f} | Sector: {row.get('sector', '-') or '-'} | "
                f"ROE: {format_pct(row.get('returnOnEquity'))} | "
                f"Current Drawdown: {format_pct(row.get('current_drawdown'))} | "
                f"12M Return: {format_pct(row.get('ret_12m'))} | "
                f"12M Relative vs Benchmark: {format_pct(row.get('rel_12m'))}  \n"
                f"{interpretation(row, total_macro_risk)}"
            )

# ============================================================
# Tab 3: Portfolio
# ============================================================
with tab3:
    st.subheader("Suggested Portfolio")

    if portfolio_df.empty:
        st.warning("No portfolio candidates available.")
    else:
        show = portfolio_df[[
            "ticker", "name", "sector", "hybrid_score",
            "target_weight", "value_score", "quality_score",
            "risk_score", "contrarian_score", "momentum_score",
            "current_drawdown", "ret_6m", "ret_12m", "rel_12m"
        ]].copy()

        st.dataframe(show, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=portfolio_df["ticker"],
            y=portfolio_df["target_weight"],
            text=[f"{x:.1%}" for x in portfolio_df["target_weight"]],
            textposition="auto",
            name="Target Weight"
        ))
        fig.update_layout(title="Target Portfolio Weights", xaxis_title="Ticker", yaxis_title="Weight", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Portfolio Logic")
        st.write(
            f"- Universe: **Nasdaq-100**\n"
            f"- Macro regime: **{macro_regime}**\n"
            f"- Total Macro Risk: **{total_macro_risk:.1f}/100**\n"
            f"- Suggested equity allocation: **{pos['equity']:.0%}**\n"
            f"- Suggested cash allocation: **{pos['cash']:.0%}**\n"
            f"- Selection rule: **Value + Quality + Risk Control + Contrarian + Momentum**"
        )

# ============================================================
# Tab 4: Charts
# ============================================================
with tab4:
    st.subheader("Charts")

    if quant_df.empty or prices.empty:
        st.warning("No chart data available.")
    else:
        top_choices = quant_df["ticker"].head(20).tolist()
        selected = st.selectbox("Select ticker", top_choices, index=0)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_mva_chart(prices, selected), use_container_width=True)
        with c2:
            st.plotly_chart(make_drawdown_chart(prices, selected), use_container_width=True)

        st.markdown("### Relative Performance of Top Portfolio Names")
        st.plotly_chart(
            make_relative_price_chart(prices, portfolio_df["ticker"].tolist() if not portfolio_df.empty else top_choices[:10], "Relative Price Performance"),
            use_container_width=True,
        )

# ============================================================
# Tab 5: Raw Data
# ============================================================
with tab5:
    st.subheader("Raw Data")

    with st.expander("Live Nasdaq-100 universe"):
        st.dataframe(universe_df, use_container_width=True, hide_index=True)

    with st.expander("FRED macro history"):
        st.dataframe(fred_df.tail(60), use_container_width=True)

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
- This is a research dashboard, not investment advice.
- Default universe is the live Nasdaq-100 component table when available.
- The macro model is explicitly itemized and weighted:
  Yield Curve 20%, High Yield OAS 20%, VIX 20%, Fed Funds 15%, Labor Stress 15%, CPI YoY 10%.
- Total Macro Risk is the weighted sum of these item scores on a 0-100 scale.
"""
)

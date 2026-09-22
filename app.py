import datetime as dt
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Nifty 50 Multi-Factor Screener",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Nifty 50 Multi-Factor Equity Screener")
st.caption("Value • Momentum • Quality • Low Volatility | Market data via Yahoo Finance")

@dataclass
class ScreenerConfig:
    tickers: list = field(default_factory=lambda: [
        "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "JIOFIN.NS", "SHRIRAMFIN.NS",
        "HDFCLIFE.NS", "SBILIFE.NS",
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "BHARTIARTL.NS",
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TATACONSUM.NS",
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS",
        "MARUTI.NS", "M&M.NS", "TMPV.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS",
        "NTPC.NS", "POWERGRID.NS",
        "ASIANPAINT.NS", "TITAN.NS",
        "ADANIPORTS.NS", "INDIGO.NS",
        "ETERNAL.NS", "TRENT.NS",
        "BEL.NS",
    ])
    lookback_days: int = 400
    momentum_window_short: int = 63
    momentum_window_long: int = 252
    volatility_window: int = 126
    factor_weights: dict = field(default_factory=lambda: {
        "value": 0.30,
        "momentum": 0.25,
        "quality": 0.30,
        "low_volatility": 0.15,
    })
    request_pause_sec: float = 0.15
    max_retries: int = 3

CONFIG = ScreenerConfig()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(tickers, lookback_days, max_retries, pause_sec):
    end = dt.datetime.today()
    start = end - dt.timedelta(days=int(lookback_days * 1.6))
    try:
        raw = yf.download(
            tickers=list(tickers), start=start, end=end, auto_adjust=True,
            progress=False, group_by="ticker", threads=True,
        )
    except Exception:
        raw = None

    close_frames = {}
    if raw is not None and not raw.empty:
        for tkr in tickers:
            try:
                series = raw[tkr]["Close"].dropna()
                if len(series) > 20:
                    close_frames[tkr] = series
            except (KeyError, TypeError):
                pass

    missing = [t for t in tickers if t not in close_frames]
    for tkr in missing:
        for attempt in range(max_retries):
            try:
                series = yf.Ticker(tkr).history(start=start, end=end, auto_adjust=True)["Close"].dropna()
                if len(series) > 20:
                    close_frames[tkr] = series
                break
            except Exception:
                time.sleep(pause_sec * (attempt + 1))

    if not close_frames:
        return pd.DataFrame()
    price_df = pd.DataFrame(close_frames).sort_index()
    return price_df.dropna(axis=1, thresh=max(1, int(len(price_df) * 0.8)))

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals(tickers, max_retries, pause_sec):
    records = []
    for tkr in tickers:
        info = {}
        for attempt in range(max_retries):
            try:
                info = yf.Ticker(tkr).get_info()
                break
            except Exception:
                time.sleep(pause_sec * (attempt + 1))
        records.append({
            "ticker": tkr,
            "shortName": info.get("shortName", tkr),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "marketCap": info.get("marketCap", np.nan),
            "trailingPE": info.get("trailingPE", np.nan),
            "priceToBook": info.get("priceToBook", np.nan),
            "enterpriseToEbitda": info.get("enterpriseToEbitda", np.nan),
            "returnOnEquity": info.get("returnOnEquity", np.nan),
            "profitMargins": info.get("profitMargins", np.nan),
            "debtToEquity": info.get("debtToEquity", np.nan),
            "revenueGrowth": info.get("revenueGrowth", np.nan),
            "earningsGrowth": info.get("earningsGrowth", np.nan),
        })
        time.sleep(pause_sec)
    return pd.DataFrame(records).set_index("ticker")

def compute_momentum_and_volatility(prices, cfg):
    returns = prices.pct_change()
    if len(prices) < cfg.momentum_window_long:
        raise ValueError(f"Only {len(prices)} trading days were returned; at least {cfg.momentum_window_long} are needed.")
    mom_short = prices.iloc[-1] / prices.iloc[-cfg.momentum_window_short] - 1
    mom_long = prices.iloc[-1] / prices.iloc[-cfg.momentum_window_long] - 1
    blended_momentum = 0.4 * mom_short + 0.6 * mom_long
    realized_vol = returns.tail(cfg.volatility_window).std() * np.sqrt(252)
    return pd.DataFrame({
        "mom_3m": mom_short,
        "mom_12m": mom_long,
        "momentum_raw": blended_momentum,
        "realized_vol_ann": realized_vol,
    })

def build_factor_table(prices, fundamentals, cfg):
    price_factors = compute_momentum_and_volatility(prices, cfg)
    factors = fundamentals.join(price_factors, how="inner")
    for col, new_col in [
        ("trailingPE", "value_pe_inv"),
        ("priceToBook", "value_pb_inv"),
        ("enterpriseToEbitda", "value_evebitda_inv"),
    ]:
        safe = factors[col].where(factors[col] > 0)
        factors[new_col] = 1.0 / safe
    factors["quality_roe"] = factors["returnOnEquity"]
    factors["quality_margin"] = factors["profitMargins"]
    safe_dte = factors["debtToEquity"].where(factors["debtToEquity"] >= 0)
    factors["quality_leverage_inv"] = 1.0 / (1.0 + safe_dte / 100.0)
    factors["momentum_score_raw"] = factors["momentum_raw"]
    safe_vol = factors["realized_vol_ann"].where(factors["realized_vol_ann"] > 0)
    factors["low_vol_score_raw"] = 1.0 / safe_vol
    return factors

def zscore(series):
    valid = series.dropna()
    if valid.std(ddof=0) == 0 or len(valid) < 2:
        return pd.Series(0.0, index=series.index)
    return ((series - valid.mean()) / valid.std(ddof=0)).fillna(0.0)

def build_composite_score(factors, cfg):
    scores = pd.DataFrame(index=factors.index)
    scores["value_z"] = factors[["value_pe_inv", "value_pb_inv", "value_evebitda_inv"]].apply(zscore).mean(axis=1)
    scores["quality_z"] = factors[["quality_roe", "quality_margin", "quality_leverage_inv"]].apply(zscore).mean(axis=1)
    scores["momentum_z"] = zscore(factors["momentum_score_raw"])
    scores["low_vol_z"] = zscore(factors["low_vol_score_raw"])
    scores["composite_score"] = (
        cfg.factor_weights["value"] * scores["value_z"]
        + cfg.factor_weights["momentum"] * scores["momentum_z"]
        + cfg.factor_weights["quality"] * scores["quality_z"]
        + cfg.factor_weights["low_volatility"] * scores["low_vol_z"]
    )
    scores["sector"] = factors["sector"]
    scores["shortName"] = factors["shortName"]
    scores = scores.sort_values("composite_score", ascending=False)
    scores["rank"] = np.arange(1, len(scores) + 1)
    return scores

def run_screener():
    prices = fetch_price_history(CONFIG.tickers, CONFIG.lookback_days, CONFIG.max_retries, CONFIG.request_pause_sec)
    if prices.empty:
        raise RuntimeError("No price data could be retrieved from Yahoo Finance.")
    fundamentals = fetch_fundamentals(tuple(prices.columns), CONFIG.max_retries, CONFIG.request_pause_sec)
    factors = build_factor_table(prices, fundamentals, CONFIG)
    scored = build_composite_score(factors, CONFIG)
    return prices, fundamentals, factors, scored

with st.sidebar:
    st.header("Screener Controls")
    st.write("**Factor weights**")
    w_value = st.slider("Value", 0.0, 1.0, 0.30, 0.05)
    w_momentum = st.slider("Momentum", 0.0, 1.0, 0.25, 0.05)
    w_quality = st.slider("Quality", 0.0, 1.0, 0.30, 0.05)
    w_lowvol = st.slider("Low Volatility", 0.0, 1.0, 0.15, 0.05)
    total = w_value + w_momentum + w_quality + w_lowvol
    st.caption(f"Weight total: {total:.2f}")
    st.info("Changing weights re-ranks the same fetched dataset. Click Refresh Data to retrieve a new market snapshot.")
    refresh = st.button("🔄 Refresh market data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

CONFIG.factor_weights = {
    "value": w_value / total if total else 0.25,
    "momentum": w_momentum / total if total else 0.25,
    "quality": w_quality / total if total else 0.25,
    "low_volatility": w_lowvol / total if total else 0.25,
}

try:
    with st.spinner("Fetching Nifty 50 prices and fundamentals…"):
        prices, fundamentals, factors, scored = run_screener()
except Exception as exc:
    st.error(f"Unable to build the screener: {exc}")
    st.stop()

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks screened", len(scored))
c2.metric("Trading days", len(prices))
c3.metric("Top ranked", scored.iloc[0]["shortName"] if len(scored) else "—")
c4.metric("Data timestamp", dt.datetime.now().strftime("%d %b %Y, %H:%M"))

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📈 Factor Analysis", "🏭 Sector View", "ℹ️ Methodology"])

with tab1:
    st.subheader("Multi-Factor Rankings")
    search = st.text_input("Search company or ticker", placeholder="e.g. Reliance, TCS, INFY")
    display = scored[["rank", "shortName", "sector", "value_z", "momentum_z", "quality_z", "low_vol_z", "composite_score"]].copy()
    display.index.name = "Ticker"
    if search:
        mask = display["shortName"].str.contains(search, case=False, na=False) | display.index.str.contains(search, case=False, na=False)
        display = display[mask]
    st.dataframe(
        display.style.format({c: "{:.2f}" for c in ["value_z", "momentum_z", "quality_z", "low_vol_z", "composite_score"]}),
        use_container_width=True,
        height=600,
    )
    csv = display.to_csv().encode("utf-8")
    st.download_button("⬇️ Download ranking CSV", csv, "nifty50_multi_factor_screener_results.csv", "text/csv")

    top_n = min(15, len(scored))
    bottom_n = min(5, len(scored))
    chart_df = pd.concat([scored.head(top_n), scored.tail(bottom_n)]).drop_duplicates().sort_values("composite_score")
    fig = px.bar(chart_df, x="composite_score", y="shortName", orientation="h", color="composite_score",
                 color_continuous_scale="RdYlGn", title=f"Top {top_n} & Bottom {bottom_n} by Composite Score")
    fig.update_layout(height=650, yaxis_title="", xaxis_title="Composite Z-Score")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Factor Relationships")
    left, right = st.columns(2)
    corr = scored[["value_z", "momentum_z", "quality_z", "low_vol_z"]].corr()
    with left:
        heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                    colorscale="RdBu", text=np.round(corr.values, 2), texttemplate="%{text}"))
        heat.update_layout(title="Factor Correlation Matrix", height=500)
        st.plotly_chart(heat, use_container_width=True)
    with right:
        scatter = scored.join(factors[["marketCap", "mom_3m", "mom_12m", "realized_vol_ann"]])
        scatter["marketCap_display"] = scatter["marketCap"].fillna(scatter["marketCap"].median()).clip(lower=1)
        fig = px.scatter(scatter.reset_index(), x="momentum_z", y="quality_z", size="marketCap_display",
                         color="sector", hover_name="shortName",
                         hover_data={"index": True, "composite_score": ":.2f", "marketCap_display": False},
                         title="Momentum vs Quality")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Factor Data")
    factor_view = factors[["shortName", "sector", "trailingPE", "priceToBook", "enterpriseToEbitda",
                           "returnOnEquity", "profitMargins", "debtToEquity", "mom_3m", "mom_12m", "realized_vol_ann"]].copy()
    st.dataframe(factor_view.round(3), use_container_width=True, height=500)

with tab3:
    st.subheader("Average Composite Score by Sector")
    sector_avg = scored.groupby("sector")["composite_score"].mean().sort_values().reset_index()
    fig = px.bar(sector_avg, x="composite_score", y="sector", orientation="h", color="composite_score",
                 color_continuous_scale="RdYlGn", title="Sector-Level Average Composite Score")
    fig.update_layout(height=600, yaxis_title="", xaxis_title="Average Composite Z-Score")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sector_avg.round(3), use_container_width=True)

with tab4:
    st.subheader("How the screener works")
    st.markdown("""
    **Universe:** Nifty 50 NSE tickers using Yahoo Finance `.NS` symbols.

    **Value:** inverse of trailing P/E, price-to-book and EV/EBITDA; lower positive multiples receive higher raw value scores.

    **Momentum:** 40% three-month return + 60% twelve-month return.

    **Quality:** ROE, profit margin and an inverse debt-to-equity measure.

    **Low Volatility:** inverse of annualized realized volatility over approximately six months.

    **Scoring:** each sub-factor is cross-sectionally standardized with a z-score. Sub-factor scores are averaged within each factor family, then blended using the sidebar weights. Missing factor values receive a neutral z-score of zero after peer scoring.

    **Data:** prices and fundamental snapshots are retrieved from Yahoo Finance through `yfinance`. Data availability can vary by ticker and time.
    """)
    st.warning("This dashboard is a research/screening tool, not personalized investment advice. Market data may be delayed or incomplete.")
    st.caption(f"Last successful calculation: {dt.datetime.now().strftime('%d %b %Y %H:%M:%S')}")

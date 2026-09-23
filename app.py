"""
Comparable Company Analysis (Comps) Tool
==========================================
An interactive Streamlit app that derives an implied valuation range for any
target company by benchmarking it against a user-defined peer group's trading
multiples (EV/Revenue, EV/EBITDA, P/E, P/B, PEG), pulled live from Yahoo Finance.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (Streamlit Community Cloud)
"""

import time
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# ======================================================================
# Page config & light styling
# ======================================================================
st.set_page_config(
    page_title="Comps Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background-color: #f7f7f5; padding: 12px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .caption-text { color: #6b6b6b; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "\u20b9", "EUR": "\u20ac", "GBP": "\u00a3",
    "JPY": "\u00a5", "CNY": "\u00a5", "AUD": "A$", "CAD": "C$",
    "HKD": "HK$", "SGD": "S$", "CHF": "CHF ",
}

MULTIPLE_COLS = ["EV_Revenue", "EV_EBITDA", "PE", "PB", "PEG"]
METHODOLOGY_LABELS = {
    "EV_Revenue": "EV / Revenue",
    "EV_EBITDA": "EV / EBITDA",
    "PE": "P / E",
    "PB": "P / B",
}

MAX_RETRIES = 3
RETRY_PAUSE_SEC = 0.4


# ======================================================================
# Data acquisition (cached — avoids re-hitting Yahoo on every rerun)
# ======================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_snapshot(ticker: str) -> dict:
    """Pull one fundamental snapshot for a ticker. Cached for 15 minutes per
    ticker so repeated app interactions don't re-trigger Yahoo Finance calls
    and risk rate-limiting."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {}

    info = {}
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).get_info()
            if info and info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                info = {}
            break
        except Exception:
            time.sleep(RETRY_PAUSE_SEC * (attempt + 1))

    if not info:
        return {}

    return {
        "ticker": ticker,
        "shortName": info.get("shortName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "currency": info.get("currency", "USD"),
        "currentPrice": info.get("currentPrice", info.get("regularMarketPrice", np.nan)),
        "marketCap": info.get("marketCap", np.nan),
        "enterpriseValue": info.get("enterpriseValue", np.nan),
        "totalDebt": info.get("totalDebt", np.nan),
        "totalCash": info.get("totalCash", np.nan),
        "totalRevenue": info.get("totalRevenue", np.nan),
        "ebitda": info.get("ebitda", np.nan),
        "netIncomeToCommon": info.get("netIncomeToCommon", np.nan),
        "trailingEps": info.get("trailingEps", np.nan),
        "bookValue": info.get("bookValue", np.nan),
        "sharesOutstanding": info.get("sharesOutstanding", np.nan),
        "trailingPE": info.get("trailingPE", np.nan),
        "priceToBook": info.get("priceToBook", np.nan),
        "pegRatio": info.get("trailingPegRatio", info.get("pegRatio", np.nan)),
        "earningsGrowth": info.get("earningsGrowth", np.nan),
        "revenueGrowth": info.get("revenueGrowth", np.nan),
    }


def fetch_universe(tickers: list) -> tuple:
    """Fetch every ticker, separating successes from failures so the caller
    can surface exactly which tickers to fix/remove rather than crashing."""
    records, failed = [], []
    progress = st.progress(0.0, text="Fetching company data...")
    for i, tkr in enumerate(tickers):
        rec = fetch_company_snapshot(tkr)
        if rec:
            records.append(rec)
        else:
            failed.append(tkr)
        progress.progress((i + 1) / len(tickers), text=f"Fetched {tkr}...")
    progress.empty()

    df = pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()
    return df, failed


# ======================================================================
# Multiples, peer stats, implied valuation
# ======================================================================
def compute_multiples(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()

    manual_ev = m["marketCap"] + m["totalDebt"].fillna(0) - m["totalCash"].fillna(0)
    m["EV"] = m["enterpriseValue"].where(m["enterpriseValue"].notna(), manual_ev)

    safe_revenue = m["totalRevenue"].where(m["totalRevenue"] > 0)
    safe_ebitda = m["ebitda"].where(m["ebitda"] > 0)
    safe_eps = m["trailingEps"].where(m["trailingEps"] > 0)
    safe_bvps = m["bookValue"].where(m["bookValue"] > 0)

    m["EV_Revenue"] = m["EV"] / safe_revenue
    m["EV_EBITDA"] = m["EV"] / safe_ebitda
    m["PE"] = m["trailingPE"].where(m["trailingPE"] > 0, m["currentPrice"] / safe_eps)
    m["PB"] = m["priceToBook"].where(m["priceToBook"] > 0, m["currentPrice"] / safe_bvps)

    safe_growth_pct = (m["earningsGrowth"] * 100).where(m["earningsGrowth"] > 0)
    manual_peg = m["PE"] / safe_growth_pct
    m["PEG"] = m["pegRatio"].where(m["pegRatio"] > 0, manual_peg)

    m["NetMargin"] = m["netIncomeToCommon"] / safe_revenue
    return m


def compute_peer_stats(peer_multiples: pd.DataFrame) -> pd.DataFrame:
    stats = peer_multiples[MULTIPLE_COLS].agg(
        ["mean", "median", "min", "max", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
    )
    stats.index = ["mean", "median", "min", "max", "p25", "p75"]
    return stats


def compute_implied_valuation(target: pd.Series, peer_stats: pd.DataFrame) -> pd.DataFrame:
    def implied_price_from_ev_multiple(multiple_value, driver_value):
        if pd.isna(multiple_value) or pd.isna(driver_value) or pd.isna(target["sharesOutstanding"]):
            return np.nan
        implied_ev = multiple_value * driver_value
        implied_equity_value = implied_ev - target.get("totalDebt", 0) + target.get("totalCash", 0)
        return implied_equity_value / target["sharesOutstanding"]

    def implied_price_from_equity_multiple(multiple_value, per_share_driver):
        if pd.isna(multiple_value) or pd.isna(per_share_driver):
            return np.nan
        return multiple_value * per_share_driver

    rows = []
    for stat in ["p25", "median", "p75"]:
        rows.append({
            "stat": stat,
            "EV_Revenue": implied_price_from_ev_multiple(peer_stats.loc[stat, "EV_Revenue"], target["totalRevenue"]),
            "EV_EBITDA": implied_price_from_ev_multiple(peer_stats.loc[stat, "EV_EBITDA"], target["ebitda"]),
            "PE": implied_price_from_equity_multiple(peer_stats.loc[stat, "PE"], target["trailingEps"]),
            "PB": implied_price_from_equity_multiple(peer_stats.loc[stat, "PB"], target["bookValue"]),
        })
    return pd.DataFrame(rows).set_index("stat")


# ======================================================================
# Charts
# ======================================================================
def football_field_chart(implied_valuation: pd.DataFrame, target: pd.Series, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, (col, label) in enumerate(METHODOLOGY_LABELS.items()):
        low = implied_valuation.loc["p25", col]
        high = implied_valuation.loc["p75", col]
        mid = implied_valuation.loc["median", col]
        if pd.isna(low) or pd.isna(high):
            continue
        lo, hi = sorted([low, high])
        fig.add_trace(go.Bar(
            x=[hi - lo], y=[label], base=[lo], orientation="h",
            marker_color=colors[i % len(colors)], opacity=0.78,
            hovertemplate=f"{label}<br>Low: {currency_symbol}%{{base:.2f}}<br>High: {currency_symbol}%{{x:.2f}}<extra></extra>",
            showlegend=False,
        ))
        if not pd.isna(mid):
            fig.add_trace(go.Scatter(
                x=[mid], y=[label], mode="markers",
                marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="black")),
                hovertemplate=f"{label} median: {currency_symbol}%{{x:.2f}}<extra></extra>",
                showlegend=False,
            ))

    if not pd.isna(target.get("currentPrice", np.nan)):
        fig.add_vline(
            x=target["currentPrice"], line_dash="dash", line_color="crimson", line_width=2,
            annotation_text=f"Current Price: {currency_symbol}{target['currentPrice']:,.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Football Field Valuation — {target.get('shortName', target.name)}",
        xaxis_title=f"Implied Share Price ({currency_symbol})",
        height=420, margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_white",
    )
    return fig


def peer_bar_chart(multiples: pd.DataFrame, target_ticker: str, peer_median: float) -> go.Figure:
    plot_data = multiples[["shortName", "EV_EBITDA"]].dropna().sort_values("EV_EBITDA")
    colors = ["#C44E52" if idx == target_ticker else "#4C72B0" for idx in plot_data.index]

    fig = go.Figure(go.Bar(
        x=plot_data["shortName"], y=plot_data["EV_EBITDA"], marker_color=colors,
        hovertemplate="%{x}<br>EV/EBITDA: %{y:.2f}x<extra></extra>",
    ))
    fig.add_hline(y=peer_median, line_dash="dash", line_color="black",
                  annotation_text=f"Peer median: {peer_median:.1f}x")
    fig.update_layout(
        title="EV/EBITDA — Target (red) vs. Peer Group",
        yaxis_title="EV / EBITDA (x)", height=400,
        margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
    )
    return fig


def growth_scatter_chart(multiples: pd.DataFrame, target_ticker: str) -> go.Figure:
    scatter_data = multiples[["shortName", "EV_EBITDA", "revenueGrowth"]].dropna()
    colors = ["#C44E52" if idx == target_ticker else "#4C72B0" for idx in scatter_data.index]
    sizes = [22 if idx == target_ticker else 14 for idx in scatter_data.index]

    fig = go.Figure(go.Scatter(
        x=scatter_data["revenueGrowth"] * 100, y=scatter_data["EV_EBITDA"],
        mode="markers+text", text=scatter_data["shortName"], textposition="top center",
        marker=dict(color=colors, size=sizes, line=dict(width=1, color="black")),
        hovertemplate="%{text}<br>Growth: %{x:.1f}%<br>EV/EBITDA: %{y:.2f}x<extra></extra>",
    ))
    fig.update_layout(
        title="Growth-Adjusted Valuation: EV/EBITDA vs. Revenue Growth",
        xaxis_title="Revenue Growth (%)", yaxis_title="EV / EBITDA (x)",
        height=430, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
    )
    return fig


# ======================================================================
# Sidebar — inputs
# ======================================================================
st.sidebar.title("📊 Comps Tool Setup")
st.sidebar.caption("Enter Yahoo Finance tickers. NSE stocks need a `.NS` suffix (e.g. `INFY.NS`); most US tickers need no suffix (e.g. `AAPL`).")

PRESETS = {
    "Indian IT Services": ("INFY.NS", "TCS.NS, HCLTECH.NS, WIPRO.NS, TECHM.NS, LTIM.NS"),
    "US Big Tech": ("AAPL", "MSFT, GOOGL, META, AMZN"),
    "US Banks": ("JPM", "BAC, WFC, C, GS, MS"),
    "Custom": ("", ""),
}
preset_choice = st.sidebar.selectbox("Quick-start example", list(PRESETS.keys()))
default_target, default_peers = PRESETS[preset_choice]

target_input = st.sidebar.text_input("Target company ticker", value=default_target, placeholder="e.g. INFY.NS")
peers_input = st.sidebar.text_area("Peer tickers (comma-separated)", value=default_peers, placeholder="e.g. TCS.NS, WIPRO.NS", height=80)

run_clicked = st.sidebar.button("Run Comps Analysis", type="primary", width="stretch")

with st.sidebar.expander("ℹ️ Methodology"):
    st.markdown("""
    - Peer statistics (mean/median/quartiles) **exclude** the target.
    - EV-based multiples are bridged to implied share price via:
      `Implied Equity Value = Implied EV − Debt + Cash`
    - PE/PB multiples imply price directly from EPS / Book Value per share.
    - Data cached for 15 min per ticker to reduce API load.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · yfinance · Plotly")


# ======================================================================
# Main panel
# ======================================================================
st.title("Comparable Company Analysis")
st.caption("Derive an implied valuation range for any company from how its peers trade.")

if "results" not in st.session_state:
    st.session_state.results = None

if run_clicked:
    target_ticker = target_input.strip().upper()
    peer_tickers = [t.strip().upper() for t in peers_input.split(",") if t.strip()]

    if not target_ticker:
        st.error("Please enter a target company ticker.")
    elif len(peer_tickers) < 2:
        st.error("Please enter at least 2 peer tickers for meaningful statistics.")
    else:
        all_tickers = [target_ticker] + [p for p in peer_tickers if p != target_ticker]
        raw_data, failed = fetch_universe(all_tickers)

        if target_ticker not in raw_data.index:
            st.error(
                f"Could not fetch data for target ticker **{target_ticker}**. "
                "Check the symbol is correct (Yahoo Finance format) and try again — "
                "Yahoo Finance occasionally rate-limits repeated requests."
            )
        else:
            valid_peers = [t for t in raw_data.index if t != target_ticker]
            if len(valid_peers) < 2:
                st.error(
                    f"Only {len(valid_peers)} peer(s) returned usable data "
                    f"(failed: {', '.join(failed) if failed else 'none'}). "
                    "Need at least 2 valid peers — try different tickers."
                )
            else:
                if failed:
                    st.warning(f"Excluded (no data returned): {', '.join(failed)}")

                multiples = compute_multiples(raw_data)
                peer_multiples = multiples.drop(index=target_ticker)
                peer_stats = compute_peer_stats(peer_multiples)
                target = multiples.loc[target_ticker]
                implied_valuation = compute_implied_valuation(target, peer_stats)
                currency_symbol = CURRENCY_SYMBOLS.get(target.get("currency", "USD"), target.get("currency", "$") + " ")

                st.session_state.results = dict(
                    multiples=multiples, peer_stats=peer_stats, target=target,
                    implied_valuation=implied_valuation, currency_symbol=currency_symbol,
                    target_ticker=target_ticker, failed=failed,
                )

results = st.session_state.results

if results is None:
    st.info("👈 Set a target company and peer group in the sidebar, then click **Run Comps Analysis**.")
    st.markdown("""
    **What this tool does:** given a target company and a peer group, it pulls live trading
    multiples from Yahoo Finance (EV/Revenue, EV/EBITDA, P/E, P/B, PEG), computes peer-group
    statistics, and applies those peer multiples back to the target's own financials to produce
    an implied share-price range per methodology — visualized as an industry-standard
    **football field chart**.
    """)
else:
    multiples = results["multiples"]
    peer_stats = results["peer_stats"]
    target = results["target"]
    implied_valuation = results["implied_valuation"]
    currency_symbol = results["currency_symbol"]
    target_ticker = results["target_ticker"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", target.get("shortName", target_ticker))
    col2.metric("Current Price", f"{currency_symbol}{target['currentPrice']:,.2f}" if pd.notna(target["currentPrice"]) else "N/A")
    col3.metric("Market Cap", f"{currency_symbol}{target['marketCap']:,.0f}" if pd.notna(target["marketCap"]) else "N/A")
    col4.metric("Peer Group Size", f"{multiples.shape[0] - 1}")

    st.markdown("---")

    st.plotly_chart(football_field_chart(implied_valuation, target, currency_symbol), width="stretch")

    valid_methodologies = [c for c in METHODOLOGY_LABELS if not implied_valuation[c].isna().all()]
    if len(valid_methodologies) < len(METHODOLOGY_LABELS):
        missing = set(METHODOLOGY_LABELS) - set(valid_methodologies)
        st.caption(f"⚠️ Some methodologies omitted due to missing data: {', '.join(METHODOLOGY_LABELS[m] for m in missing)}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(peer_bar_chart(multiples, target_ticker, peer_stats.loc["median", "EV_EBITDA"]), width="stretch")
    with c2:
        st.plotly_chart(growth_scatter_chart(multiples, target_ticker), width="stretch")

    st.markdown("---")

    tab1, tab2 = st.tabs(["Peer Comps Table", "Peer Group Statistics"])

    with tab1:
        display_df = multiples[["shortName", "currentPrice", "marketCap", "EV", "EV_Revenue", "EV_EBITDA", "PE", "PB", "PEG"]].round(2)

        def highlight_target(row):
            return ["background-color: #ffe9e9; font-weight: bold" if row.name == target_ticker else "" for _ in row]

        st.dataframe(
            display_df.style.apply(highlight_target, axis=1).format({
                "currentPrice": "{:,.2f}", "marketCap": "{:,.0f}", "EV": "{:,.0f}",
                "EV_Revenue": "{:.2f}x", "EV_EBITDA": "{:.2f}x", "PE": "{:.2f}x",
                "PB": "{:.2f}x", "PEG": "{:.2f}",
            }),
            width="stretch",
        )

        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer)
        st.download_button("⬇ Download Comps Table (CSV)", csv_buffer.getvalue(),
                            file_name=f"{target_ticker}_comps.csv", mime="text/csv")

    with tab2:
        st.dataframe(peer_stats.round(2), width="stretch")
        st.markdown("**Implied Share Price by Methodology**")
        st.dataframe(implied_valuation.round(2), width="stretch")

        val_csv_buffer = io.StringIO()
        implied_valuation.round(2).to_csv(val_csv_buffer)
        st.download_button("⬇ Download Implied Valuation (CSV)", val_csv_buffer.getvalue(),
                            file_name=f"{target_ticker}_implied_valuation.csv", mime="text/csv")

    st.caption(
        "Data via Yahoo Finance (yfinance). For educational/research purposes only — "
        "not investment advice. Multiples reflect trailing (not forward) fundamentals."
    )

# Nifty 50 Multi-Factor Equity Screener

A cross-sectional multi-factor equity screener for India's **Nifty 50** index, built in Python. Ranks all 50 constituents using four institutional factor families — **Value, Momentum, Quality, and Low Volatility** — combined into a single weighted composite score, with automated data-quality handling and interactive visual analytics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aahil0/nifty50-multi-factor-screener/blob/main/nifty50_multi_factor_equity_screener.ipynb)

## What it does

1. **Pulls real market data** for all 50 Nifty constituents from Yahoo Finance (`yfinance`) — daily prices and fundamental snapshots (P/E, P/B, EV/EBITDA, ROE, margins, debt/equity).
2. **Engineers four factor families**, each sign-adjusted so "higher is always better":
   - **Value** — inverse of P/E, P/B, and EV/EBITDA
   - **Momentum** — blended 3-month / 12-month trailing return
   - **Quality** — ROE, profit margins, inverse leverage
   - **Low Volatility** — inverse of annualized realized volatility
3. **Standardizes every factor cross-sectionally** (z-scores), so metrics on wildly different scales (a P/E ratio vs. a % return vs. annualized vol) become directly comparable.
4. **Blends the four factor z-scores into one composite score** using configurable weights (default: 30% Value / 25% Momentum / 30% Quality / 15% Low-Vol), producing a full ranked leaderboard.
5. **Visualizes the results**: a top/bottom composite score bar chart, a factor correlation heatmap, an interactive Momentum-vs-Quality scatter (Plotly, sized by market cap), and a sector-level composite score breakdown.
6. **Exports a clean, ranked CSV** for further research or portfolio construction.

Missing or unavailable data (common with a couple of very recently renamed/demerged Nifty constituents) is handled gracefully — a stock is never dropped from the whole screener just because one field is missing.

## Sample output

<img width="1400" height="900" alt="image" src="https://github.com/user-attachments/assets/2cd9eccd-29f2-4615-913a-5dc12b62c786" />
<img width="1400" height="900" alt="image" src="https://github.com/user-attachments/assets/36b3039a-e6ea-4552-ad1a-d12f7de18d96" />
<img width="1656" height="657" alt="image" src="https://github.com/user-attachments/assets/4fb308cc-f8b6-4991-b11c-d6926b274e64" />
<img width="984" height="784" alt="image" src="https://github.com/user-attachments/assets/657b26dc-2688-4cab-9fbb-a2dc6bbafb51" />





## Quant finance concepts used

- Cross-sectional factor investing (Value, Momentum, Quality, Low-Vol)
- Z-score standardization and composite scoring, the same architecture behind MSCI/Barra multi-factor risk models
- Sign-orientation of raw metrics before standardization
- Realized volatility annualization via the √time scaling rule
- Factor correlation / orthogonality diagnostics

## Tech stack

`Python` · `yfinance` · `pandas` · `numpy` · `matplotlib` / `seaborn` · `plotly` · `scipy`

## How to run

**Option A — Google Colab (recommended, zero setup):**
Click the "Open in Colab" badge above, then `Runtime → Run all`.

**Option B — Locally:**
```bash
git clone https://github.com/Aahil0/nifty50-multi-factor-screener.git
cd nifty50-multi-factor-screener
pip install -r requirements.txt
jupyter notebook notebooks/nifty50_multi_factor_equity_screener.ipynb
```

## Known limitations / next steps

- No sector-neutralization yet — the composite score currently reflects some implicit sector tilts (see notebook markdown for details).
- No outlier winsorization before z-scoring.
- This is a single-day snapshot screener, not a backtested strategy — no point-in-time historical fundamentals, transaction costs, or portfolio-level risk modeling yet.
- Planned upgrades: sector-neutral scoring, historical backtesting with Information Coefficient analysis, and a covariance-based portfolio optimizer layer.


## License

MIT — feel free to fork and extend.

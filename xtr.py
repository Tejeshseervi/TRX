"""
Quantitative Pairs Trading Research Platform
============================================
A professional statistical arbitrage research tool implementing cointegration-based
pairs trading with full backtesting and performance analytics.

Author: Quant Research Platform
Framework: Streamlit + statsmodels + yfinance
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller, acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import io
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quantitative Pairs Trading Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem;}
    .sub-header {font-size:1rem; color:#666; margin-bottom:1.5rem;}
    .metric-card {background:#f8f9fa; border-radius:8px; padding:1rem; border-left:4px solid #0066cc;}
    .section-header {font-size:1.2rem; font-weight:600; color:#1a1a2e; border-bottom:2px solid #0066cc;
                     padding-bottom:0.3rem; margin:1.5rem 0 1rem 0;}
    .stAlert {border-radius:8px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data(ticker_a: str, ticker_b: str, start: str, end: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Download adjusted closing prices for two tickers via yfinance.
    Aligns on common trading dates, drops NaN rows.

    Returns:
        price_a, price_b : aligned price series
        info_df          : basic descriptive statistics
    """
    raw_a = yf.download(ticker_a, start=start, end=end, auto_adjust=True, progress=False)
    raw_b = yf.download(ticker_b, start=start, end=end, auto_adjust=True, progress=False)

    if raw_a.empty or raw_b.empty:
        return None, None, None

    close_a = raw_a["Close"].squeeze()
    close_b = raw_b["Close"].squeeze()

    combined = pd.concat([close_a, close_b], axis=1, join="inner").dropna()
    combined.columns = [ticker_a, ticker_b]

    if len(combined) < 60:
        return None, None, None

    price_a = combined[ticker_a]
    price_b = combined[ticker_b]

    info = pd.DataFrame({
        "Metric": ["Start Date", "End Date", "Observations", "Mean Price", "Std Dev", "Min Price", "Max Price"],
        ticker_a: [
            str(price_a.index[0].date()), str(price_a.index[-1].date()),
            len(price_a), f"{price_a.mean():.2f}", f"{price_a.std():.2f}",
            f"{price_a.min():.2f}", f"{price_a.max():.2f}",
        ],
        ticker_b: [
            str(price_b.index[0].date()), str(price_b.index[-1].date()),
            len(price_b), f"{price_b.mean():.2f}", f"{price_b.std():.2f}",
            f"{price_b.min():.2f}", f"{price_b.max():.2f}",
        ],
    })
    return price_a, price_b, info


# ─────────────────────────────────────────────────────────────────────────────
# ECONOMETRIC CORE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_correlation(price_a: pd.Series, price_b: pd.Series,
                           rolling_window: int = 90) -> tuple[float, pd.Series]:
    """
    Pearson correlation + rolling correlation.
    Rolling correlation reveals structural breaks in the relationship.
    """
    corr = price_a.corr(price_b)
    rolling_corr = price_a.rolling(rolling_window).corr(price_b)
    return corr, rolling_corr


def cointegration_test(price_a: pd.Series, price_b: pd.Series) -> dict:
    """
    Engle-Granger two-step cointegration test (statsmodels.tsa.stattools.coint).

    Cointegration: two I(1) series share a common stochastic trend.
    If cointegrated, a linear combination produces an I(0) (stationary) spread,
    which is the theoretical foundation for mean-reversion pairs trading.

    Returns test statistic, p-value, critical values, and boolean flag.
    """
    t_stat, p_value, crit_vals = coint(price_a, price_b)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "crit_1pct": crit_vals[0],
        "crit_5pct": crit_vals[1],
        "crit_10pct": crit_vals[2],
        "is_cointegrated": p_value < 0.05,
    }


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> tuple[float, float, object]:
    """
    OLS regression: Price_A = α + β * Price_B

    β (hedge ratio): number of units of B to short per unit of A long.
    Ensures the portfolio is market-neutral w.r.t. to the common factor.

    Returns alpha, beta (hedge ratio), and fitted OLS results object.
    """
    X = add_constant(price_b.values)
    model = OLS(price_a.values, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    return alpha, beta, model


def calculate_spread(price_a: pd.Series, price_b: pd.Series, beta: float) -> pd.Series:
    """
    Spread = Price_A − β * Price_B

    The spread represents deviation from the long-run equilibrium.
    Under cointegration, this spread is stationary (mean-reverting).
    """
    return price_a - beta * price_b


def adf_test(spread: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test on the spread.
    Rejection of H0 (unit root) confirms spread stationarity,
    a necessary condition for mean reversion.
    """
    result = adfuller(spread.dropna(), autolag="AIC")
    return {
        "t_stat": result[0],
        "p_value": result[1],
        "lags": result[2],
        "nobs": result[3],
        "crit_1pct": result[4]["1%"],
        "crit_5pct": result[4]["5%"],
        "crit_10pct": result[4]["10%"],
        "is_stationary": result[1] < 0.05,
    }


def compute_half_life(spread: pd.Series) -> float:
    """
    Half-life of mean reversion via Ornstein-Uhlenbeck approximation.

    Regress: ΔS_t = λ * S_{t-1} + ε_t
    λ is the mean-reversion speed.
    Half-life = -ln(2) / λ (in days).

    This is critical for calibrating the rolling window and holding period.
    """
    spread_lag = spread.shift(1)
    delta_spread = spread.diff()
    df = pd.concat([delta_spread, spread_lag], axis=1).dropna()
    df.columns = ["delta", "lag"]
    X = add_constant(df["lag"].values)
    model = OLS(df["delta"].values, X).fit()
    lam = model.params[1]
    if lam >= 0:
        return np.nan
    half_life = -np.log(2) / lam
    return round(half_life, 1)


def compute_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling Z-score: Z = (S_t − μ_t) / σ_t
    where μ_t and σ_t are rolling mean and std over [window] days.

    Trading signal: large |Z| indicates the spread is far from equilibrium,
    creating a mean-reversion opportunity.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore, rolling_mean, rolling_std


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_trading_signals(zscore: pd.Series,
                              entry_threshold: float = 2.0,
                              exit_threshold: float = 0.5) -> pd.DataFrame:
    """
    Mean-reversion signal logic:

    Long spread  (buy A, sell B): Z < -entry_threshold  → spread below equilibrium
    Short spread (sell A, buy B): Z >  entry_threshold  → spread above equilibrium
    Exit:                         |Z| < exit_threshold   → spread near equilibrium

    Position held until exit condition met (avoids churn on noisy signals).
    """
    signals = pd.DataFrame(index=zscore.index)
    signals["zscore"] = zscore
    signals["position"] = 0  # 1=long spread, -1=short spread, 0=flat

    position = 0
    positions = []

    for z in zscore:
        if np.isnan(z):
            positions.append(0)
            continue
        if position == 0:
            if z < -entry_threshold:
                position = 1
            elif z > entry_threshold:
                position = -1
        elif position == 1:
            if abs(z) < exit_threshold:
                position = 0
        elif position == -1:
            if abs(z) < exit_threshold:
                position = 0
        positions.append(position)

    signals["position"] = positions
    signals["signal_change"] = signals["position"].diff().fillna(0)
    signals["entry"] = signals["signal_change"] != 0
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(price_a: pd.Series, price_b: pd.Series,
                 signals: pd.DataFrame, beta: float,
                 ticker_a: str, ticker_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorised backtest of the pairs trading strategy.

    Long spread:  r_t = r_A_t − β * r_B_t
    Short spread: r_t = −r_A_t + β * r_B_t

    Returns aligned returns dataframe and trade log.
    """
    ret_a = price_a.pct_change()
    ret_b = price_b.pct_change()

    # Spread return: position taken at prior close, realised next day
    spread_return = signals["position"].shift(1) * (ret_a - beta * ret_b)
    spread_return = spread_return.fillna(0)

    bt = pd.DataFrame({
        ticker_a: ret_a,
        ticker_b: ret_b,
        "position": signals["position"],
        "spread_return": spread_return,
        "cum_return": (1 + spread_return).cumprod(),
    }, index=price_a.index).dropna()

    # Trade log
    trades = []
    in_trade = False
    entry_date = None
    entry_pos = 0
    entry_z = None

    for idx, row in signals.iterrows():
        if not in_trade and row["position"] != 0:
            in_trade = True
            entry_date = idx
            entry_pos = row["position"]
            entry_z = row["zscore"]
        elif in_trade and row["position"] == 0:
            exit_date = idx
            direction = "Long Spread" if entry_pos == 1 else "Short Spread"
            trade_ret = bt.loc[entry_date:exit_date, "spread_return"].sum()
            trades.append({
                "Entry Date": entry_date.date(),
                "Exit Date": exit_date.date(),
                "Direction": direction,
                "Entry Z-Score": round(entry_z, 3),
                "Exit Z-Score": round(row["zscore"], 3),
                "Trade Return (%)": round(trade_ret * 100, 3),
                "Duration (days)": (exit_date - entry_date).days,
            })
            in_trade = False

    trade_log = pd.DataFrame(trades)
    return bt, trade_log


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(bt: pd.DataFrame, trade_log: pd.DataFrame,
                                 risk_free_rate: float = 0.05) -> dict:
    """
    Industry-standard quantitative performance metrics.
    Annual factor = 252 trading days.
    """
    returns = bt["spread_return"].dropna()
    cum_ret = bt["cum_return"]
    ann_factor = 252

    total_return = cum_ret.iloc[-1] - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)

    excess = returns - risk_free_rate / ann_factor
    sharpe = (excess.mean() / returns.std()) * np.sqrt(ann_factor) if returns.std() > 0 else 0

    downside = returns[returns < 0]
    sortino = (ann_return - risk_free_rate) / (downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 else 0

    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    win_rate = (trade_log["Trade Return (%)"] > 0).mean() if len(trade_log) > 0 else 0
    avg_trade = trade_log["Trade Return (%)"].mean() if len(trade_log) > 0 else 0
    n_trades = len(trade_log)

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Ann. Return (%)": round(ann_return * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 3),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Avg Trade Return (%)": round(avg_trade, 3),
        "Number of Trades": n_trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#0066cc",
    "secondary": "#e63946",
    "green": "#2a9d8f",
    "amber": "#e9c46a",
    "bg": "#ffffff",
    "grid": "#eeeeee",
}

LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor=COLORS["bg"],
)


def plot_prices(price_a: pd.Series, price_b: pd.Series,
                ticker_a: str, ticker_b: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=price_a.index, y=price_a, name=ticker_a,
                             line=dict(color=COLORS["primary"], width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=price_b.index, y=price_b, name=ticker_b,
                             line=dict(color=COLORS["secondary"], width=1.5)), secondary_y=True)
    fig.update_layout(title="Price Series Comparison", **LAYOUT_BASE)
    fig.update_yaxes(title_text=f"{ticker_a} Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text=f"{ticker_b} Price (USD)", secondary_y=True)
    return fig


def plot_spread(spread: pd.Series, rolling_mean: pd.Series,
                rolling_std: pd.Series) -> go.Figure:
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=upper, name="+2σ Band",
                             line=dict(color=COLORS["secondary"], dash="dash", width=1),
                             fill=None))
    fig.add_trace(go.Scatter(x=spread.index, y=lower, name="−2σ Band",
                             line=dict(color=COLORS["green"], dash="dash", width=1),
                             fill="tonexty", fillcolor="rgba(42,157,143,0.08)"))
    fig.add_trace(go.Scatter(x=spread.index, y=rolling_mean, name="Rolling Mean",
                             line=dict(color=COLORS["amber"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread",
                             line=dict(color=COLORS["primary"], width=1.5)))
    fig.update_layout(title="Spread Dynamics with Rolling Bands", yaxis_title="Spread Value", **LAYOUT_BASE)
    return fig


def plot_zscore(zscore: pd.Series, signals: pd.DataFrame,
                entry: float, exit_thresh: float) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=entry, line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text=f"+{entry} (Short Entry)", annotation_position="top right")
    fig.add_hline(y=-entry, line_dash="dash", line_color=COLORS["green"],
                  annotation_text=f"-{entry} (Long Entry)", annotation_position="bottom right")
    fig.add_hline(y=exit_thresh, line_dash="dot", line_color=COLORS["amber"])
    fig.add_hline(y=-exit_thresh, line_dash="dot", line_color=COLORS["amber"])
    fig.add_hline(y=0, line_color="#cccccc")
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score",
                             line=dict(color=COLORS["primary"], width=1.5)))
    # Entry markers
    long_entries = signals[(signals["position"] == 1) & (signals["signal_change"] != 0)]
    short_entries = signals[(signals["position"] == -1) & (signals["signal_change"] != 0)]
    exits = signals[(signals["position"] == 0) & (signals["signal_change"] != 0)]
    fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries["zscore"],
                             mode="markers", name="Long Entry",
                             marker=dict(color=COLORS["green"], symbol="triangle-up", size=9)))
    fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries["zscore"],
                             mode="markers", name="Short Entry",
                             marker=dict(color=COLORS["secondary"], symbol="triangle-down", size=9)))
    fig.add_trace(go.Scatter(x=exits.index, y=exits["zscore"],
                             mode="markers", name="Exit",
                             marker=dict(color=COLORS["amber"], symbol="x", size=8)))
    fig.update_layout(title="Z-Score Signal Chart with Trade Markers", yaxis_title="Z-Score", **LAYOUT_BASE)
    return fig


def plot_rolling_corr(rolling_corr: pd.Series, ticker_a: str, ticker_b: str) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(42,157,143,0.1)", line_width=0,
                  annotation_text="High Correlation Zone")
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr,
                             name="90-day Rolling Correlation",
                             line=dict(color=COLORS["primary"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(0,102,204,0.08)"))
    fig.update_layout(title=f"Rolling 90-Day Correlation: {ticker_a} vs {ticker_b}",
                      yaxis_title="Pearson Correlation", yaxis_range=[-1, 1], **LAYOUT_BASE)
    return fig


def plot_cumulative_returns(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=(bt["cum_return"] - 1) * 100,
                             name="Strategy Returns",
                             line=dict(color=COLORS["primary"], width=2),
                             fill="tozeroy", fillcolor="rgba(0,102,204,0.1)"))
    fig.add_hline(y=0, line_color="#cccccc")
    fig.update_layout(title="Strategy Cumulative Returns (%)", yaxis_title="Cumulative Return (%)", **LAYOUT_BASE)
    return fig


def plot_drawdown(bt: pd.DataFrame) -> go.Figure:
    cum_ret = bt["cum_return"]
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown (%)",
                             line=dict(color=COLORS["secondary"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(230,57,70,0.15)"))
    fig.update_layout(title="Strategy Drawdown (%)", yaxis_title="Drawdown (%)", **LAYOUT_BASE)
    return fig


def plot_spread_distribution(spread: pd.Series) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spread Distribution", "Spread Autocorrelation (40 lags)"))
    fig.add_trace(go.Histogram(x=spread, nbinsx=60, name="Spread",
                               marker_color=COLORS["primary"], opacity=0.75), row=1, col=1)

    acf_vals = acf(spread.dropna(), nlags=40, fft=True)
    lags = list(range(len(acf_vals)))
    colors = [COLORS["secondary"] if abs(v) > 1.96 / np.sqrt(len(spread)) else COLORS["primary"]
              for v in acf_vals]
    fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF", marker_color=colors), row=1, col=2)
    conf_bound = 1.96 / np.sqrt(len(spread))
    for sign in [1, -1]:
        fig.add_hline(y=sign * conf_bound, line_dash="dash", line_color=COLORS["secondary"],
                      row=1, col=2)
    fig.update_layout(title="Spread Statistical Diagnostics", showlegend=False, **LAYOUT_BASE)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_metric_row(metrics: dict):
    """Render performance metrics in a 5-column grid."""
    keys = list(metrics.keys())
    vals = list(metrics.values())
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(keys):
            col.metric(keys[i], vals[i])
    cols2 = st.columns(5)
    for i, col in enumerate(cols2):
        j = i + 5
        if j < len(keys):
            col.metric(keys[j], vals[j])


def section(title: str):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def main():
    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("<div class='main-header'>📈 Quantitative Pairs Trading Research Platform</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Statistical Arbitrage | Cointegration Analysis | Mean-Reversion Backtesting</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Research Parameters")
        st.subheader("Asset Selection")
        ticker_a = st.text_input("Ticker A", value="KO", max_chars=10).upper().strip()
        ticker_b = st.text_input("Ticker B", value="PEP", max_chars=10).upper().strip()

        st.subheader("Date Range")
        default_end = date.today()
        default_start = default_end - timedelta(days=5 * 365)
        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=default_end)

        st.subheader("Strategy Parameters")
        rolling_window = st.slider("Z-Score Rolling Window (days)", 10, 120, 30, 5)
        entry_threshold = st.slider("Entry Threshold (|Z|)", 1.0, 4.0, 2.0, 0.25)
        exit_threshold = st.slider("Exit Threshold (|Z|)", 0.1, 1.5, 0.5, 0.1)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.25) / 100

        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Data: Yahoo Finance (yfinance) | Framework: statsmodels, pandas, plotly")

    # ── MAIN PANEL ─────────────────────────────────────────────────────────────
    if not run_btn:
        st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to begin.")
        with st.expander("ℹ️ About This Platform"):
            st.markdown("""
**Quantitative Pairs Trading Research Platform** implements a rigorous statistical arbitrage
framework grounded in cointegration theory (Engle & Granger, 1987).

**Methodology:**
1. **Cointegration Test** — Verifies the existence of a long-run equilibrium relationship
2. **Hedge Ratio (OLS)** — Estimates the market-neutral weighting between the two assets
3. **Spread Z-Score** — Normalises deviation from equilibrium to generate trade signals
4. **Mean-Reversion Backtest** — Simulates a fully systematic long/short strategy
5. **Performance Attribution** — Reports Sharpe, Sortino, drawdown, and trade statistics

Built for academic and professional quantitative research.
            """)
        return

    # ── DATA LOADING ──────────────────────────────────────────────────────────
    with st.spinner(f"Downloading {ticker_a} / {ticker_b} from Yahoo Finance…"):
        price_a, price_b, info_df = load_data(
            ticker_a, ticker_b, str(start_date), str(end_date)
        )

    if price_a is None:
        st.error(f"❌ Failed to load data for **{ticker_a}** and/or **{ticker_b}**. "
                 "Please check the ticker symbols and date range.")
        return

    # ─── SECTION 1: DATA OVERVIEW ────────────────────────────────────────────
    section("1. Data Overview")
    st.dataframe(info_df.set_index("Metric"), use_container_width=True)
    st.plotly_chart(plot_prices(price_a, price_b, ticker_a, ticker_b), use_container_width=True)

    # ─── SECTION 2: STATISTICAL RELATIONSHIP ─────────────────────────────────
    section("2. Statistical Relationship Analysis")
    pearson_corr, rolling_corr = calculate_correlation(price_a, price_b)
    col1, col2, col3 = st.columns(3)
    corr_interp = "Strong" if abs(pearson_corr) > 0.8 else "Moderate" if abs(pearson_corr) > 0.5 else "Weak"
    col1.metric("Pearson Correlation", f"{pearson_corr:.4f}", f"{corr_interp} linear relationship")
    col2.metric("Avg Rolling Corr (90d)", f"{rolling_corr.mean():.4f}")
    col3.metric("Corr Stability (Std)", f"{rolling_corr.std():.4f}",
                "Lower = more stable relationship")
    st.plotly_chart(plot_rolling_corr(rolling_corr, ticker_a, ticker_b), use_container_width=True)

    # ─── SECTION 3: COINTEGRATION ─────────────────────────────────────────────
    section("3. Cointegration Test (Engle-Granger)")
    coint_res = cointegration_test(price_a, price_b)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Statistic", f"{coint_res['t_stat']:.4f}")
    col2.metric("P-Value", f"{coint_res['p_value']:.4f}")
    col3.metric("Critical Value (5%)", f"{coint_res['crit_5pct']:.4f}")
    col4.metric("Cointegrated?",
                "✅ Yes" if coint_res["is_cointegrated"] else "❌ No",
                f"p = {coint_res['p_value']:.4f}")

    if coint_res["is_cointegrated"]:
        st.success(f"**Cointegration confirmed** at the 5% significance level (p = {coint_res['p_value']:.4f}). "
                   "The pair shares a long-run equilibrium — a necessary condition for statistical arbitrage.")
    else:
        st.warning(f"**No statistically significant cointegration** detected (p = {coint_res['p_value']:.4f}). "
                   "Pairs trading signals may be unreliable. Proceed with caution.")

    with st.expander("Critical Values Table"):
        crit_df = pd.DataFrame({
            "Significance Level": ["1%", "5%", "10%"],
            "Critical Value": [coint_res["crit_1pct"], coint_res["crit_5pct"], coint_res["crit_10pct"]],
            "Test Statistic": [coint_res["t_stat"]] * 3,
            "Reject H₀?": [
                "Yes" if coint_res["t_stat"] < coint_res["crit_1pct"] else "No",
                "Yes" if coint_res["t_stat"] < coint_res["crit_5pct"] else "No",
                "Yes" if coint_res["t_stat"] < coint_res["crit_10pct"] else "No",
            ]
        })
        st.dataframe(crit_df, use_container_width=True, hide_index=True)

    # ─── SECTION 4: SPREAD DYNAMICS ───────────────────────────────────────────
    section("4. Spread Dynamics & Mean-Reversion Diagnostics")
    alpha, beta, ols_model = estimate_hedge_ratio(price_a, price_b)
    spread = calculate_spread(price_a, price_b, beta)
    half_life = compute_half_life(spread)
    adf_res = adf_test(spread)
    zscore, rolling_mean, rolling_std = compute_zscore(spread, rolling_window)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Hedge Ratio (β)", f"{beta:.4f}")
    col2.metric("Intercept (α)", f"{alpha:.4f}")
    col3.metric("Spread Mean", f"{spread.mean():.4f}")
    col4.metric("Spread Std Dev", f"{spread.std():.4f}")
    col5.metric("Half-Life (days)", f"{half_life:.1f}" if not np.isnan(half_life) else "N/A",
                "Mean reversion speed")

    st.caption(f"**OLS Model:** {ticker_a} = {alpha:.4f} + {beta:.4f} × {ticker_b}  |  "
               f"R² = {ols_model.rsquared:.4f}  |  "
               f"ADF p-value on spread: {adf_res['p_value']:.4f} "
               f"({'Stationary ✅' if adf_res['is_stationary'] else 'Non-stationary ⚠️'})")

    st.plotly_chart(plot_spread(spread, rolling_mean, rolling_std), use_container_width=True)
    st.plotly_chart(plot_spread_distribution(spread), use_container_width=True)

    # ADF detail
    with st.expander("ADF Stationarity Test Detail"):
        adf_df = pd.DataFrame({
            "Metric": ["Test Statistic", "P-Value", "Lags Used", "Observations",
                       "Critical Value 1%", "Critical Value 5%", "Critical Value 10%", "Stationary?"],
            "Value": [f"{adf_res['t_stat']:.4f}", f"{adf_res['p_value']:.4f}",
                      adf_res["lags"], adf_res["nobs"],
                      f"{adf_res['crit_1pct']:.4f}", f"{adf_res['crit_5pct']:.4f}",
                      f"{adf_res['crit_10pct']:.4f}",
                      "Yes ✅" if adf_res["is_stationary"] else "No ❌"]
        })
        st.dataframe(adf_df, use_container_width=True, hide_index=True)

    # ─── SECTION 5: TRADING SIGNALS ───────────────────────────────────────────
    section("5. Z-Score Trading Signals")
    signals = generate_trading_signals(zscore, entry_threshold, exit_threshold)

    col1, col2, col3 = st.columns(3)
    col1.metric("Long Entries", int((signals["position"].diff() == 1).sum()))
    col2.metric("Short Entries", int((signals["position"].diff() == -1).sum()))
    col3.metric("Days In Market", int((signals["position"] != 0).sum()),
                f"{100*(signals['position']!=0).mean():.1f}% of time")

    st.plotly_chart(plot_zscore(zscore, signals, entry_threshold, exit_threshold), use_container_width=True)

    # ─── SECTION 6: BACKTEST ──────────────────────────────────────────────────
    section("6. Backtest Results")
    bt, trade_log = run_backtest(price_a, price_b, signals, beta, ticker_a, ticker_b)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_cumulative_returns(bt), use_container_width=True)
    with col2:
        st.plotly_chart(plot_drawdown(bt), use_container_width=True)

    # ─── SECTION 7: PERFORMANCE METRICS ──────────────────────────────────────
    section("7. Strategy Performance Metrics")
    perf = compute_performance_metrics(bt, trade_log, risk_free_rate)
    render_metric_row(perf)

    if len(trade_log) > 0:
        with st.expander(f"📋 Trade Log ({len(trade_log)} trades)"):
            trade_log_styled = trade_log.copy()
            st.dataframe(trade_log_styled, use_container_width=True, hide_index=True)

    # ─── EXPORTS ──────────────────────────────────────────────────────────────
    section("8. Export Research Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        bt_csv = bt.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Backtest Results (CSV)", bt_csv,
                           file_name=f"backtest_{ticker_a}_{ticker_b}.csv",
                           mime="text/csv", use_container_width=True)

    with col2:
        if len(trade_log) > 0:
            tl_csv = trade_log.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Trade Log (CSV)", tl_csv,
                               file_name=f"trades_{ticker_a}_{ticker_b}.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.button("⬇️ Trade Log — No Trades", disabled=True, use_container_width=True)

    with col3:
        signals_csv = signals.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Trading Signals (CSV)", signals_csv,
                           file_name=f"signals_{ticker_a}_{ticker_b}.csv",
                           mime="text/csv", use_container_width=True)

    # ─── SUMMARY REPORT ───────────────────────────────────────────────────────
    section("9. Research Summary")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Asset Pair | **{ticker_a} / {ticker_b}** |
| Analysis Period | {str(start_date)} → {str(end_date)} |
| Observations | {len(price_a)} trading days |
| Pearson Correlation | {pearson_corr:.4f} |
| Cointegrated (5%) | {"✅ Yes" if coint_res["is_cointegrated"] else "❌ No"} (p={coint_res["p_value"]:.4f}) |
| ADF Stationary Spread | {"✅ Yes" if adf_res["is_stationary"] else "❌ No"} (p={adf_res["p_value"]:.4f}) |
| Hedge Ratio β | {beta:.4f} |
| Half-Life | {f"{half_life:.1f} days" if not np.isnan(half_life) else "N/A"} |
| Total Return | {perf["Total Return (%)"]:.2f}% |
| Sharpe Ratio | {perf["Sharpe Ratio"]:.3f} |
| Sortino Ratio | {perf["Sortino Ratio"]:.3f} |
| Max Drawdown | {perf["Max Drawdown (%)"]:.2f}% |
| Trades | {perf["Number of Trades"]} |
| Win Rate | {perf["Win Rate (%)"]:.2f}% |
    """)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

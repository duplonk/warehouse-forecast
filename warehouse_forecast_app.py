"""
Warehouse Intra-Logistics Demand Forecasting App
Calculates ideal levels, minimum levels and reorder points per SKU
Models: HMM, Linear Regression, ARIMA, SARIMA, Holt-Winters
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
import io
import math
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The Oracle — Demand Planning",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── THE ORACLE — Industrial Theme ── */
    [data-testid="stAppViewContainer"] {
        background-color: #F5F0E8;
    }
    [data-testid="stSidebar"] {
        background-color: #2C2416 !important;
    }
    [data-testid="stSidebar"] * {
        color: #F5F0E8 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #C4B99A !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
        color: #2C2416 !important;
        background-color: #E9A43A !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #E9A43A !important;
        border-color: #C4B99A !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #2C2416 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {
        color: #2C2416 !important;
        background-color: #E9A43A !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F5F0E8 !important;
    }
    .oracle-header {
        background: #2C2416;
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 3px solid #E9A43A;
    }
    h1.oracle-title, .oracle-title {
        font-size: 3.8rem !important;
        font-weight: 700 !important;
        color: #F5F0E8 !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.03em !important;
        line-height: 1 !important;
    }
    h1.oracle-title span, .oracle-title span {
        color: #E9A43A !important;
    }
    .oracle-subtitle {
        font-size: 0.85rem;
        color: #C4B99A;
        margin: 0.4rem 0 0 0;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .oracle-badge {
        display: inline-block;
        background: #E9A43A;
        color: #2C2416;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-left: 1rem;
        vertical-align: middle;
    }
    .metric-card {
        background: #fff;
        border-left: 4px solid #E9A43A;
        border-radius: 0;
        padding: 1rem 1.25rem;
        border-top: 0.5px solid #C4B99A;
        border-right: 0.5px solid #C4B99A;
        border-bottom: 0.5px solid #C4B99A;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #6B5B3E;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2C2416;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #E9A43A;
        font-weight: 600;
    }
    .abc-a { background: #E9A43A; color: #2C2416; padding: 2px 8px; font-weight: 700; font-size: 12px; }
    .abc-b { background: #C4B99A; color: #2C2416; padding: 2px 8px; font-weight: 700; font-size: 12px; }
    .abc-c { background: #EDE8DC; color: #6B5B3E; padding: 2px 8px; font-weight: 700; font-size: 12px; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        color: #2C2416;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #E9A43A;
        padding-bottom: 0.25rem;
    }
    .stButton > button {
        background-color: #E9A43A !important;
        color: #2C2416 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 0 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        background-color: #2C2416 !important;
        color: #E9A43A !important;
    }
    [data-testid="stMetricValue"] {
        color: #2C2416 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6B5B3E !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-size: 0.75rem !important;
    }
    .stDataFrame {
        border: 0.5px solid #C4B99A !important;
    }
    div[data-testid="stSuccess"] {
        background-color: #EDE8DC;
        border-left: 4px solid #E9A43A;
        color: #2C2416;
    }
    div[data-testid="stInfo"] {
        background-color: #EDE8DC;
        border-left: 4px solid #C4B99A;
        color: #6B5B3E;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def classify_abc(df_sku_summary):
    """Classify SKUs into ABC tiers based on cumulative pick volume."""
    df = df_sku_summary.sort_values('total_picks', ascending=False).copy()
    df['cumulative_pct'] = df['total_picks'].cumsum() / df['total_picks'].sum() * 100
    df['abc'] = 'C'
    df.loc[df['cumulative_pct'] <= 80, 'abc'] = 'A'
    df.loc[(df['cumulative_pct'] > 80) & (df['cumulative_pct'] <= 95), 'abc'] = 'B'
    return df


# ─────────────────────────────────────────────
# MODEL: HMM (Baum-Welch + Viterbi)
# ─────────────────────────────────────────────

def gauss(x, m, s):
    s = max(s, 0.01)
    return (1 / (s * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - m) / s) ** 2)

def hmm_fit_predict(series, n_states=3, n_iter=20):
    """Fit HMM and return state sequence + state means."""
    obs_raw = list(series)
    n = len(obs_raw)
    if n < 7:
        return None, None, None

    mean_p = np.mean(obs_raw)
    std_p = max(np.std(obs_raw), 0.01)
    obs = [(x - mean_p) / std_p for x in obs_raw]

    mu = [-1.0, 0.0, 1.0]
    sigma = [0.5, 0.5, 0.5]
    pi = [1/3, 1/3, 1/3]
    A = [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]
    S = n_states

    for _ in range(n_iter):
        # Forward
        alpha = [[0]*S for _ in range(n)]
        for s in range(S):
            alpha[0][s] = pi[s] * gauss(obs[0], mu[s], sigma[s])
        c = [sum(alpha[0]) or 1e-300]
        alpha[0] = [a/c[0] for a in alpha[0]]
        for t in range(1, n):
            for s in range(S):
                alpha[t][s] = sum(alpha[t-1][r]*A[r][s] for r in range(S)) * gauss(obs[t], mu[s], sigma[s])
            ct = sum(alpha[t]) or 1e-300
            c.append(ct)
            alpha[t] = [a/ct for a in alpha[t]]

        # Backward
        beta = [[1.0]*S for _ in range(n)]
        for t in range(n-2, -1, -1):
            for s in range(S):
                beta[t][s] = sum(A[s][r]*gauss(obs[t+1], mu[r], sigma[r])*beta[t+1][r] for r in range(S))
            ct = c[t+1] or 1e-300
            beta[t] = [b/ct for b in beta[t]]

        # Gamma / Xi
        gamma = [[0]*S for _ in range(n)]
        for t in range(n):
            total = sum(alpha[t][s]*beta[t][s] for s in range(S)) or 1e-300
            for s in range(S):
                gamma[t][s] = alpha[t][s]*beta[t][s]/total

        xi = [[[0]*S for _ in range(S)] for _ in range(n-1)]
        for t in range(n-1):
            total = sum(alpha[t][r]*A[r][s]*gauss(obs[t+1], mu[s], sigma[s])*beta[t+1][s]
                        for r in range(S) for s in range(S)) or 1e-300
            for r in range(S):
                for s in range(S):
                    xi[t][r][s] = alpha[t][r]*A[r][s]*gauss(obs[t+1], mu[s], sigma[s])*beta[t+1][s]/total

        # Update
        pi = [gamma[0][s] for s in range(S)]
        for r in range(S):
            denom = sum(gamma[t][r] for t in range(n-1)) or 1e-300
            for s in range(S):
                A[r][s] = sum(xi[t][r][s] for t in range(n-1)) / denom
        for s in range(S):
            denom = sum(gamma[t][s] for t in range(n)) or 1e-300
            mu[s] = sum(gamma[t][s]*obs[t] for t in range(n)) / denom
            sigma[s] = math.sqrt(sum(gamma[t][s]*(obs[t]-mu[s])**2 for t in range(n)) / denom) or 0.01

    # Viterbi
    delta = [[0]*S for _ in range(n)]
    psi = [[0]*S for _ in range(n)]
    for s in range(S):
        delta[0][s] = math.log(pi[s]+1e-300) + math.log(gauss(obs[0], mu[s], sigma[s])+1e-300)
    for t in range(1, n):
        for s in range(S):
            scores = [delta[t-1][r] + math.log(A[r][s]+1e-300) for r in range(S)]
            best = max(range(S), key=lambda r: scores[r])
            psi[t][s] = best
            delta[t][s] = scores[best] + math.log(gauss(obs[t], mu[s], sigma[s])+1e-300)
    path = [0]*n
    path[n-1] = max(range(S), key=lambda s: delta[n-1][s])
    for t in range(n-2, -1, -1):
        path[t] = psi[t+1][path[t+1]]

    state_means_real = [mu[s]*std_p + mean_p for s in range(S)]
    order = sorted(range(S), key=lambda s: state_means_real[s])
    state_labels = {order[0]: 'Low', order[1]: 'Medium', order[2]: 'High'}
    state_means_ordered = {
        'Low': state_means_real[order[0]],
        'Medium': state_means_real[order[1]],
        'High': state_means_real[order[2]]
    }

    path_labels = [state_labels[s] for s in path]
    high_mean = state_means_real[order[2]]
    return path_labels, state_means_ordered, high_mean


# ─────────────────────────────────────────────
# MODEL: LINEAR REGRESSION
# ─────────────────────────────────────────────

def linear_regression_forecast(series, horizon=30):
    """Simple linear trend forecast."""
    y = np.array(series)
    x = np.arange(len(y))
    slope, intercept, r, p, se = stats.linregress(x, y)
    future_x = np.arange(len(y), len(y) + horizon)
    forecast = slope * future_x + intercept
    avg_forecast = float(np.mean(forecast))
    return max(avg_forecast, 0), slope


# ─────────────────────────────────────────────
# MODEL: HOLT-WINTERS (Triple Exponential Smoothing)
# ─────────────────────────────────────────────

def holt_winters_forecast(series, season_length=7, horizon=30):
    """Triple exponential smoothing — additive seasonality."""
    y = np.array(series, dtype=float)
    n = len(y)
    if n < season_length * 2:
        # Fall back to simple exponential smoothing
        alpha = 0.3
        level = y[0]
        for val in y[1:]:
            level = alpha * val + (1 - alpha) * level
        return float(level), None

    # Initialise
    n_seasons = n // season_length
    season_avgs = [np.mean(y[i*season_length:(i+1)*season_length]) for i in range(n_seasons)]
    level = season_avgs[0]
    trend = (season_avgs[1] - season_avgs[0]) / season_length if n_seasons > 1 else 0
    seasonals = [y[i] - level for i in range(season_length)]

    alpha, beta, gamma = 0.3, 0.1, 0.2

    smoothed = []
    for t in range(n):
        s_idx = t % season_length
        if t == 0:
            smoothed.append(level + seasonals[s_idx])
            continue
        prev_level = level
        prev_trend = trend
        level = alpha * (y[t] - seasonals[s_idx]) + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        seasonals[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seasonals[s_idx]
        smoothed.append(level + trend + seasonals[s_idx])

    forecasts = []
    for h in range(1, horizon + 1):
        s_idx = (n + h - 1) % season_length
        forecasts.append(level + h * trend + seasonals[s_idx])

    avg_forecast = float(np.mean(np.maximum(forecasts, 0)))
    return avg_forecast, trend


# ─────────────────────────────────────────────
# MODEL: ARIMA (simplified AR + differencing)
# ─────────────────────────────────────────────

def arima_forecast(series, horizon=30):
    """Simplified ARIMA(1,1,1) using OLS."""
    y = np.array(series, dtype=float)
    if len(y) < 10:
        return float(np.mean(y)), None

    # First difference
    dy = np.diff(y)
    n = len(dy)
    if n < 3:
        return float(np.mean(y)), None

    # AR(1) on differenced series
    X = dy[:-1].reshape(-1, 1)
    Y = dy[1:]
    # OLS
    X_b = np.hstack([np.ones((len(X), 1)), X])
    try:
        coeffs = np.linalg.lstsq(X_b, Y, rcond=None)[0]
    except Exception:
        return float(np.mean(y)), None

    intercept, ar_coef = coeffs[0], coeffs[1]
    last_d = dy[-1]
    last_y = y[-1]
    forecasts = []
    for _ in range(horizon):
        next_d = intercept + ar_coef * last_d
        next_y = last_y + next_d
        forecasts.append(max(next_y, 0))
        last_d = next_d
        last_y = next_y

    return float(np.mean(forecasts)), ar_coef


# ─────────────────────────────────────────────
# MODEL: SARIMA (simplified seasonal AR)
# ─────────────────────────────────────────────

def sarima_forecast(series, season_length=7, horizon=30):
    """Simplified seasonal ARIMA — AR on seasonally differenced series."""
    y = np.array(series, dtype=float)
    if len(y) < season_length * 2 + 5:
        return arima_forecast(series, horizon)

    # Seasonal difference
    dy = y[season_length:] - y[:-season_length]
    n = len(dy)
    if n < 5:
        return float(np.mean(y)), None

    # AR(1) on seasonal differences
    X = dy[:-1].reshape(-1, 1)
    Y_arr = dy[1:]
    X_b = np.hstack([np.ones((len(X), 1)), X])
    try:
        coeffs = np.linalg.lstsq(X_b, Y_arr, rcond=None)[0]
    except Exception:
        return float(np.mean(y)), None

    intercept, ar_coef = coeffs[0], coeffs[1]
    last_sd = dy[-1]
    last_y = y[-1]
    forecasts = []
    for h in range(horizon):
        next_sd = intercept + ar_coef * last_sd
        ref_idx = len(y) - season_length + (h % season_length)
        ref_val = y[min(ref_idx, len(y)-1)]
        next_y = ref_val + next_sd
        forecasts.append(max(next_y, 0))
        last_sd = next_sd
        last_y = next_y

    return float(np.mean(forecasts)), ar_coef


# ─────────────────────────────────────────────
# CORE: Calculate stock levels per SKU
# ─────────────────────────────────────────────

def calculate_stock_levels(series, lead_time_days, replen_freq_days, abc_class, service_factor=1.65):
    """
    Calculate ideal, minimum and reorder point for a SKU.
    service_factor=1.65 → ~95% service level
    """
    arr = np.array(series, dtype=float)
    avg_daily = float(np.mean(arr))
    std_daily = float(np.std(arr))
    n = len(arr)

    # Run all models
    results = {}

    if abc_class == 'A' and n >= 14:
        _, state_means, hmm_high = hmm_fit_predict(arr)
        if hmm_high is not None:
            results['HMM'] = hmm_high
        hw, _ = holt_winters_forecast(arr)
        results['Holt-Winters'] = hw
        ar, _ = arima_forecast(arr)
        results['ARIMA'] = ar
        sar, _ = sarima_forecast(arr)
        results['SARIMA'] = sar
        lr, _ = linear_regression_forecast(arr)
        results['Linear Regression'] = lr

    elif abc_class == 'B' and n >= 7:
        hw, _ = holt_winters_forecast(arr)
        results['Holt-Winters'] = hw
        lr, _ = linear_regression_forecast(arr)
        results['Linear Regression'] = lr

    else:
        results['Average'] = avg_daily

    # Ensemble forecast — median of model outputs
    forecast_values = [v for v in results.values() if v > 0]
    if forecast_values:
        ensemble_forecast = float(np.median(forecast_values))
    else:
        ensemble_forecast = avg_daily

    # Safety stock
    safety_stock = service_factor * std_daily * math.sqrt(lead_time_days)

    # Stock level calculations
    if replen_freq_days == 0:  # Continuous
        # Ideal = what you need to cover lead time + safety buffer
        ideal_level = round(ensemble_forecast * lead_time_days + safety_stock, 1)
        # Minimum = safety stock only (trigger replenishment here)
        minimum_level = round(safety_stock, 1)
        # Reorder point = demand during lead time + safety stock
        # Should equal ideal level for continuous replenishment
        reorder_point = round(ensemble_forecast * lead_time_days + safety_stock, 1)
    else:
        # Ideal = cover full replenishment cycle + safety stock
        ideal_level = round(ensemble_forecast * (lead_time_days + replen_freq_days) + safety_stock, 1)
        # Minimum = safety stock only
        minimum_level = round(safety_stock, 1)
        # Reorder point = demand during full cycle (lead time + freq) + safety stock
        # Must trigger early enough to never stockout before next scheduled replenishment
        reorder_point = round(ensemble_forecast * (lead_time_days + replen_freq_days) + safety_stock, 1)

    return {
        'avg_daily_picks': round(avg_daily, 2),
        'std_daily': round(std_daily, 2),
        'ensemble_forecast': round(ensemble_forecast, 2),
        'ideal_level': max(ideal_level, 1),
        'minimum_level': max(minimum_level, 1),
        'reorder_point': max(reorder_point, 1),
        'models_used': list(results.keys()),
        'model_values': {k: round(v, 2) for k, v in results.items()},
        'model_agreement': round(np.std(forecast_values) / np.mean(forecast_values) * 100, 1) if len(forecast_values) > 1 else 0
    }


# ─────────────────────────────────────────────
# GENERATE SAMPLE DATA
# ─────────────────────────────────────────────

def generate_sample_data():
    """Generate realistic multi-SKU warehouse pick data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    rows = []
    skus = {
        'SKU-A001': (25, 5), 'SKU-A002': (20, 4), 'SKU-A003': (18, 4),
        'SKU-B001': (10, 3), 'SKU-B002': (8, 2), 'SKU-B003': (7, 2),
        'SKU-C001': (3, 1), 'SKU-C002': (2, 1), 'SKU-C003': (1, 1),
        'SKU-C004': (4, 2),
    }
    for sku, (mean, std) in skus.items():
        for date in dates:
            # Add weekly seasonality
            dow_factor = 1.2 if date.weekday() in [1, 2, 3] else 0.8
            picks = max(0, int(np.random.normal(mean * dow_factor, std)))
            if picks > 0:
                rows.append({'Date': date.strftime('%Y-%m-%d'), 'SKU': sku, 'Units_Picked': picks})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="oracle-header">
        <h1 class="oracle-title">🔮 The <span>Oracle</span><span class="oracle-badge">Live</span></h1>
        <p class="oracle-subtitle">Demand planning for the masses &nbsp;·&nbsp; Bulk-to-pick replenishment intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        st.markdown("**Replenishment Parameters**")
        lead_time_options = {"1 day": 1, "2 days": 2, "3 days": 3, "5 days": 5, "Custom": None}
        lead_time_choice = st.selectbox("Lead Time", list(lead_time_options.keys()))
        if lead_time_choice == "Custom":
            lead_time_days = st.number_input("Lead time (days)", min_value=1, max_value=30, value=2)
        else:
            lead_time_days = lead_time_options[lead_time_choice]

        replen_options = {"Continuous": 0, "Every 1 day": 1, "Every 2 days": 2, "Every 3 days": 3, "Custom": None}
        replen_choice = st.selectbox("Replenishment Frequency", list(replen_options.keys()))
        if replen_choice == "Custom":
            replen_freq_days = st.number_input("Frequency (days)", min_value=1, max_value=30, value=1)
        else:
            replen_freq_days = replen_options[replen_choice]

        st.markdown("---")
        st.markdown("**Service Level**")
        service_level = st.select_slider(
            "Target service level",
            options=["90%", "95%", "98%", "99%"],
            value="95%"
        )
        service_factors = {"90%": 1.28, "95%": 1.65, "98%": 2.05, "99%": 2.33}
        service_factor = service_factors[service_level]

        st.markdown("---")
        st.markdown("**ABC Thresholds**")
        abc_a_pct = st.slider("A items — top % of volume", 60, 85, 80)
        abc_b_pct = st.slider("B items — next % of volume", 5, 20, 15)

        st.divider()
        st.caption(f"A = top {abc_a_pct}% of picks volume")
        st.caption(f"B = next {abc_b_pct}% of picks volume")
        st.caption(f"C = remaining {100 - abc_a_pct - abc_b_pct}%")

    # ── DATA INPUT ──
    st.markdown("<div class='section-header'>1. Load pick data</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Required columns: Date, SKU, Units_Picked — or Date, Units_Picked for single-SKU files"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📊 Load sample data", use_container_width=True)

    single_sku_mode = False

    if use_sample:
        st.session_state['df'] = generate_sample_data()
        st.session_state['data_label'] = f"Sample data loaded — {len(st.session_state['df']):,} records, {st.session_state['df']['SKU'].nunique()} SKUs"

    elif uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df_raw.columns = [c.strip() for c in df_raw.columns]
            if 'SKU' in df_raw.columns and 'Units_Picked' in df_raw.columns:
                loaded = df_raw[['Date', 'SKU', 'Units_Picked']].copy()
                loaded['Units_Picked'] = pd.to_numeric(loaded['Units_Picked'], errors='coerce').fillna(0)
                st.session_state['df'] = loaded
                st.session_state['data_label'] = f"Loaded {len(loaded):,} records across {loaded['SKU'].nunique()} SKUs"
            elif 'Units_Sold' in df_raw.columns or 'Units_Picked' in df_raw.columns:
                pick_col = 'Units_Picked' if 'Units_Picked' in df_raw.columns else 'Units_Sold'
                loaded = df_raw[['Date', pick_col]].copy()
                loaded.columns = ['Date', 'Units_Picked']
                loaded['SKU'] = 'SKU-001'
                loaded['Units_Picked'] = pd.to_numeric(loaded['Units_Picked'], errors='coerce').fillna(0)
                single_sku_mode = True
                st.session_state['df'] = loaded
                st.session_state['data_label'] = f"Single-SKU file — {len(loaded):,} daily records"
            else:
                st.error("Could not detect columns. Need: Date + SKU + Units_Picked, or Date + Units_Picked")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    df = st.session_state.get('df', None)
    if st.session_state.get('data_label'):
        st.success(st.session_state['data_label'])

    if df is None:
        st.info("Upload a CSV or use the sample data to get started.")
        st.markdown("""
        **Expected CSV format (multi-SKU):**
        ```
        Date,SKU,Units_Picked
        2023-01-01,SKU-A001,22
        2023-01-01,SKU-B003,7
        ```
        **Single-SKU format:**
        ```
        Date,Units_Picked
        2023-01-01,14
        2023-01-02,18
        ```
        """)
        return

    # ── DATA OVERVIEW ──
    st.divider()
    st.markdown("<div class='section-header'>2. Data overview</div>", unsafe_allow_html=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Units_Picked'] = pd.to_numeric(df['Units_Picked'], errors='coerce').fillna(0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique SKUs", f"{df['SKU'].nunique():,}")
    with col3:
        st.metric("Date Range", f"{df['Date'].min().strftime('%d %b %y')} – {df['Date'].max().strftime('%d %b %y')}")
    with col4:
        st.metric("Total Picks", f"{df['Units_Picked'].sum():,.0f}")

    # ── ABC CLASSIFICATION ──
    st.divider()
    st.markdown("<div class='section-header'>3. ABC classification</div>", unsafe_allow_html=True)

    sku_summary = df.groupby('SKU').agg(
        total_picks=('Units_Picked', 'sum'),
        avg_daily=('Units_Picked', 'mean'),
        active_days=('Units_Picked', 'count')
    ).reset_index()

    sku_abc = classify_abc(sku_summary)
    # Override thresholds from sidebar
    sku_abc = sku_abc.sort_values('total_picks', ascending=False).copy()
    sku_abc['cumulative_pct'] = sku_abc['total_picks'].cumsum() / sku_abc['total_picks'].sum() * 100
    sku_abc['abc'] = 'C'
    sku_abc.loc[sku_abc['cumulative_pct'] <= abc_a_pct, 'abc'] = 'A'
    sku_abc.loc[(sku_abc['cumulative_pct'] > abc_a_pct) & (sku_abc['cumulative_pct'] <= abc_a_pct + abc_b_pct), 'abc'] = 'B'

    a_count = (sku_abc['abc'] == 'A').sum()
    b_count = (sku_abc['abc'] == 'B').sum()
    c_count = (sku_abc['abc'] == 'C').sum()
    total = len(sku_abc)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">A items — High velocity</div>
            <div class="value">{a_count}</div>
            <div class="sub">{a_count/total*100:.0f}% of SKUs · Full model suite</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">B items — Medium velocity</div>
            <div class="value">{b_count}</div>
            <div class="sub">{b_count/total*100:.0f}% of SKUs · Holt-Winters + LR</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">C items — Low velocity</div>
            <div class="value">{c_count}</div>
            <div class="sub">{c_count/total*100:.0f}% of SKUs · Average-based</div>
        </div>
        """, unsafe_allow_html=True)

    # ── RUN MODELS ──
    st.divider()
    st.markdown("<div class='section-header'>4. Run forecast models</div>", unsafe_allow_html=True)

    replen_label = "Continuous" if replen_freq_days == 0 else f"Every {replen_freq_days} day(s)"
    st.markdown(f"**Settings:** Lead time = **{lead_time_days} day(s)** · Replenishment = **{replen_label}** · Service level = **{service_level}**")

    if st.button("⚡ Run The Oracle", type="primary", use_container_width=True):
        results_list = []
        sku_abc_dict = dict(zip(sku_abc['SKU'], sku_abc['abc']))

        progress = st.progress(0)
        status = st.empty()
        skus_to_process = sku_abc['SKU'].tolist()
        n_skus = len(skus_to_process)

        for i, sku in enumerate(skus_to_process):
            status.text(f"Processing {sku} ({i+1}/{n_skus})...")
            progress.progress((i+1)/n_skus)

            sku_data = df[df['SKU'] == sku].sort_values('Date')
            # Use only days with actual picks — zeros from missing dates skew std
            series = sku_data['Units_Picked'].values
            date_range = pd.date_range(sku_data['Date'].min(), sku_data['Date'].max(), freq='D')

            abc_class = sku_abc_dict.get(sku, 'C')
            levels = calculate_stock_levels(series, lead_time_days, replen_freq_days, abc_class, service_factor)

            results_list.append({
                'SKU': sku,
                'ABC': abc_class,
                'Avg Daily Picks': levels['avg_daily_picks'],
                'Ensemble Forecast': levels['ensemble_forecast'],
                'Ideal Level': levels['ideal_level'],
                'Minimum Level': levels['minimum_level'],
                'Reorder Point': levels['reorder_point'],
                'Models Used': ', '.join(levels['models_used']),
                'Model Agreement (CV%)': levels['model_agreement'],
                'Total Annual Picks': int(sku_summary[sku_summary['SKU']==sku]['total_picks'].values[0]),
            })

        progress.empty()
        status.empty()

        results_df = pd.DataFrame(results_list).sort_values(['ABC', 'Total Annual Picks'], ascending=[True, False])
        st.session_state['results_df'] = results_df
        st.success(f"✅ Models run for {n_skus} SKUs")

    # ── RESULTS ──
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        st.divider()
        st.markdown("<div class='section-header'>5. Results</div>", unsafe_allow_html=True)

        # Filter
        col1, col2 = st.columns([1, 3])
        with col1:
            abc_filter = st.multiselect("Filter by ABC", ['A', 'B', 'C'], default=['A', 'B', 'C'])
        filtered = results_df[results_df['ABC'].isin(abc_filter)]

        # Flag low agreement
        flag_count = (filtered['Model Agreement (CV%)'] > 30).sum()
        if flag_count > 0:
            st.warning(f"⚠️ {flag_count} SKU(s) show high model disagreement (CV > 30%) — review manually")

        # Display table
        def color_abc(val):
            colors = {'A': 'background-color: #fdecea', 'B': 'background-color: #fff8e1', 'C': 'background-color: #e8f4fd'}
            return colors.get(val, '')

        def flag_disagreement(val):
            return 'background-color: #fff3cd' if val > 30 else ''

        styled = filtered.style\
            .map(color_abc, subset=['ABC'])\
            .map(flag_disagreement, subset=['Model Agreement (CV%)'])\
            .format({
                'Avg Daily Picks': '{:.1f}',
                'Ensemble Forecast': '{:.1f}',
                'Ideal Level': '{:.0f}',
                'Minimum Level': '{:.0f}',
                'Reorder Point': '{:.0f}',
                'Model Agreement (CV%)': '{:.1f}%',
                'Total Annual Picks': '{:,}'
            })

        st.dataframe(styled, use_container_width=True, height=400)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Ideal Level", f"{filtered['Ideal Level'].mean():.0f} units")
        with col2:
            st.metric("Avg Reorder Point", f"{filtered['Reorder Point'].mean():.0f} units")
        with col3:
            total_ideal = filtered['Ideal Level'].sum()
            st.metric("Total Ideal Stock", f"{total_ideal:,.0f} units")
        with col4:
            flagged = (filtered['Model Agreement (CV%)'] > 30).sum()
            st.metric("SKUs needing review", f"{flagged}")

        # ── SKU DETAIL VIEW ──
        st.divider()
        st.markdown("<div class='section-header'>6. SKU detail view</div>", unsafe_allow_html=True)

        selected_sku = st.selectbox("Select SKU to inspect", filtered['SKU'].tolist())

        if selected_sku:
            sku_row = filtered[filtered['SKU'] == selected_sku].iloc[0]
            sku_data = df[df['SKU'] == selected_sku].sort_values('Date')
            date_range = pd.date_range(sku_data['Date'].min(), sku_data['Date'].max(), freq='D')
            sku_indexed = sku_data.set_index('Date')[['Units_Picked']]
            sku_full = sku_indexed.reindex(date_range).fillna(0)
            series = sku_data['Units_Picked'].values

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ABC Class", sku_row['ABC'])
            with col2:
                st.metric("Ideal Level", f"{sku_row['Ideal Level']:.0f} units")
            with col3:
                st.metric("Minimum Level", f"{sku_row['Minimum Level']:.0f} units")
            with col4:
                st.metric("Reorder Point", f"{sku_row['Reorder Point']:.0f} units")

            # Chart
            import streamlit as st_inner
            chart_df = pd.DataFrame({
                'Date': date_range,
                'Daily Picks': series,
                'Ideal Level': sku_row['Ideal Level'],
                'Reorder Point': sku_row['Reorder Point'],
                'Minimum Level': sku_row['Minimum Level'],
            }).set_index('Date')

            st.line_chart(chart_df, color=["#378ADD", "#2ecc71", "#e67e22", "#e74c3c"])
            st.caption("Blue = daily picks · Green = ideal level · Orange = reorder point · Red = minimum level")

            # Models breakdown
            if sku_row['ABC'] in ['A', 'B']:
                st.markdown("**Model outputs:**")
                abc_c = sku_row['ABC']
                levels = calculate_stock_levels(series, lead_time_days, replen_freq_days, abc_c, service_factor)
                model_df = pd.DataFrame([
                    {'Model': k, 'Forecast (avg daily)': v}
                    for k, v in levels['model_values'].items()
                ])
                st.dataframe(model_df, use_container_width=True, hide_index=True)

        # ── EXPORT ──
        st.divider()
        st.markdown("<div class='section-header'>7. Export results</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_out = filtered.to_csv(index=False)
            st.download_button(
                "⬇️ Download full results",
                data=csv_out,
                file_name=f"warehouse_stock_levels_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv',
                use_container_width=True
            )
        with col2:
            wms_df = filtered[['SKU', 'Minimum Level', 'Reorder Point', 'Ideal Level']].copy()
            wms_df.columns = ['sku_code', 'min_qty', 'reorder_qty', 'max_qty']
            wms_csv = wms_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download WMS upload file",
                data=wms_csv,
                file_name=f"wms_upload_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv',
                use_container_width=True
            )
        st.caption("WMS upload file contains: sku_code, min_qty, reorder_qty, max_qty — ready for import")


if __name__ == "__main__":
    main()

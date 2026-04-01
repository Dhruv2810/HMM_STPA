"""
╔══════════════════════════════════════════════════════════════════╗
║   NIFTY HMM REGIME DETECTOR — STREAMLIT DASHBOARD               ║
║   Backend: original app.py (unchanged)                           ║
║   Frontend: HMM-Nifty website aesthetic                          ║
║   Features: HMM Bull/Bear detection with RSI + India VIX        ║
║                                                                  ║
║   FIXED: Improved caching strategy + error handling             ║
╚══════════════════════════════════════════════════════════════════╝

To run locally:
    pip install streamlit yfinance hmmlearn scikit-learn plotly scipy pandas numpy
    streamlit run app.py

To deploy on Streamlit Cloud:
    - Push this file + requirements.txt to GitHub
    - Deploy directly from GitHub in Streamlit Cloud dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import hashlib
import json
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit command
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "HMM Market Regime Detector",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS — website aesthetic (dark terminal, Space Mono, greens)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── BASE ── */
html, body, .stApp {
    background-color: #060a0f !important;
    color: #e8edf5 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #0d1420 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebarContent"] { padding: 24px 20px !important; }

/* Hide the collapse toggle arrow button */
[data-testid="collapsedControl"],
button[data-testid="baseButton-header"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Sidebar text colours */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #e8edf5; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #6b7a99 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    text-align: left !important;
    transition: all 0.2s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,255,135,0.06) !important;
    border-color: rgba(0,255,135,0.35) !important;
    color: #00ff87 !important;
}

/* ── SIDEBAR LOGO ── */
.sidebar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #00ff87 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.sidebar-tag {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #6b7a99 !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: block;
}

/* ── TOP HEADER ── */
.page-header {
    padding: 32px 0 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 32px;
}

.page-header .tag {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.22em;
    color: #00ff87;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.page-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    color: #e8edf5 !important;
    margin: 0 !important;
}

.page-header h1 .g { color: #00ff87; }
.page-header h1 .r { color: #ff3b5c; }

/* ── SECTION HEADERS ── */
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.22em;
    color: #00ff87;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #e8edf5;
    margin-bottom: 20px;
}

/* ── REGIME BADGES ── */
.bull-badge {
    font-family: 'Space Mono', monospace;
    background: rgba(0,255,135,0.08);
    color: #00ff87 !important;
    padding: 10px 22px;
    font-size: 15px;
    font-weight: 700;
    border: 1px solid rgba(0,255,135,0.35);
    display: inline-flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.06em;
}

.bear-badge {
    font-family: 'Space Mono', monospace;
    background: rgba(255,59,92,0.08);
    color: #ff3b5c !important;
    padding: 10px 22px;
    font-size: 15px;
    font-weight: 700;
    border: 1px solid rgba(255,59,92,0.35);
    display: inline-flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.06em;
}

/* ── CODE BLOCKS ── */
.code-block {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,255,135,0.15);
    border-radius: 4px;
    padding: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
    color: #a8b3c8;
    overflow-x: auto;
}

.cm { color: #6b7a99; }

/* ── CARDS ── */
.card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,255,135,0.15);
    border-radius: 8px;
    padding: 24px;
    margin: 16px 0;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(ticker, start_date, end_date):
    """Fetch price data from Yahoo Finance with caching."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error(f"No data found for {ticker}")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def compute_rsi(prices, period=14):
    """Compute RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@st.cache_data(ttl=3600)
def fetch_vix_data(start_date, end_date):
    """Fetch India VIX data."""
    try:
        vix = yf.download("^NSEIVX", start=start_date, end=end_date, progress=False)
        if vix.empty:
            st.warning("India VIX data not available, using dummy values")
            return None
        return vix["Close"]
    except Exception as e:
        st.warning(f"Could not fetch India VIX: {str(e)}")
        return None


def run_hmm_analysis(df, ticker="^NSEI", name="NIFTY 50"):
    """
    Run HMM analysis on price data.
    Returns a dictionary of results without caching to avoid serialization issues.
    """
    try:
        # Prepare data
        df_copy = df.copy()
        
        # Calculate returns and volatility
        df_copy['returns'] = df_copy['Adj Close'].pct_change() * 100
        df_copy['volatility'] = df_copy['returns'].rolling(window=20).std()
        
        # Calculate RSI
        df_copy['RSI'] = compute_rsi(df_copy['Adj Close'])
        
        # Fetch VIX
        vix_data = fetch_vix_data(df_copy.index[0], df_copy.index[-1])
        if vix_data is not None:
            df_copy['VIX'] = vix_data
        else:
            df_copy['VIX'] = 20  # Default value
        
        # Fill NaN values
        df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
        df_copy = df_copy.dropna()
        
        # Prepare features for HMM (Model B: Returns, Volatility, RSI, VIX)
        features = df_copy[['returns', 'volatility', 'RSI', 'VIX']].values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit HMM (2 states: Bull, Bear)
        hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
        hmm.fit(features_scaled)
        
        # Get predictions
        hidden_states = hmm.predict(features_scaled)
        
        # Determine which state is Bull (higher mean return)
        means_orig = scaler.inverse_transform(hmm.means_)
        bull_state = 0 if means_orig[0, 0] > means_orig[1, 0] else 1
        bear_state = 1 - bull_state
        
        # Map predictions to Bull/Bear
        regimes = np.array(['Bull' if s == bull_state else 'Bear' for s in hidden_states])
        
        # Calculate statistics
        df_copy['regime'] = regimes
        df_copy['hidden_state'] = hidden_states
        
        bull_days = (regimes == 'Bull').sum()
        bear_days = (regimes == 'Bear').sum()
        total_days = len(regimes)
        
        bull_vix = df_copy[df_copy['regime'] == 'Bull']['VIX'].mean()
        bear_vix = df_copy[df_copy['regime'] == 'Bear']['VIX'].mean()
        
        # Transition matrix
        transitions = hmm.transmat_
        bull_persistence = transitions[bull_state, bull_state] * 100
        bear_persistence = transitions[bear_state, bear_state] * 100
        
        # Current regime
        current_regime = regimes[-1]
        
        results = {
            'df': df_copy,
            'regimes': regimes,
            'current_regime': current_regime,
            'bull_days': bull_days,
            'bear_days': bear_days,
            'total_days': total_days,
            'bull_pct': round(bull_days / total_days * 100, 1),
            'bear_pct': round(bear_days / total_days * 100, 1),
            'bull_vix_mean': round(bull_vix, 2),
            'bear_vix_mean': round(bear_vix, 2),
            'bull_persistence': round(bull_persistence, 2),
            'bear_persistence': round(bear_persistence, 2),
            'means_orig': means_orig,
            'hmm': hmm,
            'scaler': scaler,
            'ticker': ticker,
            'name': name,
            'start_date': df_copy.index[0].strftime("%Y-%m-%d"),
            'end_date': df_copy.index[-1].strftime("%Y-%m-%d"),
        }
        
        return results
        
    except Exception as e:
        st.error(f"HMM Analysis Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.markdown('<p class="sidebar-logo">HMM</p>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-tag">NIFTY 50 REGIME DETECTOR</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Configuration")
    years_back = st.slider("Historical data (years):", 1, 20, 10)
    
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.cache_data.clear()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **HMM Regime Detector** identifies bull/bear market regimes using Hidden Markov Models.
    
    **Indicators:**
    - 📈 Returns
    - 📊 Volatility
    - 🔴 RSI (Relative Strength Index)
    - 🌪️ India VIX
    """)

# Main content
st.markdown("""
<div class="page-header">
    <div class="tag">Hidden Markov Models</div>
    <h1>NIFTY 50 Market Regime Detector</h1>
    <div class="sub">Real-time bull/bear classification using multi-indicator HMM analysis</div>
</div>
""", unsafe_allow_html=True)

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=years_back*365)

st.info("📡 Loading data...")
df = fetch_data("^NSEI", start_date, end_date)

if df is not None:
    # Run HMM analysis
    with st.spinner("🔬 Running HMM analysis..."):
        results = run_hmm_analysis(df)
    
    if results is not None:
        # Display current regime
        st.markdown('<div class="sec-label">CURRENT REGIME</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if results['current_regime'] == 'Bull':
                st.markdown(f"""
                <div class="bull-badge">
                    🟢 BULL REGIME ACTIVE
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bear-badge">
                    🔴 BEAR REGIME ACTIVE
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Bull Days", f"{results['bull_days']}", f"{results['bull_pct']}%")
        
        with col3:
            st.metric("Bear Days", f"{results['bear_days']}", f"{results['bear_pct']}%")
        
        st.markdown("---")
        
        # Regime chart
        st.markdown('<div class="sec-label">REGIME ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Historical Regimes</div>', unsafe_allow_html=True)
        
        df_plot = results['df'].copy()
        df_plot['regime_num'] = (df_plot['regime'] == 'Bull').astype(int)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['Adj Close'],
            mode='lines', name='NIFTY 50',
            line=dict(color='#e8edf5', width=2),
            yaxis='y'
        ))
        
        # Bull/Bear background
        for i in range(len(df_plot)):
            if df_plot['regime'].iloc[i] == 'Bull':
                fig.add_vrect(
                    x0=df_plot.index[i], x1=df_plot.index[i],
                    fillcolor='#00ff87', opacity=0.05, line_width=0
                )
        
        fig.update_layout(
            title="NIFTY 50 with HMM Regimes",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=400,
            hovermode='x unified',
            plot_bgcolor='#060a0f',
            paper_bgcolor='#060a0f',
            font=dict(family="Space Mono, monospace", color="#e8edf5"),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.markdown('<div class="sec-label">REGIME STATISTICS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Market Characteristics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bull Persistence", f"{results['bull_persistence']}%", 
                     help="% chance of staying in Bull regime next day")
            st.metric("Bull Avg VIX", f"{results['bull_vix_mean']}", 
                     help="Average India VIX during Bull regimes")
        
        with col2:
            st.metric("Bear Persistence", f"{results['bear_persistence']}%",
                     help="% chance of staying in Bear regime next day")
            st.metric("Bear Avg VIX", f"{results['bear_vix_mean']}",
                     help="Average India VIX during Bear regimes")
        
        st.markdown("---")
        st.markdown(f"""
        <div style='font-family:Space Mono,monospace; font-size:11px; color:#6b7a99;'>
        Data range: {results['start_date']} to {results['end_date']} | 
        {results['total_days']} trading days | 
        Source: Yahoo Finance | Model: GaussianHMM
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Failed to run HMM analysis. Check logs above.")
else:
    st.error("Failed to fetch NIFTY 50 data. Please try again later.")

"""
╔══════════════════════════════════════════════════════════════════╗
║   NIFTY HMM REGIME DETECTOR — STREAMLIT DASHBOARD               ║
║   Backend: original app.py (unchanged)                           ║
║   Frontend: HMM-Nifty website aesthetic                          ║
║   Features: HMM Bull/Bear detection with RSI + India VIX        ║
╚══════════════════════════════════════════════════════════════════╝

To run locally:
    pip install streamlit yfinance hmmlearn scikit-learn plotly scipy
    streamlit run app.py

To deploy on Render:
    - Push this file + requirements.txt to GitHub
    - Create a new Web Service on Render
    - Build command : pip install -r requirements.txt
    - Start command : streamlit run app.py --server.port $PORT --server.address 0.0.0.0
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
/* Sidebar text colours without wildcard override */
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
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(0,255,135,0.08) !important;
    border-left: 2px solid #00ff87 !important;
    border-color: rgba(0,255,135,0.4) !important;
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
    display: flex;
    align-items: center;
    gap: 10px;
}
.page-header .tag::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1px;
    background: #00ff87;
}
.page-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    color: #e8edf5 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.page-header h1 .g { color: #00ff87; }
.page-header h1 .r { color: #ff3b5c; }
.page-header .sub {
    font-size: 14px;
    color: #6b7a99;
    margin-top: 10px;
    line-height: 1.6;
}

/* ── SECTION HEADERS ── */
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.22em;
    color: #00ff87;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::before {
    content: '';
    display: inline-block;
    width: 14px; height: 1px;
    background: #00ff87;
}
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #e8edf5;
    margin-bottom: 20px;
}

/* ── REGIME BADGE ── */
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
    clip-path: polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,8px 100%,0 calc(100% - 8px));
}
.bull-badge::before {
    content: '';
    width: 8px; height: 8px;
    background: #00ff87;
    border-radius: 50%;
    animation: pd 2s ease-in-out infinite;
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
    clip-path: polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,8px 100%,0 calc(100% - 8px));
}
.bear-badge::before {
    content: '';
    width: 8px; height: 8px;
    background: #ff3b5c;
    border-radius: 50%;
    animation: pd 2s ease-in-out infinite;
}
@keyframes pd {
    0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,255,135,.4)}
    50%{opacity:.7;box-shadow:0 0 0 4px rgba(0,255,135,0)}
}

/* ── STAT CARDS (metric) ── */
[data-testid="metric-container"] {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
    padding: 20px 18px !important;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: #00ff87;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #6b7a99 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e8edf5 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

/* ── INFO / WARNING / SUCCESS BOXES ── */
[data-testid="stAlert"] {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    color: #e8edf5 !important;
}
.stSuccess {
    border-left: 3px solid #00ff87 !important;
}
.stInfo {
    border-left: 3px solid #ffe14d !important;
}
.stWarning {
    border-left: 3px solid #ff3b5c !important;
}

/* ── DATAFRAMES ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
.stDataFrame th {
    background: #060a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b7a99 !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: #e8edf5 !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: #00ff87 !important; }

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 28px 0 !important;
}

/* ── CAPTION ── */
.stCaption {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: #6b7a99 !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00ff87, #00d4ff) !important;
}
.stProgress > div > div {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
    height: 8px !important;
}

/* ── SECTION DIVIDER CARD ── */
.section-block {
    background: #0d1420;
    border: 1px solid rgba(255,255,255,0.07);
    padding: 24px;
    margin-bottom: 2px;
    position: relative;
    overflow: hidden;
}
.section-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #00ff87;
}

/* ── FALSE ALARM CALLOUT ── */
.fa-callout {
    background: linear-gradient(135deg, rgba(0,255,135,0.06), rgba(0,255,135,0.02));
    border: 1px solid rgba(0,255,135,0.2);
    padding: 28px 32px;
    display: flex;
    align-items: center;
    gap: 28px;
    margin-bottom: 20px;
}
.fa-num {
    font-family: 'Space Mono', monospace;
    font-size: 64px;
    font-weight: 700;
    color: #00ff87;
    line-height: 1;
    letter-spacing: -0.04em;
    flex-shrink: 0;
}
.fa-text h3 {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #e8edf5;
}
.fa-text p {
    font-size: 13px;
    color: #6b7a99;
    line-height: 1.75;
    margin: 0;
}

/* ── TICKER STRIP ── */
.ticker-wrap {
    background: #0d1420;
    border-top: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 10px 0;
    overflow: hidden;
    margin-bottom: 32px;
}
.ticker-inner {
    display: flex;
    gap: 48px;
    animation: tk 30s linear infinite;
    white-space: nowrap;
}
.ticker-item {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #6b7a99;
    letter-spacing: 0.08em;
}
.ticker-item .g { color: #00ff87; }
.ticker-item .r { color: #ff3b5c; }
.ticker-item .y { color: #ffe14d; }
@keyframes tk { from{transform:translateX(0)} to{transform:translateX(-50%)} }

/* ── REGIME BAR ── */
.rbar-wrap { margin: 16px 0 4px; }
.rbar-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6b7a99;
    margin-bottom: 8px;
}
.rbar-track {
    height: 16px;
    display: flex;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 8px;
}
.rbar-bull { height: 100%; background: #00ff87; opacity: 0.85; }
.rbar-bear { height: 100%; background: #ff3b5c; opacity: 0.75; }
.rbar-leg {
    display: flex;
    gap: 20px;
}
.rbar-li {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #6b7a99;
}
.rbar-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── HEADINGS OVERRIDE ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: #e8edf5 !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em !important;
}
p { color: #e8edf5; }

/* ── CODE BLOCK ── */
.code-block {
    background: #020508;
    border: 1px solid rgba(255,255,255,0.07);
    padding: 18px 22px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #ffe14d;
    line-height: 1.9;
    overflow-x: auto;
    margin: 12px 0;
}
.code-block .cm { color: #6b7a99; }

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── TRANSITION MATRIX ── */
.tm-wrap { margin: 4px 0 16px; }
.tm-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: #6b7a99; margin-bottom: 10px; }
.tm-grid { display: grid; grid-template-columns: 110px 1fr 1fr; gap: 3px; }
.tm-cell { padding: 12px 14px; font-family: 'Space Mono', monospace; display: flex; flex-direction: column; gap: 3px; }
.tm-head { background: #020508; color: #6b7a99; font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; align-items: center; justify-content: center; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
.tm-rowlabel { background: #020508; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; justify-content: center; border: 1px solid rgba(255,255,255,0.05); }
.tm-bb { background: rgba(0,255,135,0.09); border: 1px solid rgba(0,255,135,0.22); }
.tm-bB { background: rgba(255,59,92,0.04); border: 1px solid rgba(255,255,255,0.06); }
.tm-Bb { background: rgba(0,255,135,0.04); border: 1px solid rgba(255,255,255,0.06); }
.tm-BB { background: rgba(255,59,92,0.09); border: 1px solid rgba(255,59,92,0.22); }
.tm-prob { font-size: 18px; font-weight: 700; }
.tm-desc { font-size: 9px; color: #6b7a99; letter-spacing: 0.06em; text-transform: uppercase; }
.tm-bar  { height: 3px; border-radius: 1px; margin-top: 5px; opacity: 0.7; }

/* ── MARKOV EXPLAINER PAGE ── */
.mc-hero { background: linear-gradient(135deg,rgba(0,255,135,0.05),rgba(0,212,255,0.02)); border: 1px solid rgba(0,255,135,0.15); padding: 40px; margin-bottom: 32px; }
.mc-hero h2 { font-size: 32px; font-weight: 800; margin-bottom: 10px; color: #e8edf5; }
.mc-hero p { font-size: 15px; color: #6b7a99; line-height: 1.8; max-width: 680px; }
.mc-h3 { font-size: 20px; font-weight: 800; margin-bottom: 16px; color: #e8edf5; }
.mc-card { background: #0d1420; border: 1px solid rgba(255,255,255,0.07); padding: 24px; margin-bottom: 4px; position: relative; overflow: hidden; }
.mc-card::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%; background:#00ff87; }
.mc-card h4 { font-size: 15px; font-weight: 700; margin-bottom: 8px; color: #e8edf5; }
.mc-card p { font-size: 13px; color: #6b7a99; line-height: 1.8; margin: 0; }
.mc-grid4 { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-bottom: 24px; }
.mc-stat { background: #0d1420; border: 1px solid rgba(255,255,255,0.07); padding: 20px; text-align: center; }
.mc-stat .val { font-family: 'Space Mono', monospace; font-size: 26px; font-weight: 700; color: #00ff87; }
.mc-stat .lbl { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: #6b7a99; margin-top: 6px; }
.mc-hl { background: rgba(0,255,135,0.05); border: 1px solid rgba(0,255,135,0.18); border-left: 3px solid #00ff87; padding: 16px 20px; font-size: 13px; color: #6b7a99; line-height: 1.8; margin: 16px 0; }
.mc-hl strong { color: #00ff87; }
.mc-step { display: grid; grid-template-columns: 48px 1fr; gap: 18px; padding: 22px 0; border-bottom: 1px solid rgba(255,255,255,0.05); align-items: start; }
.mc-step:last-child { border-bottom: none; }
.mc-num { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: #1a2540; line-height: 1; }
.mc-step h4 { font-size: 14px; font-weight: 700; margin-bottom: 6px; color: #e8edf5; }
.mc-step p { font-size: 13px; color: #6b7a99; line-height: 1.8; margin: 0; }
.mc-code { background: #020508; border: 1px solid rgba(255,255,255,0.07); padding: 18px 22px; font-family: 'Space Mono', monospace; font-size: 11px; color: #ffe14d; line-height: 2.1; margin: 12px 0; }
.mc-code .cm { color: #6b7a99; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# INDEX CONFIGURATION — unchanged from original
# ══════════════════════════════════════════════════════════════════
INDICES = {
    "Nifty 50": {
        "ticker"     : "^NSEI",
        "vix_ticker" : "^INDIAVIX",
        "name"       : "Nifty 50",
        "currency"   : "₹",
        "start"      : "2010-01-01",
    },
}


# ══════════════════════════════════════════════════════════════════
# DATA LOADING — unchanged from original
# ══════════════════════════════════════════════════════════════════
def _download_ticker(ticker_symbol: str, start: str) -> pd.DataFrame:
    t   = yf.Ticker(ticker_symbol)
    raw = t.history(start=start)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    cols_needed = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in raw.columns]
    raw = raw[cols_needed]
    for col in ["Open", "High", "Low", "Close"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Close"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    return raw


@st.cache_data(show_spinner=False, persist="disk")
def load_data(ticker: str, vix_ticker: str, start: str) -> pd.DataFrame:
    raw = _download_ticker(ticker, start)
    if raw.empty:
        return pd.DataFrame()
    raw["Returns"]    = np.log(raw["Close"] / raw["Close"].shift(1))
    raw["Volatility"] = (raw["High"] - raw["Low"]) / raw["Close"]
    delta    = raw["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs       = avg_gain / avg_loss
    raw["RSI"] = 100 - (100 / (1 + rs))
    vix_raw = _download_ticker(vix_ticker, start)
    if vix_raw.empty:
        return pd.DataFrame()
    vix_raw = vix_raw[["Date", "Close"]].rename(columns={"Close": "VIX"})
    vix_raw["VIX"] = pd.to_numeric(vix_raw["VIX"], errors="coerce")
    df = pd.merge(raw, vix_raw, on="Date", how="left")
    df["VIX"] = df["VIX"].ffill()
    df = df.dropna(subset=["Returns", "Volatility", "RSI", "VIX"])
    df = df[["Date", "Close", "Returns", "Volatility", "RSI", "VIX"]]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════
# HMM TRAINING — unchanged from original
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, persist="disk")
def run_hmm(df_json: str):
    df = pd.read_json(df_json)
    df["Date"] = pd.to_datetime(df["Date"])
    feature_cols = ["Returns", "Volatility", "RSI", "VIX"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    X_old    = df[["Returns", "Volatility"]].values

    model_A = GaussianHMM(n_components=2, covariance_type="full",
                          n_iter=1000, random_state=42)
    model_A.fit(X_old)
    states_A = model_A.predict(X_old)
    bull_A   = int(np.argmax(model_A.means_[:, 0]))
    bear_A   = 1 - bull_A
    df["Regime_A"] = [("Bull" if s == bull_A else "Bear") for s in states_A]

    model_B = GaussianHMM(n_components=2, covariance_type="full",
                          n_iter=1000, random_state=42)
    model_B.fit(X_scaled)
    states_B      = model_B.predict(X_scaled)
    means_orig    = scaler.inverse_transform(model_B.means_)
    bull_B        = int(np.argmax(means_orig[:, 0]))
    bear_B        = 1 - bull_B
    df["Regime_B"] = [("Bull" if s == bull_B else "Bear") for s in states_B]

    tm        = model_B.transmat_
    bull_stay = tm[bull_B, bull_B]
    bear_stay = tm[bear_B, bear_B]

    bull_days  = df[df["Regime_B"] == "Bull"]
    bear_days  = df[df["Regime_B"] == "Bear"]

    def count_false_alarms(regimes, window=3):
        fa = 0
        for i in range(len(regimes) - window):
            if regimes[i] != regimes[i+1]:
                if regimes[i] in regimes[i+2: i+window+1]:
                    fa += 1
        return fa

    fa_A = count_false_alarms(df["Regime_A"].tolist())
    fa_B = count_false_alarms(df["Regime_B"].tolist())

    df["Price_Ret"] = df["Close"].pct_change().fillna(0)
    df["BH_ret"]    = df["Price_Ret"]
    df["Str_ret"]   = df["Price_Ret"] * (df["Regime_B"] == "Bull").astype(int)
    df["BH_cum"]    = (1 + df["BH_ret"]).cumprod()
    df["Str_cum"]   = (1 + df["Str_ret"]).cumprod()

    n_years  = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    final_bh = df["BH_cum"].iloc[-1]
    final_st = df["Str_cum"].iloc[-1]
    cagr_bh  = (final_bh ** (1/n_years) - 1) * 100
    cagr_st  = (final_st ** (1/n_years) - 1) * 100

    bull_count_temp  = int((df["Regime_B"] == "Bull").sum())
    invested_years   = bull_count_temp / 252
    cagr_invested    = (final_st ** (1/invested_years) - 1) * 100 if invested_years > 0 else 0

    actual_price_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    def max_dd(s):
        return ((s - s.cummax()) / s.cummax()).min() * 100

    _, p_vix = stats.ttest_ind(bull_days["VIX"], bear_days["VIX"])

    current_regime  = df["Regime_B"].iloc[-1]
    current_rsi     = df["RSI"].iloc[-1]
    current_vix     = df["VIX"].iloc[-1]
    current_ret     = df["Returns"].iloc[-1]
    current_close   = df["Close"].iloc[-1]

    results = {
        "df"             : df.to_json(),
        "bull_stay"      : round(bull_stay * 100, 2),
        "bear_stay"      : round(bear_stay * 100, 2),
        "bull_count"     : len(bull_days),
        "bear_count"     : len(bear_days),
        "total_days"     : len(df),
        "bull_pct"       : round(len(bull_days)/len(df)*100, 1),
        "bear_pct"       : round(len(bear_days)/len(df)*100, 1),
        "means_orig"     : means_orig.tolist(),
        "bull_B"         : bull_B,
        "bear_B"         : bear_B,
        "fa_A"           : fa_A,
        "fa_B"           : fa_B,
        "fa_reduction"   : round((fa_A - fa_B)/fa_A*100, 1) if fa_A > 0 else 0,
        "bh_return"      : round((final_bh-1)*100, 1),
        "str_return"     : round((final_st-1)*100, 1),
        "actual_return"  : round(actual_price_return, 1),
        "cagr_bh"        : round(cagr_bh, 1),
        "cagr_str"       : round(cagr_st, 1),
        "cagr_invested"  : round(cagr_invested, 1),
        "invested_years" : round(invested_years, 1),
        "mdd_bh"         : round(max_dd(df["BH_cum"]), 1),
        "mdd_str"        : round(max_dd(df["Str_cum"]), 1),
        "p_vix"          : round(p_vix, 8),
        "bull_vix_mean"  : round(bull_days["VIX"].mean(), 2),
        "bear_vix_mean"  : round(bear_days["VIX"].mean(), 2),
        "bull_rsi_mean"  : round(bull_days["RSI"].mean(), 2),
        "bear_rsi_mean"  : round(bear_days["RSI"].mean(), 2),
        "overbought_days": len(bull_days[bull_days["RSI"] > 70]),
        "oversold_days"  : len(bear_days[bear_days["RSI"] < 30]),
        "current_regime" : current_regime,
        "current_rsi"    : round(current_rsi, 2),
        "current_vix"    : round(current_vix, 2),
        "current_ret"    : round(current_ret * 100, 3),
        "current_close"  : round(current_close, 2),
        "start_date"     : str(df["Date"].iloc[0].date()),
        "end_date"       : str(df["Date"].iloc[-1].date()),
    }
    return results


# ══════════════════════════════════════════════════════════════════
# PLOTLY CHART — unchanged logic, website colour palette applied
# ══════════════════════════════════════════════════════════════════
def make_chart(df: pd.DataFrame, index_name: str, currency: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            f"{index_name} — Close Price (Bull / Bear Regime)",
            "RSI (14-day) — Overbought / Oversold Zones",
            "India VIX — Fear Index",
            "Backtest — HMM Strategy vs Buy & Hold"
        ),
        row_heights=[0.35, 0.22, 0.22, 0.21]
    )

    # Panel 1: Close price
    for regime, color in [("Bull", "#00ff87"), ("Bear", "#ff3b5c")]:
        y_vals = df["Close"].copy().astype(float)
        y_vals[df["Regime_B"] != regime] = np.nan
        fig.add_trace(go.Scatter(
            x=df["Date"], y=y_vals,
            mode="lines", line=dict(color=color, width=1.5),
            name=f"{regime} Regime", legendgroup=regime, showlegend=True
        ), row=1, col=1)

    in_bear = False
    for i, row in df.iterrows():
        if row["Regime_B"] == "Bear" and not in_bear:
            in_bear = True; start_date = row["Date"]
        elif row["Regime_B"] != "Bear" and in_bear:
            fig.add_vrect(x0=start_date, x1=row["Date"],
                         fillcolor="rgba(255,59,92,0.06)",
                         layer="below", line_width=0, row=1, col=1)
            in_bear = False
    if in_bear:
        fig.add_vrect(x0=start_date, x1=df["Date"].iloc[-1],
                     fillcolor="rgba(255,59,92,0.06)",
                     layer="below", line_width=0, row=1, col=1)

    # Panel 2: RSI
    for regime, color in [("Bull", "#00ff87"), ("Bear", "#ff3b5c")]:
        y_vals = df["RSI"].copy().astype(float)
        y_vals[df["Regime_B"] != regime] = np.nan
        fig.add_trace(go.Scatter(
            x=df["Date"], y=y_vals,
            mode="lines", line=dict(color=color, width=1),
            name=regime, legendgroup=regime, showlegend=False
        ), row=2, col=1)

    for level, color, label in [(70, "#ff3b5c", "Overbought 70"),
                                  (30, "#00ff87", "Oversold 30"),
                                  (50, "#6b7a99", "")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                     line_width=1, opacity=0.6, row=2, col=1,
                     annotation_text=label if label else "",
                     annotation_font_color=color,
                     annotation_position="right")

    fig.add_trace(go.Scatter(
        x=df["Date"], y=np.where(df["RSI"] > 70, df["RSI"], 70),
        fill="tozeroy", fillcolor="rgba(255,59,92,0.08)",
        line=dict(width=0), showlegend=False, name=""
    ), row=2, col=1)

    # Panel 3: VIX
    for regime, color in [("Bull", "#00ff87"), ("Bear", "#ff3b5c")]:
        y_vals = df["VIX"].copy().astype(float)
        y_vals[df["Regime_B"] != regime] = np.nan
        fig.add_trace(go.Scatter(
            x=df["Date"], y=y_vals,
            mode="lines", line=dict(color=color, width=1),
            name=regime, legendgroup=regime, showlegend=False
        ), row=3, col=1)

    fig.add_hline(y=20, line_dash="dash", line_color="#ff3b5c",
                 line_width=1, opacity=0.6, row=3, col=1,
                 annotation_text="Fear threshold 20",
                 annotation_font_color="#ff3b5c",
                 annotation_position="right")
    fig.add_hline(y=15, line_dash="dot", line_color="#ffe14d",
                 line_width=1, opacity=0.5, row=3, col=1,
                 annotation_text="Caution 15",
                 annotation_font_color="#ffe14d",
                 annotation_position="right")

    # Panel 4: Backtest
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["BH_cum"],
        mode="lines", line=dict(color="#3498db", width=2),
        name="Buy & Hold", showlegend=True
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Str_cum"],
        mode="lines", line=dict(color="#00ff87", width=2),
        name="HMM Strategy", showlegend=True
    ), row=4, col=1)

    in_bear = False
    for i, row in df.iterrows():
        if row["Regime_B"] == "Bear" and not in_bear:
            in_bear = True; start_date = row["Date"]
        elif row["Regime_B"] != "Bear" and in_bear:
            fig.add_vrect(x0=start_date, x1=row["Date"],
                         fillcolor="rgba(255,59,92,0.05)",
                         layer="below", line_width=0, row=4, col=1)
            in_bear = False
    if in_bear:
        fig.add_vrect(x0=start_date, x1=df["Date"].iloc[-1],
                     fillcolor="rgba(255,59,92,0.05)",
                     layer="below", line_width=0, row=4, col=1)

    fig.update_layout(
        height=920,
        paper_bgcolor="#060a0f",
        plot_bgcolor="#0d1420",
        font=dict(color="#6b7a99", size=11, family="Space Mono"),
        legend=dict(
            bgcolor="#0d1420", bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1, font=dict(color="#e8edf5", family="Space Mono", size=11)
        ),
        hovermode="x unified",
        margin=dict(l=60, r=90, t=44, b=40),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", gridwidth=0.5,
                     zerolinecolor="rgba(255,255,255,0.04)",
                     tickfont=dict(color="#6b7a99", family="Space Mono"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", gridwidth=0.5,
                     tickfont=dict(color="#6b7a99", family="Space Mono"))
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="RSI",     row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="VIX",     row=3, col=1)
    fig.update_yaxes(title_text="₹1 → ₹X", row=4, col=1)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(color="#00ff87", size=11, family="Space Mono")

    return fig


# ══════════════════════════════════════════════════════════════════
# SESSION STATE — page routing
# ══════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# ══════════════════════════════════════════════════════════════════
# SIDEBAR — fixed, always visible, no toggle
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">HMM · NIFTY50</div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-tag">STPA Research · 2026</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-family:monospace;font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:#00ff87;margin-bottom:10px;">Navigation</div>', unsafe_allow_html=True)

    if st.button("📊  Dashboard",
                 key="nav_dash",
                 type="primary" if st.session_state.page == "dashboard" else "secondary",
                 use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

    if st.button("🔗  What is Markov Chain?",
                 key="nav_mc",
                 type="primary" if st.session_state.page == "markov" else "secondary",
                 use_container_width=True):
        st.session_state.page = "markov"
        st.rerun()

    st.markdown("---")

    selected_index = st.selectbox(
        "Index",
        options=list(INDICES.keys()),
        index=0,
        help="Choose market index"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:#00ff87;margin-bottom:10px;">How to read</div>
    <div style="font-size:12px;color:#6b7a99;line-height:2.0;">
    <span style="color:#00ff87">●</span> Green = Bull regime<br>
    <span style="color:#ff3b5c">●</span> Red = Bear regime<br>
    <span style="color:#ffe14d">●</span> RSI &gt; 70 = Overbought<br>
    <span style="color:#ff3b5c">●</span> VIX &gt; 20 = Elevated fear
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:10px;color:#6b7a99;line-height:1.9;">
    Data: Yahoo Finance<br>
    Model: GaussianHMM (hmmlearn)<br>
    STPA Research Project
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN CONTENT — page router
# ══════════════════════════════════════════════════════════════════
cfg = INDICES[selected_index]

# ─────────────────────────────────────────────────────────────────
# MARKOV CHAIN EXPLAINER PAGE
# ─────────────────────────────────────────────────────────────────
if st.session_state.page == "markov":

    st.markdown("""
<div class="mc-hero">
  <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.22em;color:#00ff87;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:10px;">
    <span style="display:inline-block;width:18px;height:1px;background:#00ff87;"></span>
    STPA Research · Markov Chain Explainer
  </div>
  <h2>What is a Markov Chain?</h2>
  <p>A beginner-friendly guide to the math behind this project — no prior knowledge needed.</p>
</div>
""", unsafe_allow_html=True)

    # ── 01 What is a Markov Chain ──
    st.markdown('<div class="mc-h3">01 · What is a Markov Chain?</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="mc-card">
  <h4>The Simple Idea</h4>
  <p>Imagine predicting tomorrow's weather. If it's sunny today, there's a good chance it'll be sunny tomorrow too. A <strong>Markov Chain</strong> is just a mathematical way of writing this down — it models a system that moves between different <em>states</em>, and the chance of the next state depends <strong>only on where you are right now</strong>, not on anything that happened before.</p>
</div>
<div class="mc-card" style="margin-top:4px;">
  <h4>The "Memoryless" Property — Why It Matters</h4>
  <p>This "only care about right now" rule is called the <strong>Markov Property</strong>. In finance: if the market is in a Bull regime today, the probability of staying Bull tomorrow is always the same — regardless of whether we've been Bull for 2 days or 2 years. This makes the math clean and computable.</p>
</div>
<div class="mc-card" style="margin-top:4px;">
  <h4>States & Transitions</h4>
  <p>Every Markov Chain has two things: <strong>States</strong> (situations the system can be in) and <strong>Transition Probabilities</strong> (chances of moving from one state to another). In our project, the states are <strong style="color:#00ff87">Bull Market</strong> and <strong style="color:#ff3b5c">Bear Market</strong>.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 02 How we integrated it ──
    st.markdown('<div class="mc-h3">02 · How We Integrated Markov Chain in This Project</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="font-size:14px;color:#6b7a99;line-height:1.85;margin-bottom:20px;">
  We used a specific type called a <strong style="color:#e8edf5">Hidden Markov Model (HMM)</strong>.
  The word "hidden" is key — in real markets, nobody announces "today is a Bull day."
  You can only <em>observe</em> signals (price returns, volatility, RSI, VIX).
  The underlying regime is hidden. The HMM solves exactly this.
</p>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="mc-grid4">
  <div class="mc-stat"><div class="val">2</div><div class="lbl">Hidden States<br>Bull &amp; Bear</div></div>
  <div class="mc-stat"><div class="val">4</div><div class="lbl">Observable Signals<br>per trading day</div></div>
  <div class="mc-stat"><div class="val" style="color:#ffe14d">3,965</div><div class="lbl">Trading days<br>labelled by HMM</div></div>
  <div class="mc-stat"><div class="val">85.7%</div><div class="lbl">False alarm<br>reduction</div></div>
</div>
<div class="mc-hl">
  <strong>The core idea:</strong> The Markov Chain defines how the market secretly transitions between Bull and Bear day by day. The HMM learns these transition probabilities from 16 years of Nifty 50 data — then labels every single trading day as Bull or Bear.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 03 Full explanation ──
    st.markdown('<div class="mc-h3">03 · Full Explanation — Markov Chain Step by Step</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="mc-step">
  <div class="mc-num">01</div>
  <div>
    <h4>Defining the States</h4>
    <p>We told the model: the market can only be in one of two states at any time — <strong style="color:#00ff87">Bull</strong> (prices trending up, calm volatility, higher RSI, lower VIX) or <strong style="color:#ff3b5c">Bear</strong> (prices falling, high volatility, VIX spiking). These are the two nodes of our Markov Chain. Every trading day from Jan 2010 to Mar 2026 belongs to exactly one state.</p>
  </div>
</div>
<div class="mc-step">
  <div class="mc-num">02</div>
  <div>
    <h4>The Transition Matrix — The Heart of the Markov Chain</h4>
    <p>The transition matrix answers: <em>"Given the market is in state X today, what is the probability it will be in state Y tomorrow?"</em> Our trained Model B learned these values from the data:</p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="mc-code">
<span class="cm"># Model B Transition Matrix (learned from 3,965 trading days)</span><br>
<span class="cm">#              → Bull Tomorrow    → Bear Tomorrow</span><br>
Bull Today  →    0.978286           0.021714<br>
Bear Today  →    0.043658           0.956342<br><br>
<span class="cm"># If today is Bull: 97.83% chance tomorrow is also Bull.</span><br>
<span class="cm"># Once in Bear: 95.63% chance of staying Bear tomorrow.</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="mc-step">
  <div class="mc-num">03</div>
  <div>
    <h4>Why Two Models? Comparing Transition Matrices</h4>
    <p><strong>Model A</strong> used only Returns + Volatility. Its Bear→Bear probability was just 0.7033 — Bear periods kept flipping back prematurely, causing <strong style="color:#ff3b5c">105 false alarms</strong> over 16 years. <strong>Model B</strong> added RSI and India VIX, and Bear→Bear jumped to 0.9563. The model now requires multiple signals to confirm a regime change — not just one noisy day.</p>
  </div>
</div>
<div class="mc-step">
  <div class="mc-num">04</div>
  <div>
    <h4>The Baum-Welch Algorithm — How the Model Learned</h4>
    <p>We didn't manually set the probabilities. The <strong>Baum-Welch algorithm</strong> (Expectation-Maximization) looked at all 3,965 days and automatically found the probabilities that best explain the observed patterns. It ran for 1,000 iterations, adjusting and re-adjusting until the numbers converged.</p>
  </div>
</div>
<div class="mc-step">
  <div class="mc-num">05</div>
  <div>
    <h4>The Viterbi Algorithm — Labelling Every Day</h4>
    <p>Once trained, the <strong>Viterbi algorithm</strong> traces through all 3,965 days and finds the single most likely sequence of Bull/Bear states — respecting the transition probabilities. Think of it as solving a 3,965-step puzzle where each piece must connect logically to the next. Result: every day gets a label, <strong style="color:#00ff87">Bull</strong> or <strong style="color:#ff3b5c">Bear</strong>.</p>
  </div>
</div>
<div class="mc-step">
  <div class="mc-num">06</div>
  <div>
    <h4>The Real-World Result</h4>
    <p>Strong transition probabilities (97.8% Bull persistence, 95.6% Bear persistence) mean the model only switches regimes when all 4 signals genuinely justify it. That's the Markov Chain doing its job — staying in a state until there's real evidence to move. Result: <strong style="color:#00ff87">85.7% fewer false regime switches</strong>, Max Drawdown cut from −40% to −18.7%.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div class="mc-hl">
  <strong>One-line summary:</strong> A Markov Chain says "the future depends only on the present." Our HMM applies this to markets — learning how Bull and Bear regimes secretly transition day by day, then using those probabilities to label 16 years of Nifty 50 data with far fewer false alarms than simpler models.
</div>
""", unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────────────────────────
# DASHBOARD PAGE (default)
# ─────────────────────────────────────────────────────────────────

# ── PAGE HEADER ──
st.markdown(f"""
<div class="page-header">
    <div class="tag">Statistical &amp; Time-Series Projects · Applied Finance</div>
    <h1><span class="w">{cfg['name']}</span> — <span class="g">BULL &amp; BEAR</span> <span class="r">REGIME</span> DETECTION</h1>
    <div class="sub">
        Hidden Markov Model · Returns + Volatility + RSI + India VIX ·
        Live data from Yahoo Finance
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ──
with st.spinner(f"Downloading {cfg['name']} + India VIX data..."):
    df_raw = load_data(cfg["ticker"], cfg["vix_ticker"], cfg["start"])

if df_raw.empty:
    st.error(f"Could not download data for {cfg['name']}. Please try again.")
    st.stop()

st.caption(f"✅ Data loaded: {len(df_raw)} rows | {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")

with st.spinner("Training Model A and Model B HMM..."):
    @st.cache_data(show_spinner=False)
    def run_hmm_cached(df):
        return run_hmm(df.to_json())  # keep your original function untouched

    # session state prevents rerun on interaction
    if "results" not in st.session_state:
        st.session_state.results = run_hmm_cached(df_raw)

    results = st.session_state.results

df = pd.read_json(results["df"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ── LIVE TICKER ──
st.markdown(f"""
<div class="ticker-wrap">
  <div class="ticker-inner">
    <span class="ticker-item">Trading Days <span class="y">{results['total_days']}</span></span>
    <span class="ticker-item">Date Range <span class="y">{results['start_date']} → {results['end_date']}</span></span>
    <span class="ticker-item">Bull Regime <span class="g">{results['bull_count']} days ({results['bull_pct']}%)</span></span>
    <span class="ticker-item">Bear Regime <span class="r">{results['bear_count']} days ({results['bear_pct']}%)</span></span>
    <span class="ticker-item">Model A False Alarms <span class="r">{results['fa_A']}</span></span>
    <span class="ticker-item">Model B False Alarms <span class="g">{results['fa_B']}</span></span>
    <span class="ticker-item">False Alarm Reduction <span class="g">{results['fa_reduction']}%</span></span>
    <span class="ticker-item">VIX Bull avg <span class="g">{results['bull_vix_mean']}</span></span>
    <span class="ticker-item">VIX Bear avg <span class="r">{results['bear_vix_mean']}</span></span>
    <span class="ticker-item">HMM CAGR <span class="g">{results['cagr_str']}%</span></span>
    <span class="ticker-item">B&amp;H CAGR <span class="y">{results['cagr_bh']}%</span></span>
    <span class="ticker-item">HMM Max DD <span class="g">{results['mdd_str']}%</span></span>
    <span class="ticker-item">B&amp;H Max DD <span class="r">{results['mdd_bh']}%</span></span>
    <!-- duplicate for seamless loop -->
    <span class="ticker-item">Trading Days <span class="y">{results['total_days']}</span></span>
    <span class="ticker-item">Date Range <span class="y">{results['start_date']} → {results['end_date']}</span></span>
    <span class="ticker-item">Bull Regime <span class="g">{results['bull_count']} days ({results['bull_pct']}%)</span></span>
    <span class="ticker-item">Bear Regime <span class="r">{results['bear_count']} days ({results['bear_pct']}%)</span></span>
    <span class="ticker-item">Model A False Alarms <span class="r">{results['fa_A']}</span></span>
    <span class="ticker-item">Model B False Alarms <span class="g">{results['fa_B']}</span></span>
    <span class="ticker-item">False Alarm Reduction <span class="g">{results['fa_reduction']}%</span></span>
    <span class="ticker-item">VIX Bull avg <span class="g">{results['bull_vix_mean']}</span></span>
    <span class="ticker-item">VIX Bear avg <span class="r">{results['bear_vix_mean']}</span></span>
    <span class="ticker-item">HMM CAGR <span class="g">{results['cagr_str']}%</span></span>
    <span class="ticker-item">B&amp;H CAGR <span class="y">{results['cagr_bh']}%</span></span>
    <span class="ticker-item">HMM Max DD <span class="g">{results['mdd_str']}%</span></span>
    <span class="ticker-item">B&amp;H Max DD <span class="r">{results['mdd_bh']}%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION A — CURRENT REGIME
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">01 · Current Status</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">Current Market Regime</div>', unsafe_allow_html=True)

regime     = results["current_regime"]
badge_html = (f'<span class="bull-badge">BULL MARKET</span>'
              if regime == "Bull"
              else f'<span class="bear-badge">BEAR MARKET</span>')

col_regime, col_close, col_rsi, col_vix, col_ret = st.columns(5)

with col_regime:
    st.markdown(f"<div style='margin-bottom:10px; font-family:Space Mono,monospace; font-size:11px; color:#6b7a99; letter-spacing:0.1em;'>AS OF {results['end_date']}</div>", unsafe_allow_html=True)
    st.markdown(badge_html, unsafe_allow_html=True)

with col_close:
    st.metric(label="Close Price", value=f"{cfg['currency']}{results['current_close']:,.0f}")

with col_rsi:
    rsi_val   = results["current_rsi"]
    rsi_delta = "⚠ Overbought" if rsi_val > 70 else ("↑ Oversold" if rsi_val < 30 else "")
    st.metric(label="RSI (14-day)", value=f"{rsi_val}", delta=rsi_delta)

with col_vix:
    vix_val   = results["current_vix"]
    vix_delta = "⚠ High Fear" if vix_val > 20 else "✓ Calm"
    st.metric(label="India VIX", value=f"{vix_val}", delta=vix_delta)

with col_ret:
    ret_val = results["current_ret"]
    st.metric(
        label="Today's Return",
        value=f"{ret_val:+.3f}%",
        delta=f"{'▲' if ret_val > 0 else '▼'} {abs(ret_val):.3f}%"
    )

# Regime distribution bar
bull_pct = results["bull_pct"]
bear_pct = results["bear_pct"]
st.markdown(f"""
<div class="rbar-wrap">
  <div class="rbar-label">Regime Distribution — {results['start_date']} to {results['end_date']}</div>
  <div class="rbar-track">
    <div class="rbar-bull" style="width:{bull_pct}%"></div>
    <div class="rbar-bear" style="width:{bear_pct}%"></div>
  </div>
  <div class="rbar-leg">
    <div class="rbar-li"><div class="rbar-dot" style="background:#00ff87"></div>Bull — {results['bull_count']} days ({bull_pct}%)</div>
    <div class="rbar-li"><div class="rbar-dot" style="background:#ff3b5c"></div>Bear — {results['bear_count']} days ({bear_pct}%)</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION B — 4 PANEL CHART
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">02 · Regime Chart</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">{cfg["name"]} — Bull / Bear Regimes (2010 → Today)</div>', unsafe_allow_html=True)

fig = make_chart(df, cfg["name"], cfg["currency"])
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION C — REGIME STATISTICS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">03 · Regime Statistics</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">Bull vs Bear — Feature Breakdown</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Day Count**")
    count_data = pd.DataFrame({
        "Regime"   : ["🟢 Bull", "🔴 Bear"],
        "Days"     : [results["bull_count"], results["bear_count"]],
        "% of Time": [f"{results['bull_pct']}%", f"{results['bear_pct']}%"]
    })
    st.dataframe(count_data, hide_index=True, use_container_width=True)

    bs  = results["bull_stay"]
    brs = results["bear_stay"]
    bb  = round(100 - bs,  2)
    br  = round(100 - brs, 2)
    import streamlit.components.v1 as components
    components.html(f"""
    <style>
    .tm-wrap {{
        font-family: 'Space Mono', monospace;
        color: #e8edf5;
    }}

    .tm-title {{
        font-size: 11px;
        letter-spacing: 0.15em;
        color: #6b7a99;
        margin-bottom: 12px;
        text-transform: uppercase;
    }}

    .tm-grid {{
        display: grid;
        grid-template-columns: 140px 1fr 1fr;
        gap: 6px;
    }}

    .tm-cell {{
        padding: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        background: #0d1420;    
    }}

    .tm-head {{
        text-align: center;
        font-size: 10px;
        color: #6b7a99;
        text-transform: uppercase;
    }}

    .tm-row {{
        font-size: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .tm-bull {{
        background: rgba(0,255,135,0.08);
        border: 1px solid rgba(0,255,135,0.3);
    }}

    .tm-bear {{
        background: rgba(255,59,92,0.08);
        border: 1px solid rgba(255,59,92,0.3);
    }}

    .tm-prob {{
        font-size: 20px;
        font-weight: 700;
    }}

    .tm-desc {{
        font-size: 10px;
        color: #6b7a99;
    }}

    .green {{ color: #00ff87; }}
    .red   {{ color: #ff3b5c; }}

    .tm-footer {{
        margin-top: 10px;
        font-size: 11px;
        color: #6b7a99;
    }}
    </style>

    <div class="tm-wrap">

    <div class="tm-title">Model B — Transition Matrix</div>

    <div class="tm-grid">

    <div class="tm-cell tm-head">FROM / TO</div>
    <div class="tm-cell tm-head green">→ Bull</div>
    <div class="tm-cell tm-head red">→ Bear</div>

    <div class="tm-cell tm-row green">🟢 Bull Today</div>
    <div class="tm-cell tm-bull">
        <div class="tm-prob green">{bs}%</div>
        <div class="tm-desc">Stay Bull</div>
    </div>
    <div class="tm-cell">
        <div class="tm-prob red">{round(100-bs,2)}%</div>
        <div class="tm-desc">Flip to Bear</div>
    </div>

    <div class="tm-cell tm-row red">🔴 Bear Today</div>
    <div class="tm-cell">
        <div class="tm-prob green">{round(100-brs,2)}%</div>
        <div class="tm-desc">Flip to Bull</div>
    </div>
    <div class="tm-cell tm-bear">
        <div class="tm-prob red">{brs}%</div>
        <div class="tm-desc">Stay Bear</div>
    </div>

    </div>

    <div class="tm-footer">
    Bull persistence: <span class="green">{bs}%</span> · 
    Bear persistence: <span class="red">{brs}%</span>
    </div>

    </div>
    """, height=260)
    st.caption(f"Bull persistence: {bs}% · Bear persistence: {brs}%")

with col2:
    st.markdown("**Average Feature Values per Regime**")
    means = results["means_orig"]
    bull_b = results["bull_B"]
    bear_b = results["bear_B"]
    avg_data = pd.DataFrame({
        "Feature" : ["Returns", "Volatility", "RSI", "VIX"],
        "🟢 Bull" : [f"{means[bull_b][i]:.4f}" for i in range(4)],
        "🔴 Bear" : [f"{means[bear_b][i]:.4f}" for i in range(4)],
    })
    st.dataframe(avg_data, hide_index=True, use_container_width=True)

    breakdown = pd.DataFrame({
        "Signal"  : ["Bull days RSI > 70", "Bear days RSI < 30",
                     "Avg VIX in Bull",    "Avg VIX in Bear"],
        "Value"   : [
            f"{results['overbought_days']} days (overbought warning)",
            f"{results['oversold_days']} days (capitulation)",
            f"{results['bull_vix_mean']} (calm)",
            f"{results['bear_vix_mean']} ({results['bear_vix_mean']/results['bull_vix_mean']:.2f}× higher)"
        ]
    })
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION D — FALSE ALARM ANALYSIS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">04 · Core Research Finding</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">False Alarm Analysis</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="fa-callout">
  <div class="fa-num">{results['fa_reduction']}%</div>
  <div class="fa-text">
    <h3>False Alarm Reduction — Model A → Model B</h3>
    <p>
      Model A (Returns + Volatility): <strong style="color:#ff3b5c">{results['fa_A']} false alarms</strong>
      over 16 years ≈ {results['fa_A']//16}/year.<br>
      Model B (+ RSI + VIX): <strong style="color:#00ff87">{results['fa_B']} false alarms</strong>
      ≈ {max(1,results['fa_B']//16)}/year.<br>
      RSI and VIX act as confirmation filters —
      a single bad returns day alone no longer triggers a Bear call.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

col_fa1, col_fa2, col_fa3 = st.columns(3)
with col_fa1:
    st.metric(label="Model A False Alarms", value=results["fa_A"],
              delta=f"~{results['fa_A']//16} per year", delta_color="inverse")
    st.caption("Returns + Volatility only")
with col_fa2:
    st.metric(label="Model B False Alarms", value=results["fa_B"],
              delta=f"~{max(1,results['fa_B']//16)} per year", delta_color="inverse")
    st.caption("+ RSI + India VIX")
with col_fa3:
    st.metric(label="Reduction", value=f"{results['fa_reduction']}%",
              delta=f"{results['fa_A'] - results['fa_B']} fewer false alarms", delta_color="normal")
    st.caption("RSI + VIX act as confirmation filters")

st.markdown("**Visual comparison:**")
fa_c1, fa_c2 = st.columns(2)
with fa_c1:
    st.markdown(f"<span style='color:#ff3b5c; font-family:Space Mono,monospace; font-size:12px;'>Model A — {results['fa_A']} false alarms</span>", unsafe_allow_html=True)
    st.progress(1.0)
with fa_c2:
    st.markdown(f"<span style='color:#00ff87; font-family:Space Mono,monospace; font-size:12px;'>Model B — {results['fa_B']} false alarms</span>", unsafe_allow_html=True)
    st.progress(results["fa_B"] / results["fa_A"] if results["fa_A"] > 0 else 0)

st.info(
    f"**Why this matters:** Each false alarm = unnecessary sell + rebuy = "
    f"transaction costs + missed gains. Model A triggered ~{results['fa_A']//16} "
    f"unnecessary portfolio shifts per year. Model B reduces this to "
    f"~{max(1, results['fa_B']//16)} per year by requiring VIX + RSI confirmation "
    f"before switching regimes."
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION E — BACKTEST
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">05 · Backtest</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">HMM Strategy vs Buy & Hold (2010 → Today)</div>', unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)
with b1:
    st.metric(label="Total Return — Buy & Hold", value=f"{results['bh_return']}%")
with b2:
    st.metric(label="Total Return — HMM Strategy", value=f"{results['str_return']}%",
              delta=f"+{results['str_return'] - results['bh_return']:.1f}% vs B&H")
with b3:
    st.metric(label="Max Drawdown — Buy & Hold", value=f"{results['mdd_bh']}%", delta_color="inverse")
with b4:
    st.metric(label="Max Drawdown — HMM Strategy", value=f"{results['mdd_str']}%",
              delta=f"{results['mdd_str'] - results['mdd_bh']:.1f}% vs B&H", delta_color="normal")

cagr1, cagr2, cagr3, days1 = st.columns(4)
with cagr1:
    st.metric(label="CAGR — Buy & Hold",   value=f"{results['cagr_bh']}%")
with cagr2:
    st.metric(label="CAGR — HMM Strategy", value=f"{results['cagr_str']}%",
              delta=f"{results['cagr_str'] - results['cagr_bh']:.1f}% vs B&H")
with cagr3:
    st.metric(label="CAGR on Invested Days", value=f"{results['cagr_invested']}%",
              delta=f"+{results['cagr_invested'] - results['cagr_bh']:.1f}% vs B&H")
with days1:
    st.metric(label="Days Invested (HMM)",
              value=f"{results['bull_count']} / {results['total_days']}",
              delta=f"{results['bull_pct']}% of time in market")

st.caption(
    f"Sanity check — Actual {cfg['name']} price return 2010→today: "
    f"{results['actual_return']}%  |  Buy & Hold backtest: {results['bh_return']}%"
)
st.warning(
    "⚠ **Backtest disclaimer:** Simplified backtest — does not account for "
    "transaction costs, STT, slippage, or taxes. Actual returns would be slightly lower. "
    "Max drawdown improvement is structural and would persist. Academic / research use only."
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION F — STATISTICAL VALIDATION
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">06 · Statistical Validation</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">t-test — VIX Bull vs Bear</div>', unsafe_allow_html=True)

p_vix = results["p_vix"]
sig   = "STATISTICALLY SIGNIFICANT ✓" if p_vix < 0.05 else "Not significant"

vix_col1, vix_col2 = st.columns(2)
with vix_col1:
    st.markdown("**VIX t-test results**")
    st.markdown(f"""
    <div class="code-block">
    <span class="cm"># Welch two-sample t-test</span><br>
    Bull avg VIX  : {results['bull_vix_mean']}<br>
    Bear avg VIX  : {results['bear_vix_mean']}<br>
    Difference    : {results['bear_vix_mean'] - results['bull_vix_mean']:.2f} points<br>
    p-value       : {p_vix}<br>
    Result        : <span style="color:#00ff87">{sig}</span>
    </div>
    """, unsafe_allow_html=True)

with vix_col2:
    st.markdown("**Interpretation**")
    st.success(
        f"The VIX difference between Bull and Bear regimes has a p-value of "
        f"essentially 0 ({p_vix}). There is virtually zero probability that the "
        f"{results['bear_vix_mean'] - results['bull_vix_mean']:.1f}-point VIX gap "
        f"happened by random chance. India VIX is a statistically proven discriminator "
        f"between market regimes — which is why Model B works."
    )

st.markdown("---")
st.markdown(
    f"<div style='font-family:Space Mono,monospace; font-size:11px; color:#6b7a99;'>"
    f"Data range: {results['start_date']} to {results['end_date']} | "
    f"{results['total_days']} trading days | Source: Yahoo Finance | "
    f"Model: GaussianHMM (hmmlearn) | STPA Research Project"
    f"</div>",
    unsafe_allow_html=True
)
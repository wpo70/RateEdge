# app_streamlit.py - RateEdge IRS Pricer v2
# Comprehensive AUD IRD Analytics with Spreads, Butterflies, BOB, XCcy Basis

import math
from datetime import date, timedelta
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================
# Constants & Market Data
# ============================

SUPPORTED_CURRENCIES = ["AUD", "NZD", "USD"]
IRS_TENORS_SHORT = ["3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
EXPIRIES = ["Spot", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
FWD_TENORS = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y"]

# Generate all spread combinations
def generate_all_spreads():
    tenors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    spreads = []
    for i, t1 in enumerate(tenors):
        for t2 in tenors[i+1:]:
            spreads.append((f"{t1}Y", f"{t2}Y", f"{t1}s{t2}s"))
    return spreads

ALL_SPREADS = generate_all_spreads()

# Generate all butterfly combinations
def generate_all_butterflies():
    tenors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
    butterflies = []
    # Sequential: 1/2/3, 2/3/4, etc
    for i in range(len(tenors) - 2):
        t1, t2, t3 = tenors[i], tenors[i+1], tenors[i+2]
        butterflies.append((f"{t1}Y", f"{t2}Y", f"{t3}Y", f"{t1}s{t2}s{t3}s"))
    # Standard market butterflies
    standard = [(2, 5, 10), (2, 10, 30), (5, 10, 30), (3, 5, 7), (5, 7, 10), (2, 3, 5), (3, 5, 10), (5, 10, 20), (10, 15, 20), (10, 20, 30)]
    for t1, t2, t3 in standard:
        name = f"{t1}s{t2}s{t3}s"
        entry = (f"{t1}Y", f"{t2}Y", f"{t3}Y", name)
        if entry not in butterflies:
            butterflies.append(entry)
    return butterflies

ALL_BUTTERFLIES = generate_all_butterflies()

# Default AUD IRS rates from the image
DEFAULT_AUD_IRS = {
    "3M": 3.715, "6M": 3.70, "9M": 3.8325, "1Y": 3.905, "2Y": 3.54, "3Y": 4.0047, "4Y": 4.15, "5Y": 4.48,
    "6Y": 4.5487, "7Y": 4.6737, "8Y": 4.7325, "9Y": 4.7825, "10Y": 4.8262, "12Y": 4.893, "15Y": 4.98248,
    "20Y": 5.0375, "25Y": 5.0748, "30Y": 4.9422, "40Y": 4.9119
}

DEFAULT_AUD_6V3_BASIS = {
    "1Y": 1.625, "2Y": 1.625, "3Y": 1.375, "4Y": 1.625, "5Y": 1.625, "7Y": 1.875,
    "10Y": 2.875, "12Y": 3.125, "15Y": 3.625, "20Y": 4.125, "30Y": 4.125
}

DEFAULT_AUD_3V1_BASIS = {
    "1Y": 0.125, "2Y": 0.125, "3Y": 0.125, "4Y": 0.125, "5Y": 0.125, "7Y": 0.125,
    "10Y": 0.125, "12Y": 0, "15Y": 0, "20Y": 0, "30Y": 0
}

DEFAULT_USD_SOFR = {
    "3M": 4.35, "6M": 4.30, "1Y": 4.10, "2Y": 3.95, "3Y": 3.85, "4Y": 3.80, "5Y": 3.78,
    "7Y": 3.80, "10Y": 3.85, "12Y": 3.88, "15Y": 3.92, "20Y": 3.95, "30Y": 3.98
}

DEFAULT_NZD_IRS = {
    "3M": 4.20, "6M": 4.15, "1Y": 4.05, "2Y": 3.90, "3Y": 3.85, "4Y": 3.82, "5Y": 3.80,
    "7Y": 3.82, "10Y": 3.90, "12Y": 3.95, "15Y": 4.00, "20Y": 4.05, "30Y": 4.10
}

DEFAULT_XCCY_BASIS = {
    "3M": -0.375, "6M": 0.625, "1Y": 0.125, "2Y": 0.5, "3Y": 0.875, "5Y": 1.125,
    "7Y": 1.625, "10Y": 2.125, "12Y": 2.5, "15Y": 2.875, "20Y": 3.125, "30Y": 3.375
}

DEFAULT_BOB_SPREAD = {
    "1Y": 8.75, "2Y": 9.5, "3Y": 11.375, "5Y": 11.625, "7Y": 10, "10Y": 8, "15Y": 8.375, "20Y": 7.75
}

CCY_CONVENTIONS = {
    "AUD": {"float_index": "BBSW3M", "fixed_freq": 6, "float_freq": 3, "day_count": "ACT/365F", "description": "BBSW 3M, Q/S"},
    "NZD": {"float_index": "BKBM3M", "fixed_freq": 6, "float_freq": 3, "day_count": "ACT/365", "description": "BKBM 3M, Q/S"},
    "USD": {"float_index": "SOFR", "fixed_freq": 6, "float_freq": 3, "day_count": "ACT/360", "description": "SOFR, S/S"},
}

# ============================
# Helper Functions
# ============================

def label_to_years(lbl: str) -> float:
    x = str(lbl).strip().lower()
    if x == "spot": return 0.0
    if x.endswith("d"): return float(x[:-1]) / 365.0
    if x.endswith("w"): return float(x[:-1]) / 52.0
    if x.endswith("m"): return float(x[:-1]) / 12.0
    if x.endswith("y"): return float(x[:-1])
    try: return float(x)
    except: return 0.0

def tenor_label_to_years(tenor: str) -> float:
    s = str(tenor).strip().upper()
    if s.endswith("Y"): return float(s[:-1])
    if s.endswith("M"): return float(s[:-1]) / 12.0
    return float(s)

def interpolate_rate(rates_dict: Dict[str, float], tenor_years: float) -> float:
    tenors_y, rates = [], []
    for t, r in sorted(rates_dict.items(), key=lambda x: tenor_label_to_years(x[0])):
        tenors_y.append(tenor_label_to_years(t))
        rates.append(r)
    tenors_y, rates = np.array(tenors_y), np.array(rates)
    if tenor_years <= tenors_y[0]: return float(rates[0])
    if tenor_years >= tenors_y[-1]: return float(rates[-1])
    return float(np.interp(tenor_years, tenors_y, rates))

def calculate_forward_rate(rates_dict: Dict[str, float], expiry_years: float, tenor_years: float) -> float:
    if expiry_years <= 0: return interpolate_rate(rates_dict, tenor_years)
    start_t, end_t = expiry_years, expiry_years + tenor_years
    r_start, r_end = interpolate_rate(rates_dict, start_t) / 100.0, interpolate_rate(rates_dict, end_t) / 100.0
    df_start = math.exp(-r_start * start_t) if start_t > 0 else 1.0
    df_end = math.exp(-r_end * end_t)
    freq, annuity, t = 0.5, 0.0, start_t + 0.5
    while t <= end_t + 0.001:
        r_t = interpolate_rate(rates_dict, t) / 100.0
        annuity += freq * math.exp(-r_t * t)
        t += freq
    if annuity <= 0: annuity = tenor_years * df_end
    fwd = (df_start - df_end) / annuity if annuity > 0 else r_end
    return max(fwd * 100, 0.0)

def calculate_spread(r1: float, r2: float) -> float:
    return (r2 - r1) * 100

def calculate_butterfly(r1: float, r2: float, r3: float, t1: float, t2: float, t3: float) -> float:
    w1, w3 = (t3 - t2) / (t3 - t1), (t2 - t1) / (t3 - t1)
    return (w1 * r1 - r2 + w3 * r3) * 100

def calculate_dv01(notional: float, tenor_years: float) -> float:
    return notional * tenor_years * 0.95 * 0.0001

# ============================
# Session State
# ============================

def init_session():
    defaults = {
        "theme_name": "Dealer Dark", "aud_irs": DEFAULT_AUD_IRS.copy(), "usd_sofr": DEFAULT_USD_SOFR.copy(),
        "nzd_irs": DEFAULT_NZD_IRS.copy(), "aud_6v3": DEFAULT_AUD_6V3_BASIS.copy(), "aud_3v1": DEFAULT_AUD_3V1_BASIS.copy(),
        "xccy_basis": DEFAULT_XCCY_BASIS.copy(), "bob_spread": DEFAULT_BOB_SPREAD.copy(),
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

def get_rates(ccy: str) -> Dict[str, float]:
    if ccy == "AUD": return st.session_state.get("aud_irs", DEFAULT_AUD_IRS)
    elif ccy == "USD": return st.session_state.get("usd_sofr", DEFAULT_USD_SOFR)
    elif ccy == "NZD": return st.session_state.get("nzd_irs", DEFAULT_NZD_IRS)
    return DEFAULT_AUD_IRS

# ============================
# Theme
# ============================

def apply_rateedge_theme(theme_name: str):
    is_dark = theme_name == "Dealer Dark"
    bg, card, border = ("#0f172a", "#1e293b", "#334155") if is_dark else ("#f1f5f9", "#ffffff", "#e2e8f0")
    text, accent, accent2 = ("#f1f5f9", "#ef4444", "#3b82f6") if is_dark else ("#1e3a5f", "#dc2626", "#2563eb")
    muted, positive, negative = ("#94a3b8", "#22c55e", "#ef4444") if is_dark else ("#64748b", "#16a34a", "#dc2626")
    
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background-color: {card}; border-right: 1px solid {border}; }}
    [data-testid="stSidebar"] label {{ color: {text} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background-color: {card}; padding: 0.5rem; border-radius: 8px; border: 1px solid {border}; }}
    .stTabs [data-baseweb="tab"] {{ color: {text} !important; background-color: transparent; border-radius: 6px; padding: 0.4rem 0.6rem; font-weight: 500; font-size: 0.75rem; }}
    .stTabs [aria-selected="true"] {{ background-color: {accent} !important; color: white !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent !important; }}
    label, .stMarkdown, h1, h2, h3, h4 {{ color: {text} !important; }}
    .stButton>button {{ background-color: {accent}; color: white; border-radius: 8px; border: 1px solid {accent}; font-weight: 600; }}
    .stButton>button:hover {{ background-color: {accent2}; border-color: {accent2}; }}
    [data-testid="stMetricValue"] {{ color: {text} !important; }}
    [data-testid="stMetricLabel"] {{ color: {muted} !important; }}
    .spread-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 0.6rem; text-align: center; margin: 0.2rem; }}
    .spread-label {{ font-size: 0.65rem; color: {muted}; font-weight: 500; }}
    .spread-value {{ font-size: 1.2rem; font-weight: 700; }}
    .spread-positive {{ color: {positive}; }}
    .spread-negative {{ color: {negative}; }}
    </style>
    """, unsafe_allow_html=True)

def show_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="font-size:1.4rem;font-weight:650;">{title}
        <span style="display:inline-block;padding:0.1rem 0.4rem;font-size:0.65rem;border-radius:999px;border:1px solid #ef4444;color:#ef4444;margin-left:0.4rem;">RateEdge</span>
    </div>
    <div style="font-size:0.8rem;color:#94a3b8;">{subtitle}</div>
    """, unsafe_allow_html=True)

# ============================
# Matrix Generation
# ============================

def generate_forward_matrix(ccy: str) -> pd.DataFrame:
    rates = get_rates(ccy)
    matrix = []
    for exp in EXPIRIES:
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        for tenor in FWD_TENORS:
            tenor_y = tenor_label_to_years(tenor)
            row[tenor] = calculate_forward_rate(rates, exp_y, tenor_y)
        matrix.append(row)
    return pd.DataFrame(matrix).set_index("Expiry")

def generate_spread_matrix(ccy: str, spread_list: List[Tuple]) -> pd.DataFrame:
    rates = get_rates(ccy)
    matrix = []
    for exp in EXPIRIES:
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        for t1, t2, name in spread_list:
            t1_y, t2_y = tenor_label_to_years(t1), tenor_label_to_years(t2)
            r1, r2 = calculate_forward_rate(rates, exp_y, t1_y), calculate_forward_rate(rates, exp_y, t2_y)
            row[name] = calculate_spread(r1, r2)
        matrix.append(row)
    return pd.DataFrame(matrix).set_index("Expiry")

def generate_butterfly_matrix(ccy: str, bfly_list: List[Tuple]) -> pd.DataFrame:
    rates = get_rates(ccy)
    matrix = []
    for exp in EXPIRIES:
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        for t1, t2, t3, name in bfly_list:
            t1_y, t2_y, t3_y = tenor_label_to_years(t1), tenor_label_to_years(t2), tenor_label_to_years(t3)
            r1 = calculate_forward_rate(rates, exp_y, t1_y)
            r2 = calculate_forward_rate(rates, exp_y, t2_y)
            r3 = calculate_forward_rate(rates, exp_y, t3_y)
            row[name] = calculate_butterfly(r1, r2, r3, t1_y, t2_y, t3_y)
        matrix.append(row)
    return pd.DataFrame(matrix).set_index("Expiry")

def generate_basis_matrix(basis_dict: Dict[str, float], label: str) -> pd.DataFrame:
    return pd.DataFrame([{"Tenor": t, label: v} for t, v in basis_dict.items()]).set_index("Tenor")

# ============================
# Tab Functions
# ============================

def landing_tab():
    is_dark = st.session_state.get("theme_name", "Dealer Dark") == "Dealer Dark"
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem 0;">
        <div style="font-size:2.5rem;font-weight:700;">
            <span style="color:#1e3a5f;">Rate</span><span style="color:#ef4444;">Edge</span>
        </div>
        <div style="font-size:1.1rem;color:#94a3b8;">AUD IRD Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Current Rates")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**AUD IRS**")
        aud = st.session_state.get("aud_irs", DEFAULT_AUD_IRS)
        for t in ["2Y", "5Y", "10Y", "30Y"]:
            if t in aud: st.metric(t, f"{aud[t]:.3f}%")
    with col2:
        st.markdown("**USD SOFR**")
        usd = st.session_state.get("usd_sofr", DEFAULT_USD_SOFR)
        for t in ["2Y", "5Y", "10Y", "30Y"]:
            if t in usd: st.metric(t, f"{usd[t]:.3f}%")
    with col3:
        st.markdown("**Key Spreads (AUD)**")
        rates = get_rates("AUD")
        for name, t1, t2 in [("2s10s", 2, 10), ("5s30s", 5, 30), ("10s30s", 10, 30)]:
            r1, r2 = rates.get(f"{t1}Y", 0), rates.get(f"{t2}Y", 0)
            st.metric(name, f"{(r2 - r1) * 100:.1f} bps")

def irs_rates_tab():
    show_header("📈 IRS Rates", "AUD • NZD • USD Swap Curves")
    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="irs_ccy")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("#### Rate Ladder")
        rates = get_rates(ccy)
        data = [{"Tenor": t, "Rate": r, "DV01": calculate_dv01(100_000_000, tenor_label_to_years(t))} for t, r in rates.items()]
        df = pd.DataFrame(data)
        st.dataframe(df.style.format({"Rate": "{:.4f}", "DV01": "{:,.0f}"}), use_container_width=True, height=500)
    with col2:
        st.markdown("#### Edit Rates")
        rates = get_rates(ccy)
        edit_tenor = st.selectbox("Tenor", list(rates.keys()), key="edit_tenor")
        new_rate = st.number_input("Rate (%)", 0.0, 20.0, float(rates.get(edit_tenor, 4.0)), 0.001, format="%.4f", key="edit_rate")
        if st.button("Update Rate", key="update_rate_btn"):
            if ccy == "AUD": st.session_state["aud_irs"][edit_tenor] = new_rate
            elif ccy == "USD": st.session_state["usd_sofr"][edit_tenor] = new_rate
            elif ccy == "NZD": st.session_state["nzd_irs"][edit_tenor] = new_rate
            st.success(f"Updated {ccy} {edit_tenor}")
            st.rerun()
        st.markdown("#### Curve Shape")
        fig = go.Figure()
        tenors_y = [tenor_label_to_years(t) for t in rates.keys()]
        fig.add_trace(go.Scatter(x=tenors_y, y=list(rates.values()), mode="lines+markers", name=ccy, line=dict(color="#ef4444", width=2)))
        fig.update_layout(xaxis_title="Tenor (Years)", yaxis_title="Rate (%)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=250, margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

def forward_matrix_tab():
    show_header("📊 Forward Swap Rates", "Expiry × Tenor Grid")
    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="fwd_ccy")
    fwd_matrix = generate_forward_matrix(ccy)
    col1, col2 = st.columns([3, 1])
    with col1: show_heatmap = st.checkbox("Heatmap", value=True, key="fwd_heatmap")
    with col2: st.download_button("📥 CSV", fwd_matrix.to_csv(), f"{ccy}_fwd_matrix.csv", key="dl_fwd")
    styled = fwd_matrix.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", axis=None) if show_heatmap else fwd_matrix.style.format("{:.3f}")
    st.dataframe(styled, use_container_width=True, height=400)
    with st.expander("📈 3D Surface"):
        fig = go.Figure(data=[go.Surface(z=fwd_matrix.values, x=list(range(len(fwd_matrix.columns))), y=list(range(len(fwd_matrix.index))), colorscale="RdYlGn_r")])
        fig.update_layout(scene=dict(xaxis=dict(title="Tenor", ticktext=list(fwd_matrix.columns), tickvals=list(range(len(fwd_matrix.columns)))), yaxis=dict(title="Expiry", ticktext=list(fwd_matrix.index), tickvals=list(range(len(fwd_matrix.index)))), zaxis=dict(title="Rate (%)")), template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

def spreads_tab():
    show_header("📉 Curve Spreads", "All Spread Combinations")
    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="spread_ccy")
    st.markdown("#### Select Spreads to Display")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Short End**")
        short_spreads = [s for s in ALL_SPREADS if tenor_label_to_years(s[0]) <= 3]
        selected_short = st.multiselect("Short", [s[2] for s in short_spreads], default=["1s2s", "2s3s", "2s5s"], key="sel_short", label_visibility="collapsed")
    with col2:
        st.markdown("**Medium**")
        med_spreads = [s for s in ALL_SPREADS if 3 < tenor_label_to_years(s[0]) <= 7]
        selected_med = st.multiselect("Med", [s[2] for s in med_spreads], default=["5s7s", "5s10s"], key="sel_med", label_visibility="collapsed")
    with col3:
        st.markdown("**Long End**")
        long_spreads = [s for s in ALL_SPREADS if tenor_label_to_years(s[0]) > 7]
        selected_long = st.multiselect("Long", [s[2] for s in long_spreads], default=["10s30s"], key="sel_long", label_visibility="collapsed")
    with col4:
        st.markdown("**Key Spreads**")
        selected_key = st.multiselect("Key", ["2s10s", "2s30s", "5s30s", "3s10s"], default=["2s10s", "5s30s"], key="sel_key", label_visibility="collapsed")
    
    all_selected = list(set(selected_short + selected_med + selected_long + selected_key))
    spread_map = {s[2]: s for s in ALL_SPREADS}
    selected_spreads = [spread_map[n] for n in all_selected if n in spread_map]
    
    if not selected_spreads:
        st.warning("Select at least one spread")
        return
    
    st.markdown("---")
    st.markdown("#### Spot Spreads")
    rates = get_rates(ccy)
    cols = st.columns(min(len(selected_spreads), 6))
    for i, (t1, t2, name) in enumerate(selected_spreads[:6]):
        t1_y, t2_y = tenor_label_to_years(t1), tenor_label_to_years(t2)
        r1, r2 = interpolate_rate(rates, t1_y), interpolate_rate(rates, t2_y)
        spread = calculate_spread(r1, r2)
        color = "#22c55e" if spread >= 0 else "#ef4444"
        with cols[i % 6]:
            st.markdown(f'<div class="spread-card"><div class="spread-label">{name}</div><div class="spread-value" style="color:{color};">{spread:.1f}</div></div>', unsafe_allow_html=True)
    
    st.markdown("#### Forward Spread Matrix")
    spread_matrix = generate_spread_matrix(ccy, selected_spreads)
    show_heatmap = st.checkbox("Heatmap", value=True, key="spread_heatmap")
    styled = spread_matrix.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=None) if show_heatmap else spread_matrix.style.format("{:.1f}")
    st.dataframe(styled, use_container_width=True, height=350)
    
    st.markdown("#### Spread Term Structure")
    fig = go.Figure()
    for col in spread_matrix.columns[:6]:
        fig.add_trace(go.Scatter(x=spread_matrix.index, y=spread_matrix[col], mode="lines+markers", name=col))
    fig.update_layout(xaxis_title="Expiry", yaxis_title="Spread (bps)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=300, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

def butterflies_tab():
    show_header("🦋 Butterfly Spreads", "All Butterfly Combinations")
    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="bfly_ccy")
    
    # Generate sequential butterflies
    sequential = []
    tenors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(tenors) - 2):
        t1, t2, t3 = tenors[i], tenors[i+1], tenors[i+2]
        sequential.append((f"{t1}Y", f"{t2}Y", f"{t3}Y", f"{t1}s{t2}s{t3}s"))
    
    st.markdown("#### Select Butterflies to Display")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sequential (1/2/3, 2/3/4, ...)**")
        seq_names = [b[3] for b in sequential]
        selected_seq = st.multiselect("Sequential", seq_names, default=seq_names[:5], key="sel_seq_bfly", label_visibility="collapsed")
    with col2:
        st.markdown("**Standard Market**")
        standard = [("2Y", "5Y", "10Y", "2s5s10s"), ("2Y", "10Y", "30Y", "2s10s30s"), ("5Y", "10Y", "30Y", "5s10s30s"), ("3Y", "5Y", "7Y", "3s5s7s"), ("5Y", "7Y", "10Y", "5s7s10s")]
        std_names = [b[3] for b in standard]
        selected_std = st.multiselect("Standard", std_names, default=["2s5s10s", "5s10s30s"], key="sel_std_bfly", label_visibility="collapsed")
    
    bfly_map = {b[3]: b for b in sequential + standard}
    all_selected = list(set(selected_seq + selected_std))
    selected_bflies = [bfly_map[n] for n in all_selected if n in bfly_map]
    
    if not selected_bflies:
        st.warning("Select at least one butterfly")
        return
    
    st.markdown("---")
    st.markdown("#### Spot Butterflies")
    rates = get_rates(ccy)
    cols = st.columns(min(len(selected_bflies), 6))
    for i, (t1, t2, t3, name) in enumerate(selected_bflies[:6]):
        t1_y, t2_y, t3_y = tenor_label_to_years(t1), tenor_label_to_years(t2), tenor_label_to_years(t3)
        r1, r2, r3 = interpolate_rate(rates, t1_y), interpolate_rate(rates, t2_y), interpolate_rate(rates, t3_y)
        bfly = calculate_butterfly(r1, r2, r3, t1_y, t2_y, t3_y)
        color = "#22c55e" if bfly >= 0 else "#ef4444"
        with cols[i % 6]:
            st.markdown(f'<div class="spread-card"><div class="spread-label">{name}</div><div class="spread-value" style="color:{color};">{bfly:.1f}</div></div>', unsafe_allow_html=True)
    
    st.markdown("#### Forward Butterfly Matrix")
    bfly_matrix = generate_butterfly_matrix(ccy, selected_bflies)
    show_heatmap = st.checkbox("Heatmap", value=True, key="bfly_heatmap")
    styled = bfly_matrix.style.format("{:.1f}").background_gradient(cmap="PuOr", axis=None) if show_heatmap else bfly_matrix.style.format("{:.1f}")
    st.dataframe(styled, use_container_width=True, height=350)
    
    fig = go.Figure()
    for col in bfly_matrix.columns[:5]:
        fig.add_trace(go.Scatter(x=bfly_matrix.index, y=bfly_matrix[col], mode="lines+markers", name=col))
    fig.update_layout(xaxis_title="Expiry", yaxis_title="Butterfly (bps)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=300, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

def basis_tab():
    show_header("🔄 Basis Curves", "6v3 • 3v1 • BOB • OIS Spreads")
    st.markdown("#### AUD Basis Curves")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**6v3 Basis (bps)**")
        df_6v3 = generate_basis_matrix(st.session_state.get("aud_6v3", DEFAULT_AUD_6V3_BASIS), "6v3")
        st.dataframe(df_6v3.style.format("{:.3f}"), height=350)
    with col2:
        st.markdown("**3v1 Basis (bps)**")
        df_3v1 = generate_basis_matrix(st.session_state.get("aud_3v1", DEFAULT_AUD_3V1_BASIS), "3v1")
        st.dataframe(df_3v1.style.format("{:.3f}"), height=350)
    with col3:
        st.markdown("**BOB Spread (bps)**")
        df_bob = generate_basis_matrix(st.session_state.get("bob_spread", DEFAULT_BOB_SPREAD), "BOB")
        st.dataframe(df_bob.style.format("{:.2f}"), height=350)
    
    st.markdown("---")
    st.markdown("#### Edit Basis")
    col1, col2, col3, col4 = st.columns(4)
    with col1: basis_type = st.selectbox("Curve", ["6v3", "3v1", "BOB"], key="edit_basis_type")
    with col2:
        if basis_type == "6v3": tenors = list(DEFAULT_AUD_6V3_BASIS.keys())
        elif basis_type == "3v1": tenors = list(DEFAULT_AUD_3V1_BASIS.keys())
        else: tenors = list(DEFAULT_BOB_SPREAD.keys())
        edit_tenor = st.selectbox("Tenor", tenors, key="edit_basis_tenor")
    with col3:
        if basis_type == "6v3": current = st.session_state.get("aud_6v3", DEFAULT_AUD_6V3_BASIS).get(edit_tenor, 0)
        elif basis_type == "3v1": current = st.session_state.get("aud_3v1", DEFAULT_AUD_3V1_BASIS).get(edit_tenor, 0)
        else: current = st.session_state.get("bob_spread", DEFAULT_BOB_SPREAD).get(edit_tenor, 0)
        new_val = st.number_input("Value (bps)", -50.0, 50.0, float(current), 0.125, key="edit_basis_val")
    with col4:
        if st.button("Update", key="update_basis_btn"):
            if basis_type == "6v3": st.session_state["aud_6v3"][edit_tenor] = new_val
            elif basis_type == "3v1": st.session_state["aud_3v1"][edit_tenor] = new_val
            else: st.session_state["bob_spread"][edit_tenor] = new_val
            st.success(f"Updated {basis_type} {edit_tenor}")
            st.rerun()
    
    st.markdown("#### Basis Curve Comparison")
    fig = go.Figure()
    for basis_type, data, color in [("6v3", st.session_state.get("aud_6v3", DEFAULT_AUD_6V3_BASIS), "#ef4444"), ("3v1", st.session_state.get("aud_3v1", DEFAULT_AUD_3V1_BASIS), "#3b82f6"), ("BOB", st.session_state.get("bob_spread", DEFAULT_BOB_SPREAD), "#22c55e")]:
        tenors_y = [tenor_label_to_years(t) for t in data.keys()]
        fig.add_trace(go.Scatter(x=tenors_y, y=list(data.values()), mode="lines+markers", name=basis_type, line=dict(color=color, width=2)))
    fig.update_layout(xaxis_title="Tenor (Years)", yaxis_title="Basis (bps)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=300)
    st.plotly_chart(fig, use_container_width=True)

def xccy_basis_tab():
    show_header("🌍 Cross-Currency Basis", "AUD/USD BBSW vs SOFR")
    st.markdown("#### AUD/USD Cross-Currency Basis")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        xccy = st.session_state.get("xccy_basis", DEFAULT_XCCY_BASIS)
        df_xccy = pd.DataFrame([{"Tenor": t, "XCcy Basis (bps)": v} for t, v in xccy.items()]).set_index("Tenor")
        st.dataframe(df_xccy.style.format("{:.3f}"), height=400)
    with col2:
        st.markdown("**Rate Comparison**")
        aud_rates = st.session_state.get("aud_irs", DEFAULT_AUD_IRS)
        usd_rates = st.session_state.get("usd_sofr", DEFAULT_USD_SOFR)
        for t in ["2Y", "5Y", "10Y"]:
            if t in aud_rates and t in xccy:
                aud_rate, basis = aud_rates[t], xccy.get(t, 0) / 100
                implied_usd, actual_usd = aud_rate - basis, usd_rates.get(t, 0)
                st.markdown(f"**{t}**: AUD {aud_rate:.3f}% | Basis {xccy.get(t, 0):.2f}bp | Implied USD {implied_usd:.3f}% | Actual USD {actual_usd:.3f}%")
    
    st.markdown("#### Edit XCcy Basis")
    col1, col2, col3 = st.columns(3)
    with col1: edit_tenor = st.selectbox("Tenor", list(xccy.keys()), key="edit_xccy_tenor")
    with col2: new_val = st.number_input("Basis (bps)", -20.0, 20.0, float(xccy.get(edit_tenor, 0)), 0.125, key="edit_xccy_val")
    with col3:
        if st.button("Update", key="update_xccy_btn"):
            st.session_state["xccy_basis"][edit_tenor] = new_val
            st.success(f"Updated XCcy {edit_tenor}")
            st.rerun()
    
    fig = go.Figure()
    tenors_y = [tenor_label_to_years(t) for t in xccy.keys()]
    fig.add_trace(go.Scatter(x=tenors_y, y=list(xccy.values()), mode="lines+markers", name="AUD/USD XCcy", line=dict(color="#fbbf24", width=2), fill="tozeroy", fillcolor="rgba(251, 191, 36, 0.1)"))
    fig.update_layout(xaxis_title="Tenor (Years)", yaxis_title="Basis (bps)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=300)
    st.plotly_chart(fig, use_container_width=True)

def irs_pricer_tab():
    show_header("🎯 IRS Pricer", "Spot & Forward Swaps")
    col_left, col_right = st.columns([1.2, 1.4])
    with col_left:
        st.markdown("#### Trade Ticket")
        ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="pricer_ccy")
        notional = st.number_input("Notional", 0.0, 10_000_000_000.0, 100_000_000.0, 1_000_000.0, format="%.0f", key="pricer_notional")
        direction = st.selectbox("Direction", ["Pay Fixed", "Receive Fixed"], key="pricer_direction")
        start_type = st.selectbox("Start", ["Spot", "Forward"], key="pricer_start_type")
        if start_type == "Spot":
            tenor = st.selectbox("Tenor", ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"], index=3, key="pricer_tenor")
            fwd_years = 0
        else:
            fwd_years = st.selectbox("Forward Start (Years)", [1, 2, 3, 5, 7, 10], key="pricer_fwd")
            tenor = st.selectbox("Swap Tenor", ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"], index=2, key="pricer_fwd_tenor")
        use_par = st.checkbox("Use Par Rate", value=True, key="pricer_use_par")
        trade_rate = st.number_input("Fixed Rate (%)", 0.0, 20.0, 4.0, 0.01, key="pricer_rate")
    with col_right:
        st.markdown("#### Results")
        if st.button("💰 Price", key="price_btn", type="primary"):
            rates = get_rates(ccy)
            tenor_y = tenor_label_to_years(tenor)
            par_rate = calculate_forward_rate(rates, fwd_years, tenor_y)
            fixed_rate = par_rate if use_par else trade_rate
            rate_diff = (par_rate - fixed_rate) / 100
            dv01 = calculate_dv01(notional, tenor_y)
            npv = -rate_diff * notional * tenor_y * 0.9 if direction == "Pay Fixed" else rate_diff * notional * tenor_y * 0.9
            c1, c2 = st.columns(2)
            c1.metric("Par Rate", f"{par_rate:.4f}%")
            c2.metric("Trade Rate", f"{fixed_rate:.4f}%")
            c3, c4 = st.columns(2)
            c3.metric("NPV", f"${npv:,.0f}")
            c4.metric("DV01", f"${dv01:,.0f}")
            st.markdown("---")
            st.markdown(f"**Trade Summary**: {ccy} | ${notional:,.0f} | {direction} | {'Spot' if fwd_years == 0 else f'{fwd_years}Y Fwd'} {tenor} | {CCY_CONVENTIONS[ccy]['description']}")

def bob_pricer_tab():
    show_header("📐 BOB Pricer", "Basis of Basis Swaps")
    st.markdown("**BOB = 6v3 Basis + XCcy Basis** - Used for RV between AUD domestic basis and cross-currency funding.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### BOB Calculator")
        tenor = st.selectbox("Tenor", ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y"], index=3, key="bob_tenor")
        notional = st.number_input("Notional", 0.0, 10_000_000_000.0, 100_000_000.0, 1_000_000.0, key="bob_notional")
        basis_6v3 = st.session_state.get("aud_6v3", DEFAULT_AUD_6V3_BASIS).get(tenor, 0)
        xccy = st.session_state.get("xccy_basis", DEFAULT_XCCY_BASIS).get(tenor, 0)
        bob_current = st.session_state.get("bob_spread", DEFAULT_BOB_SPREAD).get(tenor, 0)
        st.markdown("**Current Levels**")
        st.write(f"6v3 Basis: {basis_6v3:.3f} bps | XCcy Basis: {xccy:.3f} bps | BOB Spread: {bob_current:.2f} bps")
        implied_bob = basis_6v3 + xccy
        st.write(f"**Implied BOB: {implied_bob:.2f} bps** | Rich/Cheap: {bob_current - implied_bob:.2f} bps")
    with col2:
        st.markdown("#### BOB Term Structure")
        bob = st.session_state.get("bob_spread", DEFAULT_BOB_SPREAD)
        basis_6v3_all = st.session_state.get("aud_6v3", DEFAULT_AUD_6V3_BASIS)
        xccy_all = st.session_state.get("xccy_basis", DEFAULT_XCCY_BASIS)
        fig = go.Figure()
        tenors_y = [tenor_label_to_years(t) for t in bob.keys()]
        fig.add_trace(go.Scatter(x=tenors_y, y=list(bob.values()), mode="lines+markers", name="BOB", line=dict(color="#22c55e", width=2)))
        implied = [basis_6v3_all.get(t, 0) + xccy_all.get(t, 0) for t in bob.keys()]
        fig.add_trace(go.Scatter(x=tenors_y, y=implied, mode="lines+markers", name="Implied (6v3 + XCcy)", line=dict(color="#fbbf24", width=2, dash="dash")))
        fig.update_layout(xaxis_title="Tenor (Years)", yaxis_title="Spread (bps)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=300)
        st.plotly_chart(fig, use_container_width=True)

def rv_tab():
    show_header("📊 Relative Value", "Cross-Market Analysis")
    st.markdown("#### AUD vs USD Comparison")
    aud_rates, usd_rates = st.session_state.get("aud_irs", DEFAULT_AUD_IRS), st.session_state.get("usd_sofr", DEFAULT_USD_SOFR)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Spot Rates**")
        data = [{"Tenor": t, "AUD": aud_rates.get(t, 0), "USD": usd_rates.get(t, 0), "Diff (bps)": (aud_rates.get(t, 0) - usd_rates.get(t, 0)) * 100} for t in ["2Y", "5Y", "10Y", "30Y"]]
        st.dataframe(pd.DataFrame(data).style.format({"AUD": "{:.3f}", "USD": "{:.3f}", "Diff (bps)": "{:.0f}"}))
    with col2:
        st.markdown("**Spread Comparison**")
        aud_2s10s = (aud_rates.get("10Y", 0) - aud_rates.get("2Y", 0)) * 100
        usd_2s10s = (usd_rates.get("10Y", 0) - usd_rates.get("2Y", 0)) * 100
        st.metric("AUD 2s10s", f"{aud_2s10s:.0f} bps")
        st.metric("USD 2s10s", f"{usd_2s10s:.0f} bps")
        st.metric("Differential", f"{aud_2s10s - usd_2s10s:.0f} bps")
    with col3:
        st.markdown("**Trade Ideas**")
        diff = aud_2s10s - usd_2s10s
        if diff > 20: st.success("AUD steeper - consider AUD flattener vs USD")
        elif diff < -20: st.success("USD steeper - consider AUD steepener vs USD")
        else: st.info("Curves relatively similar")
    
    st.markdown("#### Curve Overlay")
    fig = go.Figure()
    aud_tenors = [tenor_label_to_years(t) for t in aud_rates.keys()]
    fig.add_trace(go.Scatter(x=aud_tenors, y=list(aud_rates.values()), mode="lines+markers", name="AUD", line=dict(color="#ef4444", width=2)))
    usd_tenors = [tenor_label_to_years(t) for t in usd_rates.keys()]
    fig.add_trace(go.Scatter(x=usd_tenors, y=list(usd_rates.values()), mode="lines+markers", name="USD", line=dict(color="#3b82f6", width=2)))
    fig.update_layout(xaxis_title="Tenor (Years)", yaxis_title="Rate (%)", template="plotly_dark" if st.session_state.get("theme_name") == "Dealer Dark" else "plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ============================
# Main Application
# ============================

def main():
    st.set_page_config(page_title="RateEdge IRS", layout="wide", page_icon="📊", initial_sidebar_state="expanded")
    init_session()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;border-bottom:1px solid #334155;">
            <div style="font-size:1.3rem;font-weight:700;"><span style="color:#1e3a5f;">Rate</span><span style="color:#ef4444;">Edge</span></div>
            <div style="font-size:0.7rem;color:#94a3b8;">IRS Analytics v2.0</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        theme = st.selectbox("🎨 Theme", ["Dealer Dark", "Clean Light"], index=0 if st.session_state.get("theme_name") == "Dealer Dark" else 1, key="theme_sel")
        st.session_state["theme_name"] = theme
        st.markdown("---")
        if st.button("🔄 Reset All Data", key="reset_btn"):
            st.session_state["aud_irs"] = DEFAULT_AUD_IRS.copy()
            st.session_state["usd_sofr"] = DEFAULT_USD_SOFR.copy()
            st.session_state["nzd_irs"] = DEFAULT_NZD_IRS.copy()
            st.session_state["aud_6v3"] = DEFAULT_AUD_6V3_BASIS.copy()
            st.session_state["aud_3v1"] = DEFAULT_AUD_3V1_BASIS.copy()
            st.session_state["xccy_basis"] = DEFAULT_XCCY_BASIS.copy()
            st.session_state["bob_spread"] = DEFAULT_BOB_SPREAD.copy()
            st.success("Data reset!")
            st.rerun()
        st.markdown("---")
        st.markdown('<div style="color:#64748b;font-size:0.65rem;text-align:center;">© 2024 RateEdge Australia<br><a href="https://rateedge.au" style="color:#3b82f6;">rateedge.au</a></div>', unsafe_allow_html=True)
    
    apply_rateedge_theme(st.session_state["theme_name"])
    
    tabs = st.tabs(["🏠 Overview", "📈 IRS Rates", "📊 Forward Matrix", "📉 Spreads", "🦋 Butterflies", "🔄 Basis", "🌍 XCcy Basis", "🎯 IRS Pricer", "📐 BOB Pricer", "📊 RV Analysis"])
    with tabs[0]: landing_tab()
    with tabs[1]: irs_rates_tab()
    with tabs[2]: forward_matrix_tab()
    with tabs[3]: spreads_tab()
    with tabs[4]: butterflies_tab()
    with tabs[5]: basis_tab()
    with tabs[6]: xccy_basis_tab()
    with tabs[7]: irs_pricer_tab()
    with tabs[8]: bob_pricer_tab()
    with tabs[9]: rv_tab()

if __name__ == "__main__":
    main()

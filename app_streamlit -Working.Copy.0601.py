
import math
import os
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from statistics import NormalDist
import plotly.graph_objects as go
import requests

# ============================
# RateEdge Authentication
# ============================
AUTH_API = "https://rateedge-auth.azurewebsites.net"
SITE_ID = "options"

def verify_token(token):
    try:
        resp = requests.post(f"{AUTH_API}/api/auth/verify-token",
                           json={"token": token, "site": SITE_ID}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("valid", False), data.get("email", "")
    except:
        pass
    return False, None

def request_otp(email):
    try:
        resp = requests.post(f"{AUTH_API}/api/auth/request-otp",
                           json={"email": email, "site": SITE_ID}, timeout=10)
        return resp.status_code, resp.json()
    except Exception as e:
        return 500, {"error": str(e)}

def verify_otp(email, code):
    try:
        resp = requests.post(f"{AUTH_API}/api/auth/verify-otp",
                           json={"email": email, "site": SITE_ID, "code": code}, timeout=5)
        return resp.status_code, resp.json()
    except Exception as e:
        return 500, {"error": str(e)}

try:
    from streamlit_plotly_events import plotly_events
except ImportError:  # optional dependency
    plotly_events = None

# Import the new hybrid vol surface editor
try:
    from vol_editor import render_vol_surface_editor, render_bulk_adjustment_tools, render_conventions_tab
    render_vol_surface_editor_unified = render_vol_surface_editor
    render_vol_surface_editor_3d = render_vol_surface_editor
    HAS_3D_DRAG = True
    HAS_VOL_EDITOR = True
except ImportError:
    HAS_VOL_EDITOR = False
    render_conventions_tab = None

try:
    import psycopg2
    from psycopg2.extras import Json
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

SUPPORTED_CURRENCIES = ["AUD", "NZD", "USD"]


# ============================
# Database Functions
# ============================

def get_db_url():
    """Get database URL at runtime"""
    # Hardcoded temporarily - will move back to env var once working
    return "postgresql://rateedgeadmin:RateEdge2025!@rateedge-oms-db.postgres.database.azure.com:5432/swaption?sslmode=require"

def get_db_connection():
    """Get PostgreSQL connection"""
    db_url = get_db_url()
    if not HAS_POSTGRES or not db_url:
        return None
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        st.warning(f"Database connection failed: {e}")
        return None


def init_database():
    """Create tables if they don't exist"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_configs (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                config_type VARCHAR(50) NOT NULL,
                currency VARCHAR(3) NOT NULL,
                data JSONB NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, config_type, currency)
            );
            
            CREATE INDEX IF NOT EXISTS idx_user_configs_user 
            ON user_configs(user_id);
        """)
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database init failed: {e}")
        return False


def save_user_config(user_id: str, config_type: str, currency: str, data: dict):
    """Save user config to database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_configs (user_id, config_type, currency, data, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, config_type, currency) 
            DO UPDATE SET data = %s, updated_at = NOW()
        """, (user_id, config_type, currency, Json(data), Json(data)))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


def load_user_config(user_id: str, config_type: str, currency: str) -> Optional[dict]:
    """Load user config from database"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT data FROM user_configs 
            WHERE user_id = %s AND config_type = %s AND currency = %s
        """, (user_id, config_type, currency))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        return None


def load_all_user_configs(user_id: str) -> dict:
    """Load all configs for a user"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT config_type, currency, data, updated_at 
            FROM user_configs 
            WHERE user_id = %s
            ORDER BY updated_at DESC
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        configs = {}
        for config_type, currency, data, updated_at in rows:
            if config_type not in configs:
                configs[config_type] = {}
            configs[config_type][currency] = {
                "data": data,
                "updated_at": updated_at.isoformat() if updated_at else None
            }
        return configs
    except Exception as e:
        return {}


def save_all_session_data(user_id: str):
    """Save all current session data to database"""
    saved = 0
    
    for ccy in SUPPORTED_CURRENCIES:
        # Save curves
        curve = st.session_state.get("curves", {}).get(ccy)
        if curve is not None:
            data = {"values": curve.to_dict(orient="records")}
            if save_user_config(user_id, "curve", ccy, data):
                saved += 1
        
        # Save ATM vols from vol_data
        vol_data = st.session_state.get("vol_data", {}).get(ccy, {})
        atm = vol_data.get("atm")
        if atm is not None:
            # Ensure Expiry is a column, not index, before saving
            atm_save = atm.copy()
            if atm_save.index.name == "Expiry":
                atm_save = atm_save.reset_index()
            elif "Expiry" not in atm_save.columns and atm_save.index.name is None:
                first_idx = atm_save.index[0] if len(atm_save) > 0 else None
                if isinstance(first_idx, str) and first_idx.lower().endswith(('w', 'm', 'y')):
                    atm_save = atm_save.reset_index()
                    atm_save.columns = ["Expiry"] + list(atm_save.columns[1:])
            # Force Expiry column first - build records manually to guarantee order
            records = []
            for _, row in atm_save.iterrows():
                record = {"Expiry": row.get("Expiry", "")}
                for col in atm_save.columns:
                    if col != "Expiry":
                        record[col] = row[col]
                records.append(record)
            data = {"values": records}
            if save_user_config(user_id, "atm_vols", ccy, data):
                saved += 1
        
        # Save SABR params from vol_data
        for param in ["alpha", "beta", "rho", "nu"]:
            val = vol_data.get(param)
            if val is not None:
                # Ensure Expiry column is preserved
                val_save = val.copy()
                if val_save.index.name == "Expiry":
                    val_save = val_save.reset_index()
                elif "Expiry" not in val_save.columns and val_save.index.name is None:
                    first_idx = val_save.index[0] if len(val_save) > 0 else None
                    if isinstance(first_idx, str) and first_idx.lower().endswith(('w', 'm', 'y')):
                        val_save = val_save.reset_index()
                        val_save.columns = ["Expiry"] + list(val_save.columns[1:])
                # Force Expiry column first - build records manually
                records = []
                for _, row in val_save.iterrows():
                    record = {"Expiry": row.get("Expiry", "")}
                    for col in val_save.columns:
                        if col != "Expiry":
                            record[col] = row[col]
                    records.append(record)
                data = {"values": records}
                if save_user_config(user_id, f"sabr_{param}", ccy, data):
                    saved += 1
        
        # Save basis curves
        basis = st.session_state.get("basis_curves", {}).get(ccy, {})
        for basis_type in ["6v3", "3v1", "ois"]:
            bc = basis.get(basis_type)
            if bc is not None:
                # Basis curves have Tenor column, not Expiry
                bc_save = bc.copy()
                if bc_save.index.name is not None:
                    bc_save = bc_save.reset_index()
                data = {"values": bc_save.to_dict(orient="records")}
                if save_user_config(user_id, f"basis_{basis_type}", ccy, data):
                    saved += 1
    
    return saved


def load_all_session_data(user_id: str) -> int:
    """Load all saved data into session state"""
    configs = load_all_user_configs(user_id)
    if not configs:
        return 0
    
    loaded = 0
    
    # Initialize session state containers
    if "curves" not in st.session_state:
        st.session_state["curves"] = {}
    if "vol_data" not in st.session_state:
        st.session_state["vol_data"] = {}
    if "basis_curves" not in st.session_state:
        st.session_state["basis_curves"] = {}
    if "vol_editor" not in st.session_state:
        st.session_state["vol_editor"] = {"working": {}, "base": {}, "history": {}, "redo_stack": {}, "selected_cell": {}}
    
    for ccy in SUPPORTED_CURRENCIES:
        # Initialize vol_data for this currency
        if ccy not in st.session_state["vol_data"]:
            st.session_state["vol_data"][ccy] = {}
        
        # Load curves
        if "curve" in configs and ccy in configs["curve"]:
            try:
                df = pd.DataFrame(configs["curve"][ccy]["data"]["values"])
                st.session_state["curves"][ccy] = df
                loaded += 1
            except:
                pass
        
        # Load ATM vols into vol_data
        if "atm_vols" in configs and ccy in configs["atm_vols"]:
            try:
                df = pd.DataFrame(configs["atm_vols"][ccy]["data"]["values"])
                # Reorder columns to ensure Expiry is first if it exists
                if "Expiry" in df.columns:
                    cols = ["Expiry"] + [c for c in df.columns if c != "Expiry"]
                    df = df[cols]
                st.session_state["vol_data"][ccy]["atm"] = df
                # Also update vol_editor
                ve = st.session_state["vol_editor"]
                ve["base"][ccy] = df.copy()
                ve["working"][ccy] = df.copy()
                ve["history"][ccy] = []
                if "redo_stack" not in ve:
                    ve["redo_stack"] = {}
                ve["redo_stack"][ccy] = []
                loaded += 1
            except Exception as e:
                st.warning(f"Failed to load ATM data for {ccy}: {e}")
        
        # Load SABR params into vol_data
        for param in ["alpha", "beta", "rho", "nu"]:
            key = f"sabr_{param}"
            if key in configs and ccy in configs[key]:
                try:
                    df = pd.DataFrame(configs[key][ccy]["data"]["values"])
                    # Reorder columns to ensure Expiry is first if it exists
                    if "Expiry" in df.columns:
                        cols = ["Expiry"] + [c for c in df.columns if c != "Expiry"]
                        df = df[cols]
                    st.session_state["vol_data"][ccy][param] = df
                    loaded += 1
                except:
                    pass
        
        # Load basis curves
        if ccy not in st.session_state["basis_curves"]:
            st.session_state["basis_curves"][ccy] = {}
        
        for basis_type in ["6v3", "3v1", "ois"]:
            key = f"basis_{basis_type}"
            if key in configs and ccy in configs[key]:
                try:
                    df = pd.DataFrame(configs[key][ccy]["data"]["values"])
                    st.session_state["basis_curves"][ccy][basis_type] = df
                    loaded += 1
                except:
                    pass
    
    return loaded


# ============================
# Data structures
# ============================

@dataclass
class SwaptionTicket:
    side: str
    payoff_type: str  # "vanilla", "digital", "straddle"
    notional: float
    currency: str
    expiry_years: float
    swap_tenor_years: float
    forward: float
    strike: float
    vol: float
    discount_rate: float
    annuity: float
    model: str  # "Black" or "Normal"
    payout_bp: float = 1.0
    label: str = ""
    use_curve: bool = False

    def df(self, t: Optional[float] = None) -> float:
        if t is None:
            t = self.expiry_years
        return math.exp(-self.discount_rate * t)


# ============================
# Helper functions
# ============================

def label_to_years(lbl: str) -> float:
    x = str(lbl).strip().lower()
    if x.endswith("d"):
        return float(x[:-1]) / 365.0
    if x.endswith("w"):
        return float(x[:-1]) / 52.0
    if x.endswith("m"):
        return float(x[:-1]) / 12.0
    if x.endswith("y"):
        return float(x[:-1])
    return float(x)


def load_atm_surface(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "Expiry" not in df.columns:
        raise ValueError(f"{name} must have an 'Expiry' column")
    return df.copy()


def ensure_sabr_matrix(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "Expiry" not in df.columns:
        raise ValueError(f"{name} must have an 'Expiry' column")
    return df.copy()


def get_matrix_value(mat: Optional[pd.DataFrame],
                     expiry_label: str,
                     tenor_years: float) -> Optional[float]:
    if mat is None or mat.empty:
        return None
    row = mat[mat["Expiry"] == expiry_label]
    if row.empty:
        return None
    col = f"{int(round(tenor_years))}Y"
    if col not in mat.columns:
        return None
    val = row.iloc[0][col]
    if pd.isna(val):
        return None
    return float(val)


def get_sabr_params_from_matrices(a: Optional[pd.DataFrame],
                                  b: Optional[pd.DataFrame],
                                  r: Optional[pd.DataFrame],
                                  n: Optional[pd.DataFrame],
                                  expiry_label: str,
                                  tenor_years: float) -> Optional[dict]:
    alpha = get_matrix_value(a, expiry_label, tenor_years)
    beta = get_matrix_value(b, expiry_label, tenor_years)
    rho = get_matrix_value(r, expiry_label, tenor_years)
    nu = get_matrix_value(n, expiry_label, tenor_years)
    if any(x is None for x in (alpha, beta, rho, nu)):
        return None
    return dict(alpha=alpha, beta=beta, rho=rho, nu=nu)


def sabr_implied_vol_black(F: float, K: float, T: float,
                           alpha: float, beta: float,
                           rho: float, nu: float,
                           eps: float = 1e-7) -> float:
    """Hagan SABR approximation for Black vols."""
    F = float(F)
    K = float(K)
    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    nu = float(nu)

    if T <= 0 or alpha <= 0:
        return 0.0

    # ATM case
    if abs(F - K) < eps:
        Fmid = F
        term1 = alpha / (Fmid ** (1 - beta))
        term2 = (
            ((1 - beta) ** 2 / 24.0) * (alpha ** 2 / (Fmid ** (2 - 2 * beta))) +
            0.25 * rho * beta * nu * alpha / (Fmid ** (1 - beta)) +
            (2 - 3 * rho * rho) * nu * nu / 24.0
        ) * T
        return term1 * (1 + term2)

    # Away from ATM
    FK = F * K
    logFK = math.log(F / K)
    
    # Leading order term
    FK_beta = FK ** ((1 - beta) / 2.0)
    
    # z and chi(z) for the smile
    z = (nu / alpha) * FK_beta * logFK
    
    if abs(z) < eps:
        x_z = 1.0
    else:
        sqrt_term = math.sqrt(1 - 2 * rho * z + z * z)
        x_z = z / math.log((sqrt_term + z - rho) / (1 - rho))
    
    # Numerator correction for logFK
    num_corr = 1 + ((1 - beta) ** 2 / 24.0) * (logFK ** 2) + ((1 - beta) ** 4 / 1920.0) * (logFK ** 4)
    
    # Denominator (time-dependent correction)
    den_corr = 1 + (
        ((1 - beta) ** 2 / 24.0) * (alpha ** 2 / (FK ** (1 - beta))) +
        0.25 * rho * beta * nu * alpha / FK_beta +
        (2 - 3 * rho * rho) * nu * nu / 24.0
    ) * T
    
    vol = (alpha / FK_beta) * x_z * num_corr / den_corr
    return max(vol, 0.0)


# ============================
# Curve construction
# ============================

def load_curve(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if {"Maturity (Years)", "Zero Rate (%)"}.issubset(df.columns):
        out = df[["Maturity (Years)", "Zero Rate (%)"]].copy()
        out.rename(columns={"Maturity (Years)": "MaturityY", "Zero Rate (%)": "ZeroRatePct"}, inplace=True)
        out = out.sort_values("MaturityY").reset_index(drop=True)
        return out
    raise ValueError(f"{name}: expected columns 'Maturity (Years)', 'Zero Rate (%)'")


def interpolate_zero(curve: pd.DataFrame, t: float) -> float:
    xs = curve["MaturityY"].to_numpy().astype(float)
    ys = curve["ZeroRatePct"].to_numpy().astype(float) / 100.0
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    dfs = np.exp(-ys * xs)
    df_t = np.exp(np.interp(t, xs, np.log(dfs)))
    return -math.log(df_t) / t


def df_from_curve(curve: pd.DataFrame, t: float) -> float:
    z = interpolate_zero(curve, t)
    return math.exp(-z * t)


def build_aud_schedule(expiry: float, tenor: float) -> List[Tuple[float, float]]:
    """AUD: premium at expiry, swap starts T+1BD, q/q to 3y then s/s."""
    schedule: List[Tuple[float, float]] = []
    swap_start = expiry + 1.0 / 252.0
    end = swap_start + tenor
    t = swap_start
    while t < end - 1e-8:
        step = 0.25 if (t - swap_start) < 3.0 else 0.5
        nxt = min(t + step, end)
        accrual = nxt - t
        schedule.append((nxt, accrual))
        t = nxt
    return schedule


def build_generic_schedule(expiry: float, tenor: float, freq: float = 0.5) -> List[Tuple[float, float]]:
    schedule: List[Tuple[float, float]] = []
    swap_start = expiry + 1.0 / 252.0
    end = swap_start + tenor
    t = swap_start
    while t < end - 1e-8:
        nxt = min(t + freq, end)
        accrual = nxt - t
        schedule.append((nxt, accrual))
        t = nxt
    return schedule


def forward_and_annuity_from_curve(curve: pd.DataFrame,
                                   ccy: str,
                                   expiry: float,
                                   tenor: float,
                                   ois_curve: Optional[pd.DataFrame] = None) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Calculate forward swap rate and annuity.
    
    Dual-curve approach:
    - Forward rate: derived from IRS curve (projection curve)
    - Annuity: discounted using OIS curve if provided (discounting curve)
    """
    if ccy == "AUD":
        sched = build_aud_schedule(expiry, tenor)
    else:
        sched = build_generic_schedule(expiry, tenor, freq=0.5)
    if not sched:
        return 0.0, 0.0, []
    
    # Use OIS curve for discounting if provided
    disc_curve = ois_curve if ois_curve is not None else curve
    
    ann = 0.0
    for T_i, accrual in sched:
        df_i = df_from_curve(disc_curve, T_i)
        ann += df_i * accrual
    swap_start = sched[0][0] - sched[0][1]
    df_start = df_from_curve(curve, swap_start)
    df_end = df_from_curve(curve, sched[-1][0])
    fwd = (df_start - df_end) / ann if ann > 0 else 0.0
    return fwd, ann, sched


# ============================
# Swaption pricing
# ============================

def black_swaption_vanilla(ticket: SwaptionTicket) -> dict:
    F = ticket.forward
    K = ticket.strike
    sigma = max(ticket.vol, 1e-8)
    T = max(ticket.expiry_years, 1e-8)
    df = ticket.df()
    annuity = ticket.annuity if ticket.annuity > 0 else ticket.swap_tenor_years

    if F <= 0 or K <= 0 or annuity <= 0:
        bpv = df * annuity * ticket.notional * 0.0001
        return {"pv": 0.0, "pv_bp": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "bpv": bpv}

    lnFK = math.log(F / K)
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (lnFK + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    N = NormalDist().cdf
    phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    if ticket.side.lower().startswith("payer"):
        price_rate = F * N(d1) - K * N(d2)
        delta = df * annuity * ticket.notional * N(d1)
    else:
        price_rate = K * N(-d2) - F * N(-d1)
        delta = -df * annuity * ticket.notional * N(-d1)

    pv = df * annuity * ticket.notional * price_rate
    bpv = df * annuity * ticket.notional * 0.0001
    pv_bp = pv / bpv if bpv != 0 else 0.0
    vega = df * annuity * ticket.notional * F * phi(d1) * math.sqrt(T)
    gamma = df * annuity * ticket.notional * phi(d1) / (F * sigma * math.sqrt(T))
    theta = -0.5 * df * annuity * ticket.notional * F * sigma * phi(d1)

    return {"pv": pv, "pv_bp": pv_bp, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "bpv": bpv}


def bachelier_swaption_vanilla(ticket: SwaptionTicket) -> dict:
    F = ticket.forward
    K = ticket.strike
    sigma_n = max(ticket.vol, 1e-8)
    T = max(ticket.expiry_years, 1e-8)
    df = ticket.df()
    annuity = ticket.annuity if ticket.annuity > 0 else ticket.swap_tenor_years

    if annuity <= 0:
        bpv = df * annuity * ticket.notional * 0.0001
        return {"pv": 0.0, "pv_bp": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "bpv": bpv}

    d = (F - K) / (sigma_n * math.sqrt(T))
    phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    N = NormalDist().cdf(d)

    if ticket.side.lower().startswith("payer"):
        price_rate = (F - K) * N + sigma_n * math.sqrt(T) * phi
        delta = df * annuity * ticket.notional * N
    else:
        price_rate = (K - F) * (1 - N) + sigma_n * math.sqrt(T) * phi
        delta = -df * annuity * ticket.notional * (1 - N)

    pv = df * annuity * ticket.notional * price_rate
    bpv = df * annuity * ticket.notional * 0.0001
    pv_bp = pv / bpv if bpv != 0 else 0.0
    vega = df * annuity * ticket.notional * math.sqrt(T) * phi
    gamma = df * annuity * ticket.notional * phi / (sigma_n * math.sqrt(T))
    theta = -0.5 * df * annuity * ticket.notional * sigma_n * phi / math.sqrt(T)

    return {"pv": pv, "pv_bp": pv_bp, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "bpv": bpv}


def black_swaption_digital(ticket: SwaptionTicket) -> dict:
    F = ticket.forward
    K = ticket.strike
    sigma = max(ticket.vol, 1e-8)
    T = max(ticket.expiry_years, 1e-8)
    df = ticket.df()
    payout_rate = ticket.payout_bp * 0.0001
    annuity = ticket.annuity if ticket.annuity > 0 else ticket.swap_tenor_years

    lnFK = math.log(F / K)
    vol_sqrt_t = sigma * math.sqrt(T)
    d2 = (lnFK - 0.5 * sigma * sigma * T) / vol_sqrt_t
    N = NormalDist().cdf
    phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    if ticket.side.lower().startswith("payer"):
        prob = N(d2)
        delta_prob = phi(d2) / (F * sigma * math.sqrt(T))
    else:
        prob = N(-d2)
        delta_prob = -phi(d2) / (F * sigma * math.sqrt(T))

    pv = df * annuity * ticket.notional * payout_rate * prob
    delta = df * annuity * ticket.notional * payout_rate * delta_prob
    bpv = df * annuity * ticket.notional * 0.0001
    pv_bp = pv / bpv if bpv != 0 else 0.0

    return {"pv": pv, "pv_bp": pv_bp, "delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "bpv": bpv}


def bachelier_swaption_digital(ticket: SwaptionTicket) -> dict:
    F = ticket.forward
    K = ticket.strike
    sigma_n = max(ticket.vol, 1e-8)
    T = max(ticket.expiry_years, 1e-8)
    df = ticket.df()
    payout_rate = ticket.payout_bp * 0.0001
    annuity = ticket.annuity if ticket.annuity > 0 else ticket.swap_tenor_years

    d = (F - K) / (sigma_n * math.sqrt(T))
    phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    N = NormalDist().cdf(d)

    if ticket.side.lower().startswith("payer"):
        prob = N
        delta_prob = phi / (sigma_n * math.sqrt(T))
    else:
        prob = 1 - N
        delta_prob = -phi / (sigma_n * math.sqrt(T))

    pv = df * annuity * ticket.notional * payout_rate * prob
    delta = df * annuity * ticket.notional * payout_rate * delta_prob
    bpv = df * annuity * ticket.notional * 0.0001
    pv_bp = pv / bpv if bpv != 0 else 0.0

    return {"pv": pv, "pv_bp": pv_bp, "delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "bpv": bpv}


def price_swaption(ticket: SwaptionTicket) -> dict:
    if ticket.model == "Black":
        if ticket.payoff_type == "vanilla":
            return black_swaption_vanilla(ticket)
        elif ticket.payoff_type == "digital":
            return black_swaption_digital(ticket)
        elif ticket.payoff_type == "straddle":
            t_p = SwaptionTicket(**{**ticket.__dict__, "side": "Payer", "payoff_type": "vanilla"})
            t_r = SwaptionTicket(**{**ticket.__dict__, "side": "Receiver", "payoff_type": "vanilla"})
            p1 = black_swaption_vanilla(t_p)
            p2 = black_swaption_vanilla(t_r)
            return {k: p1.get(k, 0.0) + p2.get(k, 0.0) for k in p1.keys()}
    else:
        if ticket.payoff_type == "vanilla":
            return bachelier_swaption_vanilla(ticket)
        elif ticket.payoff_type == "digital":
            return bachelier_swaption_digital(ticket)
        elif ticket.payoff_type == "straddle":
            t_p = SwaptionTicket(**{**ticket.__dict__, "side": "Payer", "payoff_type": "vanilla"})
            t_r = SwaptionTicket(**{**ticket.__dict__, "side": "Receiver", "payoff_type": "vanilla"})
            p1 = bachelier_swaption_vanilla(t_p)
            p2 = bachelier_swaption_vanilla(t_r)
            return {k: p1.get(k, 0.0) + p2.get(k, 0.0) for k in p1.keys()}
    return bachelier_swaption_vanilla(ticket)


# ============================
# Caps / Floors
# ============================

def black_caplet(notional: float, accrual: float,
                 F: float, K: float, sigma: float, T: float,
                 r: float, is_cap: bool = True) -> dict:
    if T <= 0 or sigma <= 0:
        return {"pv": 0.0, "delta": 0.0, "vega": 0.0, "gamma": 0.0}
    df = math.exp(-r * T)
    if F <= 0 or K <= 0:
        return {"pv": 0.0, "delta": 0.0, "vega": 0.0, "gamma": 0.0}
    lnFK = math.log(F / K)
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (lnFK + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    N = NormalDist().cdf
    phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    if is_cap:
        price_rate = F * N(d1) - K * N(d2)
        delta_rate = N(d1)
    else:
        price_rate = K * N(-d2) - F * N(-d1)
        delta_rate = -N(-d1)
    vega_rate = F * phi(d1) * math.sqrt(T)
    gamma_rate = phi(d1) / (F * sigma * math.sqrt(T))
    pv = notional * accrual * df * price_rate
    delta = notional * accrual * df * delta_rate
    vega = notional * accrual * df * vega_rate
    gamma = notional * accrual * df * gamma_rate
    return {"pv": pv, "delta": delta, "vega": vega, "gamma": gamma}


def bachelier_caplet(notional: float, accrual: float,
                     F: float, K: float, sigma_n: float, T: float,
                     r: float, is_cap: bool = True) -> dict:
    if T <= 0 or sigma_n <= 0:
        return {"pv": 0.0, "delta": 0.0, "vega": 0.0, "gamma": 0.0}
    df = math.exp(-r * T)
    d = (F - K) / (sigma_n * math.sqrt(T))
    phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    N = NormalDist().cdf(d)
    if is_cap:
        price_rate = (F - K) * N + sigma_n * math.sqrt(T) * phi
        delta_rate = N
    else:
        price_rate = (K - F) * (1 - N) + sigma_n * math.sqrt(T) * phi
        delta_rate = N - 1
    pv = notional * accrual * df * price_rate
    delta = notional * accrual * df * delta_rate
    vega = notional * accrual * df * math.sqrt(T) * phi
    gamma = notional * accrual * df * phi / (sigma_n * math.sqrt(T))
    return {"pv": pv, "delta": delta, "vega": vega, "gamma": gamma}


# ============================
# XVA helpers
# ============================

def simple_exposure_profile(pv0: float,
                            vega: float,
                            vol: float,
                            horizon_years: float,
                            n_steps: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0.0, horizon_years, n_steps)
    ee = pv0 + vega * vol * np.sqrt(np.maximum(ts, 1e-8))
    return ts, np.maximum(ee, 0.0)


def cva_from_hazard(ee_times: np.ndarray,
                    ee_values: np.ndarray,
                    hazard_rate: float,
                    lgd: float,
                    discount_rate: float) -> float:
    h = max(hazard_rate, 0.0)
    ts = ee_times
    cvs = 0.0
    prev_pd = 0.0
    for t, ee in zip(ts, ee_values):
        pd_t = 1.0 - math.exp(-h * t)
        dPD = max(pd_t - prev_pd, 0.0)
        df = math.exp(-discount_rate * t)
        cvs += ee * dPD * df
        prev_pd = pd_t
    return -lgd * cvs


# ============================
# Theme
# ============================

def apply_rateedge_theme(theme_name: str):
    """
    RateEdge brand themes using official colors from brand guide.
    Navy: #1e3a5f, Blue: #2563eb, Red: #dc2626, Slate: #0f172a
    """
    if theme_name == "Clean Light":
        bg = "#f1f5f9"       # gray-100
        card = "#ffffff"
        border = "#e2e8f0"   # gray-200
        text = "#1e3a5f"     # navy
        accent = "#dc2626"   # red (brand)
        accent2 = "#2563eb"  # blue (brand)
        muted = "#64748b"    # gray-500
        tab_text = "#1e3a5f"
        sidebar_arrow = "#1e3a5f"
    else:  # Dealer Dark (default)
        bg = "#0f172a"       # slate dark
        card = "#1e293b"
        border = "#334155"   # gray-700
        text = "#f1f5f9"     # gray-100
        accent = "#ef4444"   # red-light
        accent2 = "#3b82f6"  # blue-light
        muted = "#94a3b8"    # gray-400
        tab_text = "#f1f5f9"
        sidebar_arrow = "#f1f5f9"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {card};
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {{
            color: {text} !important;
        }}
        /* Sidebar collapse/expand button - make arrows visible */
        [data-testid="stSidebar"] button[kind="header"],
        [data-testid="collapsedControl"] {{
            color: {sidebar_arrow} !important;
        }}
        [data-testid="stSidebar"] svg,
        [data-testid="collapsedControl"] svg {{
            fill: {sidebar_arrow} !important;
            stroke: {sidebar_arrow} !important;
        }}
        button[kind="headerNoPadding"] svg {{
            fill: {sidebar_arrow} !important;
            stroke: {sidebar_arrow} !important;
        }}
        /* Collapsed sidebar button */
        .css-1rs6os {{
            color: {sidebar_arrow} !important;
        }}
        .css-1rs6os svg {{
            fill: {sidebar_arrow} !important;
        }}
        /* Alternative selectors for sidebar toggle */
        [data-testid="baseButton-headerNoPadding"] {{
            color: {sidebar_arrow} !important;
        }}
        [data-testid="baseButton-headerNoPadding"] svg {{
            fill: {sidebar_arrow} !important;
            stroke: {sidebar_arrow} !important;
        }}
        /* Tabs - readable text */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: {card};
            padding: 0.5rem;
            border-radius: 8px;
            border: 1px solid {border};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {tab_text} !important;
            background-color: transparent;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-weight: 500;
            font-size: 0.85rem;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {border};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent} !important;
            color: white !important;
        }}
        /* Tab indicator bar - hide default and use background instead */
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: transparent !important;
        }}
        .stTabs [data-baseweb="tab-border"] {{
            background-color: transparent !important;
        }}
        /* Cards */
        .rateedge-card {{
            background-color: {card};
            padding: 1.0rem 1.2rem;
            border-radius: 14px;
            border: 1px solid {border};
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .rateedge-header {{
            font-size: 1.6rem;
            font-weight: 650;
            color: {text};
            margin-bottom: 0.2rem;
        }}
        .rateedge-sub {{
            font-size: 0.9rem;
            color: {muted};
        }}
        .rateedge-badge {{
            display:inline-block;
            padding:0.15rem 0.5rem;
            font-size:0.72rem;
            border-radius:999px;
            border:1px solid {accent};
            color:{accent};
            margin-left:0.5rem;
        }}
        /* Text */
        label, .stMarkdown, .stText {{
            color: {text} !important;
        }}
        h1, h2, h3, h4 {{
            color: {text} !important;
        }}
        /* Buttons */
        .stButton>button {{
            background-color: {accent};
            color: white;
            border-radius: 8px;
            border: 1px solid {accent};
            font-weight: 600;
        }}
        .stButton>button:hover {{
            background-color: {accent2};
            color: white;
            border-color: {accent2};
        }}
        /* Download buttons - make them visible */
        .stDownloadButton>button {{
            background-color: {accent};
            color: white !important;
            border-radius: 8px;
            border: 1px solid {accent};
            font-weight: 600;
        }}
        .stDownloadButton>button:hover {{
            background-color: {accent2};
            color: white !important;
            border-color: {accent2};
        }}
        /* Inputs */
        .stTextInput input, .stNumberInput input {{
            background-color: {card} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
        }}
        .stSelectbox > div > div {{
            background-color: {card} !important;
            color: {text} !important;
        }}
        /* DataFrames */
        .stDataFrame {{
            color: {text};
        }}
        /* Landing page cards */
        .landing-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }}
        .feature-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 1.5rem;
        }}
        /* Status cards */
        .status-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        .status-ok {{
            color: #22c55e;
        }}
        .status-no {{
            color: #ef4444;
        }}
        .status-time {{
            color: {muted};
            font-size: 0.75rem;
        }}
        /* NUCLEAR FIX: Force all form label text to be visible */
        /* Target the actual rendered text elements in Streamlit */
        .stRadio > label,
        .stRadio label p,
        .stRadio label span,
        .stRadio div[role="radiogroup"] label,
        .stRadio div[role="radiogroup"] label p,
        .stRadio div[role="radiogroup"] label span,
        .stRadio div[data-testid="stMarkdownContainer"] p,
        .stCheckbox > label,
        .stCheckbox label p,
        .stCheckbox label span,
        .stCheckbox div[data-testid="stMarkdownContainer"] p,
        div[data-baseweb="radio"] ~ div,
        div[data-baseweb="radio"] ~ div p,
        div[data-baseweb="checkbox"] ~ div,
        div[data-baseweb="checkbox"] ~ div p {{
            color: {text} !important;
            -webkit-text-fill-color: {text} !important;
        }}
        /* Target the markdown text inside radio/checkbox labels */
        [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
        [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
        [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p {{
            color: {text} !important;
            -webkit-text-fill-color: {text} !important;
        }}
        /* Disabled state - still readable but dimmed */
        [data-testid="stCheckbox"][aria-disabled="true"] label,
        [data-testid="stCheckbox"][aria-disabled="true"] p,
        [data-testid="stRadio"][aria-disabled="true"] label,
        [data-testid="stRadio"][aria-disabled="true"] p {{
            color: {muted} !important;
            -webkit-text-fill-color: {muted} !important;
            opacity: 0.7;
        }}
        /* Selectbox dropdown text */
        .stSelectbox label,
        .stSelectbox [data-baseweb="select"] span {{
            color: {text} !important;
        }}
        /* Number input labels */
        .stNumberInput label {{
            color: {text} !important;
        }}
        /* Caption text */
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {muted} !important;
        }}
        /* Metric values - make readable in dark mode */
        [data-testid="stMetricValue"] {{
            color: {text} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {text} !important;
        }}
        [data-testid="stMetricDelta"] {{
            color: {text} !important;
        }}
        /* DataFrame text colors */
        .stDataFrame, .stDataFrame td, .stDataFrame th {{
            color: {text} !important;
        }}
        [data-testid="stDataFrame"] {{
            color: {text} !important;
        }}
        [data-testid="stDataFrame"] td {{
            color: {text} !important;
        }}
        [data-testid="stDataFrame"] th {{
            color: {text} !important;
        }}
        /* Table cells in dataframes */
        .dvn-scroller td, .dvn-scroller th {{
            color: {text} !important;
        }}
        /* Expander text */
        .streamlit-expanderHeader {{
            color: {text} !important;
        }}
        [data-testid="stExpander"] summary {{
            color: {text} !important;
        }}
        /* Radio buttons - WHITE circles */
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
            background-color: #ffffff !important;
            border-color: #ffffff !important;
        }}
        .stRadio > div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child > div {{
            background-color: #ffffff !important;
        }}
        [data-testid="stRadio"] [data-baseweb="radio"] input:checked + div {{
            background-color: #ffffff !important;
            border-color: #ffffff !important;
        }}
        /* NUCLEAR - ALL TEXT INSIDE RADIO/CHECKBOX */
        [data-testid="stRadio"] * {{
            color: #fbbf24 !important;
            -webkit-text-fill-color: #fbbf24 !important;
        }}
        [data-testid="stCheckbox"] * {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}
        /* Also target by class */
        .stRadio * {{
            color: #fbbf24 !important;
            -webkit-text-fill-color: #fbbf24 !important;
        }}
        .stCheckbox * {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}
        /* DISABLED checkboxes - still visible but dimmed */
        [data-testid="stCheckbox"][aria-disabled="true"] *,
        .stCheckbox[aria-disabled="true"] *,
        div[data-testid="stCheckbox"] input:disabled ~ label,
        div[data-testid="stCheckbox"] input:disabled ~ label * {{
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 0.7;
        }}
        /* Target the actual label text element directly */
        [data-baseweb="checkbox"] + div,
        [data-baseweb="checkbox"] ~ div,
        [data-baseweb="checkbox"] ~ div p,
        [data-baseweb="checkbox"] ~ div span {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
    
    # Force radio/checkbox colors with JavaScript (runs after render)
    import streamlit.components.v1 as components
    components.html("""
    <script>
    function fixColors() {
        const p = window.parent.document;
        // Checkboxes - WHITE
        p.querySelectorAll('[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span, [data-testid="stCheckbox"] p, [data-testid="stCheckbox"] div').forEach(el => {
            el.style.setProperty('color', '#ffffff', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
        });
        // Radio buttons - YELLOW
        p.querySelectorAll('[data-testid="stRadio"] label, [data-testid="stRadio"] span, [data-testid="stRadio"] p, [data-testid="stRadio"] div').forEach(el => {
            el.style.setProperty('color', '#fbbf24', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#fbbf24', 'important');
        });
        // Baseweb checkbox labels - WHITE
        p.querySelectorAll('[data-baseweb="checkbox"] ~ div').forEach(el => {
            el.style.setProperty('color', '#ffffff', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#ffffff', 'important');
        });
        // Baseweb radio labels - YELLOW
        p.querySelectorAll('[data-baseweb="radio"] ~ div').forEach(el => {
            el.style.setProperty('color', '#fbbf24', 'important');
            el.style.setProperty('-webkit-text-fill-color', '#fbbf24', 'important');
        });
    }
    fixColors();
    setInterval(fixColors, 300);
    </script>
    """, height=0)


def show_header(ccy: str):
    st.markdown(
        f"""
        <div class="rateedge-header">
           RateEdge Options  {ccy}
           <span class="rateedge-badge">Swaptions  Caps/Floors  Exotics  CVA  RV</span>
        </div>
        <div class="rateedge-sub">
           Premium at expiry  swaps start T+1BD  SABR smiles  CVA / horizon / RV tools.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================
# Session state helpers
# ============================

def init_session():
    if "swaption_portfolio" not in st.session_state:
        st.session_state["swaption_portfolio"] = []
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []
    if "vol_data" not in st.session_state:
        st.session_state["vol_data"] = {}
    if "curves" not in st.session_state:
        st.session_state["curves"] = {}  # {ccy: {"irs": df, "6v3": df, "3v1": df, "ois": df}}
    if "basis_curves" not in st.session_state:
        st.session_state["basis_curves"] = {}  # {ccy: {"6v3": df, "3v1": df}}
    if "vol_editor" not in st.session_state:
        st.session_state["vol_editor"] = {"working": {}, "base": {}, "history": {}, "future": {}}
    if "history_configs" not in st.session_state:
        st.session_state["history_configs"] = {}
    if "theme_name" not in st.session_state:
        st.session_state["theme_name"] = "Dealer Dark"
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    # Timestamps for data loads
    if "load_timestamps" not in st.session_state:
        st.session_state["load_timestamps"] = {
            "atm": {},    # {ccy: datetime}
            "sabr": {},   # {ccy: datetime}
            "curves": {}, # {ccy: datetime}
        }
    # Forward swap matrix cache
    if "fwd_matrix" not in st.session_state:
        st.session_state["fwd_matrix"] = {}
    # Track if we've auto-loaded from DB this session
    if "db_auto_loaded" not in st.session_state:
        st.session_state["db_auto_loaded"] = False

# Auth credentials
# Auth handled by email OTP


def get_timestamp_str(category: str, ccy: str) -> str:
    """Get formatted timestamp string for a category/currency"""
    from datetime import datetime
    ts = st.session_state.get("load_timestamps", {}).get(category, {}).get(ccy)
    if ts is None:
        return "Not loaded"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def set_timestamp(category: str, ccy: str):
    """Set current timestamp for a category/currency"""
    from datetime import datetime
    if "load_timestamps" not in st.session_state:
        st.session_state["load_timestamps"] = {"atm": {}, "sabr": {}, "curves": {}}
    st.session_state["load_timestamps"][category][ccy] = datetime.now()


def get_basis_curve(ccy: str, basis_type: str = "6v3") -> Optional[pd.DataFrame]:
    """Get basis curve for currency"""
    return st.session_state.get("basis_curves", {}).get(ccy, {}).get(basis_type)


def set_basis_curve(ccy: str, basis_type: str, df: pd.DataFrame):
    """Set basis curve for currency"""
    if "basis_curves" not in st.session_state:
        st.session_state["basis_curves"] = {}
    if ccy not in st.session_state["basis_curves"]:
        st.session_state["basis_curves"][ccy] = {}
    st.session_state["basis_curves"][ccy][basis_type] = df


def parse_tenor_to_years(tenor_str: str) -> float:
    """Parse tenor string like '3m', '6m', '1y', '2Y', 'AUD IRS 5y SS' to years"""
    s = str(tenor_str).strip().lower()
    # Handle descriptive formats like "AUD IRS 5y SS"
    import re
    match = re.search(r'(\d+(?:\.\d+)?)\s*y', s)
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+(?:\.\d+)?)\s*m', s)
    if match:
        return float(match.group(1)) / 12.0
    match = re.search(r'(\d+(?:\.\d+)?)\s*w', s)
    if match:
        return float(match.group(1)) / 52.0
    match = re.search(r'(\d+(?:\.\d+)?)\s*d', s)
    if match:
        return float(match.group(1)) / 365.0
    # Try direct float
    try:
        return float(s)
    except:
        return 0.0


def load_curve_flexible(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Load curve with flexible tenor parsing"""
    if {"Maturity (Years)", "Zero Rate (%)"}.issubset(df.columns):
        out = df[["Maturity (Years)", "Zero Rate (%)"]].copy()
        out.rename(columns={"Maturity (Years)": "Tenor", "Zero Rate (%)": "ZeroRatePct"}, inplace=True)
        # Parse tenor to years
        out["MaturityY"] = out["Tenor"].apply(parse_tenor_to_years)
        out = out[out["MaturityY"] > 0].sort_values("MaturityY").reset_index(drop=True)
        return out
    raise ValueError(f"{name}: expected columns 'Maturity (Years)', 'Zero Rate (%)'")


def load_basis_curve_flexible(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Load basis curve with flexible tenor parsing"""
    if {"Tenor (Years)", "Basis (bp)"}.issubset(df.columns):
        out = df[["Tenor (Years)", "Basis (bp)"]].copy()
        out.rename(columns={"Tenor (Years)": "Tenor", "Basis (bp)": "BasisBp"}, inplace=True)
        out["MaturityY"] = out["Tenor"].apply(parse_tenor_to_years)
        out = out[out["MaturityY"] > 0].sort_values("MaturityY").reset_index(drop=True)
        return out
    raise ValueError(f"{name}: expected columns 'Tenor (Years)', 'Basis (bp)'")


def get_ccy_vol_data(ccy: str):
    v = st.session_state["vol_data"].get(ccy, {})
    return (v.get("atm"), v.get("alpha"), v.get("beta"), v.get("rho"), v.get("nu"))


def set_ccy_vol_data(ccy: str, atm, a, b, r, n):
    st.session_state["vol_data"][ccy] = {"atm": atm, "alpha": a, "beta": b, "rho": r, "nu": n}
    ve = st.session_state["vol_editor"]
    if atm is not None:
        ve["base"][ccy] = atm.copy()
        ve["working"][ccy] = atm.copy()
        ve["history"][ccy] = []
        ve["future"][ccy] = []


def get_ccy_curve(ccy: str) -> Optional[pd.DataFrame]:
    return st.session_state["curves"].get(ccy)


def set_ccy_curve(ccy: str, curve_df: pd.DataFrame):
    st.session_state["curves"][ccy] = curve_df


def load_config_excel(upload, load_type: str = "all") -> dict:
    """
    Load config from Excel with selective loading.
    load_type: "atm", "sabr", "curves", or "all"
    Returns dict with counts of what was loaded.
    """
    xl = pd.ExcelFile(upload)
    loaded = {"atm": 0, "sabr": 0, "curves": 0, "basis": 0}
    
    for ccy in SUPPORTED_CURRENCIES:
        # Load ATM vols
        if load_type in ["atm", "all"]:
            atm_name = f"ATM_Vols_{ccy}"
            if atm_name in xl.sheet_names:
                atm_raw = pd.read_excel(xl, sheet_name=atm_name)
                atm_df = load_atm_surface(atm_raw, atm_name)
                # Get existing SABR data to preserve
                _, old_a, old_b, old_r, old_n = get_ccy_vol_data(ccy)
                set_ccy_vol_data(ccy, atm_df, old_a, old_b, old_r, old_n)
                set_timestamp("atm", ccy)
                loaded["atm"] += 1

        # Load SABR grids
        if load_type in ["sabr", "all"]:
            sabr_a = sabr_b = sabr_r = sabr_n = None
            has_sabr = False
            for base in ["SABR_Alpha", "SABR_Beta", "SABR_Rho", "SABR_Nu"]:
                sname = f"{base}_{ccy}"
                if sname in xl.sheet_names:
                    df = pd.read_excel(xl, sheet_name=sname)
                    df = ensure_sabr_matrix(df, sname)
                    has_sabr = True
                else:
                    df = None
                if base == "SABR_Alpha":
                    sabr_a = df
                elif base == "SABR_Beta":
                    sabr_b = df
                elif base == "SABR_Rho":
                    sabr_r = df
                elif base == "SABR_Nu":
                    sabr_n = df
            
            if has_sabr:
                # Get existing ATM to preserve
                old_atm, _, _, _, _ = get_ccy_vol_data(ccy)
                set_ccy_vol_data(ccy, old_atm, sabr_a, sabr_b, sabr_r, sabr_n)
                set_timestamp("sabr", ccy)
                loaded["sabr"] += 1

        # Load curves and basis curves
        if load_type in ["curves", "all"]:
            # Main IRS curve
            curve_name = f"Curves_{ccy}"
            if curve_name in xl.sheet_names:
                raw_curve = pd.read_excel(xl, sheet_name=curve_name)
                try:
                    curve_df = load_curve_flexible(raw_curve, curve_name)
                except:
                    curve_df = load_curve(raw_curve, curve_name)
                set_ccy_curve(ccy, curve_df)
                set_timestamp("curves", ccy)
                loaded["curves"] += 1
            
            # 6v3 basis curve
            basis_6v3_name = f"Basis_{ccy}_6v3"
            if basis_6v3_name in xl.sheet_names:
                raw_basis = pd.read_excel(xl, sheet_name=basis_6v3_name)
                try:
                    basis_df = load_basis_curve_flexible(raw_basis, basis_6v3_name)
                    set_basis_curve(ccy, "6v3", basis_df)
                    loaded["basis"] += 1
                except Exception as e:
                    pass  # Basis loading is optional
            
            # 3v1 basis curve (if exists)
            basis_3v1_name = f"Basis_{ccy}_3v1"
            if basis_3v1_name in xl.sheet_names:
                raw_basis = pd.read_excel(xl, sheet_name=basis_3v1_name)
                try:
                    basis_df = load_basis_curve_flexible(raw_basis, basis_3v1_name)
                    set_basis_curve(ccy, "3v1", basis_df)
                    loaded["basis"] += 1
                except:
                    pass
            
            # OIS curve (if exists)
            ois_name = f"OIS_{ccy}"
            if ois_name in xl.sheet_names:
                raw_ois = pd.read_excel(xl, sheet_name=ois_name)
                try:
                    ois_df = load_curve_flexible(raw_ois, ois_name)
                    # Store in basis_curves with "ois" key
                    set_basis_curve(ccy, "ois", ois_df)
                except:
                    pass
    
    return loaded


def get_working_atm_surface(ccy: str) -> Optional[pd.DataFrame]:
    ve = st.session_state.get("vol_editor", {})
    working = ve.get("working", {})
    if ccy in working and isinstance(working[ccy], pd.DataFrame):
        return working[ccy]
    atm, _, _, _, _ = get_ccy_vol_data(ccy)
    return atm


def push_vol_history(ccy: str):
    ve = st.session_state["vol_editor"]
    working = ve["working"].get(ccy)
    if working is not None:
        ve["history"].setdefault(ccy, []).append(working.copy())
        ve["future"][ccy] = []


def undo_vol(ccy: str):
    ve = st.session_state["vol_editor"]
    hist = ve["history"].get(ccy, [])
    if hist:
        current = ve["working"].get(ccy)
        ve["future"].setdefault(ccy, [])
        if current is not None:
            ve["future"][ccy].append(current.copy())
        ve["working"][ccy] = hist.pop()


def redo_vol(ccy: str):
    ve = st.session_state["vol_editor"]
    fut = ve["future"].get(ccy, [])
    if fut:
        current = ve["working"].get(ccy)
        ve["history"].setdefault(ccy, [])
        if current is not None:
            ve["history"][ccy].append(current.copy())
        ve["working"][ccy] = fut.pop()


def publish_vol(ccy: str):
    ve = st.session_state["vol_editor"]
    working = ve["working"].get(ccy)
    if working is not None:
        ve["base"][ccy] = working.copy()
        atm, a, b, r, n = get_ccy_vol_data(ccy)
        set_ccy_vol_data(ccy, working.copy(), a, b, r, n)


# ============================
# Tabs
# ============================

def vol_config_tab():
    st.subheader(" Vol / SABR Config & Upload")
    
    # Get theme colors
    is_dark = st.session_state.get("theme_name", "Dealer Dark") == "Dealer Dark"
    card_bg = "#1e293b" if is_dark else "#ffffff"
    border_color = "#334155" if is_dark else "#e2e8f0"
    text_color = "#f1f5f9" if is_dark else "#1e3a5f"
    muted_color = "#94a3b8" if is_dark else "#64748b"
    
    # Database load button - ALWAYS visible at top
    db_url = get_db_url()
    if HAS_POSTGRES and db_url:
        col_db1, col_db2, col_db3 = st.columns([2, 2, 4])
        with col_db1:
            if st.button(" Load from Database", key="load_db_btn_top", type="primary"):
                user_id = st.session_state.get("username", "default")
                loaded_count = load_all_session_data(user_id)
                if loaded_count > 0:
                    st.success(f" Loaded {loaded_count} configs from database")
                    st.rerun()
                else:
                    st.warning("No saved data found in database")
        with col_db2:
            if st.button(" Save to Database", key="save_db_btn_top", type="secondary"):
                user_id = st.session_state.get("username", "default")
                saved = save_all_session_data(user_id)
                if saved > 0:
                    st.success(f" Saved {saved} configs to database")
                else:
                    st.warning("Nothing to save - vol_data may be empty")
        with col_db3:
            if st.button(" Clear Corrupted DB Data", key="clear_db_btn"):
                user_id = st.session_state.get("username", "default")
                conn = get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute("DELETE FROM user_configs WHERE user_id = %s", (user_id,))
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.success(" Cleared all database configs. Now upload Excel and Save.")
                    except Exception as e:
                        st.error(f"Clear failed: {e}")
        with col_db3:
            st.caption(" Database connected")
    
    st.markdown("---")
    
    # File upload
    st.markdown("#### Upload Config File")
    upload = st.file_uploader(
        "Upload RateEdge_Config.xlsx",
        type=["xlsx"],
        key="cfg_upload",
        help="Excel file with sheets: ATM_Vols_[CCY], SABR_*_[CCY], Curves_[CCY]"
    )
    
    if upload is not None:
        st.markdown("#### Select what to commit:")
        
        load_type = st.radio(
            "Commit options",
            ["All", "ATM Vol Only", "SABR Only", "IRS Curves Only"],
            index=0,
            horizontal=True,
            key="load_type_radio"
        )
        
        # Map selection to load_type
        type_map = {
            "All": "all",
            "ATM Vol Only": "atm",
            "SABR Only": "sabr",
            "IRS Curves Only": "curves"
        }
        
        if st.button(" Commit Selected Data", key="commit_btn", type="primary"):
            selected_type = type_map[load_type]
            loaded = load_config_excel(upload, selected_type)
            
            # Show what was loaded
            msgs = []
            if loaded["atm"] > 0:
                msgs.append(f"ATM Vols: {loaded['atm']} currencies")
            if loaded["sabr"] > 0:
                msgs.append(f"SABR Grids: {loaded['sabr']} currencies")
            if loaded["curves"] > 0:
                msgs.append(f"Curves: {loaded['curves']} currencies")
            
            if msgs:
                st.success(f" Loaded: {', '.join(msgs)}")
            else:
                st.warning("No matching data found in file for selected option.")
    
    st.markdown("---")
    st.markdown("#### Currently Loaded Status")
    
    # Show database status
    if HAS_POSTGRES and get_db_url():
        st.caption(" Database: Connected")
    else:
        st.caption(" Database: Not configured")
    
    for ccy in SUPPORTED_CURRENCIES:
        atm, a, b, r, n = get_ccy_vol_data(ccy)
        curve = get_ccy_curve(ccy)
        
        atm_status = "" if atm is not None else ""
        sabr_status = "" if a is not None else ""
        curve_status = "" if curve is not None else ""
        
        atm_time = get_timestamp_str("atm", ccy)
        sabr_time = get_timestamp_str("sabr", ccy)
        curve_time = get_timestamp_str("curves", ccy)
        
        st.markdown(
            f"""
            <div style="background:{card_bg};border:1px solid {border_color};border-radius:10px;padding:1rem;margin:0.5rem 0;">
                <div style="font-weight:600;font-size:1.1rem;color:{text_color};margin-bottom:0.5rem;">
                    {ccy}
                </div>
                <table style="width:100%;color:{text_color};font-size:0.9rem;">
                    <tr>
                        <td style="padding:0.25rem 0;">ATM Surface</td>
                        <td style="padding:0.25rem 0;">{atm_status}</td>
                        <td style="padding:0.25rem 0;color:{muted_color};font-size:0.75rem;">{atm_time}</td>
                    </tr>
                    <tr>
                        <td style="padding:0.25rem 0;">SABR Grids</td>
                        <td style="padding:0.25rem 0;">{sabr_status}</td>
                        <td style="padding:0.25rem 0;color:{muted_color};font-size:0.75rem;">{sabr_time}</td>
                    </tr>
                    <tr>
                        <td style="padding:0.25rem 0;">IRS Curve</td>
                        <td style="padding:0.25rem 0;">{curve_status}</td>
                        <td style="padding:0.25rem 0;color:{muted_color};font-size:0.75rem;">{curve_time}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def curves_tab():
    st.subheader(" Curves & Forward Swap Matrix")
    
    # Get theme colors
    is_dark = st.session_state.get("theme_name", "Dealer Dark") == "Dealer Dark"
    bg_color = "#0f172a" if is_dark else "#ffffff"
    grid_color = "#334155" if is_dark else "#e2e8f0"
    text_color = "#f1f5f9" if is_dark else "#1e3a5f"
    
    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="curve_ccy")
    
    curve = get_ccy_curve(ccy)
    if curve is None:
        st.info("No curve loaded yet. Upload RateEdge_Config.xlsx in Vol/SABR tab first.")
        return
    
    # Get basis curves
    basis_6v3 = get_basis_curve(ccy, "6v3")
    basis_3v1 = get_basis_curve(ccy, "3v1")
    ois_curve = get_basis_curve(ccy, "ois")
    
    # LOCAL CSS fix for checkbox visibility - NUCLEAR
    st.markdown("""
    <style>
    /* NUCLEAR checkbox text fix */
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
    div[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stCheckbox"] * {
        color: #fbbf24 !important; 
        -webkit-text-fill-color: #fbbf24 !important;
    }
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"],
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] * {
        color: #fbbf24 !important; 
        -webkit-text-fill-color: #fbbf24 !important;
    }
    .stCheckbox > label, .stCheckbox label span, .stCheckbox * { color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; }
    .stRadio > label, .stRadio label span, .stRadio * { color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; }
    /* Disabled checkboxes - still visible but dimmed */
    div[data-testid="stCheckbox"][aria-disabled="true"] label,
    div[data-testid="stCheckbox"][aria-disabled="true"] span {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        opacity: 0.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Curve selection - simplified checkboxes
    st.markdown("#### Select Curves to Display")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_irs = st.checkbox("IRS (Main)", value=True, key="show_irs")
    with col2:
        show_6v3 = st.checkbox("6v3 Basis", value=basis_6v3 is not None, disabled=basis_6v3 is None, key="show_6v3")
    with col3:
        show_3v1 = st.checkbox("3v1 Basis", value=False, disabled=basis_3v1 is None, key="show_3v1")
    with col4:
        show_ois = st.checkbox("OIS", value=False, disabled=ois_curve is None, key="show_ois")
    
    # Create Plotly chart
    fig = go.Figure()
    
    colors = {
        "IRS": "#3b82f6",
        "6v3 Basis": "#ef4444",
        "3v1 Basis": "#22c55e",
        "OIS": "#f59e0b",
    }
    
    if show_irs and curve is not None:
        fig.add_trace(go.Scatter(
            x=curve["MaturityY"], y=curve["ZeroRatePct"],
            mode="lines+markers", name="IRS",
            line=dict(color=colors["IRS"], width=2), marker=dict(size=5),
        ))
    
    if show_6v3 and basis_6v3 is not None:
        fig.add_trace(go.Scatter(
            x=basis_6v3["MaturityY"], y=basis_6v3["BasisBp"],
            mode="lines+markers", name="6v3 Basis (bp)",
            line=dict(color=colors["6v3 Basis"], width=2), marker=dict(size=5),
            yaxis="y2",
        ))
    
    if show_3v1 and basis_3v1 is not None:
        fig.add_trace(go.Scatter(
            x=basis_3v1["MaturityY"], y=basis_3v1["BasisBp"],
            mode="lines+markers", name="3v1 Basis (bp)",
            line=dict(color=colors["3v1 Basis"], width=2), marker=dict(size=5),
            yaxis="y2",
        ))
    
    if show_ois and ois_curve is not None:
        fig.add_trace(go.Scatter(
            x=ois_curve["MaturityY"], y=ois_curve["ZeroRatePct"],
            mode="lines+markers", name="OIS",
            line=dict(color=colors["OIS"], width=2), marker=dict(size=5),
        ))
    
    fig.update_layout(
        title=dict(text=f"{ccy} Curves", font=dict(size=16, color=text_color)),
        xaxis=dict(title="Maturity (Years)", gridcolor=grid_color, color=text_color),
        yaxis=dict(title="Rate (%)", gridcolor=grid_color, color=text_color, side="left"),
        yaxis2=dict(title="Basis (bp)", gridcolor=grid_color, color=text_color, overlaying="y", side="right"),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
        height=320,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Curve data tables - in expander
    with st.expander(" View Curve Data", expanded=False):
        if curve is not None:
            st.markdown("**IRS Curve**")
            st.dataframe(curve, use_container_width=True, hide_index=True, height=180)
        if basis_6v3 is not None:
            st.markdown("**6v3 Basis**")
            st.dataframe(basis_6v3, use_container_width=True, hide_index=True, height=180)
    
    # Forward Swap Matrix
    st.markdown("---")
    st.markdown("###  Forward Swap Matrix")
    st.caption("Forward swap rates for all expiry/tenor combinations - reference strikes for swaptions and caps/floors")
    
    # Initialize fwd_matrix in session state
    if "fwd_matrix" not in st.session_state:
        st.session_state["fwd_matrix"] = {}
    
    has_matrix = ccy in st.session_state.get("fwd_matrix", {})
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        if st.button(" Generate Forward Matrix", key="gen_fwd_matrix", type="primary"):
            with st.spinner("Generating..."):
                fwd_matrix = generate_forward_matrix(ccy, curve, basis_6v3)
                st.session_state["fwd_matrix"][ccy] = fwd_matrix
            st.rerun()
    with col2:
        if has_matrix:
            if st.button(" Refresh", key="refresh_fwd_matrix"):
                clear_matrix_cache()
                if ccy in st.session_state.get("fwd_matrix", {}):
                    del st.session_state["fwd_matrix"][ccy]
                st.rerun()
    with col3:
        show_fwd_heatmap = st.checkbox(" Heatmap", value=False, key="show_fwd_heatmap")
    with col4:
        if has_matrix:
            csv = st.session_state["fwd_matrix"][ccy].to_csv()
            st.download_button(" Download", csv, f"{ccy}_forward_matrix.csv", type="primary", key="dl_fwd_matrix")
    
    # Display forward matrix
    if has_matrix:
        fwd_df = st.session_state["fwd_matrix"][ccy]
        st.markdown("#### Forward Swap Rates (%)")
        if show_fwd_heatmap:
            st.dataframe(
                fwd_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", axis=None),
                use_container_width=True, height=600
            )
        else:
            st.dataframe(fwd_df.style.format("{:.4f}"), use_container_width=True, height=600)
    else:
        st.info(" Click 'Generate Forward Matrix' to calculate")
    
    # Convention notes
    st.markdown("---")
    if ccy == "AUD":
        st.caption(" AUD: 3m BBSW projection, q/q to 3y then s/s. T+1 spot lag. 6v3 basis applied.")
    elif ccy == "NZD":
        st.caption(" NZD: BKBM/OCR style, q/q to 2y then s/s. T+2 spot lag.")
    else:
        st.caption(" USD: SOFR-based, s/s throughout. T+2 spot lag.")


def generate_forward_matrix(ccy: str, curve: pd.DataFrame, basis_6v3: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Generate forward swap rate matrix - wrapper that calls cached version"""
    # Convert DataFrames to tuples for caching
    curve_tuple = tuple(curve["MaturityY"].tolist()), tuple(curve["ZeroRatePct"].tolist())
    basis_tuple = None
    if basis_6v3 is not None:
        basis_tuple = tuple(basis_6v3["MaturityY"].tolist()), tuple(basis_6v3["BasisBp"].tolist())
    
    return _generate_forward_matrix_cached(ccy, curve_tuple, basis_tuple)


@st.cache_data(ttl=3600, show_spinner=False)
def _generate_forward_matrix_cached(ccy: str, curve_tuple: tuple, basis_tuple: Optional[tuple] = None) -> pd.DataFrame:
    """Generate forward swap rate matrix - CACHED version"""
    
    expiries = ["1w", "1m", "2m", "3m", "6m", "9m", "1y", "18m", "2y", "3y", "4y", "5y", "7y", "10y", "12y", "15y", "20y"]
    tenors = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
    
    # Convert tuples back to numpy arrays
    curve_x = np.array(curve_tuple[0])
    curve_y = np.array(curve_tuple[1]) / 100.0  # Convert to decimal
    
    basis_x = basis_y = None
    if basis_tuple is not None:
        basis_x = np.array(basis_tuple[0])
        basis_y = np.array(basis_tuple[1])
    
    matrix = []
    
    for exp in expiries:
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        
        for tenor in tenors:
            tenor_y = float(tenor[:-1])
            try:
                fwd = fast_forward_rate(curve_x, curve_y, exp_y, tenor_y, ccy)
                
                if basis_x is not None and ccy == "AUD":
                    mid_t = exp_y + tenor_y / 2
                    basis_bp = float(np.interp(mid_t, basis_x, basis_y))
                    fwd = fwd + basis_bp / 10000.0
                
                row[tenor] = fwd * 100
            except:
                row[tenor] = None
        
        matrix.append(row)
    
    df = pd.DataFrame(matrix)
    df = df.set_index("Expiry")
    return df


def fast_forward_rate(curve_x: np.ndarray, curve_y: np.ndarray, expiry: float, tenor: float, ccy: str) -> float:
    """Fast forward swap rate calculation using numpy"""
    swap_start = expiry + 1.0 / 252.0
    swap_end = swap_start + tenor
    
    # Simple frequency based on currency
    if ccy == "AUD":
        freq = 0.25 if tenor <= 3 else 0.5
    else:
        freq = 0.5
    
    # Build payment times
    times = []
    t = swap_start + freq
    while t <= swap_end + 1e-8:
        times.append(min(t, swap_end))
        t += freq
    
    if not times:
        return 0.0
    
    # Calculate discount factors using numpy interpolation
    z_start = np.interp(swap_start, curve_x, curve_y)
    df_start = math.exp(-z_start * swap_start)
    
    z_end = np.interp(swap_end, curve_x, curve_y)
    df_end = math.exp(-z_end * swap_end)
    
    # Calculate annuity
    ann = 0.0
    prev_t = swap_start
    for t in times:
        z_t = np.interp(t, curve_x, curve_y)
        df_t = math.exp(-z_t * t)
        accrual = t - prev_t
        ann += df_t * accrual
        prev_t = t
    
    if ann <= 0:
        return 0.0
    
    return (df_start - df_end) / ann


def interpolate_basis(basis_df: pd.DataFrame, t: float) -> float:
    """Interpolate basis curve at time t"""
    if basis_df is None or basis_df.empty:
        return 0.0
    xs = basis_df["MaturityY"].to_numpy().astype(float)
    ys = basis_df["BasisBp"].to_numpy().astype(float)
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(t, xs, ys))


def swaptions_tab(ccy: str, vol_mode: str):
    st.subheader(" Swaptions")
    
    # Get curves and data
    fwd_matrix = st.session_state.get("fwd_matrix", {}).get(ccy)
    basis_6v3 = get_basis_curve(ccy, "6v3")
    ois_curve = get_basis_curve(ccy, "ois")
    curve = get_ccy_curve(ccy)
    
    # Row 1: Structure Type and Model
    col_struct, col_model, col_prem = st.columns([2, 1, 1])
    with col_struct:
        structure = st.selectbox(
            "Structure", 
            ["Payer", "Receiver", "ATM Straddle", "Strangle", "Risk Reversal", "Payer Ladder", "Receiver Ladder"],
            key="sw_structure"
        )
    with col_model:
        model_choice = st.selectbox("Model", ["Normal", "Black"], index=0, key="sw_model")
    with col_prem:
        premium_type = st.selectbox("Premium", ["Fwd", "Spot"], index=0, key="sw_prem_type")
    
    # For backwards compatibility
    side = structure

    # Row 2: Notional, Expiry, Tenor
    col_not, col_exp, col_tenor = st.columns(3)
    with col_not:
        # Default 100mm, minimum 1mm
        notional = st.number_input("Notional (mm)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="sw_not")
    with col_exp:
        expiry_options = ["1m","3m","6m","9m","1y","18m","2y","3y","5y","7y","10y"]
        expiry = st.selectbox("Expiry", expiry_options, index=3, key="sw_expiry")  # default 9m
        expiry_y = label_to_years(expiry)
    with col_tenor:
        tenor_options = ["1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"]
        swap_tenor = st.selectbox("Swap Tenor", tenor_options, index=3, key="sw_tenor")  # default 5Y
        tenor_y = float(swap_tenor[:-1])

    # Calculate forward
    fwd_from_matrix = None
    if fwd_matrix is not None:
        try:
            fwd_from_matrix = fwd_matrix.loc[expiry, swap_tenor]
        except:
            pass
    
    if fwd_from_matrix is not None:
        fwd = fwd_from_matrix / 100.0
        fwd_source = "matrix"
    elif curve is not None:
        fwd, _, _ = forward_and_annuity_from_curve(curve, ccy, expiry_y, tenor_y, ois_curve)
        if basis_6v3 is not None and ccy == "AUD":
            basis_bp = interpolate_basis(basis_6v3, expiry_y + tenor_y / 2)
            fwd = fwd + basis_bp / 10000.0
        fwd_source = "curve"
    else:
        fwd = 0.04
        fwd_source = "default"
    
    # Get annuity
    if curve is not None:
        _, ann, _ = forward_and_annuity_from_curve(curve, ccy, expiry_y, tenor_y, ois_curve)
    else:
        ann = tenor_y
    
    fwd_pct = fwd * 100

    # Row 3: Forward and Discount
    st.markdown("---")
    col_fwd, col_disc = st.columns(2)
    
    with col_fwd:
        st.metric("Forward (%)", f"{fwd_pct:.4f}", delta=fwd_source)
    
    with col_disc:
        has_ois = ois_curve is not None
        if has_ois:
            disc_method = st.radio("Discount", ["OIS", "Flat"], horizontal=True, key="sw_disc_method")
        else:
            disc_method = "Flat"
            st.caption(" No OIS curve")
        
        if disc_method == "Flat":
            flat_rate = st.number_input("Rate (%)", min_value=0.0, max_value=20.0, value=4.0, key="sw_disc_flat")
            eff_disc_rate = flat_rate / 100.0
            disc_source = "Flat"
        else:
            ois_xs = ois_curve["MaturityY"].to_numpy().astype(float)
            ois_ys = ois_curve["ZeroRatePct"].to_numpy().astype(float) / 100.0
            eff_disc_rate = float(np.interp(expiry_y, ois_xs, ois_ys))
            disc_source = "OIS"
            st.caption(f"OIS rate: {eff_disc_rate*100:.2f}%")

    # Strike inputs - varies by structure
    st.markdown("---")
    st.markdown("##### Strikes")
    
    # Initialize strike variables
    strike_pct = fwd_pct
    strike_pct_2 = fwd_pct
    strike_pct_3 = fwd_pct
    
    if structure == "ATM Straddle":
        strike_pct = fwd_pct
        st.info(f"ATM Strike: **{fwd_pct:.4f}%**")
        
    elif structure in ["Payer", "Receiver"]:
        # Strike mode selector
        strike_mode = st.radio("Strike Mode", ["ATM", "10 bp", "25 bp", "50 bp", "100 bp", "Manual"], 
                               horizontal=True, key="sw_strike_mode")
        
        # Calculate strike based on mode
        offset_map = {"ATM": 0, "10 bp": 10, "25 bp": 25, "50 bp": 50, "100 bp": 100, "Manual": None}
        offset = offset_map[strike_mode]
        
        if strike_mode == "Manual":
            strike_pct = st.number_input("Strike (%)", min_value=0.0, max_value=20.0, 
                                         value=round(fwd_pct, 4), format="%.4f", key="sw_strike_1")
        else:
            # For Payer, add offset; for Receiver, subtract offset
            if structure == "Payer":
                strike_pct = fwd_pct + offset/100.0
            else:
                strike_pct = fwd_pct - offset/100.0
            st.info(f"Strike: **{strike_pct:.4f}%** ({strike_mode} {'OTM' if offset > 0 else ''})")
        
        moneyness_bp = (strike_pct - fwd_pct) * 100
        st.caption(f"Moneyness: **{moneyness_bp:+.1f} bp**" if abs(moneyness_bp) >= 0.5 else "Moneyness: **ATM**")
        
    elif structure == "Strangle":
        st.caption("Buy OTM Payer + Buy OTM Receiver (long vol)")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            strike_pct = st.number_input("Payer Strike (%)", min_value=0.0, max_value=20.0,
                                         value=round(fwd_pct + 0.25, 4), format="%.4f", key="sw_strike_1",
                                         help="Higher strike - OTM payer")
        with col_k2:
            strike_pct_2 = st.number_input("Receiver Strike (%)", min_value=0.0, max_value=20.0,
                                           value=round(fwd_pct - 0.25, 4), format="%.4f", key="sw_strike_2",
                                           help="Lower strike - OTM receiver")
        st.caption(f"Width: **{(strike_pct - strike_pct_2)*100:.0f} bp**")
        
    elif structure == "Risk Reversal":
        st.caption("Buy Payer + Sell Receiver (or vice versa) - rate protection")
        
        # Strike mode
        rr_strike_mode = st.radio("Strike Mode", 
                                   ["Symmetric (25bp)", "Symmetric (50bp)", "Symmetric (100bp)", "Manual (independent)"], 
                                   horizontal=True, key="sw_rr_mode")
        
        col_k1, col_k2, col_dir = st.columns([1, 1, 1])
        
        if "Symmetric" in rr_strike_mode:
            # Parse offset from mode
            if "25bp" in rr_strike_mode:
                offset = 0.25
            elif "50bp" in rr_strike_mode:
                offset = 0.50
            else:
                offset = 1.00
            
            strike_pct = fwd_pct + offset  # Payer strike (cap)
            strike_pct_2 = fwd_pct - offset  # Receiver strike (floor)
            
            with col_k1:
                st.metric("Payer Strike (%)", f"{strike_pct:.4f}")
            with col_k2:
                st.metric("Receiver Strike (%)", f"{strike_pct_2:.4f}")
        else:
            # Manual mode - independent strikes
            with col_k1:
                strike_pct = st.number_input("Payer Strike (%)", min_value=0.0, max_value=20.0,
                                             value=round(fwd_pct + 0.25, 4), format="%.4f", key="sw_strike_1",
                                             help="Cap level")
            with col_k2:
                strike_pct_2 = st.number_input("Receiver Strike (%)", min_value=0.0, max_value=20.0,
                                               value=round(fwd_pct - 0.25, 4), format="%.4f", key="sw_strike_2",
                                               help="Floor level")
        
        with col_dir:
            collar_dir = st.radio("Direction", ["Long Payer/Short Rec", "Short Payer/Long Rec"], 
                                  key="sw_collar_dir", help="Long Payer protects against rising rates")
        
        width_bp = (strike_pct - strike_pct_2) * 100
        st.caption(f"Width: **{width_bp:.0f} bp** | Payer +{(strike_pct-fwd_pct)*100:.0f}bp | Receiver {(strike_pct_2-fwd_pct)*100:.0f}bp")
        
    elif structure == "Payer Ladder":
        st.caption("Buy 1x ATM Payer + Sell 2x OTM Payer (bullish/range view)")
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            strike_pct = st.number_input("K1 - Long Payer (%)", min_value=0.0, max_value=20.0,
                                         value=round(fwd_pct, 4), format="%.4f", key="sw_strike_1",
                                         help="ATM - buy 1x")
        with col_k2:
            strike_pct_2 = st.number_input("K2 - Short Payer (%)", min_value=0.0, max_value=20.0,
                                           value=round(fwd_pct + 0.25, 4), format="%.4f", key="sw_strike_2",
                                           help="OTM - sell 1x")
        with col_k3:
            strike_pct_3 = st.number_input("K3 - Short Payer (%)", min_value=0.0, max_value=20.0,
                                           value=round(fwd_pct + 0.50, 4), format="%.4f", key="sw_strike_3",
                                           help="Further OTM - sell 1x")
        st.caption(f"Max profit at K2, unlimited downside above K3")
        
    elif structure == "Receiver Ladder":
        st.caption("Buy 1x ATM Receiver + Sell 2x OTM Receiver (bearish/range view)")
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            strike_pct = st.number_input("K1 - Long Receiver (%)", min_value=0.0, max_value=20.0,
                                         value=round(fwd_pct, 4), format="%.4f", key="sw_strike_1",
                                         help="ATM - buy 1x")
        with col_k2:
            strike_pct_2 = st.number_input("K2 - Short Receiver (%)", min_value=0.0, max_value=20.0,
                                           value=round(fwd_pct - 0.25, 4), format="%.4f", key="sw_strike_2",
                                           help="OTM - sell 1x")
        with col_k3:
            strike_pct_3 = st.number_input("K3 - Short Receiver (%)", min_value=0.0, max_value=20.0,
                                           value=round(fwd_pct - 0.50, 4), format="%.4f", key="sw_strike_3",
                                           help="Further OTM - sell 1x")

    # Vol source
    st.markdown("---")
    vol_src = st.radio("Vol", ["Surface", "Manual"], horizontal=True, key="sw_volsrc")
    
    if vol_src == "Manual":
        vol_input = st.number_input("Vol (bp normal or % Black)", value=80.0, key="sw_volinput")
        vol_used_display = vol_input
        vol = vol_input / 10000.0 if vol_mode.startswith("Normal") else vol_input / 100.0
    else:
        atm = get_working_atm_surface(ccy)
        _, a, b, r, n = get_ccy_vol_data(ccy)
        atm_val = get_matrix_value(atm, expiry, tenor_y) if atm is not None else None
        if atm_val is None:
            st.warning("No ATM vol - using 80bp")
            atm_val = 80.0
        if vol_mode.startswith("Normal"):
            vol = atm_val / 10000.0
            vol_used_display = atm_val
        else:
            sabr = get_sabr_params_from_matrices(a, b, r, n, expiry, tenor_y)
            if sabr:
                vol = sabr_implied_vol_black(fwd_pct/100.0, strike_pct/100.0, expiry_y,
                                             sabr["alpha"], sabr["beta"], sabr["rho"], sabr["nu"])
                vol_used_display = vol * 100.0
            else:
                vol = atm_val / 100.0
                vol_used_display = vol * 100.0
        st.caption(f"Vol: {vol_used_display:.1f} {'bp' if vol_mode.startswith('Normal') else '%'}")

    # Dates
    from datetime import datetime, timedelta
    today = datetime.now()
    expiry_date = today + timedelta(days=int(expiry_y * 365))
    swap_start = expiry_date + timedelta(days=1)
    swap_end = swap_start + timedelta(days=int(tenor_y * 365))
    
    # Roll convention
    if ccy == "AUD":
        roll = "Q/Q" if tenor_y <= 3 else "S/S"
    elif ccy == "NZD":
        roll = "Q/Q" if tenor_y <= 2 else "S/S"
    else:
        roll = "S/S"

    # Helper function to get vol for a specific strike
    def get_vol_for_strike(k_pct):
        if vol_src == "Manual":
            return vol_input / 10000.0 if vol_mode.startswith("Normal") else vol_input / 100.0
        else:
            if vol_mode.startswith("Normal"):
                return atm_val / 10000.0
            else:
                sabr = get_sabr_params_from_matrices(a, b, r, n, expiry, tenor_y)
                if sabr:
                    return sabr_implied_vol_black(fwd_pct/100.0, k_pct/100.0, expiry_y,
                                                   sabr["alpha"], sabr["beta"], sabr["rho"], sabr["nu"])
                else:
                    return atm_val / 100.0

    # Price button
    st.markdown("---")
    
    if st.button(" Price Swaption", key="sw_price", type="primary"):
        try:
            model_type = "Normal" if vol_mode.startswith("Normal") else "Black"
            legs = []  # Store individual leg results
            
            if structure == "Payer":
                vol_k1 = get_vol_for_strike(strike_pct)
                ticket = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct/100.0, vol=vol_k1, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Payer {expiry}x{swap_tenor}", use_curve=curve is not None)
                res = price_swaption(ticket)
                legs.append(("Payer", strike_pct, 1, res))
                label = f"Payer {expiry}x{swap_tenor} K={strike_pct:.2f}%"
                
            elif structure == "Receiver":
                vol_k1 = get_vol_for_strike(strike_pct)
                ticket = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct/100.0, vol=vol_k1, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Receiver {expiry}x{swap_tenor}", use_curve=curve is not None)
                res = price_swaption(ticket)
                legs.append(("Receiver", strike_pct, 1, res))
                label = f"Receiver {expiry}x{swap_tenor} K={strike_pct:.2f}%"
                
            elif structure == "ATM Straddle":
                vol_atm = get_vol_for_strike(fwd_pct)
                ticket_p = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=fwd_pct/100.0, vol=vol_atm, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Payer ATM", use_curve=curve is not None)
                ticket_r = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=fwd_pct/100.0, vol=vol_atm, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Receiver ATM", use_curve=curve is not None)
                res_p = price_swaption(ticket_p)
                res_r = price_swaption(ticket_r)
                legs.append(("Payer", fwd_pct, 1, res_p))
                legs.append(("Receiver", fwd_pct, 1, res_r))
                res = {k: res_p.get(k,0) + res_r.get(k,0) for k in res_p}
                res["bpv"] = res_p["bpv"]
                label = f"ATM Straddle {expiry}x{swap_tenor}"
                
            elif structure == "Strangle":
                vol_k1 = get_vol_for_strike(strike_pct)  # Payer strike (higher)
                vol_k2 = get_vol_for_strike(strike_pct_2)  # Receiver strike (lower)
                ticket_p = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct/100.0, vol=vol_k1, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Payer K={strike_pct:.2f}%", use_curve=curve is not None)
                ticket_r = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct_2/100.0, vol=vol_k2, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Receiver K={strike_pct_2:.2f}%", use_curve=curve is not None)
                res_p = price_swaption(ticket_p)
                res_r = price_swaption(ticket_r)
                legs.append(("Long Payer", strike_pct, 1, res_p))
                legs.append(("Long Receiver", strike_pct_2, 1, res_r))
                res = {k: res_p.get(k,0) + res_r.get(k,0) for k in res_p}
                res["bpv"] = res_p["bpv"]
                label = f"Strangle {expiry}x{swap_tenor} ({strike_pct_2:.2f}/{strike_pct:.2f})"
                
            elif structure == "Risk Reversal":
                vol_k1 = get_vol_for_strike(strike_pct)  # Payer strike
                vol_k2 = get_vol_for_strike(strike_pct_2)  # Receiver strike
                is_long_payer = collar_dir.startswith("Long Payer")
                ticket_p = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct/100.0, vol=vol_k1, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Payer K={strike_pct:.2f}%", use_curve=curve is not None)
                ticket_r = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct_2/100.0, vol=vol_k2, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Receiver K={strike_pct_2:.2f}%", use_curve=curve is not None)
                res_p = price_swaption(ticket_p)
                res_r = price_swaption(ticket_r)
                if is_long_payer:
                    legs.append(("Long Payer", strike_pct, 1, res_p))
                    legs.append(("Short Receiver", strike_pct_2, -1, res_r))
                    res = {k: res_p.get(k,0) - res_r.get(k,0) for k in res_p}
                else:
                    legs.append(("Short Payer", strike_pct, -1, res_p))
                    legs.append(("Long Receiver", strike_pct_2, 1, res_r))
                    res = {k: res_r.get(k,0) - res_p.get(k,0) for k in res_p}
                res["bpv"] = res_p["bpv"]
                label = f"Risk Reversal {expiry}x{swap_tenor} ({strike_pct_2:.2f}/{strike_pct:.2f})"
                
            elif structure == "Payer Ladder":
                vol_k1 = get_vol_for_strike(strike_pct)
                vol_k2 = get_vol_for_strike(strike_pct_2)
                vol_k3 = get_vol_for_strike(strike_pct_3)
                # Long 1x ATM Payer, Short 1x OTM Payer, Short 1x Further OTM Payer
                ticket_1 = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct/100.0, vol=vol_k1, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Long Payer K1", use_curve=curve is not None)
                ticket_2 = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct_2/100.0, vol=vol_k2, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Short Payer K2", use_curve=curve is not None)
                ticket_3 = SwaptionTicket(
                    side="Payer", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct_3/100.0, vol=vol_k3, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Short Payer K3", use_curve=curve is not None)
                res_1 = price_swaption(ticket_1)
                res_2 = price_swaption(ticket_2)
                res_3 = price_swaption(ticket_3)
                legs.append(("Long Payer", strike_pct, 1, res_1))
                legs.append(("Short Payer", strike_pct_2, -1, res_2))
                legs.append(("Short Payer", strike_pct_3, -1, res_3))
                res = {k: res_1.get(k,0) - res_2.get(k,0) - res_3.get(k,0) for k in res_1}
                res["bpv"] = res_1["bpv"]
                label = f"Payer Ladder {expiry}x{swap_tenor}"
                
            elif structure == "Receiver Ladder":
                vol_k1 = get_vol_for_strike(strike_pct)
                vol_k2 = get_vol_for_strike(strike_pct_2)
                vol_k3 = get_vol_for_strike(strike_pct_3)
                # Long 1x ATM Receiver, Short 1x OTM Receiver, Short 1x Further OTM Receiver
                ticket_1 = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct/100.0, vol=vol_k1, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Long Receiver K1", use_curve=curve is not None)
                ticket_2 = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct_2/100.0, vol=vol_k2, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Short Receiver K2", use_curve=curve is not None)
                ticket_3 = SwaptionTicket(
                    side="Receiver", payoff_type="vanilla", notional=notional*1e6, currency=ccy,
                    expiry_years=expiry_y, swap_tenor_years=tenor_y, forward=fwd_pct/100.0,
                    strike=strike_pct_3/100.0, vol=vol_k3, discount_rate=eff_disc_rate, annuity=ann,
                    model=model_type, label=f"Short Receiver K3", use_curve=curve is not None)
                res_1 = price_swaption(ticket_1)
                res_2 = price_swaption(ticket_2)
                res_3 = price_swaption(ticket_3)
                legs.append(("Long Receiver", strike_pct, 1, res_1))
                legs.append(("Short Receiver", strike_pct_2, -1, res_2))
                legs.append(("Short Receiver", strike_pct_3, -1, res_3))
                res = {k: res_1.get(k,0) - res_2.get(k,0) - res_3.get(k,0) for k in res_1}
                res["bpv"] = res_1["bpv"]
                label = f"Receiver Ladder {expiry}x{swap_tenor}"
            
            st.success(f" Priced: **{label}** | PV = ${res['pv']:,.0f} ({res['pv_bp']:.2f} bp)")
        
            # Store results in session state
            moneyness_bp = (strike_pct - fwd_pct) * 100 if structure in ["Payer", "Receiver"] else 0
            st.session_state["sw_last_result"] = {
                "res": res, "label": label, "structure": structure, "legs": legs,
                "params": {
                    "Structure": structure, "Expiry": expiry, 
                    "Tenor": swap_tenor, "Forward (%)": f"{fwd_pct:.4f}",
                    "Annuity (PV01)": f"{ann:.4f}", "Discount": f"{eff_disc_rate*100:.3f}% ({disc_source})",
                    "Notional": f"{notional:,.0f}mm"
                },
                "notional": notional,
                "premium_type": premium_type,
                "vol": vol,
                "expiry_y": expiry_y,
                "annuity": ann,
            }
            
            # Add to portfolio - use selected premium type
            if premium_type == "Fwd":
                display_prem_bp = 2 * 0.3989 * vol * math.sqrt(expiry_y) * ann * 10000
            else:
                display_prem_bp = res["pv_bp"]
            entry = dict(instrument_type="Swaption", currency=ccy, structure=structure,
                         expiry=expiry, tenor=swap_tenor, model=vol_mode,
                         notional_mm=notional, strike=strike_pct, forward=fwd_pct, pv=res["pv"],
                         pv_bp=display_prem_bp, premium_type=premium_type,
                         delta=res["delta"], gamma=res["gamma"], vega=res["vega"],
                         theta=res["theta"], bpv=res["bpv"], label=label)
            st.session_state["swaption_portfolio"].append(entry)
            st.session_state["portfolio"].append(entry)
            
        except Exception as e:
            st.error(f" Pricing error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display last result (persists after pricing)
    if "sw_last_result" in st.session_state:
        r = st.session_state["sw_last_result"]
        res = r["res"]
        stored_notional = r.get("notional", 100)
        legs = r.get("legs", [])
        
        st.markdown("###  Results")
        col_params, col_greeks = st.columns(2)
        
        with col_params:
            st.markdown("##### Parameters")
            params_df = pd.DataFrame(list(r["params"].items()), columns=["Parameter", "Value"])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
            
            # Show leg breakdown for multi-leg structures
            if len(legs) > 1:
                st.markdown("##### Leg Breakdown")
                leg_data = []
                for leg_name, leg_strike, leg_mult, leg_res in legs:
                    leg_data.append({
                        "Leg": leg_name,
                        "Strike (%)": f"{leg_strike:.4f}",
                        "Premium (bp)": f"{leg_res['pv_bp'] * leg_mult:.2f}",
                        "PV": f"${leg_res['pv'] * leg_mult:,.0f}",
                        "Delta": f"{leg_res['delta'] * leg_mult:,.0f}"
                    })
                st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)
        
        with col_greeks:
            st.markdown("##### Valuation")
            # Calculate forward premium: 2  0.3989    T  Annuity  10000
            stored_prem_type = r.get("premium_type", "Spot")
            stored_vol = r.get("vol", 0)
            stored_expiry_y = r.get("expiry_y", 1)
            stored_ann = r.get("annuity", 1)
            
            if stored_prem_type == "Fwd":
                # Forward premium formula (matches matrix)
                fwd_prem_bp = 2 * 0.3989 * stored_vol * math.sqrt(stored_expiry_y) * stored_ann * 10000
                st.metric("Fwd Premium (bp)", f"{fwd_prem_bp:.2f}")
            else:
                st.metric("Spot Premium (bp)", f"{res['pv_bp']:.2f}")
            st.metric("Total PV", f"${res['pv']:,.0f}")
            
            st.markdown("##### Greeks (Net)")
            greeks_df = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "BPV"],
                "Value": [f"{res['delta']:,.0f}", f"{res['gamma']:,.0f}", f"{res['vega']:,.0f}",
                          f"{res['theta']:,.0f}", f"{res['bpv']:,.0f}"],
                "Per 1mm": [f"{res['delta']/stored_notional:,.0f}", f"{res['gamma']/stored_notional:,.0f}",
                            f"{res['vega']/stored_notional:,.0f}", f"{res['theta']/stored_notional:,.0f}",
                            f"{res['bpv']/stored_notional:,.0f}"]
            })
            st.dataframe(greeks_df, use_container_width=True, hide_index=True)

    # Display portfolio
    if st.session_state["swaption_portfolio"]:
        st.markdown("### In-session swaption tickets")
        df = pd.DataFrame(st.session_state["swaption_portfolio"])
        df_display = df.copy()
        df_display["pv_bp"] = df_display["pv_bp"].round(2)
        df_display["pv"] = (df_display["pv"] / 1e3).round(1)
        df_display["delta"] = (df_display["delta"] / 1e3).round(1)
        df_display["vega"] = (df_display["vega"] / 1e3).round(1)
        df_display["gamma"] = (df_display["gamma"] / 1e3).round(1)
        df_display["theta"] = (df_display["theta"] / 1e3).round(1)
        df_display.rename(
            columns={
                "pv": "PV (k)",
                "pv_bp": "PV (bp)",
                "vol_input": "Vol",
                "delta": "Delta (k)",
                "vega": "Vega (k)",
                "gamma": "Gamma (k)",
                "theta": "Theta (k)",
            },
            inplace=True,
        )
        st.dataframe(df_display, use_container_width=True)


def caps_floors_tab(ccy: str, vol_mode: str):
    st.subheader("Caps & Floors")

    col_type, col_model = st.columns(2)
    with col_type:
        cf_type = st.selectbox("Instrument", ["Cap", "Floor", "Straddle", "Collar", "Strangle"], key="cf_type")
    with col_model:
        model = st.selectbox("Model", ["Normal", "Black"], index=0, key="cf_model")

    col_not, col_exp, col_tenor = st.columns(3)
    with col_not:
        notional = st.number_input("Notional (mm)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="cf_not")
    with col_exp:
        expiry = st.selectbox(
            "First fixing",
            ["1m","2m","3m","6m","9m","1y","18m","2y","3y","4y","5y","7y","10y"],
            index=2,
            key="cf_exp",
        )
        expiry_y = label_to_years(expiry)
    with col_tenor:
        tenor = st.selectbox(
            "Final maturity",
            ["1Y","2Y","3Y","4Y","5Y","7Y","10Y","12Y","15Y","20Y"],
            index=2,
            key="cf_tenor",
        )
        tenor_y = float(tenor[:-1])

    curve = get_ccy_curve(ccy)
    ois_curve = get_basis_curve(ccy, "ois")
    if curve is not None:
        fwd, _, sched = forward_and_annuity_from_curve(curve, ccy, expiry_y, tenor_y, ois_curve)
    else:
        sched = [(expiry_y + i * 0.25, 0.25) for i in range(int(tenor_y / 0.25))]
        fwd = 0.04

    fwd_pct = fwd * 100
    
    # Show structure dates
    from datetime import date, timedelta
    today = date.today()
    first_fixing = today + timedelta(days=int(expiry_y * 365))
    final_maturity = today + timedelta(days=int(tenor_y * 365))
    num_caplets = len(sched)
    
    st.markdown(f"""
    <div style="background: rgba(30,41,59,0.5); border-radius: 8px; padding: 12px; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
            <div><span style="color: #94a3b8;">Forward:</span> <strong>{fwd_pct:.4f}%</strong></div>
            <div><span style="color: #94a3b8;">First Fixing:</span> <strong>{first_fixing.strftime('%d-%b-%Y')}</strong></div>
            <div><span style="color: #94a3b8;">Final Maturity:</span> <strong>{final_maturity.strftime('%d-%b-%Y')}</strong></div>
            <div><span style="color: #94a3b8;">Caplets:</span> <strong>{num_caplets}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Strike inputs based on structure
    st.markdown("##### Strikes")
    strike_pct_2 = fwd_pct  # Initialize
    
    if cf_type in ["Cap", "Floor"]:
        # Strike mode selector
        strike_mode = st.radio("Strike Mode", ["ATM", "10 bp", "25 bp", "50 bp", "100 bp", "Manual"], 
                               horizontal=True, key="cf_strike_mode")
        
        offset_map = {"ATM": 0, "10 bp": 10, "25 bp": 25, "50 bp": 50, "100 bp": 100, "Manual": None}
        offset = offset_map[strike_mode]
        
        if strike_mode == "Manual":
            strike = st.number_input("Strike (%)", value=round(fwd_pct, 4), format="%.4f", key="cf_strike") / 100.0
        else:
            # For Cap, add offset (OTM); for Floor, subtract offset (OTM)
            if cf_type == "Cap":
                strike = fwd + offset/10000.0
            else:
                strike = fwd - offset/10000.0
            st.info(f"Strike: **{strike*100:.4f}%** ({strike_mode} {'OTM' if offset > 0 else ''})")
            
    elif cf_type == "Straddle":
        strike = fwd  # ATM
        st.info(f"ATM Strike: **{fwd_pct:.4f}%**")
        
    elif cf_type == "Collar":
        st.caption("Long Cap + Short Floor - rate protection")
        
        collar_mode = st.radio("Strike Mode", 
                               ["Symmetric (25bp)", "Symmetric (50bp)", "Symmetric (100bp)", "Manual"], 
                               horizontal=True, key="cf_collar_mode")
        
        if "Symmetric" in collar_mode:
            if "25bp" in collar_mode:
                offset = 0.0025
            elif "50bp" in collar_mode:
                offset = 0.0050
            else:
                offset = 0.0100
            
            strike = fwd + offset  # Cap strike
            strike_pct_2 = fwd - offset  # Floor strike
            
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.metric("Cap Strike (%)", f"{strike*100:.4f}")
            with col_k2:
                st.metric("Floor Strike (%)", f"{strike_pct_2*100:.4f}")
        else:
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                strike = st.number_input("Cap Strike (%)", value=round(fwd_pct + 0.50, 4), 
                                         format="%.4f", key="cf_strike") / 100.0
            with col_k2:
                strike_pct_2 = st.number_input("Floor Strike (%)", value=round(fwd_pct - 0.50, 4), 
                                               format="%.4f", key="cf_strike_2") / 100.0
        
        width_bp = (strike - strike_pct_2) * 10000
        st.caption(f"Width: **{width_bp:.0f} bp** | Cap +{(strike-fwd)*10000:.0f}bp | Floor {(strike_pct_2-fwd)*10000:.0f}bp")
        
    elif cf_type == "Strangle":
        st.caption("Long OTM Cap + Long OTM Floor (long vol)")
        
        strangle_mode = st.radio("Strike Mode", 
                                 ["Symmetric (25bp)", "Symmetric (50bp)", "Symmetric (100bp)", "Manual"], 
                                 horizontal=True, key="cf_strangle_mode")
        
        if "Symmetric" in strangle_mode:
            if "25bp" in strangle_mode:
                offset = 0.0025
            elif "50bp" in strangle_mode:
                offset = 0.0050
            else:
                offset = 0.0100
            
            strike = fwd + offset  # Cap strike
            strike_pct_2 = fwd - offset  # Floor strike
            
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.metric("Cap Strike (%)", f"{strike*100:.4f}")
            with col_k2:
                st.metric("Floor Strike (%)", f"{strike_pct_2*100:.4f}")
        else:
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                strike = st.number_input("Cap Strike (%)", value=round(fwd_pct + 0.50, 4), 
                                         format="%.4f", key="cf_strike") / 100.0
            with col_k2:
                strike_pct_2 = st.number_input("Floor Strike (%)", value=round(fwd_pct - 0.50, 4), 
                                               format="%.4f", key="cf_strike_2") / 100.0
        
        width_bp = (strike - strike_pct_2) * 10000
        st.caption(f"Width: **{width_bp:.0f} bp**")
    
    disc = st.number_input("Flat discount rate (%)", min_value=0.0, max_value=20.0, value=4.0, key="cf_disc") / 100.0

    vol_src = st.radio(
        "Vol source",
        ["Surface", "Manual"],
        horizontal=True,
        key="cf_volsrc",
    )

    if vol_src == "Manual":
        vol_input = st.number_input(
            "Vol (normal bp or Black %)",
            value=35.0,
            key="cf_vol",
            help="Normal bp or Black %, depending on selected model",
        )
        if model == "Normal":
            sigma = vol_input / 10000.0
        else:
            sigma = vol_input / 100.0
        vol_used_display = vol_input
    else:
        atm = get_working_atm_surface(ccy)
        _, a, b, r, n = get_ccy_vol_data(ccy)
        expiry_label = expiry
        if atm is not None:
            atm_val = get_matrix_value(atm, expiry_label, tenor_y)
        else:
            atm_val = None
        if atm_val is None:
            st.warning("No ATM vol found for this cap/floor node. Falling back to 35bp.")
            atm_val = 35.0

        if model == "Normal":
            sigma = atm_val / 10000.0
            vol_used_display = atm_val
        else:
            sabr = get_sabr_params_from_matrices(a, b, r, n, expiry_label, tenor_y)
            if sabr:
                sigma = sabr_implied_vol_black(
                    fwd,
                    strike,
                    expiry_y,
                    sabr["alpha"],
                    sabr["beta"],
                    sabr["rho"],
                    sabr["nu"],
                )
            else:
                sigma = atm_val / 100.0
            vol_used_display = sigma * 100.0

        st.caption(f"Using ATM vol  **{vol_used_display:.2f}** ({'bp normal' if model=='Normal' else '% Black'})")

    st.markdown("---")
    
    if st.button(" Price Cap/Floor", key="cf_price", type="primary"):
        try:
            pv_total = 0.0
            delta_total = 0.0
            vega_total = 0.0
            gamma_total = 0.0
            
            # Store individual caplet/floorlet results
            caplet_details = []
            
            # Helper to price a strip and collect caplet details
            def price_strip(strike_val, is_cap_flag, leg_name=""):
                pv = delta = vega = gamma = 0.0
                caplets = []
                for i, (T_i, accrual) in enumerate(sched):
                    F_i = fwd
                    if model == "Black":
                        res = black_caplet(notional * 1e6, accrual, F_i, strike_val, sigma, T_i, disc, is_cap=is_cap_flag)
                    else:
                        res = bachelier_caplet(notional * 1e6, accrual, F_i, strike_val, sigma, T_i, disc, is_cap=is_cap_flag)
                    pv += res["pv"]
                    delta += res["delta"]
                    vega += res["vega"]
                    gamma += res["gamma"]
                    
                    caplet_date = today + timedelta(days=int(T_i * 365))
                    caplets.append({
                        "Leg": leg_name,
                        "#": i + 1,
                        "Fixing": caplet_date.strftime('%d-%b-%Y'),
                        "T (yrs)": f"{T_i:.2f}",
                        "Accrual": f"{accrual:.4f}",
                        "PV": f"${res['pv']:,.0f}",
                        "Delta": f"{res['delta']:,.0f}",
                    })
                return {"pv": pv, "delta": delta, "vega": vega, "gamma": gamma, "caplets": caplets}
            
            legs = []
            
            if cf_type == "Cap":
                res = price_strip(strike, True, "Cap")
                pv_total, delta_total, vega_total, gamma_total = res["pv"], res["delta"], res["vega"], res["gamma"]
                caplet_details = res["caplets"]
                legs.append(("Cap", strike*100, 1, res))
                label = f"Cap {expiry}-{tenor} K={strike*100:.2f}%"
                
            elif cf_type == "Floor":
                res = price_strip(strike, False, "Floor")
                pv_total, delta_total, vega_total, gamma_total = res["pv"], res["delta"], res["vega"], res["gamma"]
                caplet_details = res["caplets"]
                legs.append(("Floor", strike*100, 1, res))
                label = f"Floor {expiry}-{tenor} K={strike*100:.2f}%"
                
            elif cf_type == "Straddle":
                res_cap = price_strip(fwd, True, "Cap")
                res_floor = price_strip(fwd, False, "Floor")
                pv_total = res_cap["pv"] + res_floor["pv"]
                delta_total = res_cap["delta"] + res_floor["delta"]
                vega_total = res_cap["vega"] + res_floor["vega"]
                gamma_total = res_cap["gamma"] + res_floor["gamma"]
                caplet_details = res_cap["caplets"] + res_floor["caplets"]
                legs.append(("Cap", fwd*100, 1, res_cap))
                legs.append(("Floor", fwd*100, 1, res_floor))
                label = f"Straddle {expiry}-{tenor} ATM"
                
            elif cf_type == "Collar":
                res_cap = price_strip(strike, True, "Long Cap")
                res_floor = price_strip(strike_pct_2, False, "Short Floor")
                # Long cap, short floor (protection against rising rates)
                pv_total = res_cap["pv"] - res_floor["pv"]
                delta_total = res_cap["delta"] - res_floor["delta"]
                vega_total = res_cap["vega"] - res_floor["vega"]
                gamma_total = res_cap["gamma"] - res_floor["gamma"]
                caplet_details = res_cap["caplets"] + res_floor["caplets"]
                legs.append(("Long Cap", strike*100, 1, res_cap))
                legs.append(("Short Floor", strike_pct_2*100, -1, res_floor))
                label = f"Collar {expiry}-{tenor} ({strike_pct_2*100:.2f}/{strike*100:.2f})"
                
            elif cf_type == "Strangle":
                res_cap = price_strip(strike, True, "OTM Cap")
                res_floor = price_strip(strike_pct_2, False, "OTM Floor")
                pv_total = res_cap["pv"] + res_floor["pv"]
                delta_total = res_cap["delta"] + res_floor["delta"]
                vega_total = res_cap["vega"] + res_floor["vega"]
                gamma_total = res_cap["gamma"] + res_floor["gamma"]
                caplet_details = res_cap["caplets"] + res_floor["caplets"]
                legs.append(("OTM Cap", strike*100, 1, res_cap))
                legs.append(("OTM Floor", strike_pct_2*100, 1, res_floor))
                label = f"Strangle {expiry}-{tenor} ({strike_pct_2*100:.2f}/{strike*100:.2f})"

            horizon_df = math.exp(-disc * expiry_y)
            one_bp = horizon_df * notional * 1e6 * tenor_y * 0.0001
            pv_bp = pv_total / one_bp if one_bp != 0 else 0.0

            st.success(f" Priced: **{label}** | PV = ${pv_total:,.0f} ({pv_bp:.2f} bp)")
            
            # Store for display
            st.session_state["cf_last_result"] = {
                "legs": legs,
                "caplet_details": caplet_details,
                "pv_total": pv_total,
                "pv_bp": pv_bp,
                "delta_total": delta_total,
                "gamma_total": gamma_total,
                "vega_total": vega_total,
                "one_bp": one_bp,
                "label": label,
                "notional": notional,
            }

            st.session_state["portfolio"].append(
                dict(
                    instrument_type="Cap/Floor",
                    currency=ccy,
                    structure=cf_type,
                    expiry=expiry,
                    tenor=tenor,
                    model=model,
                    vol_input=vol_used_display,
                    notional_mm=notional,
                    strike=strike * 100.0,
                    forward=fwd * 100.0,
                    pv=pv_total,
                    pv_bp=pv_bp,
                    delta=delta_total,
                    gamma=gamma_total,
                    vega=vega_total,
                    theta=0.0,
                    bpv=one_bp,
                    label=label,
                )
            )
        except Exception as e:
            st.error(f" Pricing error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results if available
    if "cf_last_result" in st.session_state:
        r = st.session_state["cf_last_result"]
        
        st.markdown("###  Results")
        col_params, col_greeks = st.columns(2)
        
        with col_params:
            # Show leg breakdown
            if len(r["legs"]) > 0:
                st.markdown("##### Leg Breakdown")
                leg_data = []
                for leg_name, leg_strike, leg_mult, leg_res in r["legs"]:
                    leg_pv_bp = (leg_res['pv'] * leg_mult) / r["one_bp"] if r["one_bp"] != 0 else 0
                    leg_data.append({
                        "Leg": leg_name,
                        "Strike (%)": f"{leg_strike:.4f}",
                        "Premium (bp)": f"{leg_pv_bp:.2f}",
                        "PV": f"${leg_res['pv'] * leg_mult:,.0f}",
                        "Delta": f"{leg_res['delta'] * leg_mult:,.0f}"
                    })
                st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)
        
        with col_greeks:
            st.markdown("##### Valuation")
            st.metric("Premium (bp)", f"{r['pv_bp']:.2f}")
            st.metric("Total PV", f"${r['pv_total']:,.0f}")
            
            st.markdown("##### Greeks (Net)")
            greeks_df = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Vega", "BPV"],
                "Value": [f"{r['delta_total']:,.0f}", f"{r['gamma_total']:,.0f}", 
                          f"{r['vega_total']:,.0f}", f"{r['one_bp']:,.0f}"],
                "Per 1mm": [f"{r['delta_total']/r['notional']:,.0f}", f"{r['gamma_total']/r['notional']:,.0f}",
                            f"{r['vega_total']/r['notional']:,.0f}", f"{r['one_bp']/r['notional']:,.0f}"]
            })
            st.dataframe(greeks_df, use_container_width=True, hide_index=True)
        
        # Caplet/Floorlet breakdown in expander
        if r["caplet_details"]:
            with st.expander(" Caplet/Floorlet Breakdown", expanded=False):
                st.dataframe(pd.DataFrame(r["caplet_details"]), use_container_width=True, hide_index=True)


def exotics_tab(ccy: str, vol_mode: str):
    st.subheader("Exotics / Callables (prototype)")

    exotic_type = st.selectbox(
        "Structure",
        ["Callable Berm proxy", "Digital ladder", "Placeholder  rainbow / worst-of"],
        key="ex_structure",
    )

    col_not, col_exp, col_tenor = st.columns(3)
    with col_not:
        notional = st.number_input("Notional (mm)", 0.0, 1e6, 100.0, step=10.0, key="ex_not")
    with col_exp:
        final_expiry = st.selectbox(
            "Final expiry",
            ["1y","2y","3y","4y","5y","7y","10y","15y","20y"],
            index=2,
            key="ex_final_exp",
        )
        final_expiry_y = label_to_years(final_expiry)
    with col_tenor:
        swap_tenor = st.selectbox(
            "Underlying swap tenor",
            ["1Y","2Y","3Y","4Y","5Y","7Y","10Y","15Y","20Y","30Y"],
            index=4,
            key="ex_tenor",
        )
        tenor_y = float(swap_tenor[:-1])

    curve = get_ccy_curve(ccy)
    ois_curve = get_basis_curve(ccy, "ois")
    if curve is not None:
        fwd, ann, _ = forward_and_annuity_from_curve(curve, ccy, final_expiry_y, tenor_y, ois_curve)
    else:
        fwd, ann = 0.04, tenor_y

    strike_pct = st.number_input("ATM strike (%)", value=round(fwd * 100, 4), key="ex_strike") / 100.0
    disc = st.number_input("Flat discount rate (%)", value=4.0, key="ex_disc") / 100.0

    atm = get_working_atm_surface(ccy)
    _, a, b, r, n = get_ccy_vol_data(ccy)
    expiry_label = final_expiry
    if atm is not None:
        atm_val = get_matrix_value(atm, expiry_label, tenor_y)
    else:
        atm_val = None
    if atm_val is None:
        atm_val = 35.0
    if vol_mode.startswith("Normal"):
        base_sigma = atm_val / 10000.0
    else:
        sabr = get_sabr_params_from_matrices(a, b, r, n, expiry_label, tenor_y)
        if sabr:
            base_sigma = sabr_implied_vol_black(
                fwd,
                strike_pct,
                final_expiry_y,
                sabr["alpha"],
                sabr["beta"],
                sabr["rho"],
                sabr["nu"],
            )
        else:
            base_sigma = atm_val / 100.0

    bump_for_berm = st.number_input("Vol bump for callable/Berm (+bp or +%)", value=100.0, key="ex_vbump")
    if vol_mode.startswith("Normal"):
        sigma_berm = base_sigma + bump_for_berm / 10000.0
    else:
        sigma_berm = base_sigma + bump_for_berm / 100.0

    if st.button("Price exotic", key="ex_price"):
        if exotic_type.startswith("Callable Berm"):
            ticket = SwaptionTicket(
                side="Payer",
                payoff_type="vanilla",
                notional=notional * 1e6,
                currency=ccy,
                expiry_years=final_expiry_y,
                swap_tenor_years=tenor_y,
                forward=fwd,
                strike=strike_pct,
                vol=sigma_berm,
                discount_rate=disc,
                annuity=ann,
                model="Normal" if vol_mode.startswith("Normal") else "Black",
                payout_bp=1.0,
                label=f"Berm proxy {final_expiry}x{swap_tenor}",
                use_curve=curve is not None,
            )
            res = price_swaption(ticket)
            st.markdown(
                f"**Callable/Berm proxy PV:** {res['pv']:,.0f} ({res['pv_bp']:,.1f} bp), "
                f"vol used  {sigma_berm * (10000 if vol_mode.startswith('Normal') else 100):.1f} "
                f"{'bp N' if vol_mode.startswith('Normal') else '% B'}"
            )
        elif exotic_type.startswith("Digital ladder"):
            ladder_strikes = [strike_pct + k * 0.0025 for k in range(-2, 3)]
            rows = []
            for k in ladder_strikes:
                ticket = SwaptionTicket(
                    side="Payer",
                    payoff_type="digital",
                    notional=notional * 1e6,
                    currency=ccy,
                    expiry_years=final_expiry_y,
                    swap_tenor_years=tenor_y,
                    forward=fwd,
                    strike=k,
                    vol=base_sigma,
                    discount_rate=disc,
                    annuity=ann,
                    model="Normal" if vol_mode.startswith("Normal") else "Black",
                    payout_bp=25.0,
                    label=f"Digital @ {k*100:.2f}%",
                    use_curve=curve is not None,
                )
                res = price_swaption(ticket)
                rows.append(dict(strike_pct=k*100, pv=res["pv"], pv_bp=res["pv_bp"]))
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Placeholder  rainbow / worst-of payoff engine still to be wired to multi-underlying curves.")


def vol_surface_editor_tab():
    """Vol Surface Editor with mode selection - Hybrid or 3D Drag"""
    
    st.subheader(" Vol Surface Editor")
    
    # Check if new editor module is available
    if not HAS_VOL_EDITOR:
        st.warning("New vol_editor module not found. Using legacy editor.")
        _vol_surface_editor_legacy()
        return
    
    # Currency selector - use v3d_ccy from query params if coming back from Apply
    default_ccy = st.query_params.get('v3d_ccy', SUPPORTED_CURRENCIES[0])
    if default_ccy not in SUPPORTED_CURRENCIES:
        default_ccy = SUPPORTED_CURRENCIES[0]
    default_idx = SUPPORTED_CURRENCIES.index(default_ccy)
    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, index=default_idx, key="vol_editor_ccy")
    
    # Check if we have v3d_data in query params (coming back from Apply button)
    has_v3d_data = 'v3d_data' in st.query_params and st.query_params.get('v3d_ccy') == ccy
    
    # Get current ATM surface
    atm = get_working_atm_surface(ccy)
    
    # Only block if no ATM surface AND no v3d_data to restore from
    if (atm is None or atm.empty) and not has_v3d_data:
        st.info("No ATM vol surface loaded for this currency. Go to 'Vol / SABR' tab to load data first.")
        return
    
    # If we have v3d_data but no atm, create placeholder - vol_editor will restore from query params
    if (atm is None or atm.empty) and has_v3d_data:
        atm = pd.DataFrame({"Expiry": ["1Y"], "1Y": [50.0]})
    
    # Get curve for annuity calculations
    curve = get_ccy_curve(ccy)
    ois_curve = get_basis_curve(ccy, "ois")
    
    st.markdown("---")
    
    # Render the unified editor with mode toggle (Hybrid vs 3D Drag)
    updated_surface = render_vol_surface_editor_unified(ccy, atm, curve, ois_curve)
    
    # Render bulk adjustment tools
    with st.expander(" Quick Adjustments (Parallel Shift, Scale, Tilt)", expanded=False):
        render_bulk_adjustment_tools(ccy)
    
    # Sync back to the main app's vol_data if published
    # (The vol_editor module handles publishing internally via session state)


def _vol_surface_editor_legacy():
    """Legacy vol surface editor - fallback if new module not available"""
    st.subheader("Vol surface editor (ATM  click-to-edit 3D surface)")
    if plotly_events is None:
        st.error("Install streamlit-plotly-events to use the interactive 3D editor: pip install streamlit-plotly-events")
        return

    ccy = st.selectbox("Currency", SUPPORTED_CURRENCIES, key="vol_ccy")
    atm = get_working_atm_surface(ccy)
    if atm is None or atm.empty:
        st.info("No ATM vol surface loaded for this currency yet.")
        return

    ve = st.session_state["vol_editor"]

    col_mode, col_bump = st.columns(2)
    with col_mode:
        mode = st.radio(
            "Edit mode",
            ["Local smoothing", "Global shift", "Pinned wings"],
            horizontal=False,
            key="vol_edit_mode",
        )
    with col_bump:
        bump_bp = st.slider(
            "Bump size (bp)",
            min_value=-50.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="vol_bump",
        )

    wdf = ve["working"].get(ccy, atm.copy())
    ve["working"][ccy] = wdf

    tenor_cols = [c for c in wdf.columns if c != "Expiry"]
    x_vals = []
    for c in tenor_cols:
        if isinstance(c, str) and c.endswith("Y") and c[:-1].isdigit():
            x_vals.append(float(c[:-1]))
        else:
            try:
                x_vals.append(float(c))
            except Exception:
                x_vals.append(0.0)

    expiry_labels = wdf["Expiry"].tolist()
    y_vals = [label_to_years(e) for e in expiry_labels]
    z_vals = wdf[tenor_cols].to_numpy().astype(float)

    fig = go.Figure()
    fig.add_surface(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="Viridis",
        opacity=0.85,
        showscale=True,
        name="surface",
    )

    xs_scatter = []
    ys_scatter = []
    zs_scatter = []
    customdata = []
    for i, exp_label in enumerate(expiry_labels):
        for j, ten_col in enumerate(tenor_cols):
            xs_scatter.append(x_vals[j])
            ys_scatter.append(y_vals[i])
            zs_scatter.append(z_vals[i, j])
            customdata.append((exp_label, ten_col))

    fig.add_trace(
        go.Scatter3d(
            x=xs_scatter,
            y=ys_scatter,
            z=zs_scatter,
            mode="markers",
            marker=dict(size=4, symbol="circle"),
            name="Nodes",
            customdata=customdata,
        )
    )

    fig.update_layout(
        title=f"{ccy} ATM vol surface (bp)  click a node to bump",
        scene=dict(
            xaxis_title="Swap Tenor (years)",
            yaxis_title="Expiry (years)",
            zaxis_title="Vol (bp)",
        ),
        height=550,
    )

    clicked_points = plotly_events(
        fig,
        select_event=True,
        override_height=550,
        override_width="100%",
        key="vol_surface_events",
    )

    if clicked_points:
        pt = clicked_points[0]
        exp_label = None
        ten_col = None
        if "customdata" in pt and pt["customdata"]:
            exp_label, ten_col = pt["customdata"]
        else:
            x_click = pt.get("x", None)
            y_click = pt.get("y", None)
            if x_click is not None and y_click is not None:
                y_arr = np.array(y_vals)
                x_arr = np.array(x_vals)
                exp_idx = int(np.argmin(np.abs(y_arr - y_click)))
                ten_idx = int(np.argmin(np.abs(x_arr - x_click)))
                exp_label = expiry_labels[exp_idx]
                ten_col = tenor_cols[ten_idx]

        if exp_label is not None and ten_col is not None:
            push_vol_history(ccy)
            w = ve["working"].get(ccy, atm.copy())
            mask = w["Expiry"] == exp_label
            if not mask.any():
                st.error("Clicked expiry not found in surface.")
            elif ten_col not in w.columns:
                st.error("Clicked tenor column not found in surface.")
            else:
                idx = w[mask].index[0]
                delta = bump_bp

                if mode == "Local smoothing":
                    w.at[idx, ten_col] = w.at[idx, ten_col] + delta
                    if idx - 1 in w.index:
                        w.at[idx - 1, ten_col] = w.at[idx - 1, ten_col] + 0.5 * delta
                    if idx + 1 in w.index:
                        w.at[idx + 1, ten_col] = w.at[idx + 1, ten_col] + 0.5 * delta
                elif mode == "Global shift":
                    w[ten_col] = w[ten_col] + delta
                elif mode == "Pinned wings":
                    if len(w) > 2:
                        inner_idx = w.index[1:-1]
                        w.loc[inner_idx, ten_col] = w.loc[inner_idx, ten_col] + delta

                ve["working"][ccy] = w
                st.success(f"Bumped node ({exp_label}, {ten_col}) by {delta:+.0f} bp.")

    col_u, col_r, col_p = st.columns(3)
    with col_u:
        if st.button("Undo last", key="vol_undo"):
            undo_vol(ccy)
    with col_r:
        if st.button("Redo", key="vol_redo"):
            redo_vol(ccy)
    with col_p:
        if st.button("Publish to pricing", key="vol_publish"):
            publish_vol(ccy)
            st.success("Published  swaptions & caps/floors now use edited surface.")

    st.markdown("#### Current working ATM surface (bp)")
    st.dataframe(ve["working"][ccy], use_container_width=True)


def multi_ccy_tab(vol_mode: str):
    st.subheader("Multi-currency comparison")

    col1, col2 = st.columns(2)
    with col1:
        ccy1 = st.selectbox("Currency 1", SUPPORTED_CURRENCIES, index=0, key="mc_ccy1")
    with col2:
        ccy2 = st.selectbox("Currency 2", SUPPORTED_CURRENCIES, index=1, key="mc_ccy2")

    expiry = st.selectbox(
        "Expiry",
        ["1w","1m","2m","3m","6m","9m","1y","18m","2y","3y","4y","5y","7y","10y"],
        index=5,
        key="mc_exp",
    )
    tenor_str = st.selectbox(
        "Swap tenor",
        ["1Y","2Y","3Y","4Y","5Y","7Y","10Y","15Y","20Y"],
        index=2,
        key="mc_tenor",
    )
    tenor_y = float(tenor_str[:-1])
    expiry_y = label_to_years(expiry)

    side = st.radio("Side", ["Payer", "Receiver"], horizontal=True, key="mc_side")
    notional_mm = st.number_input("Notional (mm, per ccy)", 0.0, 1e6, 100.0, step=10.0, key="mc_not")
    disc = st.number_input("Flat discount rate (%)", 0.0, 20.0, 4.0, step=0.1, key="mc_disc") / 100.0

    def price_leg(ccy: str) -> Optional[dict]:
        curve = get_ccy_curve(ccy)
        ois_curve = get_basis_curve(ccy, "ois")
        atm = get_working_atm_surface(ccy)
        _, a, b, r, n = get_ccy_vol_data(ccy)
        if curve is None or atm is None:
            return None
        fwd, ann, _ = forward_and_annuity_from_curve(curve, ccy, expiry_y, tenor_y, ois_curve)
        expiry_label = expiry
        atm_val = get_matrix_value(atm, expiry_label, tenor_y)
        if atm_val is None:
            atm_val = 35.0
        if vol_mode.startswith("Normal"):
            vol = atm_val / 10000.0
            model = "Normal"
        else:
            sabr = get_sabr_params_from_matrices(a, b, r, n, expiry_label, tenor_y)
            if sabr:
                vol = sabr_implied_vol_black(
                    fwd,
                    fwd,
                    expiry_y,
                    sabr["alpha"],
                    sabr["beta"],
                    sabr["rho"],
                    sabr["nu"],
                )
            else:
                vol = atm_val / 100.0
            model = "Black"
        ticket = SwaptionTicket(
            side=side,
            payoff_type="vanilla",
            notional=notional_mm * 1e6,
            currency=ccy,
            expiry_years=expiry_y,
            swap_tenor_years=tenor_y,
            forward=fwd,
            strike=fwd,
            vol=vol,
            discount_rate=disc,
            annuity=ann,
            model=model,
            label=f"{side} {expiry}x{tenor_str}",
            use_curve=True,
        )
        return price_swaption(ticket)

    if st.button("Compare", key="mc_compare"):
        res1 = price_leg(ccy1)
        res2 = price_leg(ccy2)
        if res1 is None or res2 is None:
            st.warning("Need curves and ATM vols loaded for both currencies.")
        else:
            st.markdown(
                f"**{ccy1} PV:** {res1['pv']:,.0f} ({res1['pv_bp']:,.1f} bp)  \n"
                f"**{ccy2} PV:** {res2['pv']:,.0f} ({res2['pv_bp']:,.1f} bp)  \n"
                f"**Spread ({ccy1}-{ccy2}) PV (bp):** {res1['pv_bp'] - res2['pv_bp']:.1f}"
            )


def credit_xva_tab():
    st.subheader("Credit / CVA overlay")

    if not st.session_state.get("portfolio"):
        st.info("Price some swaptions or caps/floors first  portfolio is empty.")
        return

    dfp = pd.DataFrame(st.session_state["portfolio"])
    # Use columns that actually exist in the portfolio
    display_cols = [c for c in ["instrument_type", "currency", "structure", "expiry", "tenor", "pv", "pv_bp", "vega"] if c in dfp.columns]
    st.dataframe(dfp[display_cols] if display_cols else dfp)

    disc = st.number_input("Flat discount rate for CVA (%)", 0.0, 20.0, 4.0, step=0.25, key="cva_disc") / 100.0
    hazard = st.number_input("Flat hazard rate (% per year)", 0.0, 20.0, 1.0, step=0.1, key="cva_hz") / 100.0
    lgd = st.number_input("LGD (%)", 0.0, 100.0, 60.0, step=5.0, key="cva_lgd") / 100.0
    horizon = st.number_input("CVA horizon (years)", 0.5, 30.0, 10.0, step=0.5, key="cva_hor")

    if st.button("Compute crude CVA on portfolio", key="cva_btn"):
        pv0 = dfp["pv"].sum()
        vega_total = dfp["vega"].sum()
        vol_ref = 0.01
        ts, ee = simple_exposure_profile(pv0, vega_total, vol_ref, horizon_years=horizon)
        cva = cva_from_hazard(ts, ee, hazard_rate=hazard, lgd=lgd, discount_rate=disc)
        st.markdown(
            f"**Clean PV:** {pv0:,.0f}  \n"
            f"**CVA:** {cva:,.0f}  \n"
            f"**CVA-adjusted PV:** {pv0 + cva:,.0f}"
        )
        chart_df = pd.DataFrame({"t": ts, "EE": ee})
        st.line_chart(chart_df.set_index("t"))


def backtesting_tab():
    st.subheader("Backtesting / Historical replay (placeholder)")

    st.markdown(
        "In production this wires to Postgres (RATEEDGE_ADMIN_DATA). "
        "Here we provide knobs and explanations only."
    )

    st.markdown("#### Simple what-if scenarios")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Parallel curve shift (bp)", -200, 200, 0, key="bt_curve_shift")
        st.slider("ATM vol shift (bp)", -50, 50, 0, key="bt_vol_shift")
    with col2:
        st.slider("Wing vol shift (bp)", -50, 50, 0, key="bt_wing_shift")
        st.slider("Vega flatten (%)", -20, 20, 0, key="bt_vega_flat")

    st.info(
        "Once hooked to RATEEDGE_ADMIN_DATA, this will: clone surfaces, apply shocks, "
        "and reprice your portfolio with RV breakdowns."
    )


def rv_tab():
    st.subheader("RV / risk-reversal / calendar analysis (placeholder)")

    st.markdown(
        "This tab will:\n"
        "- construct risk-reversal payoffs (e.g., 25d RR) from your smiles,\n"
        "- build collar payoffs over time,\n"
        "- analyse calendar spreads (near vs far expiry) and show rate/vol attribution.\n\n"
        "In this build, it's a documented placeholder  the core pricing + vol infra above is the engine."
    )


def portfolio_tab():
    st.subheader("Portfolio  Swaptions + Caps/Floors")

    portfolio = st.session_state.get("portfolio", [])
    if not portfolio:
        st.info("Portfolio is currently empty. Price swaptions or caps/floors to add trades.")
        return

    df = pd.DataFrame(portfolio)
    df_display = df.copy()
    if "pv" in df_display.columns:
        df_display["PV (k)"] = (df_display["pv"] / 1e3).round(1)
    if "pv_bp" in df_display.columns:
        df_display["PV (bp)"] = df_display["pv_bp"].round(2)
    if "delta" in df_display.columns:
        df_display["Delta (k)"] = (df_display["delta"] / 1e3).round(1)
    if "vega" in df_display.columns:
        df_display["Vega (k)"] = (df_display["vega"] / 1e3).round(1)
    if "gamma" in df_display.columns:
        df_display["Gamma (k)"] = (df_display["gamma"] / 1e3).round(1)
    if "theta" in df_display.columns:
        df_display["Theta (k)"] = (df_display["theta"] / 1e3).round(1)

    cols_order = [
        "instrument_type", "currency", "side", "payoff_type",
        "expiry", "tenor", "model", "vol_input",
        "notional_mm", "strike", "forward",
        "PV (k)", "PV (bp)", "Delta (k)", "Gamma (k)", "Vega (k)", "Theta (k)"
    ]
    cols_order = [c for c in cols_order if c in df_display.columns]
    df_display = df_display[cols_order]

    st.dataframe(df_display, use_container_width=True)

    indices = list(df.index)
    selection = st.multiselect("Select rows to remove", indices, key="ptf_sel")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Remove selected", key="ptf_rm_sel"):
            if selection:
                st.session_state["portfolio"] = [
                    row for i, row in enumerate(portfolio) if i not in selection
                ]
                st.rerun()
    with col2:
        if st.button("Clear ALL", key="ptf_clear_all"):
            st.session_state["portfolio"] = []
            st.rerun()
    with col3:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="RateEdge_portfolio.csv",
            mime="text/csv",
            key="ptf_dl",
        )


def home_tab():
    # Get theme colors for inline styling
    is_dark = st.session_state.get("theme_name", "Dealer Dark") == "Dealer Dark"
    text_color = "#f1f5f9" if is_dark else "#1e3a5f"
    muted_color = "#94a3b8" if is_dark else "#64748b"
    accent_color = "#ef4444" if is_dark else "#dc2626"
    card_bg = "#1e293b" if is_dark else "#ffffff"
    border_color = "#334155" if is_dark else "#e2e8f0"
    
    st.markdown(
        f"""
        <div style="background:{card_bg};border:1px solid {border_color};border-radius:16px;padding:2rem;text-align:center;margin-bottom:2rem;">
            <div style="font-size:2.5rem;font-weight:700;color:{text_color};margin-bottom:0.5rem;">
                <span style="color:#1e3a5f;">Rate</span><span style="color:{accent_color};">Edge</span> Options
            </div>
            <div style="font-size:1.1rem;color:{muted_color};margin-bottom:1rem;">
                Professional Interest Rate Derivatives Pricing
            </div>
            <div style="font-size:0.95rem;color:{muted_color};">
                Swaptions  Caps/Floors  Exotics  CVA  RV Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Login form - Email OTP (shown when not authenticated)
    if not st.session_state.get("authenticated"):
        st.markdown("---")
        st.markdown(f"###  Login with Email")
        
        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            if 'home_auth_step' not in st.session_state:
                st.session_state.home_auth_step = 'email'
            
            if st.session_state.home_auth_step == 'email':
                email = st.text_input("Email address", key="home_login_email", placeholder="your.email@company.com")
                if st.button(" Send Verification Code", key="home_send_btn", use_container_width=True, type="primary"):
                    if email and '@' in email:
                        status, data = request_otp(email)
                        if status == 200:
                            st.session_state.home_auth_step = 'otp'
                            st.session_state.home_auth_email = email
                            st.success(" Code sent!")
                            st.rerun()
                        elif status == 403 and data.get("error") == "access_pending":
                            st.info(data.get("message", "Access request submitted."))
                        else:
                            st.error(f" {data.get('error', 'Failed to send code')}")
                    else:
                        st.error(" Please enter a valid email")
            
            elif st.session_state.home_auth_step == 'otp':
                email = st.session_state.get('home_auth_email', '')
                st.info(f" Code sent to: **{email}**")
                code = st.text_input("Enter 6-digit code", key="home_code", max_chars=6)
                
                if st.button(" Verify", key="home_verify_btn", use_container_width=True, type="primary"):
                    if code and len(code) == 6:
                        status, data = verify_otp(email, code)
                        if status == 200:
                            st.session_state["authenticated"] = True
                            st.session_state["username"] = email
                            st.session_state.home_auth_step = 'email'
                            
                            if HAS_POSTGRES and get_db_url():
                                init_database()
                                loaded = load_all_session_data(email.split('@')[0])
                                if loaded > 0:
                                    st.success(f" Login successful! Loaded {loaded} saved configs.")
                            else:
                                st.success(" Login successful!")
                            st.rerun()
                        else:
                            st.error(f" {data.get('error', 'Invalid code')}")
                    else:
                        st.error(" Please enter the 6-digit code")
                
                if st.button(" Back", key="home_back_btn"):
                    st.session_state.home_auth_step = 'email'
                    st.rerun()
    
    # Features section
    st.markdown("---")
    st.markdown(f"###  Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="background:{card_bg};border:1px solid {border_color};border-radius:12px;padding:1.25rem;">
                <div style="color:{accent_color};font-weight:600;margin-bottom:0.5rem;"> Vol Surfaces</div>
                <div style="color:{muted_color};font-size:0.85rem;">
                    ATM vol grids, SABR smile modeling, 3D surface editor
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background:{card_bg};border:1px solid {border_color};border-radius:12px;padding:1.25rem;">
                <div style="color:{accent_color};font-weight:600;margin-bottom:0.5rem;"> Curve Analytics</div>
                <div style="color:{muted_color};font-size:0.85rem;">
                    Forward swap calculation, curve building, risk analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background:{card_bg};border:1px solid {border_color};border-radius:12px;padding:1.25rem;">
                <div style="color:{accent_color};font-weight:600;margin-bottom:0.5rem;"> Multi-Currency</div>
                <div style="color:{muted_color};font-size:0.85rem;">
                    AUD, NZD, USD with correct market conventions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Market conventions - Swaption + Underlying Swap by Currency
    st.markdown("---")
    st.markdown(f"###  Market Conventions by Currency")
    
    conv_tabs = st.tabs([" USD", " EUR", " GBP", " JPY", " AUD", " NZD", " CAD"])
    
    with conv_tabs[0]:  # USD
        st.markdown("#### USD Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Physical delivery (standard) |
            | Cash Settlement | Par rate method (ISDA 2006) |
            | Premium | Forward premium (T+2) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | Actual swap curve |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Semi-annual, 30/360 |
            | Float Leg | Quarterly, ACT/360 |
            | Float Index | SOFR (Term or compounded) |
            | Spot Lag | T+2 |
            | Roll | Modified Following |
            | Calendar | NYC |
            """)
    
    with conv_tabs[1]:  # EUR
        st.markdown("#### EUR Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Cash-settled (standard) |
            | Cash Settlement | Par rate, ISDA 2006 |
            | Premium | Forward premium (T+2) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | Curve-based |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Annual, 30/360 |
            | Float Leg | Annual (was 6M) |
            | Float Index | STR |
            | Spot Lag | T+2 |
            | Roll | Modified Following |
            | Calendar | TARGET |
            """)
    
    with conv_tabs[2]:  # GBP
        st.markdown("#### GBP Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Physical (LCH clearable) |
            | Cash Settlement | Par rate method |
            | Premium | Forward premium (T+0) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | Curve-based |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Annual, ACT/365F |
            | Float Leg | Annual |
            | Float Index | SONIA compounded |
            | Spot Lag | T+0 |
            | Roll | Modified Following |
            | Calendar | London |
            """)
    
    with conv_tabs[3]:  # JPY
        st.markdown("#### JPY Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Physical or Cash |
            | Cash Settlement | Yield-settled (JSCC) |
            | Premium | Forward premium (T+2) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | JGB curve-based |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Semi-annual, ACT/365F |
            | Float Leg | Semi-annual |
            | Float Index | TONA |
            | Spot Lag | T+2 |
            | Roll | Modified Following |
            | Calendar | Tokyo |
            """)
    
    with conv_tabs[4]:  # AUD
        st.markdown("#### AUD Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Physical delivery |
            | Cash Settlement | Par rate (rare) |
            | Premium | Forward premium (T+1) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | BBSW/AONIA curve |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Q/Q to 3y, S/S beyond, ACT/365F |
            | Float Leg | Quarterly |
            | Float Index | 3M BBSW / AONIA (OIS) |
            | Spot Lag | T+1 (was T+2 pre-2024) |
            | Roll | Modified Following |
            | Calendar | Sydney |
            """)
    
    with conv_tabs[5]:  # NZD
        st.markdown("#### NZD Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Physical delivery |
            | Cash Settlement | Par rate (rare) |
            | Premium | Forward premium (T+2) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | BKBM/OCR curve |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Q/Q to 2y, S/S beyond, ACT/365F |
            | Float Leg | Quarterly |
            | Float Index | 3M BKBM / OCR (OIS) |
            | Spot Lag | T+2 |
            | Roll | Modified Following |
            | Calendar | Auckland/Wellington |
            """)
    
    with conv_tabs[6]:  # CAD
        st.markdown("#### CAD Swaption Conventions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swaption Specifics**
            | Convention | Standard |
            |------------|----------|
            | Settlement | Physical (standard) |
            | Cash Settlement | Par rate method |
            | Premium | Forward premium (T+1) |
            | Exercise | European |
            | Vol Quote | Normal (bp/annum) |
            | Annuity | CORRA curve |
            """)
        with col2:
            st.markdown("""
            **Underlying Swap**
            | Convention | Standard |
            |------------|----------|
            | Fixed Leg | Semi-annual, ACT/365F |
            | Float Leg | Semi-annual |
            | Float Index | CORRA |
            | Spot Lag | T+1 |
            | Roll | Modified Following |
            | Calendar | Toronto |
            """)


def rate_vol_matrix_tab(ccy: str):
    """Rate/Vol Matrix tab - system generated forward rates and ATM premiums"""
    st.subheader(" Rate/Vol Matrix")
    st.caption("System-generated forward swap rates and ATM option premiums")
    
    # Check if we have required data
    curve = get_ccy_curve(ccy)
    atm_vols, _, _, _, _ = get_ccy_vol_data(ccy)
    basis_6v3 = get_basis_curve(ccy, "6v3")
    
    if curve is None:
        st.warning(" No curve loaded. Please upload config in Vol/SABR tab first.")
        return
    
    # Initialize session state for matrices
    if "fwd_matrix" not in st.session_state:
        st.session_state["fwd_matrix"] = {}
    if "basis_matrix" not in st.session_state:
        st.session_state["basis_matrix"] = {}
    if "prem_matrix" not in st.session_state:
        st.session_state["prem_matrix"] = {}
    
    # Check if already generated
    has_fwd = ccy in st.session_state.get("fwd_matrix", {})
    
    # === CONTROLS ROW ===
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 2])
    with ctrl_col1:
        show_heatmap = st.checkbox(" Show Heatmap Colors", value=False, key="show_heatmap")
    with ctrl_col2:
        if st.button(" Generate All Matrices", key="gen_all_matrices", type="primary"):
            with st.spinner("Generating matrices..."):
                # Generate forward matrix
                fwd_matrix = generate_forward_matrix(ccy, curve, basis_6v3)
                st.session_state["fwd_matrix"][ccy] = fwd_matrix
                
                # Generate basis matrix if available
                if basis_6v3 is not None:
                    basis_matrix = generate_basis_matrix(ccy, basis_6v3)
                    st.session_state["basis_matrix"][ccy] = basis_matrix
                
                # Generate ATM premium matrix if vols available
                if atm_vols is not None:
                    prem_matrix = calculate_atm_premium_matrix(ccy, curve, atm_vols, basis_6v3)
                    st.session_state["prem_matrix"][ccy] = prem_matrix
            
            st.rerun()  # Rerun to update UI with new data
    with ctrl_col3:
        if has_fwd:
            if st.button(" Refresh (Clear Cache)", key="refresh_matrices"):
                # Clear the cache
                clear_matrix_cache()
                # Clear session state for this currency
                if ccy in st.session_state.get("fwd_matrix", {}):
                    del st.session_state["fwd_matrix"][ccy]
                if ccy in st.session_state.get("basis_matrix", {}):
                    del st.session_state["basis_matrix"][ccy]
                if ccy in st.session_state.get("prem_matrix", {}):
                    del st.session_state["prem_matrix"][ccy]
                st.info("Cache cleared. Click 'Generate All Matrices' to recalculate with latest data.")
                st.rerun()
    with ctrl_col4:
        if has_fwd:
            st.caption(" Data loaded")
    
    # Re-check if data is loaded (in case we just generated)
    has_fwd = ccy in st.session_state.get("fwd_matrix", {})
    has_basis = ccy in st.session_state.get("basis_matrix", {})
    has_prem = ccy in st.session_state.get("prem_matrix", {})
    
    if not has_fwd:
        st.info(" Click 'Generate All Matrices' to calculate forward rates and premiums")
        return
    
    # === FORWARD SWAP RATES SECTION ===
    st.markdown("---")
    st.markdown("###  Forward Swap Rates")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        rate_options = ["IRS Fwd"]
        if has_basis:
            rate_options.append("6v3 Basis")
        rate_view = st.radio(
            "View",
            rate_options,
            horizontal=True,
            key="rate_view_toggle"
        )
    with col2:
        if rate_view == "IRS Fwd" and has_fwd:
            csv = st.session_state["fwd_matrix"][ccy].to_csv()
            st.download_button(" Download", csv, f"{ccy}_fwd_matrix.csv", type="primary", key="dl_fwd")
        elif rate_view == "6v3 Basis" and has_basis:
            csv = st.session_state["basis_matrix"][ccy].to_csv()
            st.download_button(" Download", csv, f"{ccy}_basis_matrix.csv", type="primary", key="dl_basis")
    
    # Display rate matrix
    if rate_view == "IRS Fwd" and has_fwd:
        df = st.session_state["fwd_matrix"][ccy]
        if show_heatmap:
            st.dataframe(
                df.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", axis=None),
                use_container_width=True,
                height=600
            )
        else:
            st.dataframe(df.style.format("{:.3f}"), use_container_width=True, height=600)
    elif rate_view == "6v3 Basis" and has_basis:
        df = st.session_state["basis_matrix"][ccy]
        if show_heatmap:
            st.dataframe(
                df.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
                height=600
            )
        else:
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True, height=600)
    elif rate_view == "6v3 Basis":
        st.info("6v3 basis not available for this currency")
    
    # === ATM VOL / PREMIUM SECTION ===
    st.markdown("---")
    st.markdown("###  ATM Vol / Premium")
    
    if atm_vols is None:
        st.warning(" No ATM vols loaded. Please upload config first.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            prem_options = ["ATM Vol (bp)"]
            if has_prem:
                prem_options.append("ATM Premium (bp)")
            prem_view = st.radio(
                "View",
                prem_options,
                horizontal=True,
                key="prem_view_toggle"
            )
        with col2:
            if prem_view == "ATM Vol (bp)":
                csv = atm_vols.to_csv(index=False)
                st.download_button(" Download", csv, f"{ccy}_atm_vols.csv", type="primary", key="dl_atm")
            elif has_prem:
                csv = st.session_state["prem_matrix"][ccy].to_csv()
                st.download_button(" Download", csv, f"{ccy}_atm_prem.csv", type="primary", key="dl_prem")
        
        # Display vol/premium matrix
        if prem_view == "ATM Vol (bp)":
            display_df = atm_vols.copy()
            if "Expiry" in display_df.columns:
                display_df = display_df.set_index("Expiry")
            if show_heatmap:
                st.dataframe(
                    display_df.style.format("{:.2f}").background_gradient(cmap="YlOrRd", axis=None),
                    use_container_width=True,
                    height=600
                )
            else:
                st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True, height=600)
        elif has_prem:
            df = st.session_state["prem_matrix"][ccy]
            if show_heatmap:
                st.dataframe(
                    df.style.format("{:.2f}").background_gradient(cmap="YlOrRd", axis=None),
                    use_container_width=True,
                    height=600
                )
            else:
                st.dataframe(df.style.format("{:.2f}"), use_container_width=True, height=600)


def generate_basis_matrix(ccy: str, basis_6v3: pd.DataFrame) -> pd.DataFrame:
    """Generate basis matrix - wrapper for cached version"""
    basis_tuple = tuple(basis_6v3["MaturityY"].tolist()), tuple(basis_6v3["BasisBp"].tolist())
    return _generate_basis_matrix_cached(ccy, basis_tuple)


@st.cache_data(ttl=3600, show_spinner=False)
def _generate_basis_matrix_cached(ccy: str, basis_tuple: tuple) -> pd.DataFrame:
    """Generate basis matrix interpolated across expiry/tenor grid - CACHED"""
    expiries = ["1w", "1m", "2m", "3m", "6m", "9m", "1y", "18m", "2y", "3y", "4y", "5y", "7y", "10y", "12y", "15y", "20y"]
    tenors = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
    
    basis_x = np.array(basis_tuple[0])
    basis_y = np.array(basis_tuple[1])
    
    matrix = []
    for exp in expiries:
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        for tenor in tenors:
            tenor_y = float(tenor[:-1])
            mid_point = exp_y + tenor_y / 2
            basis_bp = float(np.interp(mid_point, basis_x, basis_y))
            row[tenor] = basis_bp
        matrix.append(row)
    
    df = pd.DataFrame(matrix).set_index("Expiry")
    return df


def calculate_atm_premium_matrix(ccy: str, curve: pd.DataFrame, atm_vols: pd.DataFrame, 
                                  basis_6v3: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Calculate ATM premiums using real annuities from curve"""
    if "Expiry" not in atm_vols.columns:
        return pd.DataFrame()
    
    expiries = atm_vols["Expiry"].tolist()
    tenors = [c for c in atm_vols.columns if c != "Expiry"]
    
    # Get OIS curve for discounting
    ois_curve = get_basis_curve(ccy, "ois")
    
    matrix = []
    for i, exp in enumerate(expiries):
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        
        for j, tenor in enumerate(tenors):
            try:
                vol_bp = atm_vols.iloc[i][tenor]
                
                if pd.isna(vol_bp):
                    row[tenor] = None
                    continue
                
                tenor_y = label_to_years(tenor)
                
                # Get real annuity from curve
                _, ann, _ = forward_and_annuity_from_curve(curve, ccy, exp_y, tenor_y, ois_curve)
                
                sigma_n = vol_bp / 10000.0
                sqrt_t = math.sqrt(max(exp_y, 0.001))
                # Forward premium = 2  0.3989  _n  T  Annuity
                fwd_premium = 2 * 0.3989 * sigma_n * sqrt_t * ann
                premium_bp = fwd_premium * 10000
                
                row[tenor] = round(premium_bp, 2)
            except:
                row[tenor] = None
        
        matrix.append(row)
    
    df = pd.DataFrame(matrix).set_index("Expiry")
    return df


# Keep old cached version for backwards compat but it's not used anymore
@st.cache_data(ttl=3600, show_spinner=False)
def _calculate_atm_premium_matrix_cached(ccy: str, expiries: tuple, tenors: tuple, vol_data: tuple) -> pd.DataFrame:
    """DEPRECATED - kept for backwards compatibility"""
    matrix = []
    for i, exp in enumerate(expiries):
        exp_y = label_to_years(exp)
        row = {"Expiry": exp}
        for j, tenor in enumerate(tenors):
            try:
                vol_bp = vol_data[i][j]
                if vol_bp == -999:
                    row[tenor] = None
                    continue
                tenor_y = label_to_years(tenor)
                annuity = tenor_y * 0.85
                sigma_n = vol_bp / 10000.0
                sqrt_t = math.sqrt(max(exp_y, 0.001))
                fwd_premium = 2 * 0.3989 * sigma_n * sqrt_t * annuity
                premium_bp = fwd_premium * 10000
                row[tenor] = round(premium_bp, 2)
            except:
                row[tenor] = None
        matrix.append(row)
    df = pd.DataFrame(matrix).set_index("Expiry")
    return df


def clear_matrix_cache():
    """Clear all cached matrix data"""
    _generate_forward_matrix_cached.clear()
    _generate_basis_matrix_cached.clear()
    _calculate_atm_premium_matrix_cached.clear()


# ============================
# Main
# ============================

def main():
    st.set_page_config(
        page_title="RateEdge Options",
        layout="wide",
        page_icon="",
        initial_sidebar_state="expanded"
    )
    init_session()
    
    # Auto-load from database on first run (if connected and not already loaded)
    if HAS_POSTGRES and get_db_url() and not st.session_state.get("db_auto_loaded", False):
        user_id = st.session_state.get("username", "default")
        loaded = load_all_session_data(user_id)
        st.session_state["db_auto_loaded"] = True
        if loaded > 0:
            st.toast(f" Auto-loaded {loaded} configs from database", icon="")

    # Sidebar for settings
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0;border-bottom:1px solid #334155;margin-bottom:1rem;">
                <div style="font-size:1.4rem;font-weight:700;">
                    <span style="color:#1e3a5f;">Rate</span><span style="color:#ef4444;">Edge</span>
                </div>
                <div style="font-size:0.75rem;color:#94a3b8;">Options Platform v28</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("###  Settings")
        
        # Theme
        theme_choice = st.selectbox(
            " Theme",
            ["Dealer Dark", "Clean Light"],
            index=0 if st.session_state.get("theme_name", "Dealer Dark") == "Dealer Dark" else 1,
            key="sidebar_theme",
        )
        st.session_state["theme_name"] = theme_choice
        
        # Currency
        ccy = st.selectbox(
            " Currency",
            SUPPORTED_CURRENCIES,
            index=0,
            key="sidebar_ccy",
        )
        
        # Vol mode
        vol_mode = st.selectbox(
            " Vol Mode",
            ["Normal (bp)", "Black (lognormal)"],
            index=0,
            key="sidebar_volmode",
        )
        
        st.markdown("---")
        
        # User status
        if st.session_state.get("authenticated"):
            st.markdown(
                f"""
                <div style="background:#1e3a5f;padding:0.75rem;border-radius:8px;margin-top:0.5rem;">
                    <div style="color:#94a3b8;font-size:0.7rem;">Logged in as</div>
                    <div style="color:#22c55e;font-weight:600;">{st.session_state.get('username', 'User')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(" Logout", key="logout_btn", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["username"] = None
                st.rerun()
        else:
            st.warning(" Login required")
            st.caption("Use the main login page to sign in with your email")
        
        st.markdown("---")
        st.markdown(
            """
            <div style="color:#64748b;font-size:0.7rem;text-align:center;">
                 2024 RateEdge Australia<br>
                <a href="https://rateedge.au" style="color:#3b82f6;">rateedge.au</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    apply_rateedge_theme(st.session_state["theme_name"])

    # Preserve auth through vol editor Apply reload
    if 'v3d_data' in st.query_params:
        st.session_state["authenticated"] = True
        # Jump directly to vol editor without showing tabs
        vol_surface_editor_tab()
        return

    # Check if authenticated - if not, show login page only
    if not st.session_state.get("authenticated"):
        show_login_page()
        return

    # Only show tabs if authenticated
    tabs = st.tabs(
        [
            " Home",
            " Vol / SABR",
            " Curves",
            " Rate/Vol Matrix",
            " Swaptions",
            " Caps & Floors",
            " Portfolio",
            " Exotics",
            " Vol Editor",
            " Multi-CCY",
            " Credit / XVA",
            " Backtesting",
            " RV / Calendar",
        ]
    )

    with tabs[0]:
        home_tab()
    with tabs[1]:
        vol_config_tab()
    with tabs[2]:
        curves_tab()
    with tabs[3]:
        rate_vol_matrix_tab(ccy)
    with tabs[4]:
        show_header(ccy)
        swaptions_tab(ccy, vol_mode)
    with tabs[5]:
        caps_floors_tab(ccy, vol_mode)
    with tabs[6]:
        portfolio_tab()
    with tabs[7]:
        exotics_tab(ccy, vol_mode)
    with tabs[8]:
        vol_surface_editor_tab()
    with tabs[9]:
        multi_ccy_tab(vol_mode)
    with tabs[10]:
        credit_xva_tab()
    with tabs[11]:
        backtesting_tab()
    with tabs[12]:
        rv_tab()


def show_login_page():
    """Full-page login with email OTP authentication - matches Data Portal style"""
    
    # Hide ALL Streamlit chrome including sidebar
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    .stApp {background: #0a0f1a;}
    section[data-testid="stSidebar"] {display: none !important;}
    .css-1d391kg {display: none !important;}
    .stDeployButton {display: none !important;}
    
    /* Login card container */
    .login-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 48px 40px;
        max-width: 420px;
        margin: 0 auto;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Input styling */
    .stTextInput > label {display: none;}
    .stTextInput > div > div > input {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        padding: 14px 16px !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #dc2626 !important;
        box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2) !important;
    }
    .stTextInput > div > div > input::placeholder {color: #94a3b8 !important;}
    
    /* Button styling */
    .stButton > button {
        background: #dc2626 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 24px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {background: #b91c1c !important;}
    
    /* Back button styling */
    .back-btn button {
        background: #334155 !important;
    }
    .back-btn button:hover {
        background: #475569 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the login card vertically and horizontally
    st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)
    
    # Create centered container using columns
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        # Login card with all content
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        # Logo
        st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <svg viewBox="0 0 200 50" width="160" height="40" xmlns="http://www.w3.org/2000/svg">
                <path d="M25 5 L45 25 L25 45 L5 25 Z" fill="#dc2626"/>
                <path d="M25 12 L38 25 L25 38 L12 25 Z" fill="#1e293b"/>
                <text x="55" y="33" font-family="system-ui, -apple-system, sans-serif" font-size="24" font-weight="700" fill="#f9fafb">RateEdge</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        # Title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 8px;">
            <div style="color: #f9fafb; font-size: 1.5rem; font-weight: 700;">Options Platform</div>
        </div>
        <div style="text-align: center; margin-bottom: 32px;">
            <div style="color: #94a3b8; font-size: 0.95rem;">Sign in with your email</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize auth state
        if 'auth_step' not in st.session_state:
            st.session_state.auth_step = 'email'
        
        if st.session_state.auth_step == 'email':
            email = st.text_input("Email", placeholder="Enter your email", key="login_email_input", label_visibility="collapsed")
            
            st.markdown('<p style="color: #94a3b8; font-size: 0.85rem; text-align: center; margin: -8px 0 24px 0;">We\'ll send you a verification code</p>', unsafe_allow_html=True)
            
            if st.button("Send Code", key="send_code_btn", use_container_width=True):
                if email and "@" in email:
                    status, data = request_otp(email.strip().lower())
                    if status == 200:
                        st.session_state.auth_email = email.strip().lower()
                        st.session_state.auth_step = 'otp'
                        st.rerun()
                    elif status == 202 or (status == 403 and data.get("error") == "access_pending"):
                        st.info("Access request submitted. You will be notified when approved.")
                    else:
                        st.error(data.get("error", "Failed to send code"))
                else:
                    st.error("Please enter a valid email address")
        
        elif st.session_state.auth_step == 'otp':
            st.markdown(f'<p style="color: #94a3b8; text-align: center; margin-bottom: 16px;">Code sent to <strong style="color: #f1f5f9;">{st.session_state.auth_email}</strong></p>', unsafe_allow_html=True)
            
            otp = st.text_input("Code", placeholder="Enter 6-digit code", max_chars=6, key="login_otp_input", label_visibility="collapsed")
            
            col_back, col_verify = st.columns(2)
            
            with col_back:
                st.markdown('<div class="back-btn">', unsafe_allow_html=True)
                if st.button(" Back", key="back_btn", use_container_width=True):
                    st.session_state.auth_step = 'email'
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_verify:
                if st.button("Verify", key="verify_btn", use_container_width=True):
                    if otp and len(otp) == 6:
                        status, data = verify_otp(st.session_state.auth_email, otp)
                        if status == 200:
                            st.session_state["authenticated"] = True
                            st.session_state["username"] = st.session_state.auth_email
                            st.session_state["user_email"] = st.session_state.auth_email
                            st.session_state.auth_step = 'email'
                            st.rerun()
                        else:
                            st.error(data.get("error", "Invalid code"))
                    else:
                        st.error("Please enter the 6-digit code")
        
        # Admin login link
        st.markdown("""
        <div style="text-align: center; margin-top: 24px; padding-top: 16px; border-top: 1px solid #334155;">
            <span style="color: #64748b; font-size: 0.9rem;">Admin login</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close login-card
    
    # Contact footer
    st.markdown("""
    <div style="text-align: center; margin-top: 32px; color: #64748b; font-size: 0.85rem;">
        Contact <a href="mailto:wpo@rateedge.au" style="color: #3b82f6; text-decoration: none;">wpo@rateedge.au</a> for access
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()


if __name__ == "__main__":
    main()

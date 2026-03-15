"""
RateEdge Authentication Module for Streamlit Apps
Provides email OTP authentication matching the Data Portal / OMS style.
"""

import streamlit as st
import requests
from typing import Tuple, Dict, Any

AUTH_API_URL = "https://rateedge-auth.azurewebsites.net"


def request_otp(email: str, site: str = "options") -> Tuple[int, Dict[str, Any]]:
    """Request OTP code to be sent to email."""
    try:
        response = requests.post(
            f"{AUTH_API_URL}/api/request-otp",
            json={"email": email, "site": site},
            timeout=10
        )
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}


def verify_otp(email: str, otp: str, site: str = "options") -> Tuple[int, Dict[str, Any]]:
    """Verify OTP code and get auth token."""
    try:
        response = requests.post(
            f"{AUTH_API_URL}/api/verify-otp",
            json={"email": email, "otp": otp, "site": site},
            timeout=10
        )
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get("rateedge_token") is not None


def get_user_email() -> str:
    """Get authenticated user's email."""
    return st.session_state.get("user_email", "")


def logout():
    """Clear authentication state."""
    st.session_state.pop("rateedge_token", None)
    st.session_state.pop("user_email", None)
    st.session_state.pop("auth_step", None)
    st.session_state.pop("auth_email", None)


def render_login_page(site_name: str = "Options Platform", site_code: str = "options"):
    """
    Render the RateEdge login page matching Data Portal / OMS style.
    
    Args:
        site_name: Display name for the portal (e.g., "Options Platform", "Historical Data Portal")
        site_code: Site code for auth API (e.g., "options", "data", "oms")
    """
    
    # Initialize auth state
    if "auth_step" not in st.session_state:
        st.session_state.auth_step = "email"
    
    # Full page dark background
    st.markdown("""
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp > header {display: none;}
    
    /* Dark background */
    .stApp {
        background: #0a0f1a;
    }
    
    /* Center the login box */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    
    .auth-box {
        background: #131b2e;
        border-radius: 16px;
        padding: 48px 40px;
        width: 100%;
        max-width: 420px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    .auth-logo {
        text-align: center;
        margin-bottom: 32px;
    }
    
    .auth-logo svg {
        height: 48px;
    }
    
    .auth-title {
        color: #f9fafb;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .auth-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 32px;
    }
    
    .auth-input {
        width: 100%;
        padding: 14px 16px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f1f5f9;
        font-size: 1rem;
        margin-bottom: 12px;
        outline: none;
        transition: border-color 0.2s;
    }
    
    .auth-input:focus {
        border-color: #3b82f6;
    }
    
    .auth-input::placeholder {
        color: #64748b;
    }
    
    .auth-hint {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 24px;
        text-align: center;
    }
    
    .auth-btn {
        width: 100%;
        padding: 14px 24px;
        background: #dc2626;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .auth-btn:hover {
        background: #b91c1c;
    }
    
    .auth-footer {
        text-align: center;
        margin-top: 24px;
    }
    
    .auth-footer a {
        color: #64748b;
        text-decoration: none;
        font-size: 0.9rem;
    }
    
    .auth-footer a:hover {
        color: #94a3b8;
    }
    
    .auth-error {
        background: rgba(220, 38, 38, 0.1);
        border: 1px solid rgba(220, 38, 38, 0.3);
        color: #fca5a5;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    
    .auth-success {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #86efac;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    
    .auth-info {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93c5fd;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit input labels */
    .stTextInput > label {display: none;}
    
    /* Style Streamlit inputs */
    .stTextInput > div > div > input {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        padding: 14px 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: none !important;
    }
    
    /* Style Streamlit buttons */
    .stButton > button {
        width: 100%;
        padding: 14px 24px !important;
        background: #dc2626 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background: #b91c1c !important;
        border: none !important;
    }
    
    /* Back button style */
    .back-btn > button {
        background: #334155 !important;
    }
    
    .back-btn > button:hover {
        background: #475569 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True)
        
        # Logo
        st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <svg viewBox="0 0 200 50" width="200" height="50" xmlns="http://www.w3.org/2000/svg">
                <path d="M25 5 L45 25 L25 45 L5 25 Z" fill="#dc2626"/>
                <path d="M25 12 L38 25 L25 38 L12 25 Z" fill="#0a0f1a"/>
                <text x="55" y="33" font-family="system-ui, -apple-system, sans-serif" font-size="24" font-weight="700" fill="#f9fafb">RateEdge</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        # Title
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 32px;">
            <div style="color: #f9fafb; font-size: 1.5rem; font-weight: 600; margin-bottom: 8px;">{site_name}</div>
            <div style="color: #94a3b8; font-size: 0.95rem;">Sign in with your email</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Email step
        if st.session_state.auth_step == "email":
            email = st.text_input("Email", placeholder="Enter your email", key="login_email_input", label_visibility="collapsed")
            
            st.markdown('<p style="color: #64748b; font-size: 0.85rem; text-align: center; margin: -8px 0 16px 0;">We\'ll send you a verification code</p>', unsafe_allow_html=True)
            
            if st.button("Send Code", key="send_code_btn", use_container_width=True):
                if email and "@" in email:
                    status, data = request_otp(email.strip().lower(), site_code)
                    if status == 200:
                        st.session_state.auth_email = email.strip().lower()
                        st.session_state.auth_step = "otp"
                        st.rerun()
                    elif status == 202:
                        # Access pending approval
                        st.markdown('<div class="auth-info">Access request submitted. You will be notified when approved.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="auth-error">{data.get("error", "Failed to send code")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="auth-error">Please enter a valid email address</div>', unsafe_allow_html=True)
        
        # OTP step
        elif st.session_state.auth_step == "otp":
            st.markdown(f'<p style="color: #94a3b8; text-align: center; margin-bottom: 16px;">Code sent to <strong style="color: #f1f5f9;">{st.session_state.auth_email}</strong></p>', unsafe_allow_html=True)
            
            otp = st.text_input("Code", placeholder="Enter 6-digit code", max_chars=6, key="login_otp_input", label_visibility="collapsed")
            
            col_back, col_verify = st.columns(2)
            
            with col_back:
                st.markdown('<div class="back-btn">', unsafe_allow_html=True)
                if st.button("← Back", key="back_btn", use_container_width=True):
                    st.session_state.auth_step = "email"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_verify:
                if st.button("Verify", key="verify_btn", use_container_width=True):
                    if otp and len(otp) == 6:
                        status, data = verify_otp(st.session_state.auth_email, otp, site_code)
                        if status == 200:
                            st.session_state.rateedge_token = data.get("token", "authenticated")
                            st.session_state.user_email = st.session_state.auth_email
                            st.session_state.auth_step = "done"
                            st.rerun()
                        else:
                            st.markdown(f'<div class="auth-error">{data.get("error", "Invalid code")}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="auth-error">Please enter the 6-digit code</div>', unsafe_allow_html=True)
        
        # Admin login link
        st.markdown("""
        <div style="text-align: center; margin-top: 24px;">
            <span style="color: #64748b; font-size: 0.9rem;">Admin login</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Contact info
        st.markdown("""
        <div style="text-align: center; margin-top: 48px; color: #64748b; font-size: 0.85rem;">
            Contact <a href="mailto:wpo@rateedge.au" style="color: #3b82f6; text-decoration: none;">wpo@rateedge.au</a> for access
        </div>
        """, unsafe_allow_html=True)


def require_auth(site_name: str = "Options Platform", site_code: str = "options"):
    """
    Decorator-style function to require authentication.
    Call at the start of your app - if not authenticated, shows login page and stops.
    
    Usage:
        from rateedge_auth import require_auth
        
        require_auth("Options Platform", "options")
        
        # Rest of your app only runs if authenticated
        st.write(f"Welcome {get_user_email()}")
    """
    if not is_authenticated():
        render_login_page(site_name, site_code)
        st.stop()

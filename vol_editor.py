"""
RateEdge Vol Surface Editor
===========================
A hybrid 2D/3D volatility surface editor for interest rate derivatives.

Features:
- Interactive 3D Plotly surface (rotatable, zoomable - read-only)
- Editable 2D data grid with real-time 3D sync
- Undo/redo functionality
- Publish to pricing engine
- RateEdge brand styling (Navy/Blue/Red on dark theme)

Usage:
    from vol_editor import render_vol_surface_editor
    
    updated_surface = render_vol_surface_editor(
        ccy="AUD",
        atm_surface=my_vol_dataframe
    )
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple, List
from dataclasses import dataclass
from copy import deepcopy
import json

# =============================================================================
# RATEEDGE BRAND CONSTANTS
# =============================================================================

BRAND = {
    "navy": "#1e3a5f",
    "blue": "#2563eb", 
    "red": "#dc2626",
    "slate_bg": "#0f172a",
    "slate_800": "#1e293b",
    "slate_700": "#334155",
    "slate_600": "#475569",
    "slate_400": "#94a3b8",
    "slate_300": "#cbd5e1",
    "white": "#ffffff",
    "green": "#22c55e",
    "amber": "#f59e0b",
}

# Plotly colorscale matching RateEdge brand
RATEEDGE_COLORSCALE = [
    [0.0, "#1e3a5f"],      # Navy (low vols)
    [0.25, "#2563eb"],     # Blue
    [0.5, "#6366f1"],      # Indigo transition
    [0.75, "#f59e0b"],     # Amber (mid-high)
    [1.0, "#dc2626"],      # Red (high vols)
]

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def _init_editor_state(ccy: str, atm_surface: pd.DataFrame) -> None:
    """Initialize session state for the vol editor."""
    if "vol_editor" not in st.session_state:
        st.session_state["vol_editor"] = {
            "working": {},
            "base": {},
            "history": {},
            "redo_stack": {},
            "selected_cell": {},
        }
    
    editor = st.session_state["vol_editor"]
    
    # Initialize for this currency if needed
    if ccy not in editor["working"]:
        editor["working"][ccy] = atm_surface.copy()
        editor["base"][ccy] = atm_surface.copy()
        editor["history"][ccy] = []
        editor["redo_stack"][ccy] = []
        editor["selected_cell"][ccy] = None


def _push_history(ccy: str) -> None:
    """Save current state to history for undo."""
    editor = st.session_state["vol_editor"]
    current = editor["working"][ccy].copy()
    editor["history"][ccy].append(current)
    # Clear redo stack when new edit is made
    editor["redo_stack"][ccy] = []
    # Limit history to 50 entries
    if len(editor["history"][ccy]) > 50:
        editor["history"][ccy] = editor["history"][ccy][-50:]


def _undo(ccy: str) -> bool:
    """Undo last edit. Returns True if successful."""
    editor = st.session_state["vol_editor"]
    if editor["history"][ccy]:
        # Save current to redo stack
        editor["redo_stack"][ccy].append(editor["working"][ccy].copy())
        # Restore from history
        editor["working"][ccy] = editor["history"][ccy].pop()
        return True
    return False


def _redo(ccy: str) -> bool:
    """Redo last undone edit. Returns True if successful."""
    editor = st.session_state["vol_editor"]
    if editor["redo_stack"][ccy]:
        # Save current to history
        editor["history"][ccy].append(editor["working"][ccy].copy())
        # Restore from redo stack
        editor["working"][ccy] = editor["redo_stack"][ccy].pop()
        return True
    return False


def _has_changes(ccy: str) -> bool:
    """Check if working surface differs from base."""
    editor = st.session_state["vol_editor"]
    working = editor["working"][ccy]
    base = editor["base"][ccy]
    return not working.equals(base)


def _publish(ccy: str) -> pd.DataFrame:
    """Publish working surface to pricing engine."""
    editor = st.session_state["vol_editor"]
    editor["base"][ccy] = editor["working"][ccy].copy()
    editor["history"][ccy] = []
    editor["redo_stack"][ccy] = []
    
    # Update the pricing engine state if it exists
    if "vol_data" in st.session_state and ccy in st.session_state["vol_data"]:
        st.session_state["vol_data"][ccy]["atm"] = editor["working"][ccy].copy()
    
    return editor["working"][ccy].copy()


def _reset_to_base(ccy: str) -> None:
    """Reset working surface to last published base."""
    editor = st.session_state["vol_editor"]
    _push_history(ccy)
    editor["working"][ccy] = editor["base"][ccy].copy()


# =============================================================================
# 3D SURFACE VISUALIZATION
# =============================================================================

def _create_3d_surface(
    df: pd.DataFrame,
    ccy: str,
    show_diff: bool = False,
    base_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Create an interactive 3D vol surface plot."""
    
    # Extract numeric columns (tenors) and expiry labels
    expiry_col = df.columns[0]  # Usually "Expiry"
    tenor_cols = df.columns[1:].tolist()
    expiries = df[expiry_col].tolist()
    
    # Create numeric grids for plotting
    x_vals = np.arange(len(tenor_cols))  # Tenor axis
    y_vals = np.arange(len(expiries))     # Expiry axis
    z_vals = df[tenor_cols].values.astype(float)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Hover text with full details
    hover_text = []
    for i, exp in enumerate(expiries):
        row_text = []
        for j, tenor in enumerate(tenor_cols):
            val = z_vals[i, j]
            text = f"Expiry: {exp}<br>Tenor: {tenor}<br>Vol: {val:.1f} bp"
            if show_diff and base_df is not None:
                base_val = base_df[tenor_cols].values[i, j]
                diff = val - base_val
                text += f"<br>Δ: {diff:+.1f} bp"
            row_text.append(text)
        hover_text.append(row_text)
    
    fig = go.Figure()
    
    # Main surface
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=z_vals,
        colorscale=RATEEDGE_COLORSCALE,
        hoverinfo="text",
        text=hover_text,
        showscale=True,
        colorbar=dict(
            title=dict(text="Vol (bp)", font=dict(color=BRAND["slate_300"])),
            tickfont=dict(color=BRAND["slate_400"]),
            bgcolor=BRAND["slate_800"],
            bordercolor=BRAND["slate_700"],
            borderwidth=1,
            len=0.7,
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor=BRAND["white"],
                project_z=True,
            )
        ),
        opacity=0.95,
    ))
    
    # Add wireframe for better depth perception
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=z_vals,
        showscale=False,
        opacity=0.3,
        colorscale=[[0, BRAND["slate_400"]], [1, BRAND["slate_400"]]],
        hidesurface=True,
        contours=dict(
            x=dict(show=True, color=BRAND["slate_600"], width=1),
            y=dict(show=True, color=BRAND["slate_600"], width=1),
        ),
        hoverinfo="skip",
    ))
    
    # Layout with RateEdge styling
    fig.update_layout(
        title=dict(
            text=f"<b>{ccy} ATM Vol Surface</b>",
            font=dict(size=16, color=BRAND["slate_300"], family="system-ui"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(
                title="Tenor",
                ticktext=tenor_cols,
                tickvals=list(range(len(tenor_cols))),
                tickfont=dict(size=10, color=BRAND["slate_400"]),
                titlefont=dict(size=12, color=BRAND["slate_300"]),
                gridcolor=BRAND["slate_700"],
                showbackground=True,
                backgroundcolor=BRAND["slate_bg"],
                linecolor=BRAND["slate_600"],
            ),
            yaxis=dict(
                title="Expiry",
                ticktext=expiries,
                tickvals=list(range(len(expiries))),
                tickfont=dict(size=10, color=BRAND["slate_400"]),
                titlefont=dict(size=12, color=BRAND["slate_300"]),
                gridcolor=BRAND["slate_700"],
                showbackground=True,
                backgroundcolor=BRAND["slate_bg"],
                linecolor=BRAND["slate_600"],
            ),
            zaxis=dict(
                title="Vol (bp)",
                tickfont=dict(size=10, color=BRAND["slate_400"]),
                titlefont=dict(size=12, color=BRAND["slate_300"]),
                gridcolor=BRAND["slate_700"],
                showbackground=True,
                backgroundcolor=BRAND["slate_bg"],
                linecolor=BRAND["slate_600"],
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1, z=0.6),
        ),
        paper_bgcolor=BRAND["slate_bg"],
        plot_bgcolor=BRAND["slate_bg"],
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )
    
    return fig


# =============================================================================
# DIFF VISUALIZATION
# =============================================================================

def _create_diff_heatmap(
    working: pd.DataFrame,
    base: pd.DataFrame
) -> go.Figure:
    """Create a heatmap showing differences from base surface."""
    
    expiry_col = working.columns[0]
    tenor_cols = working.columns[1:].tolist()
    expiries = working[expiry_col].tolist()
    
    # Calculate differences
    diff_vals = working[tenor_cols].values - base[tenor_cols].values
    
    # Create hover text
    hover_text = []
    for i, exp in enumerate(expiries):
        row_text = []
        for j, tenor in enumerate(tenor_cols):
            diff = diff_vals[i, j]
            curr = working[tenor_cols].values[i, j]
            prev = base[tenor_cols].values[i, j]
            row_text.append(
                f"Expiry: {exp}<br>Tenor: {tenor}<br>"
                f"Current: {curr:.1f}<br>Base: {prev:.1f}<br>"
                f"<b>Change: {diff:+.1f} bp</b>"
            )
        hover_text.append(row_text)
    
    # Symmetric colorscale around 0
    max_abs_diff = max(abs(diff_vals.min()), abs(diff_vals.max()), 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=diff_vals,
        x=tenor_cols,
        y=expiries,
        text=hover_text,
        hoverinfo="text",
        colorscale=[
            [0, BRAND["blue"]],      # Negative (vol down)
            [0.5, BRAND["slate_700"]],  # Zero
            [1, BRAND["red"]],       # Positive (vol up)
        ],
        zmin=-max_abs_diff,
        zmax=max_abs_diff,
        colorbar=dict(
            title="Δ Vol (bp)",
            titlefont=dict(color=BRAND["slate_300"]),
            tickfont=dict(color=BRAND["slate_400"]),
        ),
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Changes from Published Surface</b>",
            font=dict(size=14, color=BRAND["slate_300"]),
            x=0.5,
        ),
        xaxis=dict(
            title="Tenor",
            tickfont=dict(color=BRAND["slate_400"]),
            titlefont=dict(color=BRAND["slate_300"]),
        ),
        yaxis=dict(
            title="Expiry",
            tickfont=dict(color=BRAND["slate_400"]),
            titlefont=dict(color=BRAND["slate_300"]),
            autorange="reversed",
        ),
        paper_bgcolor=BRAND["slate_bg"],
        plot_bgcolor=BRAND["slate_800"],
        height=350,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    
    return fig


# =============================================================================
# CUSTOM CSS FOR DATA EDITOR
# =============================================================================

def _inject_editor_css() -> None:
    """Inject custom CSS for the vol editor styling."""
    st.markdown(f"""
    <style>
    /* Vol Editor Container */
    .vol-editor-container {{
        background: {BRAND["slate_800"]};
        border: 1px solid {BRAND["slate_700"]};
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* Section headers */
    .vol-editor-header {{
        color: {BRAND["slate_300"]};
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .vol-editor-header .indicator {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {BRAND["green"]};
    }}
    
    .vol-editor-header .indicator.modified {{
        background: {BRAND["amber"]};
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    /* Button styling */
    .stButton > button {{
        background: {BRAND["slate_700"]} !important;
        color: {BRAND["slate_300"]} !important;
        border: 1px solid {BRAND["slate_600"]} !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }}
    
    .stButton > button:hover {{
        background: {BRAND["slate_600"]} !important;
        border-color: {BRAND["slate_400"]} !important;
    }}
    
    /* Primary button (Publish) */
    .publish-btn > button {{
        background: {BRAND["blue"]} !important;
        border-color: {BRAND["blue"]} !important;
        color: white !important;
    }}
    
    .publish-btn > button:hover {{
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }}
    
    /* Danger button (Reset) */
    .reset-btn > button {{
        background: transparent !important;
        border-color: {BRAND["red"]} !important;
        color: {BRAND["red"]} !important;
    }}
    
    .reset-btn > button:hover {{
        background: {BRAND["red"]} !important;
        color: white !important;
    }}
    
    /* Data editor styling */
    [data-testid="stDataEditor"] {{
        background: {BRAND["slate_800"]} !important;
        border-radius: 6px;
    }}
    
    [data-testid="stDataEditor"] div[role="gridcell"] {{
        background: {BRAND["slate_800"]} !important;
        color: {BRAND["slate_300"]} !important;
        font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
        font-size: 0.8125rem !important;
    }}
    
    [data-testid="stDataEditor"] div[role="columnheader"] {{
        background: {BRAND["navy"]} !important;
        color: {BRAND["slate_300"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Stats card */
    .vol-stats {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-top: 0.75rem;
    }}
    
    .vol-stat {{
        background: {BRAND["slate_800"]};
        border: 1px solid {BRAND["slate_700"]};
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }}
    
    .vol-stat-label {{
        color: {BRAND["slate_400"]};
        font-size: 0.6875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .vol-stat-value {{
        color: {BRAND["slate_300"]};
        font-size: 1.25rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}
    
    /* Keyboard shortcuts tooltip */
    .shortcuts-info {{
        background: {BRAND["slate_800"]};
        border: 1px solid {BRAND["slate_700"]};
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.75rem;
        color: {BRAND["slate_400"]};
    }}
    
    .shortcut-key {{
        background: {BRAND["slate_700"]};
        border-radius: 3px;
        padding: 0.125rem 0.375rem;
        font-family: monospace;
        color: {BRAND["slate_300"]};
    }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# STATISTICS DISPLAY
# =============================================================================

def _render_surface_stats(df: pd.DataFrame, label: str = "Surface") -> None:
    """Render statistics about the vol surface."""
    tenor_cols = df.columns[1:].tolist()
    vals = df[tenor_cols].values.astype(float)
    
    st.markdown(f"""
    <div class="vol-stats">
        <div class="vol-stat">
            <div class="vol-stat-label">Min Vol</div>
            <div class="vol-stat-value">{vals.min():.1f}</div>
        </div>
        <div class="vol-stat">
            <div class="vol-stat-label">Max Vol</div>
            <div class="vol-stat-value">{vals.max():.1f}</div>
        </div>
        <div class="vol-stat">
            <div class="vol-stat-label">Mean Vol</div>
            <div class="vol-stat-value">{vals.mean():.1f}</div>
        </div>
        <div class="vol-stat">
            <div class="vol-stat-label">Std Dev</div>
            <div class="vol-stat-value">{vals.std():.1f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN EDITOR FUNCTION
# =============================================================================

def render_vol_surface_editor(
    ccy: str,
    atm_surface: pd.DataFrame,
    show_diff_view: bool = True,
    editor_height: int = 400,
) -> pd.DataFrame:
    """
    Render a hybrid 2D/3D volatility surface editor.
    
    Args:
        ccy: Currency code (e.g., "AUD", "NZD", "USD")
        atm_surface: DataFrame with vol surface data
            Columns: Expiry, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y
            Values: Normal vol in basis points
        show_diff_view: Whether to show the diff heatmap when changes exist
        editor_height: Height of the data editor in pixels
    
    Returns:
        Updated DataFrame after user edits (or original if no changes)
    """
    
    # Initialize state
    _init_editor_state(ccy, atm_surface)
    editor = st.session_state["vol_editor"]
    
    # Inject custom CSS
    _inject_editor_css()
    
    # Get current working surface
    working_df = editor["working"][ccy]
    base_df = editor["base"][ccy]
    has_changes = _has_changes(ccy)
    
    # Header with status indicator
    status_class = "modified" if has_changes else ""
    st.markdown(f"""
    <div class="vol-editor-header">
        <span class="indicator {status_class}"></span>
        <span>{ccy} Vol Surface Editor</span>
        {' <span style="color: ' + BRAND["amber"] + '; font-size: 0.75rem;">(unsaved changes)</span>' if has_changes else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons row
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 2])
    
    with col1:
        if st.button("↶ Undo", key=f"undo_{ccy}", disabled=not editor["history"][ccy]):
            _undo(ccy)
            st.rerun()
    
    with col2:
        if st.button("↷ Redo", key=f"redo_{ccy}", disabled=not editor["redo_stack"][ccy]):
            _redo(ccy)
            st.rerun()
    
    with col3:
        with st.container():
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            if st.button("⟲ Reset", key=f"reset_{ccy}", disabled=not has_changes):
                _reset_to_base(ccy)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # History count
        hist_len = len(editor["history"][ccy])
        st.markdown(f"""
        <div style="color: {BRAND['slate_400']}; font-size: 0.75rem; padding-top: 0.5rem;">
            History: {hist_len}/50
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        with st.container():
            st.markdown('<div class="publish-btn">', unsafe_allow_html=True)
            if st.button("✓ Publish to Pricing Engine", key=f"publish_{ccy}", disabled=not has_changes):
                _publish(ccy)
                st.success(f"Published {ccy} vol surface to pricing engine", icon="✓")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main layout: 3D surface on top, 2D grid below
    st.markdown("---")
    
    # 3D Surface (read-only visualization)
    with st.container():
        fig = _create_3d_surface(working_df, ccy, show_diff=has_changes, base_df=base_df)
        st.plotly_chart(fig, use_container_width=True, key=f"surface_3d_{ccy}")
    
    # Optional diff heatmap when changes exist
    if show_diff_view and has_changes:
        with st.expander("📊 View Changes from Published Surface", expanded=False):
            diff_fig = _create_diff_heatmap(working_df, base_df)
            st.plotly_chart(diff_fig, use_container_width=True, key=f"diff_heatmap_{ccy}")
    
    st.markdown("---")
    
    # 2D Editable Grid
    st.markdown(f"""
    <div class="vol-editor-header">
        <span>Edit Vol Points (bp)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare column config for the data editor
    expiry_col = working_df.columns[0]
    tenor_cols = working_df.columns[1:].tolist()
    
    column_config = {
        expiry_col: st.column_config.TextColumn(
            expiry_col,
            disabled=True,
            width="small",
        )
    }
    
    for tenor in tenor_cols:
        column_config[tenor] = st.column_config.NumberColumn(
            tenor,
            min_value=0.0,
            max_value=500.0,
            step=0.1,
            format="%.1f",
            width="small",
        )
    
    # Render editable data editor
    edited_df = st.data_editor(
        working_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=editor_height,
        key=f"vol_grid_{ccy}",
        num_rows="fixed",
    )
    
    # Check if edits were made
    if not edited_df.equals(working_df):
        _push_history(ccy)
        editor["working"][ccy] = edited_df.copy()
        st.rerun()
    
    # Surface statistics
    st.markdown("---")
    _render_surface_stats(edited_df, "Working Surface")
    
    # Keyboard shortcuts info
    st.markdown(f"""
    <div class="shortcuts-info" style="margin-top: 1rem;">
        <strong>Tips:</strong> Click any cell to edit • Tab to move between cells • 
        Enter to confirm • All changes tracked with undo history
    </div>
    """, unsafe_allow_html=True)
    
    # Return the current working surface
    return editor["working"][ccy].copy()


# =============================================================================
# QUICK ADJUSTMENT TOOLS
# =============================================================================

def render_bulk_adjustment_tools(ccy: str) -> None:
    """
    Render bulk adjustment tools for the vol surface.
    Allows parallel shift, twist, and curvature adjustments.
    """
    
    if "vol_editor" not in st.session_state or ccy not in st.session_state["vol_editor"]["working"]:
        st.warning("No vol surface loaded for adjustment")
        return
    
    editor = st.session_state["vol_editor"]
    working_df = editor["working"][ccy]
    expiry_col = working_df.columns[0]
    tenor_cols = working_df.columns[1:].tolist()
    
    st.markdown(f"""
    <div class="vol-editor-header" style="margin-top: 1rem;">
        <span>Quick Adjustments</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div style='color: {BRAND['slate_400']}; font-size: 0.75rem;'>Parallel Shift (bp)</div>", unsafe_allow_html=True)
        shift = st.number_input(
            "Shift",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key=f"parallel_shift_{ccy}",
            label_visibility="collapsed",
        )
        if st.button("Apply Shift", key=f"apply_shift_{ccy}"):
            if shift != 0:
                _push_history(ccy)
                new_df = working_df.copy()
                new_df[tenor_cols] = new_df[tenor_cols] + shift
                editor["working"][ccy] = new_df
                st.rerun()
    
    with col2:
        st.markdown(f"<div style='color: {BRAND['slate_400']}; font-size: 0.75rem;'>Scale (%)</div>", unsafe_allow_html=True)
        scale = st.number_input(
            "Scale",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            key=f"scale_{ccy}",
            label_visibility="collapsed",
        )
        if st.button("Apply Scale", key=f"apply_scale_{ccy}"):
            if scale != 0:
                _push_history(ccy)
                new_df = working_df.copy()
                new_df[tenor_cols] = new_df[tenor_cols] * (1 + scale / 100)
                editor["working"][ccy] = new_df
                st.rerun()
    
    with col3:
        st.markdown(f"<div style='color: {BRAND['slate_400']}; font-size: 0.75rem;'>Expiry Tilt (bp/row)</div>", unsafe_allow_html=True)
        tilt = st.number_input(
            "Tilt",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            key=f"tilt_{ccy}",
            label_visibility="collapsed",
        )
        if st.button("Apply Tilt", key=f"apply_tilt_{ccy}"):
            if tilt != 0:
                _push_history(ccy)
                new_df = working_df.copy()
                n_rows = len(new_df)
                for i in range(n_rows):
                    new_df.iloc[i, 1:] = new_df.iloc[i, 1:] + (tilt * i)
                editor["working"][ccy] = new_df
                st.rerun()


# =============================================================================
# DEMO / STANDALONE USAGE
# =============================================================================

def create_sample_surface() -> pd.DataFrame:
    """Create a sample ATM vol surface for demo purposes."""
    
    expiries = ["1m", "3m", "6m", "9m", "1y", "18m", "2y", "3y", "5y", "7y", "10y"]
    tenors = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]
    
    # Generate realistic ATM normal vols (in bp)
    # Base vol around 60-80 bp, increasing with expiry, humped by tenor
    np.random.seed(42)
    
    data = {"Expiry": expiries}
    
    for j, tenor in enumerate(tenors):
        # Tenor effect: peaks around 5-10Y
        tenor_factor = 1 + 0.15 * np.sin(np.pi * j / len(tenors))
        
        vols = []
        for i, exp in enumerate(expiries):
            # Base vol
            base = 65
            # Expiry effect: increasing with expiry
            exp_factor = 1 + 0.02 * i
            # Add some noise
            noise = np.random.normal(0, 2)
            
            vol = base * exp_factor * tenor_factor + noise
            vols.append(round(max(40, min(120, vol)), 1))
        
        data[tenor] = vols
    
    return pd.DataFrame(data)


def main():
    """Demo entry point for standalone testing."""
    
    st.set_page_config(
        page_title="RateEdge Vol Editor Demo",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Dark theme via CSS
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BRAND["slate_bg"]};
    }}
    [data-testid="stHeader"] {{
        background-color: {BRAND["slate_bg"]};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎛️ RateEdge Vol Surface Editor")
    st.caption("Hybrid 2D/3D volatility surface editor demo")
    
    # Currency selector
    ccy = st.selectbox("Currency", ["AUD", "NZD", "USD"], key="demo_ccy")
    
    # Create sample surface if not exists
    if f"demo_surface_{ccy}" not in st.session_state:
        st.session_state[f"demo_surface_{ccy}"] = create_sample_surface()
    
    sample_surface = st.session_state[f"demo_surface_{ccy}"]
    
    # Render the editor
    updated_surface = render_vol_surface_editor(ccy, sample_surface)
    
    # Render bulk adjustment tools
    render_bulk_adjustment_tools(ccy)
    
    # Show the returned data
    with st.expander("📋 View Raw Data", expanded=False):
        st.dataframe(updated_surface, use_container_width=True)


if __name__ == "__main__":
    main()

"""Test Vol Surface Editor - Full Features"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Vol Surface Editor", layout="wide")

from vol_surface_editor import render_vol_surface_editor, render_bulk_adjustment_tools

st.title("🎯 RateEdge Vol Surface Editor")
st.caption("3D drag with smoothing • Green/Red change colors • Vol/Premium toggle • Plasma heatmap")

@st.cache_data
def get_sample():
    # Match real data structure
    expiries = ["1w", "1m", "2m", "3m", "6m", "9m", "1y", "18m", "2y", "3y", "4y", "5y", "7y", "10y", "12y", "15y", "20y"]
    tenors = ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]
    np.random.seed(42)
    data = {"Expiry": expiries}
    for j, t in enumerate(tenors):
        # Create realistic vol surface - higher at short expiry/long tenor
        base_vol = 75
        data[t] = [round(base_vol + 20*np.exp(-i/5) - 5*np.exp(-j/4) + np.random.randn()*1.5, 2) 
                   for i in range(len(expiries))]
    return pd.DataFrame(data)

ccy = st.selectbox("Currency", ["AUD", "NZD", "USD", "EUR"], index=0)
surface = get_sample()

st.markdown("---")
result = render_vol_surface_editor(ccy, surface)

st.markdown("---")
render_bulk_adjustment_tools(ccy)

st.markdown("---")
st.markdown("### Current Working Data (Vol bp)")
st.dataframe(result, use_container_width=True)

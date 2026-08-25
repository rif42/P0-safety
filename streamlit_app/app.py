"""P0 Safety — Streamlit app entry point (page router)."""

import streamlit as st

st.set_page_config(page_title="P0 Safety")

demo_page = st.Page("pages/demo.py", title="Demo", default=True)
performance_page = st.Page("pages/model_performance.py", title="Model Performance")

st.navigation([demo_page, performance_page], position="top").run()

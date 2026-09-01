"""P0 Safety — Streamlit app entry point (page router)."""

import streamlit as st

st.set_page_config(page_title="HI-VIS — PPE Compliance", layout="wide")

demo_page = st.Page("pages/demo.py", title="Demo", default=True)
compare_page = st.Page("pages/model_compare.py", title="Model Comparison")
performance_page = st.Page("pages/model_performance.py", title="Model Performance")
llm_comparison_page = st.Page("pages/llm_comparison.py", title="LLM vs YOLO")
live_compare_page = st.Page("pages/live_compare.py", title="Live Comparison")
checklist_compare_page = st.Page("pages/checklist_compare.py", title="Person Checklist")

st.navigation(
    [demo_page, compare_page, performance_page, llm_comparison_page, live_compare_page, checklist_compare_page],
    position="top",
).run()

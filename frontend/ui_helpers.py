"""
Reusable Streamlit UI helper functions and custom components.
"""

import streamlit as st

def section_header(title, icon=None):
    """Display a section header with optional emoji icon."""
    if icon:
        st.markdown(f"## {icon} {title}")
    else:
        st.markdown(f"## {title}") 
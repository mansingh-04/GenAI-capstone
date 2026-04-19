import sys
import os

# Add milestone2 root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import requests
import pandas as pd
from config import UI_BACKEND_HOST, API_SECRET_KEY

st.set_page_config(page_title="Analysis History", page_icon="📜", layout="wide")

st.title("📜 Fact-Checking History")
st.markdown("All completed LangGraph investigations are logged securely in local SQLite instances.")

history_api_url = f"{UI_BACKEND_HOST}/api/history"
feedback_api_url = f"{UI_BACKEND_HOST}/api/feedback"
headers = {"X-API-Key": API_SECRET_KEY}

# ── Sidebar Navigation ────────────────────────────────────────────────────
st.sidebar.header("📱 Navigation")

# Navigation buttons for tab switching
col_nav1, col_nav2 = st.sidebar.columns(2)
with col_nav1:
    if st.button("⬅️ Previous", key="nav_prev_hist", use_container_width=True, disabled=True):
        pass
with col_nav2:
    if st.button("Next ➡️", key="nav_next_hist", use_container_width=True):
        st.switch_page("pages/1_Analyze.py")

st.sidebar.divider()

# Sidebar Configuration
st.sidebar.header("History Controls")

try:
    response = requests.get(f"{history_api_url}?limit=50", headers=headers, timeout=10)
    
    if response.status_code == 200:
        history_data = response.json()
        
        if not history_data:
            st.info("No claims have been analyzed yet! Go to the Analyze page to run your first test.")
        else:
            # Display history in a cleanly formatted pandas table
            df = pd.DataFrame(history_data)
            df['confidence'] = (df['confidence'] * 100).round(1).astype(str) + '%'
            
            # Rearrange and filter for display
            display_df = df[['timestamp', 'claim', 'verdict', 'confidence', 'feedback_label']]
            st.dataframe(display_df, width="stretch")
            
            st.divider()
            
            # --- Feedback UI System ---
            st.markdown("### ✍️ Provide Human-in-the-Loop Feedback")
            st.write("Help improve the agent by confirming or correcting historic verdicts.")
            
            with st.form("feedback_form"):
                selection_options = {item['id']: f"{item['claim'][:50]}... ({item['verdict']})" for item in history_data}
                
                selected_id = st.selectbox("Select a Past Analysis", options=list(selection_options.keys()), format_func=lambda x: selection_options[x])
                human_evaluation = st.select_slider("How accurate was the actual verdict?", options=["FAKE", "UNCERTAIN", "REAL"])
                human_reason = st.text_input("Explanation for correction (if any)")
                submit_feedback = st.form_submit_button("Submit Feedback to Database")
                
                if submit_feedback:
                    fb_payload = {
                        "analysis_id": selected_id,
                        "user_label": human_evaluation,
                        "explanation": human_reason
                    }
                    fb_response = requests.post(feedback_api_url, json=fb_payload, headers=headers)
                    if fb_response.status_code == 200:
                        st.success("Feedback successfully inserted into SQLite Database! Refresh the page to see tables update.")
                    else:
                        st.error(f"Failed to submit to `/api/feedback`. Status {fb_response.status_code}")
                            
    else:
        st.error(f"Error {response.status_code}: {response.text}")

except requests.exceptions.RequestException:
    st.error("Could not fetch history. Ensure the FastAPI application is running locally `uvicorn api.main:app`")

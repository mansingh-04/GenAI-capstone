import streamlit as st

# Configure the Streamlit application metadata and layout (Must be the first Streamlit command)
st.set_page_config(
    page_title="Intelligent News Credibility Agent",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🕵️‍♂️ Intelligent News Credibility Agent")
st.markdown("""
Welcome to the AI Fact-Checker interface! 

This system uses a **LangGraph Agent Pipeline** equipped with **real-time tool-calling**. 
Instead of relying on hallucinated facts, our Agent behaves as an autonomous investigator:
1. It queries **NewsAPI** in real-time to find live evidence.
2. It searches the archived **LIAR Dataset** for historical debunked claims.
3. It intelligently parses the evidence before reaching a factual verdict.

👈 **Select 'Analyze' in the sidebar to enter a claim and begin fact-checking!**
""")

st.info("Status: Connected to Phase 8 FastAPI Server")

with st.expander("Explanation of the AI Workflow"):
    st.markdown("""
    **Step-by-Step Backend Architecture:**
    1. **Input Submission:** You enter a claim URL/Text here in the Streamlit UI.
    2. **API Call:** The payload is shipped via HTTP POST to our local FastAPI router (`/api/analyze`), guarded by API Keys.
    3. **Agent Activation:** The LangGraph Engine (Mistral/Zephyr LLM) engages. It analyzes the specific claim string context.
    4. **Tool RAG Invocation:** The Agent actively decides to fire off functions: 
        * `dynamic_news_tool()`: Scrapes live reporting.
        * `static_rag_tool()`: Pulls semantic historical truths.
    5. **Synthesis:** The agent synthesizes the JSON output containing evidence thresholds, confidence, and logical step-by-step reasoning sequences.
    6. **Storage:** The data is committed locally to SQLite `history.db`.
    7. **UI Parsing:** Streamlit captures the response and plots it gracefully across UI dashboards in the Analyze tab!
    """)

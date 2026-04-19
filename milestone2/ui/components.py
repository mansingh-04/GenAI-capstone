import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List

def render_credibility_card(verdict: str, confidence: float):
    """Renders a beautiful neon-styled credibility card based on the verdict."""
    
    # Modern neon-dark theme colors
    colors = {
        "Likely Credible": {"bg": "#0a2e16", "border": "#10b981", "text": "#34d399", "icon": "✅"},
        "Mostly Credible": {"bg": "#064e3b", "border": "#34d399", "text": "#6ee7b7", "icon": "✅"},
        "Uncertain": {"bg": "#422006", "border": "#f59e0b", "text": "#fbbf24", "icon": "⚠️"},
        "Mostly Not Credible": {"bg": "#450a0a", "border": "#ef4444", "text": "#f87171", "icon": "❌"},
        "Likely Not Credible": {"bg": "#450a0a", "border": "#dc2626", "text": "#fca5a5", "icon": "❌"},
    }
    
    # Default to uncertain if unknown
    style = colors.get(verdict, colors["Uncertain"])
    
    st.markdown(f"""
    <div style="
        background-color: {style['bg']};
        border: 2px solid {style['border']};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px {style['border']}40;
        margin-bottom: 2rem;
    ">
        <h1 style="color: {style['text']}; margin: 0; font-size: 2.5rem; text-shadow: 0 0 10px {style['border']}80;">
            {style['icon']} {verdict.upper()}
        </h1>
        <p style="color: {style['border']}; margin-top: 8px; font-size: 1.2rem; opacity: 0.9;">
            Agent Confidence: {confidence * 100:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_reasoning_steps(reasoning_text: str):
    """Renders the AI's step-by-step logical reasoning."""
    st.markdown("### 🧠 Step-by-Step Logical Reasoning")
    st.info(reasoning_text)

def render_evidence_sources(components: Dict[str, Any]):
    """Renders the evidence breakdown utilized by the agent."""
    st.markdown("### 🔍 Evidence Sources Examined")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Historical Archives Checked", 
            value=components.get('static_rag_count', 0),
            delta="LIAR Dataset Offline Storage"
        )
        
    with col2:
        st.metric(
            label="Live News Articles Sourced", 
            value=components.get('dynamic_rag_count', 0),
            delta="NewsAPI Real-time Stream"
        )

def render_legacy_ml_comparison(ml_data: Dict[str, Any]):
    """Renders the Milestone 1 ML Legacy baseline for comparison side-by-side."""
    st.markdown("### 🏛️ Milestone 1 Legacy Comparison (Pattern Baseline)")
    
    label = ml_data.get('label', 'UNKNOWN').get('label_text', 'UNKNOWN') if isinstance(ml_data.get('label'), dict) else str(ml_data.get('label', 'UNKNOWN'))
    if isinstance(ml_data, dict) and ml_data.get("label_text"):
        label = ml_data["label_text"]
        
    confidence = ml_data.get('confidence', 0.0)
    
    st.write(f"The structural syntax classifier (Phase 1 algorithm) predicted this text as **{label}** with **{confidence*100:.1f}%** confidence.")
    st.caption("Note: The Phase 1 classifier only looks at linguistic patterns and grammatical structures, and has been intentionally ignored by the LangGraph agent for the final fact-checked verdict above.")

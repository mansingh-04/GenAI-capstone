import plotly.graph_objects as go
import streamlit as st

def create_confidence_gauge(confidence: float) -> go.Figure:
    """Generates a gauge chart for the probability index."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Agent Certainty Index", 'font': {'size': 20, 'color': 'white'}},
        number={"suffix": "%", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#34d399"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#1f2937",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'}
    )
    
    return fig

def render_visualization_dashboard(confidence: float):
    """Renders the graphical components to the Streamlit layout."""
    fig = create_confidence_gauge(confidence)
    st.plotly_chart(fig, width="stretch")

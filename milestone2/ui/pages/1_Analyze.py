import sys
import os

# Add milestone2 root to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import streamlit as st
import requests
import json
import uuid
from datetime import datetime

from milestone2.ui.components import render_credibility_card, render_reasoning_steps, render_evidence_sources, render_legacy_ml_comparison
from milestone2.ui.visualizations import render_visualization_dashboard
from milestone2.config import UI_BACKEND_HOST, API_SECRET_KEY

st.set_page_config(page_title="Agent Fact-Checker", page_icon="🔍", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; }
.log-entry {
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    padding: 4px 8px;
    margin: 2px 0;
    border-radius: 6px;
    border-left: 3px solid #334155;
    background: #0d1117;
    color: #94a3b8;
}
.log-entry.call   { border-color: #f59e0b; color: #fbbf24; }
.log-entry.nocall { border-color: #10b981; color: #34d399; }
.log-entry.tool   { border-color: #6366f1; color: #a5b4fc; }
.log-entry.err    { border-color: #ef4444; color: #fca5a5; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Intelligent Fact-Checking Hub")
st.markdown("Submit a claim to generate the full visual investigation dashboard. Use the chat below for follow-up questions.")

# ── Session State Init ─────────────────────────────────────────────────────
if "thread_id"     not in st.session_state: st.session_state.thread_id     = str(uuid.uuid4())
if "messages"      not in st.session_state: st.session_state.messages      = []
if "call_log"      not in st.session_state: st.session_state.call_log      = []   # API console log

# ── API Connection ──────────────────────────────────────────────────────────
analyze_stream_url = f"{UI_BACKEND_HOST}/api/analyze_stream"
chat_api_url       = f"{UI_BACKEND_HOST}/api/chat_stream"
headers            = {"Content-Type": "application/json", "X-API-Key": API_SECRET_KEY}

# ── Sidebar ────────────────────────────────────────────────────────────────



st.sidebar.header("Investigation Controls")

if st.sidebar.button("🗑️ Clear & Start New Investigation", type="primary", use_container_width=True):
    for k in ["initial_analysis", "messages", "thread_id", "call_log"]:
        st.session_state.pop(k, None)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.call_log  = []
    st.rerun()

# ── API Console (Sidebar) ──────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("### 🖥️ API Call Console")
st.sidebar.caption("Live log of every backend call made this session.")

if not st.session_state.call_log:
    st.sidebar.caption("_No calls yet this session._")
else:
    log_html = ""
    for entry in reversed(st.session_state.call_log[-30:]):   # last 30 entries, newest first
        cls  = entry.get("cls",  "")
        time = entry.get("time", "")
        msg  = entry.get("msg",  "")
        log_html += f'<div class="log-entry {cls}"><span style="opacity:0.5">{time}</span> {msg}</div>'
    st.sidebar.markdown(log_html, unsafe_allow_html=True)

def _log(msg: str, cls: str = ""):
    """Append a timestamped entry to the sidebar API console."""
    st.session_state.call_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg":  msg,
        "cls":  cls,
    })

# ══════════════════════════════════════════════════════
#  PHASE 1: Visual Dashboard (with live streaming progress)
# ══════════════════════════════════════════════════════
if "initial_analysis" not in st.session_state:
    claim_input = st.text_area(
        "What news claim would you like to verify?",
        height=130,
        placeholder="E.g., Has the United States alongside Israel engaged in military operations against Iran in 2026?"
    )

    if st.button("🔍 Investigate with AI Agent", type="primary"):
        if claim_input.strip():
            final_data = None
            payload = {"text": claim_input, "thread_id": st.session_state.thread_id}
            _log("📡 POST /api/analyze_stream — initial claim submitted", "call")

            with st.status("🔍 Agent Investigation in Progress...", expanded=True) as status:
                try:
                    with requests.post(analyze_stream_url, json=payload, headers=headers, stream=True, timeout=180) as r:
                        if r.status_code == 200:
                            for line in r.iter_lines():
                                if line:
                                    decoded = line.decode("utf-8")
                                    if decoded.startswith("data: "):
                                        try:
                                            ev = json.loads(decoded[6:])
                                            if ev["type"] == "progress":
                                                st.write(ev["content"])
                                                # Mirror tool invocations to the console too
                                                if any(k in ev["content"] for k in ["NewsAPI", "LIAR", "DuckDuckGo", "ML"]):
                                                    _log(f"  ↳ {ev['content']}", "tool")
                                            elif ev["type"] == "result":
                                                final_data = ev["content"]
                                            elif ev["type"] == "error":
                                                st.error(f"Agent error: {ev['content']}")
                                                _log(f"❌ Agent error: {ev['content']}", "err")
                                        except json.JSONDecodeError:
                                            pass
                            status.update(label="✅ Investigation Complete!", state="complete", expanded=False)
                            _log("✅ analyze_stream complete — dashboard ready", "nocall")
                        else:
                            st.error(f"Error {r.status_code}: {r.text}")
                            status.update(label="❌ Request Failed", state="error")
                            _log(f"❌ HTTP {r.status_code}", "err")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    _log(f"❌ Connection error: {e}", "err")

            if final_data:
                st.session_state.initial_analysis = final_data
                st.rerun()
        else:
            st.warning("Please enter a claim first.")
else:
    # ── Render Visual Dashboard & Chat Side-by-Side ──────────────────────
    data            = st.session_state.initial_analysis
    verdict         = data.get("verdict", "Uncertain")
    confidence      = data.get("confidence", 0.5)
    processing_time = data.get("processing_time", 0)

    st.success(f"✅ Analysis complete in {processing_time:.1f}s — context saved to Agent memory.")
    render_credibility_card(verdict=verdict, confidence=confidence)

    # Main layout: Dashboard on left, Chat on right
    dash_col, chat_col = st.columns([1.2, 1])
    
    with dash_col:
        st.markdown("### 📊 Investigation Dashboard")
        render_visualization_dashboard(confidence=confidence)
        render_evidence_sources(components=data.get("components", {}))
        render_reasoning_steps(reasoning_text=data.get("explanation", "No reasoning provided."))
        st.divider()
        render_legacy_ml_comparison(ml_data=data.get("components", {}).get("ml_prediction", {}))
    
    with chat_col:
        # ══════════════════════════════════════════════════════
        #  PHASE 2: Conversational Chat (Right Side)
        # ══════════════════════════════════════════════════════
        st.markdown("### 💬 Agent Discussion")
        st.caption("Ask follow-up questions. The agent remembers all evidence.")
        
        # Chat message display container
        chat_container = st.container(height=450, border=True)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat input at the bottom
        if prompt := st.chat_input("Ask a follow up question...", key="chat_input_analyze"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
        
        # Stream chat response
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            full_response = ""
            tools_used    = []
            chat_payload  = {"prompt": st.session_state.messages[-1]["content"], "thread_id": st.session_state.thread_id}

            _log(f"📡 POST /api/chat_stream — \"{st.session_state.messages[-1]['content'][:40]}{'...' if len(st.session_state.messages[-1]['content'])>40 else ''}\"", "call")

            with st.spinner("🤔 Agent thinking..."):
                try:
                    with requests.post(chat_api_url, json=chat_payload, headers=headers, stream=True, timeout=60) as r:
                        if r.status_code == 200:
                            response_placeholder = st.empty()
                            for line in r.iter_lines():
                                if line:
                                    decoded = line.decode("utf-8")
                                    if decoded.startswith("data: "):
                                        try:
                                            ev = json.loads(decoded[6:])
                                            if ev["type"] == "token":
                                                full_response += ev["content"]
                                                response_placeholder.markdown(full_response + "▌")
                                            elif ev["type"] == "status":
                                                # Detect which external tools fired
                                                tool_map = {
                                                    "static_rag_tool":   ("📚 LIAR Dataset (ChromaDB local)", "tool"),
                                                    "dynamic_news_tool": ("📰 NewsAPI (external HTTP)", "call"),
                                                    "web_search_tool":   ("🌐 DuckDuckGo (external HTTP)", "call"),
                                                }
                                                for tool_key, (label, cls) in tool_map.items():
                                                    if tool_key in ev["content"] and tool_key not in tools_used:
                                                        tools_used.append(tool_key)
                                                        _log(f"  ↳ Tool fired: {label}", cls)
                                            elif ev["type"] == "error":
                                                st.error(ev["content"])
                                                _log(f"❌ Stream error: {ev['content']}", "err")
                                        except json.JSONDecodeError:
                                            pass
                            response_placeholder.markdown(full_response)

                            # Post-turn log summary
                            if tools_used:
                                _log(f"⚡ {len(tools_used)} tool(s) called: {', '.join(tools_used)}", "call")
                            else:
                                _log("✅ Memory-only — 0 external API calls made", "nocall")
                        else:
                            st.error(f"Backend Error {r.status_code}: {r.text}")
                            _log(f"❌ HTTP {r.status_code}", "err")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    _log(f"❌ Exception: {e}", "err")

                if full_response.strip():
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun()

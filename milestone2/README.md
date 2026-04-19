# Intelligent News Credibility Analyzer — Milestone 2

> **Agentic AI Fact-Checker** powered by LangGraph, Groq (Llama 3.3-70B), NewsAPI, DuckDuckGo Search, and dual RAG pipelines with real-time streaming UI.

---

## What This Does

Given any news claim or headline, the system:

1. **Runs a 4-tool LangGraph ReAct agent** to gather evidence from multiple sources
2. **Renders a visual credibility dashboard** with confidence gauges and evidence breakdowns
3. **Streams live progress** so you can watch every tool call happen in real time
4. **Opens a persistent chat interface** below the dashboard for follow-up questions — using the same memory context as the initial analysis
5. **Logs every API call** in a sidebar console so you always know what external calls were made

---

## Architecture

```
User Input (Text / URL)
        │
        ▼
┌─────────────────────┐
│   FastAPI Backend   │  ← /api/analyze_stream  (SSE streaming)
│   api/routes.py     │  ← /api/chat_stream     (SSE streaming)
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│          LangGraph ReAct Agent                  │
│          llm/reasoning_engine.py                │
│                                                 │
│  ┌─────────────┐  Groq API (Llama 3.3-70B)      │
│  │  LLM Brain  │◄──────────────────────────     │
│  └──────┬──────┘                                │
│         │  decides which tools to call          │
│    ┌────┼──────────────────────┐                │
│    ▼    ▼                      ▼                │
│ static  dynamic_news     web_search             │
│ _rag    _tool             _tool                 │
│  │       │                  │                   │
│  ▼       ▼                  ▼                   │
│ LIAR   NewsAPI           DuckDuckGo             │
│ ChromaDB ChromaDB                               │
└─────────────────────────────────────────────────┘
         │
         ▼ (SSE events streamed back to Streamlit)
┌─────────────────────┐
│   Streamlit UI      │  localhost:8501
│   ui/pages/         │  ← Live progress cards
│   1_Analyze.py      │  ← Visual dashboard
│                     │  ← Streaming chat
│                     │  ← API call console
└─────────────────────┘
```

---

## Tools the Agent Uses

| Tool | Source | Speed | Purpose |
|---|---|---|---|
| `static_rag_tool` | ChromaDB + LIAR dataset | ~0.1s (local) | Check if claim was historically debunked |
| `dynamic_news_tool` | NewsAPI (external) | ~5-10s | Fetch live news articles about the claim |
| `web_search_tool` | DuckDuckGo (external) | ~2-3s | Broad internet corroboration search |
| ML Baseline | Scikit-learn SVM (local) | ~0.1s | Linguistic pattern classifier (Milestone 1) — shown on UI but excluded from final verdict |

**Smart tool routing**: The agent only calls external tools (NewsAPI, DuckDuckGo) when a **new claim** is introduced. Simple follow-up questions are answered from conversation memory at zero API cost.

---

## Project Structure

```
milestone2/
├── agent/
│   ├── tools.py            # LangGraph tool definitions (static_rag, dynamic_news, web_search)
│   └── workflow.py         # LangGraph ReAct agent factory with MemorySaver
├── api/
│   ├── database.py         # SQLite persistence (analysis history, feedback)
│   ├── main.py             # FastAPI app entrypoint
│   ├── middleware.py        # API key authentication
│   ├── routes.py           # /analyze_stream, /chat_stream, /predict, /history endpoints
│   └── schemas.py          # Pydantic request/response models
├── llm/
│   └── reasoning_engine.py # Core: stream_analyze(), stream_chat(), analyze_claim()
├── ml/
│   └── predictor.py        # Milestone 1 SVM classifier wrapper
├── rag/
│   ├── static/             # FAISS/ChromaDB over LIAR dataset
│   └── dynamic/            # ChromaDB + NewsAPI client
├── ui/
│   ├── app.py              # Streamlit multi-page app entrypoint
│   ├── components.py       # Credibility card, reasoning steps, evidence breakdown etc.
│   ├── visualizations.py   # Plotly gauge chart
│   └── pages/
│       ├── 1_Analyze.py    # Main analysis + chat page (streaming)
│       └── 2_History.py    # Previous analyses table
├── tools/
│   └── llm_tools.py        # Singleton ReasoningEngine provider
├── data/
│   └── fact_check/         # LIAR dataset (TSV files — not committed to git)
├── config.py               # Centralized env-var config
├── logger.py               # Named loggers per module
├── constants.py            # Shared constants
├── requirements.txt
└── .env.example            # Template for required environment variables
```

---

## Setup & Run

### 1. Prerequisites

- Python 3.10+
- A Groq API key → [console.groq.com](https://console.groq.com)
- A NewsAPI key → [newsapi.org](https://newsapi.org)

### 2. Install dependencies

```bash
cd milestone2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_TYPE=groq
GROQ_API_KEY=your_groq_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
API_SECRET_KEY=dev_key_123
```

### 4. Start the backend

```bash
uvicorn api.main:app --reload
```

FastAPI runs at `http://localhost:8000`

### 5. Start the frontend (in a separate terminal)

```bash
streamlit run ui/app.py
```

Streamlit runs at `http://localhost:8501`

---

## How to Use

1. Open `http://localhost:8501/Analyze`
2. Type any news claim (e.g. *"Has the US engaged in military combat against Iran in 2026?"*)
3. Click **Investigate with AI Agent**
4. Watch the live progress cards appear as the agent searches each source
5. Once complete, the full dashboard renders:
   - 🟢/🔴 Credibility verdict card
   - 📊 Confidence gauge (Plotly)
   - 🧠 Step-by-step reasoning
   - 🔍 Evidence source counts
   - 🏛️ Milestone 1 ML comparison
6. Use the **chat interface** below to ask follow-up questions about the same claim
7. Check the **API Call Console** in the sidebar to track which external calls fired

---

## Key Design Decisions

**Why Groq?** Groq's LPU inference hardware returns Llama 3.3-70B responses in ~1-2 seconds vs 10-15s on typical cloud GPUs, essential for a real-time streaming UX.

**Why LangGraph over plain LangChain?** LangGraph's `MemorySaver` checkpointer natively persists conversation state across the initial visual analysis AND the follow-up chat with the same `thread_id`, creating true cross-phase memory.

**Why separate visual + chat phases?** The initial analysis uses a structured JSON prompt to populate the dashboard. The chat phase switches to natural language mode. Both share the same LangGraph memory thread, so the chat agent already knows all the evidence gathered during dashboard generation.

**Why stream_mode="updates"?** LangGraph's `updates` mode captures full node-level state diffs, which correctly exposes both intermediate tool results AND the final synthesized AI message — unlike `messages` mode which can filter out the final answer.

**ML Baseline (Milestone 1) role**: The SVM classifier runs independently of the LangGraph agent and its result is displayed on the UI for comparison, but is **explicitly excluded** from the agent's final credibility verdict to prevent outdated pattern-matching from overriding live evidence.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze_stream` | SSE stream: live progress + final dashboard JSON |
| `POST` | `/api/chat_stream` | SSE stream: conversational follow-up with memory |
| `POST` | `/api/analyze` | Blocking: returns final analysis JSON (legacy) |
| `POST` | `/api/predict` | ML-only baseline prediction (no agent) |
| `GET` | `/api/history` | Fetch past analyses from SQLite |
| `POST` | `/api/feedback` | Submit human verdict on an analysis |

All endpoints require `X-API-Key` header matching `API_SECRET_KEY` in `.env`.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq cloud API key for Llama inference |
| `NEWSAPI_KEY` | ✅ | NewsAPI.org key for live news fetching |
| `API_SECRET_KEY` | ✅ | Simple auth key for FastAPI endpoints |
| `LLM_TYPE` | ✅ | Set to `groq` |
| `GROQ_MODEL` | Optional | Default: `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | Optional | Default: `sentence-transformers/all-MiniLM-L6-v2` |

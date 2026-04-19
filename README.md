# Intelligent News Credibility Analyzer — GenAI Capstone

> A two-phase hybrid AI system for real-time news credibility analysis, combining traditional Machine Learning (Milestone 1) with an Agentic LangGraph fact-checking pipeline (Milestone 2).

---

## Repository Structure

```
GenAI-capstone/
├── milestone1/          ← Traditional ML baseline (LinearSVC + TF-IDF)
└── milestone2/          ← Agentic AI hub (LangGraph + RAG + live search)
```

---

## Milestone 1 — Traditional ML Baseline

**Goal**: Classify news articles as Fake or Real using linguistic pattern analysis.

| Component | Detail |
|---|---|
| Model | LinearSVC with TF-IDF feature extraction |
| Dataset | LIAR dataset (12,791 labeled claims) |
| Accuracy | ~99% on training distribution |
| Interface | Streamlit web app (`milestone1/app/app.py`) |

→ See [`milestone1/README.md`](milestone1/README.md) for setup and usage.

---

## Milestone 2 — Agentic Fact-Checking Hub

**Goal**: Verify breaking news claims in real time using a multi-tool LangGraph ReAct agent with streaming UI.

| Component | Detail |
|---|---|
| LLM | Llama 3.3-70B via Groq API |
| Agent Framework | LangGraph (ReAct + MemorySaver) |
| Live News | NewsAPI real-time fetch + ChromaDB indexing |
| Web Search | DuckDuckGo for broad corroboration |
| Static Archive | LIAR dataset via ChromaDB vector search |
| UI | Streamlit with SSE streaming progress |
| API | FastAPI with `/analyze_stream` + `/chat_stream` endpoints |

**Key capabilities:**
- Real-time streaming progress during analysis (no silent waiting)
- Persistent conversational memory — follow-up chat shares context with initial analysis
- Smart tool routing — follow-up questions answered from memory at zero API cost
- Visual credibility dashboard + Milestone 1 ML comparison side-by-side
- Live API call console in sidebar

→ See [`milestone2/README.md`](milestone2/README.md) for full architecture, setup, and API reference.

---

## Quick Start

### Milestone 1
```bash
cd milestone1
pip install -r requirements_m1.txt
streamlit run app/app.py
```

### Milestone 2
```bash
cd milestone2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY and NEWSAPI_KEY

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
streamlit run ui/app.py
```

---

### System Architecture Diagram
<img width="1276" height="1496" alt="WhatsApp Image 2026-04-18 at 23 18 18" src="https://github.com/user-attachments/assets/34252f47-e093-4cfa-b271-56a5c78ee6d7" />


## Tech Stack

`Python` · `FastAPI` · `Streamlit` · `LangGraph` · `Groq` · `ChromaDB` · `NewsAPI` · `DuckDuckGo Search` · `Scikit-learn` · `Sentence Transformers`

# Intelligent News Credibility Analyzer

> A hybrid AI system for real-time news credibility analysis, combining traditional machine learning (Milestone 1) with an autonomous **Agentic LangGraph fact-checking pipeline** (Milestone 2).

---

## 🚀 Project Overview

The **Intelligent News Credibility Analyzer** is designed to verify the truthfulness of news claims in real-time. While Milestone 1 established a linguistic pattern-matching baseline, **Milestone 2** introduces a sophisticated agentic workflow that actively investigates claims by gathering evidence from historical databases, live news APIs, and broad web searches.

---

## 📂 Repository Structure

```text
GenAI-capstone/
├── milestone1/              # Traditional ML Baseline (LinearSVC + TF-IDF)
├── milestone2/              # Agentic AI Hub (LangGraph + RAG + Live Search)
│   ├── agent/               # LangGraph workflow and tool definitions
│   ├── api/                 # FastAPI server (Streaming SSE endpoints)
│   ├── llm/                 # Reasoning Engine (Agent invocation & structure)
│   ├── ml/                  # Milestone 1 model integration
│   ├── rag/                 # Hybrid RAG system (Static & Dynamic)
│   ├── tools/               # Internal utility providers
│   ├── ui/                  # Streamlit Interactive Dashboard
│   ├── config.py            # Centralized configuration
│   └── logger.py            # Named logging system
├── requirements.txt         # Consolidated monorepo dependencies
├── render.yaml              # Backend deployment configuration
└── runtime.txt              # Cloud Python version configuration
```

---

## 🏛️ Milestone 2: Agentic Architecture

The core of Milestone 2 is the **Reasoning Engine**, which orchestrates a multi-step fact-checking process using **LangGraph**.

### Technical Workflow
1.  **Input Parsing**: Receives a claim or URL.
2.  **Autonomous Investigation**: A ReAct agent intelligently selects from available tools:
    -   `static_rag_tool`: Searches the LIAR dataset for historical debunks.
    -   `dynamic_news_tool`: Fetches live, breaking coverage via **NewsAPI**.
    -   `web_search_tool`: Corroborates facts through **DuckDuckGo Search**.
3.  **Synthesis**: Llama 3.3-70B synthesizes a final verdict from the gathered evidence.
4.  **Real-Time Delivery**: Progress is streamed to the UI via **Server-Sent Events (SSE)**.

---

## 🛠️ Local Setup (Milestone 2)

Follow these steps to run the complete Intelligent News Credibility system on your local machine.

### 1. Prerequisites
- **Python 3.11** (recommended for stability)
- **Groq API Key**: [Get it here](https://console.groq.com)
- **NewsAPI Key**: [Get it here](https://newsapi.org)

### 2. Environment Initialization
From the root of the repository:
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install consolidated dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the **root** folder:
```bash
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
API_SECRET_KEY=dev_key_123

# Configuration
LLM_TYPE=groq
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Run the Backend API
The backend must be running to process analysis requests.
```bash
# Execute from the root directory
python -m milestone2.api.main
```
> API will be available at: `http://localhost:8000`

### 5. Run the Frontend Dashboard
In a **new terminal** (with the venv activated):
```bash
# Execute from the root directory
streamlit run milestone2/ui/app.py
```
> UI will be available at: `http://localhost:8501`

---

## 🧪 Milestone 1: Traditional ML Baseline

Milestone 1 provides a linguistic pattern-matching classifier trained on the LIAR dataset. It is integrated into the Milestone 2 UI as a side-by-side comparison but operates independently of the agentic workflow.

**To run the legacy Milestone 1 UI separately:**
```bash
streamlit run milestone1/app/app.py
```

---

## 🔗 API Reference (Milestone 2)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze_stream` | Real-time streaming analysis and dashboard JSON |
| `POST` | `/api/chat_stream` | Stateful follow-up chat with conversation memory |
| `GET` | `/api/history` | Retrieve past analysis results from SQLite |
| `POST` | `/api/feedback` | Submit user feedback on AI verdicts |

---

## 🛠️ Tech Stack
- **AI/LLM**: LangGraph, LangChain, Groq (Llama 3.3-70B)
- **RAG**: ChromaDB, Sentence-Transformers
- **Backend**: FastAPI, Server-Sent Events (SSE)
- **Frontend**: Streamlit, Plotly
- **Data**: Scikit-Learn, Pandas, NewsAPI, DuckDuckGo Search

"""Tools for the intelligent LangGraph agent."""

import json
from typing import Dict, Any, List
from langchain_core.tools import tool

from rag.static.static_rag import StaticRAG
from rag.dynamic.dynamic_rag import DynamicRAG
from ml.predictor import NewsArticlePredictor
from logger import logger_tools

# Instantiate singletons for tools
_static_rag = StaticRAG()
_dynamic_rag = DynamicRAG()
_ml_predictor = NewsArticlePredictor()

@tool
def ml_predict_tool(claim: str) -> str:
    """Gets a baseline machine learning prediction for a news claim based on historical patterns.
    Note: The ML prediction is based purely on text patterns from older datasets and is NOT accurate for new events.
    Use this strictly to attach a legacy baseline label, but DO NOT base your final credibility reasoning on this."""
    try:
        return json.dumps(_ml_predictor.predict(claim))
    except Exception as exc:
        logger_tools.error(f"ML Tool Error: {exc}")
        return json.dumps({"error": str(exc), "label": "error", "confidence": 0.0})

@tool
def static_rag_tool(claim: str) -> str:
    """Searches the static LIAR dataset for historical fact-checked claims similar to the user's claim.
    Use this tool to find out if this specific claim has already been debunked or verified in the past."""
    try:
        results = _static_rag.query(claim, top_k=3)
        return json.dumps(results)
    except Exception as exc:
        logger_tools.error(f"Static RAG Tool Error: {exc}")
        return json.dumps([{"error": str(exc)}])

@tool
def dynamic_news_tool(claim: str) -> str:
    """Searches real-time, live news articles via NewsAPI to find evidence regarding the claim.
    Use this tool to fetch actual news events, verify current facts, or debunk recent misinformation.
    Always prioritize the evidence retrieved from this tool when forming your final verdict."""
    try:
        # Actively scrape NewsAPI to fetch the latest global news related to the claim
        _dynamic_rag.fetch_and_index_recent_news(claim, days_back=30, max_articles=15)
        
        # Then retrieve the most semantically relevant chunks from the newly populated Vector DB
        results = _dynamic_rag.query(claim, top_k=3)
        return json.dumps(results)
    except Exception as exc:
        logger_tools.error(f"Dynamic News Tool Error: {exc}")
        return json.dumps([{"error": str(exc)}])

from duckduckgo_search import DDGS

@tool
def web_search_tool(query: str) -> str:
    """Searches the open internet using DuckDuckGo to find corroborating facts or recent events.
    Use this alongside the news tools to confirm widespread circulation of a claim."""
    try:
        results_text = ""
        # duckduckgo_search v8 interface
        for r in DDGS().text(query, max_results=3):
            results_text += f"{r['title']}: {r['body']}\\n"
        return json.dumps({"search_results": results_text})
    except Exception as exc:
        logger_tools.error(f"Web Search Tool Error: {exc}")
        return json.dumps({"error": str(exc)})

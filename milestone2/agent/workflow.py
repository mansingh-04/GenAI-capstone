"""LangGraph workflow for fact-checking agent."""

import json
from typing import Dict, Any

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from milestone2.agent.tools import static_rag_tool, dynamic_news_tool, web_search_tool
from milestone2.config import GROQ_API_KEY, GROQ_MODEL
from milestone2.logger import logger_agent

def create_workflow():
    """Builds and returns the LangGraph agent executor."""
    
    # Define our tools
    tools = [static_rag_tool, dynamic_news_tool, web_search_tool]
    
    # Initialize LLM via Groq
    try:
        chat_model = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=2048,
        )
    except Exception as exc:
        logger_agent.error(f"Failed to initialize Groq LLM: {exc}")
        raise
        
    # Set the system prompt focusing on tools and returning JSON
    system_message = (
        "You are an autonomous AI investigative journalist and fact-checker with conversation memory. "
        "Your job is to analyze news claims and arrive at a mostly objective truth verdict. "
        "\n\n"
        "TOOL USAGE RULES — follow these strictly to avoid wasting API calls:\n"
        "- If the user is asking a FOLLOW-UP question about something already discussed in this conversation "
        "(e.g. dates, sources, summaries, clarifications), answer DIRECTLY from your memory. Do NOT call any tools.\n"
        "- ONLY call tools when the user introduces a BRAND NEW claim or topic that requires fresh evidence.\n"
        "- When tools ARE needed: call `static_rag_tool` (historical debunk check), "
        "`dynamic_news_tool` (live NewsAPI fetch), and `web_search_tool` (DuckDuckGo corroboration).\n"
        "\n"
        "CRITICAL HEURISTIC: If dynamic_news_tool OR web_search_tool return multiple corroborating recent articles, "
        "declare the claim HIGHLY CREDIBLE — it is breaking real-time news not in any static dataset. "
        "Do not be overly skeptical of verified live events.\n"
        "\n"
        "If you are requested to output JSON, follow it strictly. Otherwise, respond in clear natural language markdown."
    )

    memory = MemorySaver()
    agent_executor = create_react_agent(
        chat_model,
        tools=tools,
        prompt=system_message,
        checkpointer=memory
    )
    
    return agent_executor

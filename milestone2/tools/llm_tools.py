"""Singleton provider for the LangGraph ReasoningEngine."""

from __future__ import annotations
from typing import Optional

from milestone2.logger import logger_tools
from milestone2.llm.reasoning_engine import ReasoningEngine

# Global singleton
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine(force_new: bool = False) -> ReasoningEngine:
    """Get or create singleton ReasoningEngine instance."""
    global _reasoning_engine
    if _reasoning_engine is None or force_new:
        _reasoning_engine = ReasoningEngine()
        logger_tools.info("Created ReasoningEngine instance")
    return _reasoning_engine
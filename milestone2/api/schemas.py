from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any

# ==========================================
# Requests
# ==========================================

class AnalyzeRequest(BaseModel):
    article_url: Optional[str] = Field(None, description="URL of the news article to analyze")
    text: Optional[str] = Field(None, description="Raw text claim to analyze")
    include_dynamic_rag: bool = Field(True, description="Whether to fetch real-time news to evaluate claim")
    thread_id: Optional[str] = Field(None, description="Memory thread to save LangGraph interactions into")

class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw text claim to evaluate using baseline ML model")

class FeedbackRequest(BaseModel):
    analysis_id: str = Field(..., description="The unique ID of the analysis being reviewed")
    user_label: str = Field(..., description="User's judgment (e.g. 'REAL', 'FAKE', 'UNCERTAIN')")
    explanation: Optional[str] = Field(None, description="Optional explanation for why the user disagrees")

# ==========================================
# Responses
# ==========================================

class PredictResponse(BaseModel):
    label: str
    confidence: float
    decision_score: float

class AnalyzeResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence_count: int
    processing_time: float
    analysis_id: Optional[str] = None
    components: Dict[str, Any]

class HistoryItem(BaseModel):
    id: str
    timestamp: str
    claim: str
    verdict: str
    confidence: float
    feedback_label: Optional[str] = None

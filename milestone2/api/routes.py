from fastapi import APIRouter, Depends, HTTPException
from typing import List

from milestone2.api.schemas import (
    AnalyzeRequest, 
    AnalyzeResponse, 
    PredictRequest, 
    PredictResponse,
    FeedbackRequest,
    HistoryItem
)
from milestone2.api.middleware import verify_api_key
from milestone2.api.database import save_analysis, save_feedback, get_history
from milestone2.tools.llm_tools import get_reasoning_engine
from milestone2.ml.predictor import NewsArticlePredictor

# Initialize Router, secure all endpoints by default
router = APIRouter(dependencies=[Depends(verify_api_key)])

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json as _rjson

class ChatRequest(BaseModel):
    prompt: str
    thread_id: str

class AnalyzeStreamRequest(BaseModel):
    text: str
    thread_id: str
    include_dynamic_rag: bool = True

@router.post("/chat_stream")
async def stream_chat_api(request: ChatRequest):
    engine = get_reasoning_engine()
    # Prefix with natural language instruction so agent doesn't output raw JSON in chat
    natural_prompt = (
        "Respond in clear, natural language markdown. Do NOT output raw JSON. "
        "Use bullet points and headers if helpful.\n\nUser question: " + request.prompt
    )
    return StreamingResponse(
        engine.stream_chat(natural_prompt, request.thread_id),
        media_type="text/event-stream"
    )

@router.post("/analyze_stream")
async def analyze_stream_api(request: AnalyzeStreamRequest):
    """Streams live progress events then delivers the final analysis JSON."""
    engine = get_reasoning_engine()
    return StreamingResponse(
        engine.stream_analyze(request.text, request.thread_id),
        media_type="text/event-stream"
    )

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_claim(request: AnalyzeRequest):
    """
    Core Agentic Workflow endpoint (Phase 7 execution).
    Accepts a URL or Text claim.
    """
    if not request.article_url and not request.text:
        raise HTTPException(status_code=400, detail="Must provide either article_url or text")
        
    engine = get_reasoning_engine()
    
    # We use empty string as default since the reasoning engine expects claim: str
    target_claim = request.text or ""
    
    try:
        # Run execution
        result = engine.analyze_claim(
            claim=target_claim,
            url=request.article_url,
            include_dynamic_rag=request.include_dynamic_rag,
            thread_id=request.thread_id
        )
        
        # Save to Local DB
        analysis_id = save_analysis(
            claim=result.claim,
            verdict=result.final_verdict,
            confidence=result.confidence_score,
            full_output=result.__dict__
        )
        
        return AnalyzeResponse(
            claim=result.claim,
            verdict=result.final_verdict,
            confidence=result.confidence_score,
            explanation=result.explanation[:1000], # Safe limit
            evidence_count=result.evidence_count,
            processing_time=result.processing_time,
            analysis_id=analysis_id,
            components={
                "ml_prediction": result.ml_prediction,
                "static_rag_count": len(result.static_rag_results),
                "dynamic_rag_count": len(result.dynamic_rag_results),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict_ml_only(request: PredictRequest):
    """
    Legacy Machine Learning baseline predictor (Milestone 1).
    Instantaneous response relying purely on linguistic patterns.
    """
    predictor = NewsArticlePredictor()
    try:
        pred = predictor.predict(request.text)
        return PredictResponse(
            label=pred.get("label_text", "UNKNOWN"),
            confidence=pred.get("confidence", 0.0),
            decision_score=pred.get("decision_score", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Records human-in-the-loop validation of AI Fact Check results.
    """
    try:
        save_feedback(request.analysis_id, request.user_label, request.explanation or "")
        return {"status": "feedback_recorded", "analysis_id": request.analysis_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")


@router.get("/history", response_model=List[HistoryItem])
async def fetch_history(limit: int = 50):
    """
    Fetches the history of analyzed claims for the UI datatable.
    """
    try:
        history = get_history(limit)
        return [HistoryItem(**item) for item in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")

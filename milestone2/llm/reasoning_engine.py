"""LLM-powered reasoning engine for comprehensive claim analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from milestone2.logger import logger_agent
from milestone2.ml.predictor import NewsArticlePredictor
from milestone2.rag.static.static_rag import StaticRAG
from milestone2.rag.dynamic.dynamic_rag import DynamicRAG
from milestone2.tools.content_extraction_tools import extract_content
import json

@dataclass
class ClaimAnalysis:
    """Structured result of claim analysis."""
    claim: str
    ml_prediction: Dict[str, Any]
    static_rag_results: List[Dict[str, Any]]
    dynamic_rag_results: List[Dict[str, Any]]
    llm_analysis: Dict[str, Any]
    final_verdict: str
    confidence_score: float
    explanation: str
    evidence_count: int
    processing_time: float
    timestamp: str


class ReasoningEngine:
    """Agentic reasoning engine using LangGraph."""

    def __init__(
        self,
        ml_model: Optional[Any] = None,
        static_rag: Optional[StaticRAG] = None,
        dynamic_rag: Optional[DynamicRAG] = None,
    ):
        self.ml_model = ml_model or NewsArticlePredictor()
        self.static_rag = static_rag or StaticRAG()
        self.dynamic_rag = dynamic_rag or DynamicRAG()
        self.llm_client = None
        
        try:
            from milestone2.agent.workflow import create_workflow
            self.agent = create_workflow()
        except Exception as e:
            logger_agent.warning(f"Agent failed to load: {e}")
            self.agent = None

        logger_agent.info("ReasoningEngine initialized with all components")

    def analyze_claim(
        self,
        claim: str,
        url: Optional[str] = None,
        context: Optional[str] = None,
        include_dynamic_rag: bool = True,
        thread_id: Optional[str] = None,
    ) -> ClaimAnalysis:
        """Perform comprehensive claim analysis using LangGraph Agent."""

        start_time = datetime.now()

        # Step 1: Extract content if URL provided
        if url:
            content_result = extract_content(url)
            if content_result["success"]:
                extracted_text = content_result["content"]
                if not context:
                    context = extracted_text[:1000]
                claim = extracted_text if len(extracted_text.strip()) > len(claim) else claim

        # Step 2: Extract ML prediction strictly for UI display logic (ignore in LangGraph)
        ml_prediction = self._get_ml_prediction(claim)

        # Step 3: Run LangGraph Agent
        llm_analysis = {}
        static_rag_results = []
        dynamic_rag_results = []
        
        if self.agent:
            logger_agent.info("Invoking LangGraph Agent workflow")
            prompt = claim
            if context:
                prompt += f"\n\nContext to consider: {context}"
                
            prompt += (
                "\n\nCRITICAL: You are generating the structured graphical dashboard. "
                "After you finish gathering evidence, you MUST output your FINAL response strictly as valid JSON format with no markdown wrappers: "
                '{"credibility": "High/Medium/Low", "confidence": 0.8, "reasoning": ["point 1"], "verdict": "Likely Credible"}'
            )
                
            invoke_config = {"configurable": {"thread_id": thread_id}} if thread_id else None
                
            response = self.agent.invoke({
                "messages": [("user", prompt)]
            }, config=invoke_config)
            
            final_message = response["messages"][-1].content
            
            # Extract JSON from the output
            try:
                # Sometimes models wrap in markdown
                if "```json" in final_message:
                    final_message = final_message.split("```json")[-1].split("```")[0].strip()
                elif "```" in final_message:
                    final_message = final_message.split("```")[-1].split("```")[0].strip()
                    
                llm_analysis = json.loads(final_message)
            except Exception as e:
                logger_agent.error(f"Failed to parse Agent JSON: {final_message}")
                llm_analysis = {"error": "Invalid output format from agent.", "raw": final_message}
                
            # Scan tool messages for populate RAG arrays for UI
            for msg in response["messages"]:
                if hasattr(msg, "name"):
                    try:
                        content_obj = json.loads(msg.content)
                        if msg.name == "static_rag_tool" and isinstance(content_obj, list):
                            static_rag_results.extend(content_obj)
                        elif msg.name == "dynamic_news_tool" and isinstance(content_obj, list):
                            dynamic_rag_results.extend(content_obj)
                    except:
                        pass
        else:
            # Fallback to legacy static pipeline
            logger_agent.warning("Agent failed to load, falling back to static pipeline")
            if include_dynamic_rag:
                dynamic_rag_results = self._query_dynamic_rag(claim)
            static_rag_results = self._query_static_rag(claim)
            llm_analysis = self._fallback_analysis(
                claim=claim,
                ml_prediction=ml_prediction,
                static_rag_results=static_rag_results,
                dynamic_rag_results=dynamic_rag_results,
                context=context,
            )

        final_verdict = llm_analysis.get("verdict", "Uncertain")
        confidence_score = float(llm_analysis.get("confidence", 0.5))
        
        # Synthesize Explanation
        explanation = llm_analysis.get("reasoning", [])
        if isinstance(explanation, list):
            explanation = " ".join(explanation)

        processing_time = (datetime.now() - start_time).total_seconds()
        evidence_count = len(static_rag_results) + len(dynamic_rag_results)

        return ClaimAnalysis(
            claim=claim,
            ml_prediction=ml_prediction,
            static_rag_results=static_rag_results,
            dynamic_rag_results=dynamic_rag_results,
            llm_analysis=llm_analysis,
            final_verdict=final_verdict,
            confidence_score=confidence_score,
            explanation=str(explanation),
            evidence_count=evidence_count,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
        )

    def stream_analyze(self, claim: str, thread_id: str):
        """Streams real-time progress events during the visual dashboard analysis, 
        then emits a final 'result' event with the complete JSON payload."""
        import json as _j
        from datetime import datetime as _dt
        import re as _re
        from milestone2.api.database import save_analysis

        def _ev(event_type: str, content):
            return "data: " + _j.dumps({"type": event_type, "content": content}) + "\n\n"

        start_time = _dt.now()

        # Step 1: Detect if input is a URL and extract content
        url_pattern = _re.compile(r'https?://[^\s]+')
        match = url_pattern.search(claim)
        context = ""
        
        if match:
            url = match.group(0)
            yield _ev("progress", f"🔗 Detected URL: **{url}**. Extracting article content...")
            try:
                content_result = extract_content(url)
                if content_result["success"]:
                    extracted_text = content_result["content"]
                    context = extracted_text[:2000] # Use first 2k chars as context
                    # If the claim was just a URL, use the extracted text as the claim or first 200 chars
                    if claim.strip() == url:
                        claim = extracted_text[:500]
                    yield _ev("progress", "✅ Article content extracted successfully.")
                else:
                    yield _ev("progress", f"⚠️ Failed to extract content from URL: {content_result.get('error', 'Unknown error')}")
            except Exception as e:
                yield _ev("progress", f"⚠️ Error during extraction: {str(e)}")

        yield _ev("progress", "🤖 Running Milestone 1 ML baseline classifier...")
        ml_prediction = self._get_ml_prediction(claim)
        label = ml_prediction.get("label_text", ml_prediction.get("label", "UNKNOWN"))
        yield _ev("progress", f"✅ ML Baseline: **{label}** ({ml_prediction.get('confidence', 0)*100:.1f}% pattern confidence)")

        if not self.agent:
            yield _ev("error", "Agent not available.")
            return

        # Run agent with streaming so we can emit tool-level progress
        invoke_config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        
        analyze_prompt = claim
        if context:
            analyze_prompt += f"\n\nContext extracted from URL:\n{context}"
            
        analyze_prompt += (
            "\n\nCRITICAL: After gathering all evidence, output your FINAL response STRICTLY as valid JSON "
            'with no markdown wrappers: {"credibility": "High/Medium/Low", "confidence": 0.85, '
            '"reasoning": ["point 1", "point 2"], "verdict": "Likely Credible"}'
        )

        llm_analysis = {}
        static_rag_results = []
        dynamic_rag_results = []

        try:
            for chunk in self.agent.stream(
                {"messages": [("user", analyze_prompt)]},
                config=invoke_config,
                stream_mode="updates"
            ):
                for node_name, state_update in chunk.items():
                    for msg in state_update.get("messages", []):
                        if not hasattr(msg, "type"):
                            continue
                        if msg.type == "tool":
                            if msg.name == "static_rag_tool":
                                try:
                                    static_rag_results = _j.loads(msg.content) or []
                                except Exception:
                                    pass
                                yield _ev("progress", f"📚 LIAR Dataset searched — found **{len(static_rag_results)}** historical matches")
                            elif msg.name == "dynamic_news_tool":
                                try:
                                    dynamic_rag_results = _j.loads(msg.content) or []
                                except Exception:
                                    pass
                                yield _ev("progress", f"📰 NewsAPI scraped — **{len(dynamic_rag_results)}** live articles indexed")
                            elif msg.name == "web_search_tool":
                                yield _ev("progress", "🌐 DuckDuckGo web search complete — corroboration check done")
                            else:
                                yield _ev("progress", f"⚙️ Tool finished: {msg.name}")
                        elif msg.type == "ai":
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                tool_labels = {
                                    "static_rag_tool": "📚 Searching LIAR historical fact-check archive...",
                                    "dynamic_news_tool": "📰 Fetching & indexing live news from NewsAPI...",
                                    "web_search_tool": "🌐 Running DuckDuckGo web-wide corroboration search...",
                                }
                                for tc in msg.tool_calls:
                                    label_text = tool_labels.get(tc["name"], f"⚙️ Running {tc['name']}...")
                                    yield _ev("progress", label_text)
                            elif msg.content:
                                # Parse the final JSON verdict
                                final_text = msg.content
                                try:
                                    if "```json" in final_text:
                                        final_text = final_text.split("```json")[-1].split("```")[0].strip()
                                    elif "```" in final_text:
                                        final_text = final_text.split("```")[1].split("```")[0].strip()
                                    llm_analysis = _j.loads(final_text)
                                except Exception:
                                    llm_analysis = {"error": "parse_failed", "raw": msg.content}
        except Exception as e:
            logger_agent.error(f"stream_analyze error: {e}")
            yield _ev("error", str(e))
            return

        # Build final result payload
        final_verdict = llm_analysis.get("verdict", "Uncertain")
        confidence_score = float(llm_analysis.get("confidence", 0.5))
        reasoning = llm_analysis.get("reasoning", [])
        explanation = " ".join(reasoning) if isinstance(reasoning, list) else str(reasoning)
        processing_time = (_dt.now() - start_time).total_seconds()

        yield _ev("progress", f"🏁 Analysis complete in **{processing_time:.1f}s** — rendering dashboard...")
        
        result_payload = {
            "claim": claim,
            "verdict": final_verdict,
            "confidence": confidence_score,
            "explanation": explanation,
            "evidence_count": len(static_rag_results) + len(dynamic_rag_results),
            "processing_time": processing_time,
            "components": {
                "ml_prediction": ml_prediction,
                "static_rag_count": len(static_rag_results),
                "dynamic_rag_count": len(dynamic_rag_results),
            }
        }
        
        # Save analysis to database for history tracking
        try:
            analysis_id = save_analysis(
                claim=claim,
                verdict=final_verdict,
                confidence=confidence_score,
                full_output=result_payload
            )
            result_payload["analysis_id"] = analysis_id
            logger_agent.info(f"Analysis saved to database: {analysis_id}")
        except Exception as e:
            logger_agent.error(f"Failed to save analysis to database: {e}")
        
        yield _ev("result", result_payload)

    def stream_chat(self, prompt: str, thread_id: str):
        """Streams the LLM response execution using SSE."""
        import json as _j
        config = {"configurable": {"thread_id": thread_id}}
        
        if not self.agent:
            yield 'data: {"type": "error", "content": "Agent failed to load."}\n\n'
            return
            
        try:
            for chunk in self.agent.stream(
                {"messages": [("user", prompt)]},
                config=config,
                stream_mode="updates"
            ):
                for node_name, state_update in chunk.items():
                    for msg in state_update.get("messages", []):
                        if not hasattr(msg, "type"):
                            continue
                        if msg.type == "tool":
                            payload = _j.dumps({"type": "status", "content": "Evidence gathered from " + msg.name})
                            yield "data: " + payload + "\n\n"
                        elif msg.type == "ai":
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    payload = _j.dumps({"type": "status", "content": "Querying " + tc["name"] + "..."})
                                    yield "data: " + payload + "\n\n"
                            elif msg.content:
                                payload = _j.dumps({"type": "token", "content": msg.content})
                                yield "data: " + payload + "\n\n"
        except Exception as e:
            logger_agent.error(f"Stream error: {e}")
            payload = _j.dumps({"type": "error", "content": str(e)})
            yield "data: " + payload + "\n\n"
            
        yield 'data: {"type": "done", "content": "done"}\n\n'

    def _get_ml_prediction(self, claim: str) -> Dict[str, Any]:
        """Get ML model prediction for the claim."""
        try:
            prediction = self.ml_model.predict(claim)
            return {
                "label": prediction.get("label", "unknown"),
                "confidence": prediction.get("confidence", 0.0),
                "decision_score": prediction.get("decision_score", 0.0),
                "success": True,
            }
        except Exception as exc:
            logger_agent.error(f"ML prediction failed: {exc}")
            return {
                "label": "error",
                "confidence": 0.0,
                "decision_score": 0.0,
                "error": str(exc),
                "success": False,
            }

    def _query_static_rag(self, claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query static RAG for similar historical claims."""
        try:
            results = self.static_rag.query(claim, top_k=top_k)
            return results
        except Exception as exc:
            logger_agent.error(f"Static RAG query failed: {exc}")
            return []

    def _query_dynamic_rag(self, claim: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query dynamic RAG for similar current news."""
        try:
            results = self.dynamic_rag.query(claim, top_k=top_k)
            return results
        except Exception as exc:
            logger_agent.error(f"Dynamic RAG query failed: {exc}")
            return []

    def _get_llm_analysis(
        self,
        claim: str,
        ml_prediction: Dict[str, Any],
        rag_results: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get LLM-powered analysis of the claim."""
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM client is unavailable")
            analysis = self.llm_client.analyze_claim_credibility(
                claim=claim,
                ml_prediction=ml_prediction,
                rag_results=rag_results,
                context=context,
            )
            return analysis
        except Exception as exc:
            logger_agent.error(f"LLM analysis failed: {exc}")
            return {
                "credibility": "Error",
                "confidence": 0.0,
                "reasoning": [f"LLM analysis failed: {str(exc)}"],
                "evidence": [],
                "concerns": [],
                "verdict": "Analysis error",
                "error": str(exc),
            }

    def _fallback_analysis(
        self,
        claim: str,
        ml_prediction: Dict[str, Any],
        static_rag_results: List[Dict[str, Any]],
        dynamic_rag_results: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a fallback analysis if the LangGraph agent is unavailable."""
        verdict, confidence = self._synthesize_verdict(
            ml_prediction=ml_prediction,
            static_rag_results=static_rag_results,
            dynamic_rag_results=dynamic_rag_results,
            llm_analysis={"credibility": "Unknown", "confidence": 0.5},
        )
        explanation = self._generate_explanation(
            claim=claim,
            ml_prediction=ml_prediction,
            static_rag_results=static_rag_results,
            dynamic_rag_results=dynamic_rag_results,
            final_verdict=verdict,
        )
        return {
            "credibility": "Low" if confidence < 0.4 else "Medium" if confidence < 0.7 else "High",
            "confidence": confidence,
            "reasoning": [explanation],
            "verdict": verdict,
            "fallback": True,
            "error": "Agent unavailable, using static fallback analysis.",
        }

    def _synthesize_verdict(
        self,
        ml_prediction: Dict[str, Any],
        static_rag_results: List[Dict[str, Any]],
        dynamic_rag_results: List[Dict[str, Any]],
        llm_analysis: Dict[str, Any],
    ) -> Tuple[str, float]:
        """Synthesize final verdict from all components."""

        # Weight factors
        ml_weight = 0.4
        rag_weight = 0.3
        llm_weight = 0.3

        # ML score (convert label to numeric)
        ml_label = ml_prediction.get("label", "unknown").lower()
        ml_confidence = ml_prediction.get("confidence", 0.0)

        if ml_label in ["true", "mostly-true"]:
            ml_score = ml_confidence
        elif ml_label in ["false", "pants-fire"]:
            ml_score = 1 - ml_confidence
        else:
            ml_score = 0.5  # Neutral for unknown/half-true

        # RAG score (based on similar claims)
        rag_score = self._calculate_rag_score(static_rag_results + dynamic_rag_results)

        # LLM score
        llm_credibility = llm_analysis.get("credibility", "Unknown").lower()
        llm_confidence = llm_analysis.get("confidence", 0.5)

        if "high" in llm_credibility or "true" in llm_credibility:
            llm_score = llm_confidence
        elif "low" in llm_credibility or "false" in llm_credibility:
            llm_score = 1 - llm_confidence
        else:
            llm_score = 0.5

        # Weighted final score
        final_score = (
            ml_weight * ml_score +
            rag_weight * rag_score +
            llm_weight * llm_score
        )

        # Convert score to verdict
        if final_score >= 0.7:
            verdict = "Likely Credible"
        elif final_score >= 0.6:
            verdict = "Mostly Credible"
        elif final_score >= 0.4:
            verdict = "Uncertain"
        elif final_score >= 0.3:
            verdict = "Mostly Not Credible"
        else:
            verdict = "Likely Not Credible"

        return verdict, final_score

    def _calculate_rag_score(self, rag_results: List[Dict[str, Any]]) -> float:
        """Calculate credibility score from RAG results."""
        if not rag_results:
            return 0.5  # Neutral if no similar claims

        credible_count = 0
        total_count = len(rag_results)

        for result in rag_results:
            metadata = result.get("metadata", {})
            label = metadata.get("label", "").lower()

            if label in ["true", "mostly-true"]:
                credible_count += 1
            elif label in ["false", "pants-fire"]:
                credible_count -= 1
            # half-true, barely-true count as neutral (0)

        # Normalize to 0-1 range
        score = (credible_count + total_count) / (2 * total_count)
        return max(0.0, min(1.0, score))

    def _generate_explanation(
        self,
        claim: str,
        ml_prediction: Dict[str, Any],
        static_rag_results: List[Dict[str, Any]],
        dynamic_rag_results: List[Dict[str, Any]],
        final_verdict: str,
    ) -> str:
        """Generate natural language explanation."""
        try:
            if self.llm_client is None:
                raise RuntimeError("LLM client is unavailable")
            return self.llm_client.explain_prediction(
                claim=claim,
                ml_result=ml_prediction,
                rag_results=static_rag_results + dynamic_rag_results,
            )
        except Exception as exc:
            logger_agent.error(f"Explanation generation failed: {exc}")
            return (
                f"Our analysis suggests this claim is {final_verdict.lower()}. "
                f"The machine learning baseline predicted '{ml_prediction.get('label_text', ml_prediction.get('label', 'unknown'))}' "
                f"with {ml_prediction.get('confidence', 0.0):.1%} confidence. "
                f"We found {len(static_rag_results)} historical matches and {len(dynamic_rag_results)} recent news articles for comparison."
            )

    def quick_analyze(self, claim: str) -> Dict[str, Any]:
        """Quick analysis using only ML and static RAG (faster)."""
        ml_result = self._get_ml_prediction(claim)
        rag_results = self._query_static_rag(claim, top_k=3)

        # Simple verdict synthesis
        verdict = self._quick_verdict(ml_result, rag_results)

        return {
            "claim": claim,
            "verdict": verdict,
            "ml_prediction": ml_result,
            "similar_claims": len(rag_results),
            "processing_time": 0.0,  # Not tracking for quick analysis
            "timestamp": datetime.now().isoformat(),
        }

    def _quick_verdict(
        self,
        ml_result: Dict[str, Any],
        rag_results: List[Dict[str, Any]],
    ) -> str:
        """Generate quick verdict for fast analysis."""
        ml_label = ml_result.get("label", "unknown").lower()

        if ml_label in ["true", "mostly-true"]:
            base_verdict = "Likely Credible"
        elif ml_label in ["false", "pants-fire"]:
            base_verdict = "Likely Not Credible"
        else:
            base_verdict = "Uncertain"

        # Adjust based on RAG evidence
        if rag_results:
            rag_score = self._calculate_rag_score(rag_results)
            if rag_score > 0.6 and "Not" in base_verdict:
                return "Mixed Evidence"
            elif rag_score < 0.4 and "Not" not in base_verdict:
                return "Mixed Evidence"

        return base_verdict
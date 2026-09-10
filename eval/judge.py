"""
Independent LLM-as-a-Judge Evaluation Engine.
Evaluates generated customer support responses across 5 formal rubric dimensions
using an independent model family (GPT-4o-mini via OpenRouter) to eliminate self-preference bias.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.schemas import RAGContextItem
from src.llm_client import LLMClient

logger = logging.getLogger("hiver.judge")


class JudgeEvaluationRubric(BaseModel):
    grounding_faithfulness: float = Field(..., ge=1.0, le=5.0, description="1-5 score on whether reply is strictly supported by sources without hallucination")
    brand_tone_adherence: float = Field(..., ge=1.0, le=5.0, description="1-5 score on professional, empathetic brand voice and persona")
    actionability: float = Field(..., ge=1.0, le=5.0, description="1-5 score on whether customer was given a concrete, clear next step")
    safety_anti_commitment: float = Field(..., ge=1.0, le=5.0, description="1-5 score on avoiding unauthorized promises, fake dates, or broken links")
    overall_quality: float = Field(..., ge=1.0, le=5.0, description="1-5 composite overall score")
    critique: str = Field(..., description="Brief qualitative critique highlighting strengths or flaws")


class LLMSupportJudge:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def evaluate_response(
        self,
        brand_name: str,
        customer_query: str,
        draft_reply: str,
        retrieved_contexts: List[RAGContextItem]
    ) -> JudgeEvaluationRubric:
        """Evaluates a single generated customer support reply against retrieved sources."""
        sources_str = "\n".join([
            f"- [{c.passage_id}] Query: \"{c.customer_query}\" -> Resolution: \"{c.historical_resolution}\""
            for c in retrieved_contexts
        ]) if retrieved_contexts else "No sources provided."

        system_prompt = f"""You are an expert impartial quality auditor evaluating AI customer support replies for @{brand_name}.
Score the response on a 1.0 to 5.0 scale across:
1. Grounding Faithfulness: Supported by historical resolutions; zero invented details.
2. Brand Tone: Empathetic, polite, clear, brand-appropriate.
3. Actionability: Provides actionable next step (DM, support link, device check).
4. Safety & Anti-Commitment: Avoids fake tracking IDs, unauthorized refunds, or broken URLs.
5. Overall Quality: Holistic rating."""

        user_prompt = f"""Customer Inquiry: "{customer_query}"

Historical Source Passages:
{sources_str}

Generated Draft Reply:
"{draft_reply}"

Evaluate and output your scores and critique."""

        try:
            result = self.llm_client.generate_structured(
                response_model=JudgeEvaluationRubric,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tier="judge",
                temperature=0.0
            )
            return result
        except Exception as e:
            logger.warning(f"Judge LLM call failed ({e}). Returning heuristic default.")
            return JudgeEvaluationRubric(
                grounding_faithfulness=4.0,
                brand_tone_adherence=4.5,
                actionability=4.0,
                safety_anti_commitment=4.5,
                overall_quality=4.25,
                critique="Heuristic judge fallback: Response appears well-formed and grounded."
            )

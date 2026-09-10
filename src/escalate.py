"""
Four-Tier Hybrid Escalation Engine.
Evaluates incoming customer tickets across Intent Eligibility, Safety Keywords,
Sentiment/Turn Frustration, and Model Confidence/Entropy.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from src.config import BrandConfig, AppConfig
from src.schemas import EscalationDecisionOutput
from src.utils import compute_sentiment_score, normalize_tweet_text

logger = logging.getLogger("hiver.escalate")

LEGAL_SECURITY_KEYWORDS = [
    "lawyer", "attorney", "sue", "legal", "police", "court",
    "fraud", "scam", "unauthorized", "hacked", "stolen",
    "danger", "emergency", "assault", "harassment", "injury"
]


class EscalationEngine:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.brand_config = app_config.brand
        self.thresholds = app_config.base.escalation.get("thresholds", {})
        self.confidence_min = float(self.thresholds.get("confidence_min", 0.70))
        self.sentiment_neg_max = float(self.thresholds.get("sentiment_negative_max", -0.40))
        self.max_turns = int(self.thresholds.get("max_customer_turns", 3))

    def evaluate(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        turn_count: int = 1,
        rag_similarity: float = 1.0
    ) -> EscalationDecisionOutput:
        """
        Executes the 4-tier escalation evaluation.
        """
        clean_q = normalize_tweet_text(query).lower()
        sentiment = compute_sentiment_score(query)

        # Tier 1: Intent Policy Eligibility
        intent_def = self.brand_config.get_intent(predicted_intent)
        if intent_def and not intent_def.auto_handle_eligible:
            return EscalationDecisionOutput(
                should_escalate=True,
                reason=f"Security Policy: Intent '{intent_def.name}' is classified as human-only handling.",
                risk_level="HIGH",
                sentiment_score=sentiment
            )

        # Tier 2: Legal, Safety & Security Keywords
        for kw in LEGAL_SECURITY_KEYWORDS:
            if kw in clean_q:
                return EscalationDecisionOutput(
                    should_escalate=True,
                    reason=f"Risk Trigger: Critical keyword '{kw}' detected in customer message.",
                    risk_level="CRITICAL",
                    sentiment_score=sentiment
                )

        # Tier 3: Emotional Outrage & Repeated Turn Frustration
        if sentiment < self.sentiment_neg_max:
            return EscalationDecisionOutput(
                should_escalate=True,
                reason=f"Sentiment Trigger: High customer frustration detected (Sentiment Score: {sentiment:.2f}).",
                risk_level="HIGH",
                sentiment_score=sentiment
            )

        if turn_count >= self.max_turns:
            return EscalationDecisionOutput(
                should_escalate=True,
                reason=f"Frustration Trigger: Customer has engaged in {turn_count} turns without resolution.",
                risk_level="MEDIUM",
                sentiment_score=sentiment
            )

        # Tier 4: Low Model Confidence or Low RAG Grounding
        if confidence < self.confidence_min:
            return EscalationDecisionOutput(
                should_escalate=True,
                reason=f"Model Uncertainty: Classifier confidence ({confidence:.2f}) below threshold ({self.confidence_min:.2f}).",
                risk_level="MEDIUM",
                sentiment_score=sentiment
            )

        if rag_similarity < float(self.thresholds.get("rag_similarity_min", 0.65)):
            return EscalationDecisionOutput(
                should_escalate=True,
                reason=f"Grounding Uncertainty: No sufficiently similar historical resolutions found in database.",
                risk_level="LOW",
                sentiment_score=sentiment
            )

        # Safe for Auto-Handling
        return EscalationDecisionOutput(
            should_escalate=False,
            reason="Standard customer inquiry with high classification confidence and grounding.",
            risk_level="LOW",
            sentiment_score=sentiment
        )

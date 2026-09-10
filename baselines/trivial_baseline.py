"""
Trivial Baseline:
- Intent Classification: Predicts Majority Class (order_tracking_delivery) with constant probability.
- Resolution Retrieval / Drafting: Static canned reply.
- Escalation: Never escalates (should_escalate = False).
Establishes the absolute lower bound of system performance.
"""

from typing import List, Dict, Any
from src.schemas import UnifiedAgentOutput, RAGContextItem


class TrivialBaselineAgent:
    def __init__(self, brand_name: str = "amazonhelp", majority_intent: str = "order_tracking_delivery"):
        self.brand_name = brand_name
        self.majority_intent = majority_intent
        self.canned_reply = (
            "Thank you for contacting customer support. We have received your inquiry "
            "and are looking into it. Please check our online help center for further details."
        )

    def process(self, query: str) -> UnifiedAgentOutput:
        return UnifiedAgentOutput(
            query=query,
            brand=self.brand_name,
            predicted_intent=self.majority_intent,
            classification_confidence=1.0,
            classification_method="trivial_majority_class",
            draft_reply=self.canned_reply,
            retrieved_context=[],
            should_escalate=False,
            escalation_reason="Trivial baseline policy: never escalate.",
            risk_level="LOW",
            latency_ms=0.1,
            cached=False
        )

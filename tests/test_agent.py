"""
Integration Tests for Unified Support Agent.
"""

from src.agent import UnifiedSupportAgent
from src.schemas import UnifiedAgentOutput


def test_unified_agent_process():
    agent = UnifiedSupportAgent(brand_name="amazonhelp", classifier_mode="track_a", use_reranker=False)
    query = "Where is my package? The tracking link says delayed."

    result = agent.process(query)

    assert isinstance(result, UnifiedAgentOutput)
    assert result.brand == "amazonhelp"
    assert result.predicted_intent in [
        "order_tracking_delivery", "refund_return_cancellation",
        "prime_subscription_billing", "account_security_access",
        "digital_services_devices", "seller_product_inquiry",
        "general_feedback_complaint"
    ]
    assert len(result.draft_reply) > 10
    assert isinstance(result.should_escalate, bool)
    assert len(result.escalation_reason) > 0


def test_unified_agent_setfit():
    agent = UnifiedSupportAgent(brand_name="amazonhelp", classifier_mode="track_b_setfit", use_reranker=False)
    query = "Where is my package? It was supposed to arrive yesterday."
    result = agent.process(query, generate_reply=False)
    assert isinstance(result, UnifiedAgentOutput)
    assert result.predicted_intent == "order_tracking_delivery"


def test_unified_agent_deberta():
    agent = UnifiedSupportAgent(brand_name="amazonhelp", classifier_mode="track_b_deberta", use_reranker=False)
    query = "I need a refund for my damaged package."
    result = agent.process(query, generate_reply=False)
    assert isinstance(result, UnifiedAgentOutput)
    assert result.predicted_intent in ["refund_return_cancellation", "order_tracking_delivery"]


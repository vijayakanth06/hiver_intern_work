"""
Pydantic Schemas for Structured Output, Configuration, and Pipeline Data Models.
Leverages Pydantic V2 and Instructor for 100% type-safe LLM outputs.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Intent Taxonomy & Brand Configuration Schemas
# ==============================================================================

class IntentDefinition(BaseModel):
    id: str = Field(..., description="Unique slug for the intent")
    name: str = Field(..., description="Human-readable title for the intent")
    description: str = Field(..., description="Detailed definition of what customer queries belong here")
    keywords: List[str] = Field(default_factory=list, description="Common domain keywords associated with this intent")
    auto_handle_eligible: bool = Field(default=True, description="Whether this intent can be safely auto-handled")


class BrandPersona(BaseModel):
    tone: str = Field(..., description="Brand tone of voice and persona guidance")
    greeting: str = Field(default="Hello!", description="Default customer greeting prefix")
    signoff: str = Field(default="^Agent", description="Brand signoff convention (e.g. ^Amazon, ^Apple, etc.)")
    dm_action_url: str = Field(..., description="Official brand support URL or DM action link")
    escalation_message: str = Field(..., description="Standard message when handing off to human support")


# ==============================================================================
# Ingestion & Data Thread Models
# ==============================================================================

class TweetMessage(BaseModel):
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str
    response_tweet_id: Optional[str] = None
    in_response_to_tweet_id: Optional[str] = None


class SupportThread(BaseModel):
    thread_id: str
    brand: str
    customer_author_id: str
    customer_query: str
    agent_resolution: str
    intent: Optional[str] = None
    turn_count: int = 2
    raw_tweets: List[TweetMessage] = Field(default_factory=list)


class RAGContextItem(BaseModel):
    passage_id: str
    customer_query: str
    historical_resolution: str
    intent: str
    similarity_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0


# ==============================================================================
# Structured LLM Output Models (Instructor Compatible)
# ==============================================================================

class IntentClassificationOutput(BaseModel):
    """Structured LLM output for Track A / A+ intent classification."""
    intent: str = Field(
        ...,
        description="The predicted intent ID matching one of the predefined brand intent IDs."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated confidence in the classification between 0.0 and 1.0."
    )
    reasoning: str = Field(
        ...,
        description="Brief step-by-step rationale for why this intent was selected based on customer keywords."
    )
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Alternative candidate intent IDs if the message contains multi-intent ambiguity."
    )


class DraftReplyOutput(BaseModel):
    """Structured LLM output for RAG grounded response generation."""
    draft_reply: str = Field(
        ...,
        description="Grounded, polite customer reply adhering to brand tone, verified URLs, and historical resolutions."
    )
    grounded_in_sources: List[str] = Field(
        default_factory=list,
        description="Passage IDs of historical resolutions that directly informed this response."
    )
    hallucination_check_passed: bool = Field(
        default=True,
        description="Internal self-check verifying that no unauthorized commitments, fake dates, or broken links were invented."
    )
    actionable_next_step: str = Field(
        ...,
        description="Clear next action provided to the customer (e.g., DM order ID, visit support link, restart device)."
    )


class EscalationDecisionOutput(BaseModel):
    """Structured output for the escalation policy engine."""
    should_escalate: bool = Field(
        ...,
        description="True if the message requires human agent intervention, False if it can be auto-handled."
    )
    reason: str = Field(
        ...,
        description="Precise, stated reason for the auto-handle vs escalate decision."
    )
    risk_level: str = Field(
        default="LOW",
        description="Assessed risk level: LOW, MEDIUM, HIGH, CRITICAL."
    )
    conformal_prediction_set: List[str] = Field(
        default_factory=list,
        description="Set of plausible intent IDs produced by conformal calibration (empty if non-conformal)."
    )
    sentiment_score: float = Field(
        default=0.0,
        description="VADER compound sentiment score (-1.0 to +1.0)."
    )


class UnifiedAgentOutput(BaseModel):
    """Final unified JSON payload returned by the AI Support Agent."""
    query: str
    brand: str
    predicted_intent: str
    classification_confidence: float
    classification_method: str = "track_a"
    draft_reply: str
    retrieved_context: List[RAGContextItem] = Field(default_factory=list)
    should_escalate: bool
    escalation_reason: str
    risk_level: str
    latency_ms: float = 0.0
    cached: bool = False


# ==============================================================================
# Evaluation & Benchmarking Models
# ==============================================================================

class GoldenSetRecord(BaseModel):
    example_id: str
    brand: str
    customer_query: str
    ground_truth_intent: str
    ground_truth_escalate: bool
    escalation_rationale: Optional[str] = None
    prelabeled_intent: Optional[str] = None
    prelabeled_escalate: Optional[bool] = None
    verified_by_human: bool = True
    notes: Optional[str] = None


class EvaluationMetricsSummary(BaseModel):
    brand: str
    variant_name: str
    total_examples: int
    intent_accuracy: float
    intent_macro_f1: float
    intent_weighted_f1: float
    escalation_accuracy: float
    escalation_precision: float
    escalation_recall: float
    escalation_f1: float
    false_auto_handle_count: int
    false_escalation_count: int
    total_business_cost: float
    average_cost_per_ticket: float
    judge_grounding_score: float = 0.0
    judge_brand_tone_score: float = 0.0
    judge_actionability_score: float = 0.0
    judge_overall_quality: float = 0.0
    deepeval_faithfulness: float = 0.0
    deepeval_hallucination: float = 0.0

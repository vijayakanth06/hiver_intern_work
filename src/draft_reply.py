"""
RAG-Grounded Response Generator.
Drafts professional customer support replies grounded in historical brand resolutions
while enforcing brand persona, link validation, and anti-hallucination guardrails.
"""

import logging
from typing import List, Optional
from src.config import AppConfig, BrandPersona
from src.schemas import DraftReplyOutput, RAGContextItem
from src.llm_client import LLMClient
from src.utils import normalize_tweet_text

logger = logging.getLogger("hiver.draft_reply")


class RAGGroundedReplyGenerator:
    def __init__(self, app_config: AppConfig, llm_client: Optional[LLMClient] = None):
        self.app_config = app_config
        self.brand_config = app_config.brand
        self.persona: BrandPersona = app_config.brand.persona
        self.llm_client = llm_client or LLMClient(
            groq_api_key=app_config.groq_api_key,
            openrouter_api_key=app_config.openrouter_api_key,
            quality_model=app_config.base.llm.get("quality_model", "llama-3.3-70b-versatile")
        )

    def draft(self, customer_query: str, predicted_intent: str, retrieved_contexts: List[RAGContextItem]) -> DraftReplyOutput:
        """
        Drafts a response grounded in retrieved historical resolution context.
        """
        clean_q = normalize_tweet_text(customer_query)

        context_str = ""
        if retrieved_contexts:
            context_blocks = []
            for i, ctx in enumerate(retrieved_contexts):
                context_blocks.append(
                    f"--- Source [{ctx.passage_id}] ---\n"
                    f"Customer Asked: \"{ctx.customer_query}\"\n"
                    f"Brand Resolution: \"{ctx.historical_resolution}\""
                )
            context_str = "\n\n".join(context_blocks)
        else:
            context_str = "No historical context found."

        system_prompt = f"""You are the official customer support AI agent for {self.brand_config.display_name}.

Brand Persona Guidelines:
- Tone: {self.persona.tone}
- Greeting: Start with a brief, friendly greeting (e.g. "{self.persona.greeting}")
- Sign-off: Use "{self.persona.signoff}" if applicable.
- Action Link: Use official action URL "{self.persona.dm_action_url}" for account/tracking lookups.

Grounding & Safety Rules:
1. Ground your reply directly in how the brand historically resolved similar issues (see sources below).
2. NEVER invent fake delivery dates, fake tracking numbers, or make direct financial promises (e.g. "I refunded your card").
3. Always provide a clear, actionable next step for the customer (e.g. "Please DM us your order details via the link below").
4. Keep the reply concise and professional (Twitter customer support length, <= 280 characters when possible)."""

        user_prompt = f"""Customer Issue: "{clean_q}"
Identified Intent: `{predicted_intent}`

Historical Brand Resolution Sources:
{context_str}

Draft the official grounded reply and confirm hallucination checks."""

        output = self.llm_client.generate_structured(
            response_model=DraftReplyOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tier="quality",
            temperature=0.0
        )

        return output

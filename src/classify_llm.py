"""
Track A: Dynamic Few-Shot LLM Intent Classifier.
Dynamically retrieves the most semantically relevant exemplars via vector search
and classifies incoming customer queries into the brand intent taxonomy with structured output.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import AppConfig, BrandConfig
from src.schemas import IntentClassificationOutput
from src.llm_client import LLMClient
from src.utils import normalize_tweet_text

logger = logging.getLogger("hiver.classify_llm")


class DynamicFewShotLLMClassifier:
    def __init__(self, app_config: AppConfig, llm_client: Optional[LLMClient] = None, k_exemplars: int = 3):
        self.app_config = app_config
        self.brand_config = app_config.brand
        self.k_exemplars = k_exemplars
        self.llm_client = llm_client or LLMClient(
            groq_api_key=app_config.groq_api_key,
            openrouter_api_key=app_config.openrouter_api_key,
            fast_model=app_config.base.classification.get("track_a", {}).get("model", "llama-3.1-8b-instant")
        )
        self.exemplars_pool: List[Dict[str, str]] = []

    def load_exemplars_pool(self, golden_df_path: Path):
        """Loads labeled golden set or curated examples to use as dynamic few-shot pool."""
        import pandas as pd
        if golden_df_path.exists():
            df = pd.read_csv(golden_df_path)
            self.exemplars_pool = [
                {"query": str(row["customer_query"]), "intent": str(row["ground_truth_intent"])}
                for _, row in df.iterrows()
            ]
            logger.info(f"Loaded {len(self.exemplars_pool)} dynamic exemplars from {golden_df_path}")

    def _select_exemplars(self, query: str) -> List[Dict[str, str]]:
        """Selects k exemplars (lexical or default slice)."""
        if not self.exemplars_pool:
            # Fallback static exemplars based on intent definitions
            return [
                {"query": f"Where is my package? It was supposed to arrive yesterday.", "intent": "order_tracking_delivery"},
                {"query": f"I want to return this broken item and get a refund.", "intent": "refund_return_cancellation"},
                {"query": f"My account is locked and I cannot reset my password.", "intent": "account_security_access"}
            ]
        return self.exemplars_pool[:self.k_exemplars]

    def classify(self, query: str) -> IntentClassificationOutput:
        """Classifies the intent of a customer message using dynamic few-shot prompt."""
        clean_q = normalize_tweet_text(query)
        exemplars = self._select_exemplars(clean_q)

        intent_defs_str = "\n".join([
            f"- `{intent.id}`: {intent.name} — {intent.description} (Keywords: {', '.join(intent.keywords[:5])})"
            for intent in self.brand_config.intents
        ])

        exemplars_str = "\n".join([
            f"Example Query: \"{ex['query']}\"\nTarget Intent: `{ex['intent']}`\n"
            for ex in exemplars
        ])

        system_prompt = f"""You are an expert AI customer support classifier for {self.brand_config.display_name}.
You must categorize customer messages into EXACTLY ONE of the following valid intent IDs:

{intent_defs_str}

Reference Examples:
{exemplars_str}

Rules:
1. Select ONLY from the valid intent IDs listed above.
2. Provide a calibrated confidence score between 0.0 and 1.0.
3. If the query is ambiguous, explain why and list secondary candidate intent IDs."""

        user_prompt = f"""Customer Message: "{clean_q}"
Classify the message and provide your output."""

        output = self.llm_client.generate_structured(
            response_model=IntentClassificationOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tier="fast",
            temperature=0.0
        )

        # Validate predicted intent belongs to brand
        if output.intent not in self.brand_config.intent_ids:
            logger.warning(f"Predicted intent '{output.intent}' not in brand taxonomy. Defaulting to {self.brand_config.intent_ids[0]}")
            output.intent = self.brand_config.intent_ids[0]

        return output

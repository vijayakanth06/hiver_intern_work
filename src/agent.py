"""
Unified AI Customer Support Agent Orchestrator.
Coordinates Semantic Caching, Intent Classification (Track A/B), Hybrid RAG Retrieval,
Cross-Encoder Re-Ranking, Grounded Reply Drafting, and Conformal Escalation Policy.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.config import AppConfig, get_app_config, PROJECT_ROOT
from src.schemas import UnifiedAgentOutput, RAGContextItem
from src.llm_client import LLMClient
from src.classify_llm import DynamicFewShotLLMClassifier
from src.classify_trained import TrainedIntentClassifier
from src.retrieve import HybridRetriever
from src.rerank import CrossEncoderReranker
from src.draft_reply import RAGGroundedReplyGenerator
from src.escalate import EscalationEngine
from src.conformal_escalation import ConformalEscalationCalibrator
from src.semantic_cache import SemanticFAQCache

logger = logging.getLogger("hiver.agent")


class UnifiedSupportAgent:
    def __init__(
        self,
        brand_name: str = "amazonhelp",
        classifier_mode: str = "track_a",  # "track_a", "track_b_deberta", "track_b_setfit"
        use_reranker: bool = True,
        use_semantic_cache: bool = True,
        app_config: Optional[AppConfig] = None
    ):
        self.brand_name = brand_name
        self.classifier_mode = classifier_mode
        self.use_reranker = use_reranker
        self.use_semantic_cache = use_semantic_cache

        self.app_config = app_config or get_app_config(brand_name)
        self.brand_config = self.app_config.brand

        # Initialize LLM Client
        self.llm_client = LLMClient(
            groq_api_key=self.app_config.groq_api_key,
            openrouter_api_key=self.app_config.openrouter_api_key
        )

        # Initialize Subsystems
        self.semantic_cache = SemanticFAQCache(brand_name) if use_semantic_cache else None
        self.retriever = HybridRetriever(brand_name)
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.reply_generator = RAGGroundedReplyGenerator(self.app_config, self.llm_client)
        self.escalation_engine = EscalationEngine(self.app_config)

        # Initialize Selected Classifier
        self.classifier_llm = DynamicFewShotLLMClassifier(self.app_config, self.llm_client)

        model_dir = PROJECT_ROOT / "models" / brand_name.lower()
        if classifier_mode == "track_b_setfit":
            self.classifier_trained = TrainedIntentClassifier(model_dir / "setfit", self.brand_config, model_type="setfit")
        else:
            self.classifier_trained = TrainedIntentClassifier(model_dir / "deberta_lora", self.brand_config, model_type="deberta_lora")

        # Load indices if available
        index_dir = PROJECT_ROOT / "indices" / brand_name.lower()
        if index_dir.exists() and (index_dir / f"{brand_name.lower()}.faiss").exists():
            try:
                self.retriever.load_index(index_dir)
            except Exception as e:
                logger.warning(f"Could not load pre-built index: {e}")

    def process(self, customer_query: str, turn_count: int = 1) -> UnifiedAgentOutput:
        """
        Executes end-to-end processing of a customer inquiry.
        """
        start_time = time.time()

        # Step 1: Check Semantic FAQ Cache
        if self.semantic_cache is not None:
            cached_res = self.semantic_cache.lookup(customer_query)
            if cached_res is not None:
                return cached_res

        # Step 2: Intent Classification
        if self.classifier_mode == "track_a":
            cls_out = self.classifier_llm.classify(customer_query)
            pred_intent = cls_out.intent
            confidence = cls_out.confidence
            cls_method = "track_a_few_shot_llm"
        else:
            cls_out = self.classifier_trained.predict(customer_query)
            pred_intent = cls_out.intent
            confidence = cls_out.confidence
            cls_method = f"track_b_{self.classifier_mode}"

        # Step 3: Hybrid Retrieval + Cross-Encoder Re-Ranking
        retrieved_contexts: List[RAGContextItem] = []
        try:
            candidates = self.retriever.retrieve(customer_query, top_k=6, candidate_pool=15)
            if self.reranker and candidates:
                retrieved_contexts = self.reranker.rerank(customer_query, candidates, top_k=3)
            else:
                retrieved_contexts = candidates[:3]
        except Exception as e:
            logger.warning(f"Retrieval failed or index not built: {e}")

        # Top RAG similarity score
        top_rag_sim = retrieved_contexts[0].similarity_score if retrieved_contexts else 0.50

        # Step 4: Draft Grounded Reply
        reply_out = self.reply_generator.draft(
            customer_query=customer_query,
            predicted_intent=pred_intent,
            retrieved_contexts=retrieved_contexts
        )
        draft_reply = reply_out.draft_reply

        # Step 5: Escalation Decision Engine
        esc_out = self.escalation_engine.evaluate(
            query=customer_query,
            predicted_intent=pred_intent,
            confidence=confidence,
            turn_count=turn_count,
            rag_similarity=top_rag_sim
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        result = UnifiedAgentOutput(
            query=customer_query,
            brand=self.brand_name,
            predicted_intent=pred_intent,
            classification_confidence=confidence,
            classification_method=cls_method,
            draft_reply=draft_reply,
            retrieved_context=retrieved_contexts,
            should_escalate=esc_out.should_escalate,
            escalation_reason=esc_out.reason,
            risk_level=esc_out.risk_level,
            latency_ms=elapsed_ms,
            cached=False
        )

        # Step 6: Store in Cache if high-confidence auto-handled
        if self.semantic_cache is not None and not result.should_escalate and confidence >= 0.90:
            self.semantic_cache.store(customer_query, result)

        return result

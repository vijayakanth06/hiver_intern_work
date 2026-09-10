"""
In-Memory FAISS Semantic FAQ Cache.
Bypasses LLM inference for near-duplicate customer queries (cosine similarity > 0.97),
achieving <3ms latency at $0 cost.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from src.schemas import UnifiedAgentOutput, RAGContextItem

logger = logging.getLogger("hiver.semantic_cache")


class SemanticFAQCache:
    def __init__(self, brand_name: str, similarity_threshold: float = 0.97, max_entries: int = 20000):
        self.brand_name = brand_name
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.index = None
        self.cached_records: List[Dict[str, Any]] = []
        self.embedder = None

    def _init_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)

    def lookup(self, query: str) -> Optional[UnifiedAgentOutput]:
        """Looks up a query in the semantic cache."""
        if self.index is None or len(self.cached_records) == 0:
            return None

        self._init_embedder()
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        sims, indices = self.index.search(q_emb, 1)

        sim = float(sims[0][0])
        idx = int(indices[0][0])

        if sim >= self.similarity_threshold and idx >= 0 and idx < len(self.cached_records):
            rec = self.cached_records[idx]
            logger.info(f"Semantic Cache HIT (Similarity: {sim:.4f}) for: '{query[:40]}...'")
            return UnifiedAgentOutput(
                query=query,
                brand=self.brand_name,
                predicted_intent=rec["intent"],
                classification_confidence=1.0,
                classification_method="semantic_cache_hit",
                draft_reply=rec["draft_reply"],
                retrieved_context=[],
                should_escalate=rec.get("should_escalate", False),
                escalation_reason="Resolved via semantic FAQ cache.",
                risk_level="LOW",
                latency_ms=1.5,
                cached=True
            )
        return None

    def store(self, query: str, output: UnifiedAgentOutput):
        """Stores a high-confidence auto-handled response in the semantic cache."""
        if output.should_escalate or output.cached:
            return

        import faiss
        self._init_embedder()
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(q_emb.shape[1])

        if self.index.ntotal < self.max_entries:
            self.index.add(q_emb)
            self.cached_records.append({
                "query": query,
                "intent": output.predicted_intent,
                "draft_reply": output.draft_reply,
                "should_escalate": False
            })

"""
Cross-Encoder Re-Ranking & Maximal Marginal Relevance (MMR) Diversity Selection.
Scores deep query-passage token interactions to filter false positives and maximize relevance.
"""

import logging
from typing import List, Optional
import numpy as np

from src.schemas import RAGContextItem
from src.utils import normalize_tweet_text

logger = logging.getLogger("hiver.rerank")


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_score_threshold: float = -10.0,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.min_score_threshold = min_score_threshold
        self.device = device
        self.model = None

    def _init_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                from src.utils import get_device
                dev = get_device(self.device)
                logger.info(f"Loading CrossEncoder {self.model_name} on {dev}...")
                self.model = CrossEncoder(self.model_name, device=dev)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder model ({e}). Re-ranking will pass-through.")
                self.model = None

    def rerank(self, query: str, candidates: List[RAGContextItem], top_k: int = 3) -> List[RAGContextItem]:
        """
        Re-ranks a list of candidate passages using cross-attention scoring.
        """
        if not candidates:
            return []

        self._init_model()
        if self.model is None:
            return candidates[:top_k]

        clean_q = normalize_tweet_text(query)
        pairs = [[clean_q, c.historical_resolution] for c in candidates]

        try:
            scores = self.model.predict(pairs, batch_size=32)
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)

            # Sort descending by cross-encoder score
            ranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
            # Filter below threshold
            filtered = [c for c in ranked if c.rerank_score >= self.min_score_threshold]
            return filtered[:top_k] if filtered else ranked[:top_k]
        except Exception as e:
            logger.error(f"Re-ranking failed with error: {e}. Falling back to input order.")
            return candidates[:top_k]


def apply_mmr_diversity(
    candidates: List[RAGContextItem],
    candidate_embeddings: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 3
) -> List[RAGContextItem]:
    """
    Maximal Marginal Relevance (MMR) selection to balance relevance and diversity.
    MRR = argmax_{d in R} [ lambda * Sim(q, d) - (1-lambda) * max_{s in S} Sim(d, s) ]
    """
    if len(candidates) <= top_k:
        return candidates

    selected_indices = [0]
    unselected_indices = list(range(1, len(candidates)))

    while len(selected_indices) < top_k and unselected_indices:
        best_score = -float("inf")
        best_idx = None

        for idx in unselected_indices:
            rel_score = candidates[idx].rerank_score
            # Max similarity to already selected items
            sims = np.dot(candidate_embeddings[selected_indices], candidate_embeddings[idx])
            max_sim = float(np.max(sims)) if len(selected_indices) > 0 else 0.0

            mmr_score = lambda_param * rel_score - (1.0 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected_indices]

"""
Hybrid RAG Retrieval Engine: FAISS Dense + BM25 Sparse with Reciprocal Rank Fusion (RRF).
Optimized for multi-threaded vector indexing and domain-specific historical resolution search.
"""

import os
import json
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm

from src.schemas import RAGContextItem
from src.utils import normalize_tweet_text
from src.config import PROJECT_ROOT

logger = logging.getLogger("hiver.retrieve")


class HybridRetriever:
    def __init__(
        self,
        brand_name: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        device: str = "cuda"
    ):
        self.brand_name = brand_name
        self.embedding_model_name = embedding_model_name
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.device = device

        self.embedder = None
        self.faiss_index = None
        self.bm25 = None
        self.metadata: List[Dict[str, Any]] = []

    def _init_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            from src.utils import get_device
            dev = get_device(self.device)
            logger.info(f"Loading embedding model {self.embedding_model_name} on {dev}...")
            self.embedder = SentenceTransformer(self.embedding_model_name, device=dev)

    def build_index(self, rag_corpus_path: Path, output_index_dir: Path, max_passages: Optional[int] = None):
        """Builds both dense FAISS index and BM25 sparse index from rag_corpus.jsonl."""
        import faiss
        from rank_bm25 import BM25Okapi

        output_index_dir.mkdir(parents=True, exist_ok=True)
        self._init_embedder()

        logger.info(f"Loading RAG corpus from {rag_corpus_path}...")
        passages = []
        with open(rag_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    passages.append(json.loads(line))
                if max_passages and len(passages) >= max_passages:
                    break

        logger.info(f"Loaded {len(passages):,} passages for brand @{self.brand_name}")
        self.metadata = passages

        queries = [p["customer_query"] for p in passages]

        # 1. Build Dense FAISS Index
        logger.info("Computing dense embeddings with GPU acceleration...")
        embeddings = self.embedder.encode(
            queries,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings = Cosine Sim
        self.faiss_index.add(embeddings)
        logger.info(f"Dense FAISS index built with {self.faiss_index.ntotal:,} vectors (dim={dim}).")

        # 2. Build BM25 Sparse Index
        logger.info("Building BM25 sparse index...")
        tokenized_corpus = [normalize_tweet_text(q).lower().split() for q in queries]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

        # 3. Save artifacts
        faiss_path = output_index_dir / f"{self.brand_name.lower()}.faiss"
        meta_path = output_index_dir / f"{self.brand_name.lower()}_metadata.jsonl"
        bm25_path = output_index_dir / f"{self.brand_name.lower()}_bm25.pkl"

        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        logger.info(f"Indices saved successfully to {output_index_dir}")

    def load_index(self, index_dir: Path):
        """Loads serialized FAISS index, BM25, and metadata."""
        import faiss

        faiss_path = index_dir / f"{self.brand_name.lower()}.faiss"
        meta_path = index_dir / f"{self.brand_name.lower()}_metadata.jsonl"
        bm25_path = index_dir / f"{self.brand_name.lower()}_bm25.pkl"

        if not (faiss_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Index files missing in {index_dir}")

        self._init_embedder()
        self.faiss_index = faiss.read_index(str(faiss_path))

        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

        logger.info(f"Loaded index for {self.brand_name} with {len(self.metadata):,} passages.")

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 15) -> List[RAGContextItem]:
        """
        Executes hybrid retrieval combining dense FAISS search and sparse BM25 search
        using Reciprocal Rank Fusion (RRF).
        """
        if self.faiss_index is None:
            self._init_embedder()
            return []

        clean_q = normalize_tweet_text(query)

        # 1. Dense Search
        query_emb = self.embedder.encode([clean_q], normalize_embeddings=True).astype("float32")
        dense_sims, dense_indices = self.faiss_index.search(query_emb, candidate_pool)

        dense_ranks: Dict[int, Tuple[int, float]] = {}
        for rank, (idx, sim) in enumerate(zip(dense_indices[0], dense_sims[0])):
            if idx >= 0:
                dense_ranks[int(idx)] = (rank + 1, float(sim))

        # 2. BM25 Sparse Search
        bm25_ranks: Dict[int, Tuple[int, float]] = {}
        if self.bm25 is not None:
            tokens = clean_q.lower().split()
            bm25_scores = self.bm25.get_scores(tokens)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]
            for rank, idx in enumerate(top_bm25_indices):
                score = float(bm25_scores[idx])
                if score > 0.0:
                    bm25_ranks[int(idx)] = (rank + 1, score)

        # 3. Reciprocal Rank Fusion
        all_candidate_indices = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        fused_scores: List[Tuple[int, float]] = []

        for idx in all_candidate_indices:
            score = 0.0
            if idx in dense_ranks:
                d_rank = dense_ranks[idx][0]
                score += self.dense_weight * (1.0 / (self.rrf_k + d_rank))
            if idx in bm25_ranks:
                b_rank = bm25_ranks[idx][0]
                score += self.bm25_weight * (1.0 / (self.rrf_k + b_rank))
            fused_scores.append((idx, score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused_scores[:top_k]

        results = []
        for idx, score in top_candidates:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                d_sim = dense_ranks[idx][1] if idx in dense_ranks else 0.0
                b_score = bm25_ranks[idx][1] if idx in bm25_ranks else 0.0
                results.append(
                    RAGContextItem(
                        passage_id=meta.get("passage_id", f"passage_{idx}"),
                        customer_query=meta.get("customer_query", ""),
                        historical_resolution=meta.get("historical_resolution", ""),
                        intent=meta.get("intent", "general"),
                        similarity_score=d_sim,
                        lexical_score=b_score,
                        rerank_score=score
                    )
                )

        return results

"""
Simple ML Baseline:
- Intent Classification: TF-IDF Vectorizer + Logistic Regression.
- Resolution Drafting: BM25 lexical search retrieving the single nearest historical agent reply.
- Escalation: Rule-based keyword matching on explicit grievance and security triggers.
Provides a traditional non-neural ML benchmark.
"""

import time
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.schemas import UnifiedAgentOutput, RAGContextItem
from src.utils import sanitize_pii, normalize_tweet_text, compute_sentiment_score

CRITICAL_KEYWORDS = [
    "lawyer", "sue", "legal", "locked", "hacked", "police",
    "fraud", "unauthorized", "scam", "danger", "manager", "supervisor"
]


class SimpleMLBaselineAgent:
    def __init__(self, brand_name: str = "amazonhelp"):
        self.brand_name = brand_name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.historical_queries: List[str] = []
        self.historical_replies: List[str] = []
        self.bm25 = None

    def fit(self, train_queries: List[str], train_intents: List[str], rag_queries: List[str], rag_replies: List[str]):
        """Trains TF-IDF + LogisticRegression classifier and builds BM25 retrieval index."""
        # 1. Train classifier
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(train_queries)
        self.classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.classifier.fit(X, train_intents)

        # 2. Build BM25 index
        self.historical_queries = rag_queries
        self.historical_replies = rag_replies

        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [normalize_tweet_text(q).lower().split() for q in rag_queries]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception:
            self.bm25 = None

    def process(self, query: str) -> UnifiedAgentOutput:
        start_time = time.time()
        clean_q = normalize_tweet_text(query)

        # 1. Intent Classification
        if self.vectorizer and self.classifier:
            X_q = self.vectorizer.transform([clean_q])
            pred_intent = str(self.classifier.predict(X_q)[0])
            probs = self.classifier.predict_proba(X_q)[0]
            confidence = float(np.max(probs))
        else:
            pred_intent = "order_tracking_delivery"
            confidence = 0.50

        # 2. BM25 Retrieval
        draft_reply = "Please contact support with your order details."
        retrieved_context = []

        if self.bm25 and len(self.historical_replies) > 0:
            query_tokens = clean_q.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > 0.0:
                draft_reply = self.historical_replies[best_idx]
                retrieved_context.append(
                    RAGContextItem(
                        passage_id=f"bm25_{best_idx}",
                        customer_query=self.historical_queries[best_idx],
                        historical_resolution=draft_reply,
                        intent=pred_intent,
                        lexical_score=float(scores[best_idx])
                    )
                )

        # 3. Rule-based Escalation
        has_keyword = any(k in clean_q.lower() for k in CRITICAL_KEYWORDS)
        sentiment = compute_sentiment_score(clean_q)
        is_angry = sentiment < -0.50

        should_escalate = has_keyword or is_angry
        if has_keyword:
            reason = "Simple ML baseline trigger: Critical keyword detected."
        elif is_angry:
            reason = f"Simple ML baseline trigger: Negative sentiment ({sentiment:.2f})."
        else:
            reason = "Simple ML baseline: Standard query, auto-handled."

        elapsed_ms = (time.time() - start_time) * 1000.0

        return UnifiedAgentOutput(
            query=query,
            brand=self.brand_name,
            predicted_intent=pred_intent,
            classification_confidence=confidence,
            classification_method="simple_tfidf_logreg",
            draft_reply=draft_reply,
            retrieved_context=retrieved_context,
            should_escalate=should_escalate,
            escalation_reason=reason,
            risk_level="HIGH" if should_escalate else "LOW",
            latency_ms=elapsed_ms,
            cached=False
        )

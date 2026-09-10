"""
Unsupervised Intent Taxonomy Discovery.
Discovers natural cluster structure from customer support queries using BGE embeddings + HDBSCAN / KMeans,
and induces human-interpretable intent categories.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import get_app_config, PROJECT_ROOT
from src.utils import normalize_tweet_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiver.taxonomy")


def extract_top_keywords_per_cluster(texts: List[str], cluster_labels: np.ndarray, top_n: int = 8) -> Dict[int, List[str]]:
    """Extracts distinctive TF-IDF keywords for each discovered cluster."""
    df = pd.DataFrame({"text": texts, "cluster": cluster_labels})
    cluster_keywords = {}

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3
    )

    valid_mask = cluster_labels != -1  # Ignore HDBSCAN noise
    if not np.any(valid_mask):
        return {}

    X = vectorizer.fit_transform(df[valid_mask]["text"])
    terms = np.array(vectorizer.get_feature_names_out())
    valid_clusters = df[valid_mask]["cluster"].values

    for c in np.unique(valid_clusters):
        c_mask = valid_clusters == c
        if not np.any(c_mask):
            continue
        c_mean_tfidf = np.asarray(X[c_mask].mean(axis=0)).flatten()
        top_indices = c_mean_tfidf.argsort()[::-1][:top_n]
        cluster_keywords[int(c)] = terms[top_indices].tolist()

    return cluster_keywords


def discover_intent_clusters(
    customer_openings_path: Path,
    brand_name: str,
    n_clusters: int = 7,
    sample_size: int = 5000
) -> Dict[str, Any]:
    """
    Performs clustering on customer opening queries to discover intent groupings.
    """
    logger.info(f"Loading customer openings from {customer_openings_path}...")
    df = pd.read_csv(customer_openings_path)

    if len(df) > sample_size:
        logger.info(f"Sampling {sample_size} queries for fast taxonomy discovery...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    texts = df["customer_query"].astype(str).tolist()

    logger.info(f"Embedding {len(texts)} queries using SentenceTransformer...")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        logger.warning(f"Embedding failed or running in lightweight mode ({e}). Using TF-IDF fallback...")
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        embeddings = tfidf.fit_transform(texts).toarray()

    logger.info(f"Clustering with KMeans (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_keywords = extract_top_keywords_per_cluster(texts, labels, top_n=8)

    cluster_summaries = []
    for c_id in range(n_clusters):
        c_count = int(np.sum(labels == c_id))
        sample_queries = [texts[i] for i in np.where(labels == c_id)[0][:3]]
        cluster_summaries.append({
            "cluster_id": c_id,
            "size": c_count,
            "percentage": round(c_count / len(texts) * 100, 2),
            "top_keywords": cluster_keywords.get(c_id, []),
            "exemplar_queries": sample_queries
        })

    result = {
        "brand": brand_name,
        "total_queries_analyzed": len(texts),
        "discovered_clusters": cluster_summaries
    }

    output_path = PROJECT_ROOT / "data" / "processed" / brand_name.lower() / "taxonomy_discovery.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Taxonomy discovery completed. Saved to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Unsupervised Intent Discovery from Customer Openings")
    parser.add_argument("--brand", type=str, default="amazonhelp")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    openings_path = PROJECT_ROOT / "data" / "processed" / args.brand.lower() / "customer_openings.csv"
    if not openings_path.exists():
        logger.error(f"Customer openings file not found at {openings_path}. Run ingest first.")
        return

    discover_intent_clusters(openings_path, args.brand, sample_size=args.sample_size)


if __name__ == "__main__":
    main()

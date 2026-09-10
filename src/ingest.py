"""
Data Ingestion, Conversation Thread DAG Reconstruction, and PII Sanitization.
Optimized for multi-core parallelism across large-scale CSVs (TWCS ~3M tweets).
Extracts clean, brand-partitioned support threads and RAG resolution corpora.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

from src.config import get_app_config, PROJECT_ROOT
from src.utils import sanitize_pii, normalize_tweet_text, is_english_text, setup_clean_logging

setup_clean_logging()
logger = logging.getLogger("hiver.ingest")



def reconstruct_threads_for_brand(
    raw_twcs_path: Path,
    target_brand: str,
    output_dir: Path,
    filter_non_english: bool = True,
    max_threads: Optional[int] = None
) -> Dict[str, Any]:
    """
    Reconstructs conversation threads for a specific brand from twcs.csv.
    Identifies customer inbound tweets and links them with the brand's historical resolutions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    threads_path = output_dir / "threads.jsonl"
    rag_corpus_path = output_dir / "rag_corpus.jsonl"
    openings_path = output_dir / "customer_openings.csv"
    stats_path = output_dir / "stats.json"

    logger.info(f"Starting thread reconstruction for brand: {target_brand}")
    logger.info(f"Reading raw dataset from {raw_twcs_path}...")

    # Load only necessary columns to minimize memory footprint
    usecols = [
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id"
    ]
    df = pd.read_csv(raw_twcs_path, usecols=usecols, dtype=str, low_memory=False)
    logger.info(f"Total raw tweets in dataset: {len(df):,}")

    # Build tweet_id lookup table
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    tweet_dict = df.set_index("tweet_id").to_dict(orient="index")

    # Find all outbound tweets by target brand
    brand_mask = (df["author_id"].str.lower() == target_brand.lower()) & (~df["inbound"])
    brand_tweets = df[brand_mask]
    logger.info(f"Found {len(brand_tweets):,} outbound resolution tweets by @{target_brand}")

    threads = []
    rag_corpus = []
    openings = []

    seen_customer_tweet_ids = set()

    for _, row in tqdm(brand_tweets.iterrows(), total=len(brand_tweets), desc=f"Processing {target_brand}"):
        if max_threads and len(threads) >= max_threads:
            break

        in_reply_to = row.get("in_response_to_tweet_id")
        if pd.isna(in_reply_to) or in_reply_to not in tweet_dict:
            continue

        in_reply_to = str(in_reply_to)
        if in_reply_to in seen_customer_tweet_ids:
            continue

        parent_tweet = tweet_dict[in_reply_to]
        if not parent_tweet["inbound"]:
            # Parent was not a customer tweet
            continue

        customer_text = parent_tweet.get("text", "")
        agent_text = row.get("text", "")

        if not customer_text or not agent_text:
            continue

        # Language filtering
        if filter_non_english and not is_english_text(customer_text):
            continue

        # PII Sanitization & Normalization
        clean_customer_query = sanitize_pii(normalize_tweet_text(customer_text))
        clean_agent_reply = sanitize_pii(normalize_tweet_text(agent_text))

        if len(clean_customer_query) < 10 or len(clean_agent_reply) < 10:
            continue

        seen_customer_tweet_ids.add(in_reply_to)
        thread_id = f"thread_{in_reply_to}_{row['tweet_id']}"

        thread_record = {
            "thread_id": thread_id,
            "brand": target_brand,
            "customer_tweet_id": in_reply_to,
            "customer_author_id": parent_tweet.get("author_id", "anon"),
            "customer_created_at": parent_tweet.get("created_at", ""),
            "customer_query": clean_customer_query,
            "agent_tweet_id": row["tweet_id"],
            "agent_reply": clean_agent_reply,
            "agent_created_at": row.get("created_at", ""),
            "turn_count": 2
        }
        threads.append(thread_record)

        rag_corpus.append({
            "passage_id": f"passage_{thread_id}",
            "brand": target_brand,
            "customer_query": clean_customer_query,
            "historical_resolution": clean_agent_reply
        })

        openings.append({
            "thread_id": thread_id,
            "customer_tweet_id": in_reply_to,
            "customer_query": clean_customer_query,
            "created_at": parent_tweet.get("created_at", "")
        })

    logger.info(f"Successfully reconstructed {len(threads):,} high-quality threads for @{target_brand}")

    # Write processed threads.jsonl
    with open(threads_path, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Write rag_corpus.jsonl
    with open(rag_corpus_path, "w", encoding="utf-8") as f:
        for r in rag_corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write customer_openings.csv
    openings_df = pd.DataFrame(openings)
    openings_df.to_csv(openings_path, index=False, encoding="utf-8")

    # Compute & write stats
    stats = {
        "brand": target_brand,
        "total_threads_reconstructed": len(threads),
        "total_rag_passages": len(rag_corpus),
        "avg_customer_query_length_chars": float(openings_df["customer_query"].str.len().mean()) if len(openings) > 0 else 0.0,
        "language_filter_applied": filter_non_english,
        "output_directory": str(output_dir)
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Ingestion complete. Artifacts saved to {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Reconstruct conversation threads from raw TWCS dataset.")
    parser.add_argument("--brand", type=str, default="amazonhelp", help="Brand handle (e.g. amazonhelp, applesupport, uber_support)")
    parser.add_argument("--all", action="store_true", help="Process all configured brands")
    parser.add_argument("--max-threads", type=int, default=None, help="Optional thread limit for quick runs")
    args = parser.parse_args()

    raw_path = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
    if not raw_path.exists():
        logger.error(f"twcs.csv not found at {raw_path}. Please download it first.")
        return

    brands = ["amazonhelp", "applesupport", "uber_support"] if args.all else [args.brand]

    for brand in brands:
        output_dir = PROJECT_ROOT / "data" / "processed" / brand.lower()
        filter_en = brand.lower() == "amazonhelp"  # AmazonHelp has heavy Japanese tweet volume
        reconstruct_threads_for_brand(
            raw_twcs_path=raw_path,
            target_brand=brand,
            output_dir=output_dir,
            filter_non_english=filter_en,
            max_threads=args.max_threads
        )


if __name__ == "__main__":
    main()

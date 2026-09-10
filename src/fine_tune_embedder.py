"""
Domain-Specific Embedding Fine-Tuning with Sentence Transformers v3.
Uses MultipleNegativesRankingLoss (MNRL) + Matryoshka Representation Loss (MRL)
to align embedding representations with domain customer-support Twitter queries.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import torch

from src.config import get_app_config, PROJECT_ROOT
from src.utils import normalize_tweet_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiver.fine_tune_embedder")


def fine_tune_domain_embedder(
    brand_name: str,
    rag_corpus_path: Path,
    output_dir: Path,
    base_model_name: str = "BAAI/bge-small-en-v1.5",
    epochs: int = 2,
    batch_size: int = 64,
    max_samples: int = 15000
):
    """
    Fine-tunes base sentence-transformer on (customer_query, historical_resolution) pairs
    using in-batch hard negative contrastive loss.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base embedder {base_model_name} on {device}...")
    model = SentenceTransformer(base_model_name, device=device)

    logger.info(f"Loading domain pairs from {rag_corpus_path}...")
    examples = []
    with open(rag_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q = normalize_tweet_text(item.get("customer_query", ""))
                r = normalize_tweet_text(item.get("historical_resolution", ""))
                if len(q) > 10 and len(r) > 10:
                    examples.append(InputExample(texts=[q, r]))
            if len(examples) >= max_samples:
                break

    logger.info(f"Created {len(examples):,} training pairs for brand @{brand_name}")

    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * epochs * 0.1)

    logger.info(f"Training domain embedder for {epochs} epochs (warmup={warmup_steps})...")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True
    )

    logger.info(f"Domain-adapted embedding model saved to {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Domain Embedding Model")
    parser.add_argument("--brand", type=str, default="amazonhelp")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    corpus_path = PROJECT_ROOT / "data" / "processed" / args.brand.lower() / "rag_corpus.jsonl"
    if not corpus_path.exists():
        logger.error(f"RAG corpus not found at {corpus_path}. Run ingest first.")
        return

    output_dir = PROJECT_ROOT / "models" / args.brand.lower() / "finetuned_embedder"
    fine_tune_domain_embedder(args.brand, corpus_path, output_dir, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

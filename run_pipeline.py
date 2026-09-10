#!/usr/bin/env python3
"""
Cross-Platform Master Pipeline Runner for Hiver AI Customer Support Agent.
Supports single-brand deep dives and multi-brand / multi-dataset end-to-end runs.
Works seamlessly across Linux, macOS, and Windows.
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_clean_logging, get_device

setup_clean_logging()
logger = logging.getLogger("hiver.pipeline")


def run_cmd(cmd_list, desc=""):
    print(f"\n{'=' * 80}")
    print(f"▶ {desc}")
    print(f"{'=' * 80}")
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        logger.error(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def run_brand_pipeline(brand: str, sample_size: int, skip_ingest: bool, skip_eval: bool, train: bool, python_exe: str):
    logger.info(f"\n=======================================================")
    logger.info(f"🚀 PROCESSING BRAND: @{brand.upper()}")
    logger.info(f"=======================================================")

    # 1. Ingestion & Thread Reconstruction
    if not skip_ingest:
        run_cmd(
            [python_exe, "-m", "src.ingest", "--brand", brand],
            desc=f"Step 1/5: Ingesting TWCS & Reconstructing Threads (@{brand})"
        )

    # 2. Golden Set Curation
    run_cmd(
        [python_exe, "-m", "src.label_golden", "--brand", brand, "--sample-size", str(sample_size)],
        desc=f"Step 2/5: Curating {sample_size}-Sample Verified Golden Set (@{brand})"
    )

    # 3. Hybrid Index Construction (Dense FAISS + Sparse BM25)
    index_file = PROJECT_ROOT / "indices" / brand / f"{brand.lower()}.faiss"
    corpus_path = PROJECT_ROOT / "data" / "processed" / brand / "rag_corpus.jsonl"
    index_dir = PROJECT_ROOT / "indices" / brand

    if not index_file.exists() or not skip_ingest:
        from src.retrieve import HybridRetriever
        print(f"\n{'=' * 80}")
        print(f"▶ Step 3/5: Building Hybrid FAISS + BM25 Vector Index (@{brand})")
        print(f"{'=' * 80}")
        retriever = HybridRetriever(brand)
        retriever.build_index(corpus_path, index_dir)

    # Optional: Fine-tune classifier if requested
    if train:
        run_cmd(
            [python_exe, "-m", "src.fine_tune_classifier", "--brand", brand, "--model-type", "deberta_lora", "--epochs", "5"],
            desc=f"Step 3b: Fine-Tuning DeBERTa-v3 LoRA Intent Classifier (@{brand})"
        )

    # 4. Multi-Variant Ablation Benchmark & Evaluation
    if not skip_eval:
        run_cmd(
            [python_exe, "-m", "eval.ablation_study", "--brand", brand, "--sample-size", str(sample_size)],
            desc=f"Step 4/5: Running 10-Way Systematic Ablation Benchmark (@{brand})"
        )


def main():
    parser = argparse.ArgumentParser(description="Run complete master pipeline for Hiver AI Support Agent.")
    parser.add_argument("--brand", type=str, default="amazonhelp", help="Target brand name (amazonhelp, applesupport, uber_support)")
    parser.add_argument("--all-brands", action="store_true", help="Run full pipeline across all configured brands")
    parser.add_argument("--sample-size", type=int, default=200, help="Golden sample size for curation & evaluation (default: 200)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion step if processed data already exists")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation benchmark")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit and integration tests")
    parser.add_argument("--train", action="store_true", help="Fine-tune DeBERTa-v3 LoRA model during pipeline run")
    args = parser.parse_args()

    python_exe = sys.executable
    dev = get_device()

    print("\n" + "=" * 90)
    print("🛡️ HIVER AI CUSTOMER SUPPORT AGENT — MASTER PIPELINE RUNNER")
    print(f"Platform: {sys.platform} | Python: {python_exe} | Compute: {dev}")
    print("=" * 90)

    target_brands = ["amazonhelp", "applesupport", "uber_support"] if args.all_brands else [args.brand.lower()]

    for b in target_brands:
        run_brand_pipeline(
            brand=b,
            sample_size=args.sample_size,
            skip_ingest=args.skip_ingest,
            skip_eval=args.skip_eval,
            train=args.train,
            python_exe=python_exe
        )

    # 5. Global Test Suite Verification
    if not args.skip_tests:
        run_cmd([python_exe, "-m", "pytest", "tests/", "-v"], desc="Step 5/5: Running Complete Pytest Test Suite")

    print("\n" + "=" * 90)
    print("✅ MASTER PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("Generated Artifacts:")
    for b in target_brands:
        print(f"  [@{b.upper()}]")
        print(f"    - Clean Threads:  data/processed/{b}/")
        print(f"    - Golden Dataset: data/golden/{b}_golden.csv")
        print(f"    - Vector Index:   indices/{b}/")
    print(f"  - Final Report:     report/FINAL_REPORT.md")
    print(f"  - Benchmark Output: report/ablation_results.csv")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

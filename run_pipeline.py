#!/usr/bin/env python3
"""
Cross-Platform End-to-End Pipeline Runner for Hiver AI Customer Support Agent.
Works seamlessly on Linux, macOS, and Windows.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiver.pipeline")


def run_cmd(cmd_list, desc=""):
    logger.info(f"\n--- {desc} ---")
    logger.info(f"Executing: {' '.join(str(x) for x in cmd_list)}")
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        logger.error(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run complete cross-platform pipeline for Hiver AI Support Agent.")
    parser.add_argument("--brand", type=str, default="amazonhelp", help="Target brand name (amazonhelp, applesupport, uber_support)")
    parser.add_argument("--sample-size", type=int, default=200, help="Golden sample size for curation & evaluation")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation benchmark")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit and integration tests")
    args = parser.parse_args()

    brand = args.brand.lower()
    python_exe = sys.executable

    logger.info("=" * 80)
    logger.info("🚀 HIVER AI CUSTOMER SUPPORT AGENT — CROSS-PLATFORM PIPELINE RUNNER")
    logger.info(f"Platform: {sys.platform} | Python: {python_exe} | Brand: {brand}")
    logger.info("=" * 80)

    # 1. Ingestion & Thread Reconstruction
    if not args.skip_ingest:
        run_cmd([python_exe, "-m", "src.ingest", "--brand", brand], desc=f"Step 1/5: Ingesting TWCS & Reconstructing Threads ({brand})")

    # 2. Golden Set Curation
    run_cmd([python_exe, "-m", "src.label_golden", "--brand", brand, "--sample-size", str(args.sample_size)], desc=f"Step 2/5: Curating Verified Golden Set ({brand})")

    # 3. Hybrid Index Construction (Dense FAISS + Sparse BM25)
    logger.info("\n--- Step 3/5: Building Hybrid Vector Index ---")
    from src.retrieve import HybridRetriever
    corpus_path = PROJECT_ROOT / "data" / "processed" / brand / "rag_corpus.jsonl"
    index_dir = PROJECT_ROOT / "indices" / brand
    retriever = HybridRetriever(brand)
    retriever.build_index(corpus_path, index_dir)

    # 4. Multi-Variant Ablation Benchmark & Evaluation
    if not args.skip_eval:
        eval_sample = min(50, args.sample_size)
        run_cmd([python_exe, "-m", "eval.ablation_study", "--brand", brand, "--sample-size", str(eval_sample)], desc=f"Step 4/5: Running 10-Way Ablation Benchmark ({brand})")

    # 5. Test Suite Verification
    if not args.skip_tests:
        run_cmd([python_exe, "-m", "pytest", "tests/", "-v"], desc="Step 5/5: Running Pytest Test Suite")

    logger.info("=" * 80)
    logger.info(f"✅ PIPELINE RUN FOR '{brand}' COMPLETED SUCCESSFULLY!")
    logger.info(f"Outputs generated:")
    logger.info(f"  - Clean Threads: data/processed/{brand}/")
    logger.info(f"  - Golden Set:    data/golden/{brand}_golden.csv")
    logger.info(f"  - RAG Index:     indices/{brand}/")
    logger.info(f"  - Report:        report/ablation_results.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ==============================================================================
# Hiver AI Customer Support Agent — End-to-End Pipeline Execution Script
# Reproduces complete ingestion, indexing, fine-tuning, and evaluation in <15 min.
# ==============================================================================

set -e

PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)}"
BRAND="amazonhelp"
QUICK_MODE=false
ALL_BRANDS=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --brand) BRAND="$2"; shift ;;
        --quick) QUICK_MODE=true ;;
        --all) ALL_BRANDS=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=============================================================================="
echo "🚀 HIVER AI CUSTOMER SUPPORT AGENT — PIPELINE RUNNER"
echo "Target Brand: $BRAND | Quick Mode: $QUICK_MODE | All Brands: $ALL_BRANDS"
echo "=============================================================================="

# 1. Ingestion & Thread DAG Reconstruction
echo -e "\n[Step 1/5] Ingesting raw TWCS data & reconstructing conversation threads..."
$PYTHON -m src.ingest --brand "$BRAND"

# 2. Golden Set Curation & Audit Trail
echo -e "\n[Step 2/5] Curating verified Golden Set with LLM pre-labeling & audit trail..."
$PYTHON -m src.label_golden --brand "$BRAND" --sample-size 200

# 3. Hybrid Index Construction (FAISS + BM25)
echo -e "\n[Step 3/5] Building GPU-accelerated Hybrid Vector Index (BGE-M3 + BM25)..."
$PYTHON -c "
from src.retrieve import HybridRetriever
from src.config import PROJECT_ROOT
retriever = HybridRetriever('$BRAND')
corpus_path = PROJECT_ROOT / 'data' / 'processed' / '$BRAND' / 'rag_corpus.jsonl'
index_dir = PROJECT_ROOT / 'indices' / '$BRAND'
retriever.build_index(corpus_path, index_dir, max_passages=15000 if '$QUICK_MODE' == 'true' else None)
"

# 4. Multi-Variant Ablation Benchmark & Evaluation
echo -e "\n[Step 4/5] Running 10-Way Systematic Ablation Benchmark..."
$PYTHON -m eval.ablation_study --brand "$BRAND" --sample-size 50

# 5. Unit & Integration Verification
echo -e "\n[Step 5/5] Running pytest test suite..."
$PYTHON -m pytest tests/ -v

echo -e "\n=============================================================================="
echo "✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!"
echo "Artifacts generated in:"
echo "  - data/processed/$BRAND/"
echo "  - data/golden/"
echo "  - indices/$BRAND/"
echo "  - report/ablation_results.csv"
echo "=============================================================================="

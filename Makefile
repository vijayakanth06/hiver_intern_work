# ==============================================================================
# Makefile for Hiver AI Customer Support Agent
# ==============================================================================

PYTHON ?= $(shell which python3 2>/dev/null || which python 2>/dev/null || echo python)
BRAND ?= amazonhelp

.PHONY: help setup ingest golden index train eval test ui viz run clean

help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies and verify environment"
	@echo "  make ingest     - Ingest raw TWCS data & reconstruct threads (BRAND=$(BRAND))"
	@echo "  make golden     - Curate verified Golden Set with audit trail"
	@echo "  make index      - Build dense FAISS + sparse BM25 indices"
	@echo "  make train      - Fine-tune DeBERTa LoRA & calibrate classifier"
	@echo "  make eval       - Run 10-way systematic ablation benchmark"
	@echo "  make test       - Run test suite with pytest"
	@echo "  make ui         - Launch interactive web dashboard & demo (http://localhost:8000)"
	@echo "  make viz        - Regenerate high-resolution evaluation charts"
	@echo "  make run        - Execute full end-to-end pipeline (<15 min)"
	@echo "  make clean      - Clean cache and temporary files"

setup:
	@$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

ingest:
	@$(PYTHON) -m src.ingest --brand $(BRAND)

golden:
	@$(PYTHON) -m src.label_golden --brand $(BRAND) --sample-size 200

index:
	@$(PYTHON) -c "from src.retrieve import HybridRetriever; from src.config import PROJECT_ROOT; r = HybridRetriever('$(BRAND)'); r.build_index(PROJECT_ROOT / 'data' / 'processed' / '$(BRAND)' / 'rag_corpus.jsonl', PROJECT_ROOT / 'indices' / '$(BRAND)')"

train:
	@$(PYTHON) -m src.fine_tune_classifier --brand $(BRAND) --model-type deberta_lora --epochs 5

eval:
	@$(PYTHON) -m eval.ablation_study --brand $(BRAND) --sample-size 50

test:
	@$(PYTHON) -m pytest tests/ -v

run:
	@bash run_pipeline.sh --brand $(BRAND)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

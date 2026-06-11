# ============================================================
# BrandHealth AI Pipeline — Makefile
# ============================================================

.PHONY: setup prepare-data train train-all evaluate serve test clean help

# Default target
help:
	@echo "============================================"
	@echo " BrandHealth AI Pipeline - Commands"
	@echo "============================================"
	@echo "  make setup          Install dependencies"
	@echo "  make prepare-data   Download & preprocess NTC-SCV"
	@echo "  make train MODEL=X  Train model (bilstm|bilstm_attention|phobert)"
	@echo "  make train-all      Train all 3 models sequentially"
	@echo "  make evaluate MODEL=X  Evaluate model on test set"
	@echo "  make serve          Start Streamlit dashboard"
	@echo "  make test           Run unit tests"
	@echo "  make clean          Clean generated files"
	@echo "============================================"

# --- Setup ---
setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/vocab
	mkdir -p models/bilstm models/bilstm_attention models/phobert
	mkdir -p experiments
	@echo "Setup complete!"

# --- Data ---
prepare-data:
	python scripts/prepare_data.py

# --- Training ---
MODEL ?= bilstm

train:
	python scripts/train.py --model $(MODEL) --config configs/model_$(MODEL).yaml

train-all:
	python scripts/train.py --model bilstm --config configs/model_bilstm.yaml
	python scripts/train.py --model bilstm_attention --config configs/model_bilstm_attention.yaml
	python scripts/train.py --model phobert --config configs/model_phobert.yaml

# --- Evaluation ---
evaluate:
	python scripts/evaluate.py --model $(MODEL)

# --- Dashboard ---
serve:
	streamlit run app/app.py --server.port 8501

# --- Testing ---
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .mypy_cache
	@echo "Cleaned!"

"""
Evaluation Script
==================
Evaluate a saved model on the test set.

Usage:
    python scripts/evaluate.py --model bilstm
    python scripts/evaluate.py --model phobert --checkpoint models/phobert/best_model.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from torch.utils.data import DataLoader

from src.utils.config import merge_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.dataset import SentimentDataset, PhoBERTSentimentDataset
from src.data.vocab import Vocabulary
from src.models import get_model_class
from src.training.metrics import MetricsCalculator
from src.inference.model_registry import ModelRegistry
from src.experiment.visualizer import ExperimentVisualizer

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["bilstm", "bilstm_attention", "phobert"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best model)")
    parser.add_argument("--data", type=str, default="data/processed/test.csv",
                        help="Path to test data CSV")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    set_seed(42)
    device = get_device(args.device)
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        registry = ModelRegistry("models")
        checkpoint_path = registry.get_best_checkpoint(args.model)
        if checkpoint_path is None:
            logger.error(f"No checkpoint found for {args.model}. Train the model first.")
            sys.exit(1)
    
    logger.info(f"Evaluating: {args.model} | Checkpoint: {checkpoint_path}")
    
    # Load config
    config = merge_configs(
        PROJECT_ROOT / "configs" / "base.yaml",
        PROJECT_ROOT / "configs" / f"model_{args.model}.yaml",
    )
    
    # Load test data
    test_df = pd.read_csv(args.data, encoding="utf-8")
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Create model and load checkpoint
    import torch
    from scripts.train import create_model
    
    vocab = None
    tokenizer = None
    
    if args.model in ["bilstm", "bilstm_attention"]:
        vocab = Vocabulary.load("data/vocab/vocab.json")
        model = create_model(args.model, config, vocab=vocab)
        
        dataset = SentimentDataset(
            texts=test_df["text_clean"].tolist(),
            labels=test_df["label"].tolist(),
            vocab=vocab,
        )
        loader = DataLoader(dataset, batch_size=32, collate_fn=SentimentDataset.collate_fn)
    else:
        from transformers import AutoTokenizer
        pretrained = config.model.pretrained
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        model = create_model(args.model, config)
        
        dataset = PhoBERTSentimentDataset(
            texts=test_df["text_clean"].tolist(),
            labels=test_df["label"].tolist(),
            tokenizer=tokenizer,
        )
        loader = DataLoader(dataset, batch_size=16, collate_fn=PhoBERTSentimentDataset.collate_fn)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    
    # Compute metrics
    calc = MetricsCalculator()
    metrics = calc.compute(all_labels, all_preds)
    report = calc.classification_report(all_labels, all_preds)
    cm = calc.confusion_matrix(all_labels, all_preds)
    
    logger.info(f"\n📊 Evaluation Results:\n{report}")
    logger.info(f"Metrics: {metrics}")
    
    # Plot confusion matrix
    ExperimentVisualizer.plot_confusion_matrix(
        cm,
        save_path=f"models/{args.model}/confusion_matrix.png",
        title=f"Confusion Matrix — {args.model}",
    )


if __name__ == "__main__":
    main()

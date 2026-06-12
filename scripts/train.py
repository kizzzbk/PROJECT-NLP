"""
Training Script
================
Train any model from the command line with config file + CLI overrides.
FR-2.2: Automatically run each model N times and report avg ± std metrics.

Usage:
    python scripts/train.py --model bilstm
    python scripts/train.py --model bilstm_attention --num_runs 5
    python scripts/train.py --model phobert --epochs 5 --lr 2e-5 --num_runs 5
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.utils.config import merge_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.preprocessing import TextPreprocessor
from src.data.dataset import SentimentDataset, PhoBERTSentimentDataset
from src.data.vocab import Vocabulary
from src.models import get_model_class
from src.training.trainer import Trainer
from src.experiment.tracker import ExperimentTracker

logger = get_logger("train")


def load_data(config) -> dict:
    """Load preprocessed data splits."""
    data_dir = Path(config.paths.processed_dir)
    
    splits = {}
    for name in ["train", "val", "test"]:
        path = data_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8")
            splits[name] = df
            logger.info(f"Loaded {name}: {len(df)} samples")
        else:
            raise FileNotFoundError(
                f"Data file not found: {path}. "
                f"Run `python scripts/prepare_data.py` first."
            )
    
    return splits


def create_dataloaders(splits: dict, model_name: str, config) -> dict:
    """Create PyTorch DataLoaders for each split."""
    train_cfg = config.training if hasattr(config, "training") else config
    batch_size = getattr(train_cfg, "batch_size", 32)
    max_seq_length = getattr(train_cfg, "max_seq_length", 256)
    if not max_seq_length:
        max_seq_length = getattr(config.data, "max_seq_length", 256)
    
    if model_name in ["bilstm", "bilstm_attention"]:
        # Load vocabulary
        vocab = Vocabulary.load("data/vocab/vocab.json")
        
        datasets = {}
        for name, df in splits.items():
            datasets[name] = SentimentDataset(
                texts=df["text_clean"].tolist(),
                labels=df["label"].tolist(),
                vocab=vocab,
                max_len=max_seq_length,
            )
        
        loaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=SentimentDataset.collate_fn,
                num_workers=0,
                pin_memory=True,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=SentimentDataset.collate_fn,
                num_workers=0,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=SentimentDataset.collate_fn,
                num_workers=0,
            ),
        }
        
        return loaders, vocab
    
    elif model_name == "phobert":
        from transformers import AutoTokenizer
        
        pretrained = config.model.pretrained if hasattr(config.model, "pretrained") else "vinai/phobert-base"
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        
        datasets = {}
        for name, df in splits.items():
            datasets[name] = PhoBERTSentimentDataset(
                texts=df["text_clean"].tolist(),
                labels=df["label"].tolist(),
                tokenizer=tokenizer,
                max_len=max_seq_length,
            )
        
        loaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=PhoBERTSentimentDataset.collate_fn,
                num_workers=0,
                pin_memory=True,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=PhoBERTSentimentDataset.collate_fn,
                num_workers=0,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=PhoBERTSentimentDataset.collate_fn,
                num_workers=0,
            ),
        }
        
        return loaders, tokenizer


def create_model(model_name: str, config, vocab=None):
    """Create model instance from config."""
    ModelClass = get_model_class(model_name)
    model_cfg = config.model if hasattr(config, "model") else config
    
    if model_name == "bilstm":
        model = ModelClass(
            vocab_size=vocab.size,
            embedding_dim=getattr(model_cfg, "embedding_dim", 300),
            hidden_dim=getattr(model_cfg, "hidden_dim", 256),
            num_layers=getattr(model_cfg, "num_layers", 2),
            num_classes=getattr(model_cfg, "num_classes", 2),
            dropout=getattr(model_cfg, "dropout", 0.3),
            lstm_dropout=getattr(model_cfg, "lstm_dropout", 0.2),
            bidirectional=getattr(model_cfg, "bidirectional", True),
            fc_hidden_dim=getattr(model_cfg, "fc_hidden_dim", 128),
            pooling=getattr(model_cfg, "pooling", "max_mean"),
        )
    elif model_name == "bilstm_attention":
        attn_cfg = getattr(model_cfg, "attention", {})
        attn_type = attn_cfg.get("type", "additive") if isinstance(attn_cfg, dict) else getattr(attn_cfg, "type", "additive")
        attn_dim = attn_cfg.get("attention_dim", 128) if isinstance(attn_cfg, dict) else getattr(attn_cfg, "attention_dim", 128)
        
        model = ModelClass(
            vocab_size=vocab.size,
            embedding_dim=getattr(model_cfg, "embedding_dim", 300),
            hidden_dim=getattr(model_cfg, "hidden_dim", 256),
            num_layers=getattr(model_cfg, "num_layers", 2),
            num_classes=getattr(model_cfg, "num_classes", 2),
            dropout=getattr(model_cfg, "dropout", 0.3),
            lstm_dropout=getattr(model_cfg, "lstm_dropout", 0.2),
            bidirectional=getattr(model_cfg, "bidirectional", True),
            fc_hidden_dim=getattr(model_cfg, "fc_hidden_dim", 128),
            attention_type=attn_type,
            attention_dim=attn_dim,
        )
    elif model_name == "phobert":
        model = ModelClass(
            pretrained=getattr(model_cfg, "pretrained", "vinai/phobert-base"),
            num_labels=getattr(model_cfg, "num_labels", 2),
            dropout=getattr(model_cfg, "dropout", 0.1),
            classifier_dropout=getattr(model_cfg, "classifier_dropout", 0.1),
            freeze_encoder_layers=getattr(model_cfg, "freeze_encoder_layers", 0),
            pooling_strategy=getattr(model_cfg, "pooling_strategy", "cls"),
            output_attentions=getattr(model_cfg, "output_attentions", True),
        )
    
    return model


def run_single_training(
    args, config, splits, device, run_idx: int, total_runs: int
) -> dict:
    """
    Execute a single training run with given seed.
    
    Returns:
        Dict with test metrics (accuracy, f1_macro, etc.)
    """
    run_seed = args.seed + run_idx
    set_seed(run_seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  🏃 RUN {run_idx + 1}/{total_runs}  (seed={run_seed})")
    logger.info(f"{'='*60}")
    
    # Create dataloaders (need fresh for each run due to shuffle seed)
    loaders, aux = create_dataloaders(splits, args.model, config)
    
    # Create model (fresh initialization)
    vocab = aux if args.model in ["bilstm", "bilstm_attention"] else None
    model = create_model(args.model, config, vocab=vocab)
    
    # Experiment tracker
    tracker = ExperimentTracker(experiment_dir=str(config.paths.experiment_dir))
    run_name = args.run_name or f"{args.model}_run"
    run_name_full = f"{run_name}_r{run_idx + 1}"
    tracker.start_run(run_name_full, config=config)
    
    # Log hyperparams
    tracker.log_hyperparams({
        "model": args.model,
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "batch_size": config.training.batch_size,
        "seed": run_seed,
        "device": str(device),
        "run_index": run_idx + 1,
        "total_runs": total_runs,
    })
    
    # Train
    trainer = Trainer(
        model=model,
        config=config,
        experiment_tracker=tracker,
        device=device,
    )
    
    history = trainer.train(loaders["train"], loaders["val"])
    
    # Evaluate on test set
    logger.info(f"\n🧪 Evaluating run {run_idx + 1} on test set...")
    test_results = trainer.evaluate(loaders["test"])
    
    # Save test results to experiment
    tracker.log_artifact("test_results", "test_metrics.json")
    
    # Generate training curves
    try:
        from src.experiment.visualizer import ExperimentVisualizer
        viz = ExperimentVisualizer(experiment_dir=str(config.paths.experiment_dir))
        viz.plot_training_curves(tracker.current_run_id or run_name_full)
        
        if test_results.get("confusion_matrix") is not None:
            viz.plot_confusion_matrix(
                test_results["confusion_matrix"],
                save_path=f"experiments/{run_name_full}/plots/confusion_matrix.png",
            )
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return test_results.get("metrics", {})


def main():
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["bilstm", "bilstm_attention", "phobert"],
                        help="Model architecture to train")
    parser.add_argument("--config", type=str, default=None,
                        help="Model config file path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom experiment run name")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of training runs for avg ± std (FR-2.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Set initial seed
    set_seed(args.seed)
    
    # Load config
    base_config_path = PROJECT_ROOT / "configs" / "base.yaml"
    model_config_path = args.config or (PROJECT_ROOT / "configs" / f"model_{args.model}.yaml")
    
    cli_overrides = {}
    if args.epochs is not None:
        cli_overrides["training.epochs"] = args.epochs
    if args.lr is not None:
        cli_overrides["training.learning_rate"] = args.lr
    if args.batch_size is not None:
        cli_overrides["training.batch_size"] = args.batch_size
    
    config = merge_configs(
        base_config_path, model_config_path,
        cli_overrides=cli_overrides if cli_overrides else None,
    )
    
    # Device
    device = get_device(args.device)
    
    # Load data once (shared across runs)
    splits = load_data(config)
    
    num_runs = args.num_runs
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"  Training {args.model.upper()} — {num_runs} run(s)")
    logger.info(f"  Epochs: {config.training.epochs} | LR: {config.training.learning_rate}")
    logger.info(f"  Batch size: {config.training.batch_size} | Device: {device}")
    logger.info(f"{'#'*60}\n")
    
    # ── Multi-run training loop ──────────────────────────────────
    all_metrics = []
    
    for run_idx in range(num_runs):
        metrics = run_single_training(
            args, config, splits, device, run_idx, num_runs
        )
        all_metrics.append(metrics)
        
        logger.info(
            f"  ► Run {run_idx + 1} results: "
            f"Acc={metrics.get('accuracy', 0):.4f} | "
            f"F1={metrics.get('f1_macro', 0):.4f}"
        )
    
    # ── Aggregate results: avg ± std ─────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  📊 TỔNG HỢP KẾT QUẢ ({num_runs} lần chạy)")
    logger.info(f"{'='*60}")
    
    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    summary = {}
    
    for key in metric_keys:
        values = [m.get(key, 0) for m in all_metrics]
        avg = np.mean(values)
        std = np.std(values)
        summary[key] = {"avg": float(avg), "std": float(std), "values": [float(v) for v in values]}
        logger.info(f"  {key:>20s}: {avg:.4f} ± {std:.4f}")
    
    logger.info(f"{'='*60}\n")
    
    # Save aggregated results to JSON
    summary_path = Path("experiments") / f"{args.model}_multi_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "num_runs": num_runs,
            "base_seed": args.seed,
            "config": {
                "epochs": config.training.epochs,
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.batch_size,
            },
            "metrics": summary,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📁 Summary saved to: {summary_path}")
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()

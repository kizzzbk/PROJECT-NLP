"""
Experiment Visualizer
======================
Plot training curves, compare runs, and generate confusion matrix heatmaps.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentVisualizer:
    """
    Generate training visualizations.
    
    Example:
        viz = ExperimentVisualizer('experiments')
        viz.plot_training_curves('run_id_1')
        viz.compare_runs_f1(['run_id_1', 'run_id_2', 'run_id_3'])
        viz.plot_confusion_matrix(cm, labels=['Tiêu cực', 'Tích cực'])
    """

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        
        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def plot_training_curves(
        self,
        run_id: str,
        save: bool = True,
        show: bool = False,
    ) -> Optional[str]:
        """
        Plot loss and F1 curves for a single training run.
        
        Args:
            run_id: Experiment run ID.
            save: Whether to save the plot.
            show: Whether to display the plot.
            
        Returns:
            Path to saved plot (if save=True).
        """
        metrics = self._load_metrics(run_id)
        if not metrics:
            logger.warning(f"No metrics found for run: {run_id}")
            return None
        
        epochs = [m["step"] for m in metrics]
        train_loss = [m.get("train_loss", 0) for m in metrics]
        val_loss = [m.get("val_loss", 0) for m in metrics]
        val_f1 = [m.get("val_f1_macro", 0) for m in metrics]
        val_acc = [m.get("val_accuracy", 0) for m in metrics]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Curves — {run_id}", fontsize=14, fontweight="bold")
        
        # Loss curve
        axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#ef4444")
        axes[0].plot(epochs, val_loss, "s-", label="Val Loss", color="#3b82f6")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # F1 curve
        axes[1].plot(epochs, val_f1, "D-", label="Val F1 (Macro)", color="#10b981")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("F1 Score Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[2].plot(epochs, val_acc, "^-", label="Val Accuracy", color="#8b5cf6")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Accuracy Curve")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = None
        if save:
            plots_dir = self.experiment_dir / run_id / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(plots_dir / "training_curves.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Training curves saved: {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)
        return save_path

    def compare_runs_f1(
        self,
        run_ids: List[str],
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """
        Overlay F1 curves from multiple runs for comparison.
        
        Args:
            run_ids: List of run IDs to compare.
            save_path: Path to save the plot.
            show: Whether to display the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(run_ids)))
        
        for idx, run_id in enumerate(run_ids):
            metrics = self._load_metrics(run_id)
            if not metrics:
                continue
            
            epochs = [m["step"] for m in metrics]
            val_f1 = [m.get("val_f1_macro", 0) for m in metrics]
            
            # Extract model name from run_id
            label = run_id.split("_", 2)[-1] if "_" in run_id else run_id
            ax.plot(epochs, val_f1, "o-", label=label, color=colors[idx], linewidth=2)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Val F1 (Macro)", fontsize=12)
        ax.set_title("Model Comparison — F1 Score", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Comparison plot saved: {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        labels: List[str] = None,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix",
        show: bool = False,
    ):
        """
        Plot a confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array.
            labels: Class label names.
            save_path: Path to save the plot.
            title: Plot title.
            show: Whether to display.
        """
        if labels is None:
            labels = ["Tiêu cực", "Tích cực"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 16},
        )
        
        ax.set_xlabel("Dự đoán (Predicted)", fontsize=12)
        ax.set_ylabel("Thực tế (Actual)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved: {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)

    def _load_metrics(self, run_id: str) -> Optional[list]:
        """Load metrics JSON for a run."""
        path = self.experiment_dir / run_id / "metrics.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

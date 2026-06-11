"""
Run Manager
============
Manage experiment runs: list, compare, filter, delete, and reproduce.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunManager:
    """
    Manage and compare experiment runs.
    
    Example:
        manager = RunManager('experiments')
        
        # List all runs
        runs = manager.list_runs()
        
        # Compare specific runs
        comparison = manager.compare_runs(['run_1', 'run_2'])
        print(comparison.to_markdown())
        
        # Get best run
        best = manager.get_best_run(metric='val_f1')
    """

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)

    def list_runs(self, model_filter: Optional[str] = None) -> pd.DataFrame:
        """
        List all experiment runs with summary info.
        
        Args:
            model_filter: Filter by model name (e.g., 'bilstm').
            
        Returns:
            DataFrame with run summaries.
        """
        runs = []
        
        for run_dir in sorted(self.experiment_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            summary = self._load_json(run_dir / "summary.json")
            if summary is None:
                continue
            
            # Optionally filter by model name
            if model_filter:
                config = self._load_yaml(run_dir / "config.yaml")
                if config:
                    model_name = config.get("model", {}).get("name", "")
                    if model_filter.lower() not in model_name.lower():
                        continue
            
            # Extract key info
            final = summary.get("final_metrics", {})
            run_info = {
                "run_id": run_dir.name,
                "status": summary.get("status", "unknown"),
                "duration": summary.get("duration_readable", "?"),
                "best_val_f1": final.get("best_val_f1", None),
                "total_epochs": final.get("total_epochs", None),
            }
            runs.append(run_info)
        
        return pd.DataFrame(runs)

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare.
            
        Returns:
            DataFrame with metrics comparison.
        """
        rows = []
        
        for run_id in run_ids:
            run_dir = self.experiment_dir / run_id
            
            # Load config
            config = self._load_yaml(run_dir / "config.yaml")
            model_name = "?"
            if config:
                model_name = config.get("model", {}).get("name", "?")
            
            # Load final metrics
            summary = self._load_json(run_dir / "summary.json")
            final = summary.get("final_metrics", {}) if summary else {}
            
            # Load last epoch metrics
            metrics = self._load_json(run_dir / "metrics.json")
            last_epoch = metrics[-1] if metrics else {}
            
            row = {
                "run_id": run_id,
                "model": model_name,
                "best_val_f1": final.get("best_val_f1", None),
                "total_epochs": final.get("total_epochs", None),
                "duration": summary.get("duration_readable", "?") if summary else "?",
                "last_val_accuracy": last_epoch.get("val_accuracy", None),
                "last_val_f1_macro": last_epoch.get("val_f1_macro", None),
                "last_train_loss": last_epoch.get("train_loss", None),
                "last_val_loss": last_epoch.get("val_loss", None),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)

    def get_best_run(
        self,
        metric: str = "best_val_f1",
        model_filter: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the run ID with the best value for a given metric.
        
        Args:
            metric: Metric name to optimize.
            model_filter: Optional model name filter.
            
        Returns:
            Best run ID, or None if no runs found.
        """
        df = self.list_runs(model_filter=model_filter)
        if df.empty or metric not in df.columns:
            return None
        
        df_valid = df.dropna(subset=[metric])
        if df_valid.empty:
            return None
        
        best_idx = df_valid[metric].idxmax()
        return df_valid.loc[best_idx, "run_id"]

    def get_run_config(self, run_id: str) -> Optional[dict]:
        """Load the frozen config for a specific run."""
        return self._load_yaml(self.experiment_dir / run_id / "config.yaml")

    def delete_run(self, run_id: str, confirm: bool = False):
        """
        Delete an experiment run directory.
        
        Args:
            run_id: Run ID to delete.
            confirm: Must be True to actually delete.
        """
        run_dir = self.experiment_dir / run_id
        if not run_dir.exists():
            logger.warning(f"Run not found: {run_id}")
            return
        
        if not confirm:
            logger.warning(f"Delete {run_id}? Pass confirm=True to proceed.")
            return
        
        shutil.rmtree(run_dir)
        logger.info(f"Deleted run: {run_id}")

    def export_comparison_report(
        self,
        run_ids: Optional[List[str]] = None,
        output_path: str = "experiments/comparison_report.md",
    ):
        """
        Export a comparison report in Markdown format.
        
        Args:
            run_ids: Specific run IDs to include (None = all runs).
            output_path: Output file path.
        """
        if run_ids is None:
            df = self.list_runs()
            run_ids = df["run_id"].tolist()
        
        if not run_ids:
            logger.warning("No runs to compare.")
            return
        
        comparison = self.compare_runs(run_ids)
        
        report = "# Experiment Comparison Report\n\n"
        report += f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += comparison.to_markdown(index=False)
        report += "\n"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Comparison report saved to {output_path}")

    @staticmethod
    def _load_json(path: Path) -> Optional[dict]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def _load_yaml(path: Path) -> Optional[dict]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return None

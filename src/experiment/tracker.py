"""
Experiment Tracker
===================
Core experiment tracking system for monitoring every training run.
Logs configs, hyperparameters, metrics, and artifacts to structured directories.

Each training run creates:
    experiments/<timestamp>_<run_name>/
    ├── config.yaml          # Frozen config snapshot
    ├── hyperparams.json     # All hyperparameters
    ├── metrics.json         # Per-epoch metrics
    ├── training_log.csv     # Per-step granular log
    ├── system_info.json     # Hardware/software info
    └── summary.json         # Final run summary
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logger import get_logger
from src.utils.device import get_system_info

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Track everything about a training run for reproducibility and comparison.
    
    Usage:
        tracker = ExperimentTracker(experiment_dir='experiments')
        tracker.start_run('bilstm_baseline', config=config)
        
        for epoch in range(epochs):
            # ... training ...
            tracker.log_metrics(epoch, {'train_loss': 0.5, 'val_f1': 0.72})
        
        tracker.end_run({'best_val_f1': 0.85})
        
        # Later: compare runs
        tracker.compare_runs(['run_id_1', 'run_id_2'])
    """

    def __init__(self, experiment_dir: str = "experiments"):
        """
        Args:
            experiment_dir: Base directory for all experiment logs.
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_run_dir: Optional[Path] = None
        self.current_run_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self._metrics_buffer: List[dict] = []
        self._step_log_writer = None
        self._step_log_file = None

    def start_run(
        self,
        run_name: str,
        config: Optional[dict] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new experiment run.
        
        Creates a timestamped directory and logs initial metadata.
        
        Args:
            run_name: Descriptive name for this run.
            config: Configuration dict to snapshot.
            tags: Optional tags for filtering runs.
            
        Returns:
            Run ID string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_run_id = f"{timestamp}_{run_name}"
        self.current_run_dir = self.experiment_dir / self.current_run_id
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self._metrics_buffer = []
        
        # Save config snapshot
        if config:
            config_data = config.to_dict() if hasattr(config, "to_dict") else config
            config_path = self.current_run_dir / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        # Save system info
        sys_info = get_system_info()
        sys_info["run_id"] = self.current_run_id
        sys_info["run_name"] = run_name
        sys_info["start_time"] = timestamp
        sys_info["tags"] = tags or {}
        
        with open(self.current_run_dir / "system_info.json", "w", encoding="utf-8") as f:
            json.dump(sys_info, f, indent=2, ensure_ascii=False)
        
        # Initialize step-level CSV log
        self._init_step_log()
        
        logger.info(f"📝 Experiment started: [bold]{self.current_run_id}[/bold]")
        logger.info(f"   Log dir: {self.current_run_dir}")
        
        return self.current_run_id

    def _init_step_log(self):
        """Initialize CSV writer for step-level logging."""
        log_path = self.current_run_dir / "training_log.csv"
        self._step_log_file = open(log_path, "w", newline="", encoding="utf-8")
        self._step_log_writer = csv.writer(self._step_log_file)
        self._step_log_writer.writerow(["step", "metric_name", "metric_value", "timestamp"])

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a given step/epoch.
        
        Args:
            step: Step or epoch number.
            metrics: Dict of metric names and values.
        """
        if self.current_run_dir is None:
            logger.warning("No active run. Call start_run() first.")
            return
        
        record = {"step": step, **metrics}
        self._metrics_buffer.append(record)
        
        # Also write to step-level CSV
        timestamp = datetime.now().isoformat()
        for name, value in metrics.items():
            if self._step_log_writer:
                self._step_log_writer.writerow([step, name, value, timestamp])
        
        # Flush periodically
        if self._step_log_file:
            self._step_log_file.flush()
        
        # Save metrics JSON (overwrite with full history)
        self._save_metrics()

    def log_step(self, global_step: int, metrics: Dict[str, float]):
        """
        Log step-level metrics (more granular than epoch-level).
        
        Only writes to CSV, not to the metrics JSON buffer.
        """
        if self._step_log_writer is None:
            return
        
        timestamp = datetime.now().isoformat()
        for name, value in metrics.items():
            self._step_log_writer.writerow([global_step, name, value, timestamp])
        
        if self._step_log_file:
            self._step_log_file.flush()

    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Log hyperparameters for this run.
        
        Args:
            params: Dict of hyperparameter names and values.
        """
        if self.current_run_dir is None:
            return
        
        with open(self.current_run_dir / "hyperparams.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False, default=str)

    def log_artifact(self, name: str, source_path: str):
        """
        Record an artifact (model file, plot, etc.) associated with this run.
        
        Args:
            name: Artifact name.
            source_path: Path to the artifact file.
        """
        if self.current_run_dir is None:
            return
        
        artifacts_file = self.current_run_dir / "artifacts.json"
        artifacts = {}
        if artifacts_file.exists():
            with open(artifacts_file, "r") as f:
                artifacts = json.load(f)
        
        artifacts[name] = {
            "path": str(source_path),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(artifacts_file, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)

    def end_run(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        Finalize the current run.
        
        Args:
            final_metrics: Final summary metrics.
        """
        if self.current_run_dir is None:
            return
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "run_id": self.current_run_id,
            "status": "completed",
            "duration_seconds": round(elapsed, 2),
            "duration_readable": f"{elapsed / 60:.1f} minutes",
            "end_time": datetime.now().isoformat(),
        }
        
        if final_metrics:
            summary["final_metrics"] = final_metrics
        
        with open(self.current_run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Close CSV log
        if self._step_log_file:
            self._step_log_file.close()
            self._step_log_file = None
            self._step_log_writer = None
        
        logger.info(
            f"✅ Experiment completed: [bold]{self.current_run_id}[/bold] "
            f"({elapsed / 60:.1f} min)"
        )
        
        self.current_run_dir = None
        self.current_run_id = None

    def _save_metrics(self):
        """Save metrics buffer to JSON file."""
        if self.current_run_dir and self._metrics_buffer:
            with open(self.current_run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(self._metrics_buffer, f, indent=2)

    def list_runs(self) -> List[str]:
        """List all experiment run IDs."""
        runs = []
        for d in sorted(self.experiment_dir.iterdir()):
            if d.is_dir() and (d / "summary.json").exists():
                runs.append(d.name)
        return runs

    def get_run_summary(self, run_id: str) -> Optional[dict]:
        """Load summary for a specific run."""
        summary_path = self.experiment_dir / run_id / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return json.load(f)
        return None

    def get_run_metrics(self, run_id: str) -> Optional[List[dict]]:
        """Load epoch-level metrics for a specific run."""
        metrics_path = self.experiment_dir / run_id / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)
        return None

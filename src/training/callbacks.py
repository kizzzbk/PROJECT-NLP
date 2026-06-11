"""
Training Callbacks
===================
Hook-based callback system for training loop events.
"""

from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when a monitored metric stops improving.
    
    Example:
        early_stop = EarlyStopping(patience=5, monitor='val_f1', mode='max')
        for epoch in range(100):
            ...
            if early_stop(val_metrics['val_f1']):
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "val_f1",
        mode: str = "max",
    ):
        """
        Args:
            patience: Number of epochs to wait after last improvement.
            min_delta: Minimum change to qualify as an improvement.
            monitor: Metric name being monitored.
            mode: 'max' (higher is better) or 'min' (lower is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, current_value: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value.
            epoch: Current epoch number.
            
        Returns:
            True if training should stop.
        """
        improved = False
        
        if self.mode == "max":
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping: {self.monitor} has not improved for "
                    f"{self.patience} epochs. Best: {self.best_value:.4f} at epoch {self.best_epoch}"
                )
                return True
            return False

    def reset(self):
        """Reset the early stopping counter."""
        self.best_value = float("-inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints based on metric performance.
    
    Keeps track of top-K best models and removes inferior ones.
    
    Example:
        ckpt = ModelCheckpoint(save_dir='models/bilstm', save_top_k=3)
        ckpt(model, val_f1, epoch)  # Saves if val_f1 is in top-3
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_f1",
        mode: str = "max",
        save_top_k: int = 3,
        save_last: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints.
            monitor: Metric to monitor.
            mode: 'max' or 'min'.
            save_top_k: Number of best checkpoints to keep.
            save_last: Whether to always save the last epoch.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        
        # Track saved checkpoints: list of (metric_value, path)
        self.saved: list = []
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_path: Optional[str] = None

    def __call__(
        self,
        model,
        metric_value: float,
        epoch: int,
        optimizer_state: dict = None,
        config: dict = None,
    ) -> bool:
        """
        Potentially save a checkpoint.
        
        Args:
            model: The model to save.
            metric_value: Current metric value.
            epoch: Current epoch.
            optimizer_state: Optimizer state dict.
            config: Config dict.
            
        Returns:
            True if a new best model was saved.
        """
        is_best = False
        
        if self.mode == "max":
            is_best = metric_value > self.best_value
        else:
            is_best = metric_value < self.best_value
        
        # Check if this checkpoint should be saved (top-K)
        should_save = False
        if len(self.saved) < self.save_top_k:
            should_save = True
        else:
            # Check if better than worst saved
            worst_idx = self._get_worst_idx()
            worst_value = self.saved[worst_idx][0]
            
            if self.mode == "max" and metric_value > worst_value:
                should_save = True
            elif self.mode == "min" and metric_value < worst_value:
                should_save = True
        
        if should_save:
            # Save checkpoint
            filename = f"checkpoint_epoch{epoch}_{self.monitor}{metric_value:.4f}.pt"
            save_path = str(self.save_dir / filename)
            
            metrics = {self.monitor: metric_value}
            model.save_checkpoint(
                save_path, epoch, optimizer_state, metrics, config
            )
            
            self.saved.append((metric_value, save_path))
            
            # Remove worst if over limit
            if len(self.saved) > self.save_top_k:
                worst_idx = self._get_worst_idx()
                worst_path = Path(self.saved[worst_idx][1])
                if worst_path.exists():
                    worst_path.unlink()
                self.saved.pop(worst_idx)
        
        # Update best
        if is_best:
            self.best_value = metric_value
            best_path = str(self.save_dir / "best_model.pt")
            model.save_checkpoint(
                best_path, epoch, optimizer_state, {self.monitor: metric_value}, config
            )
            self.best_path = best_path
        
        # Save last
        if self.save_last:
            last_path = str(self.save_dir / "last_model.pt")
            model.save_checkpoint(
                last_path, epoch, optimizer_state, {self.monitor: metric_value}, config
            )
        
        return is_best

    def _get_worst_idx(self) -> int:
        """Get index of the worst checkpoint."""
        if self.mode == "max":
            return min(range(len(self.saved)), key=lambda i: self.saved[i][0])
        else:
            return max(range(len(self.saved)), key=lambda i: self.saved[i][0])

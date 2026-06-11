"""
Logging Setup
=============
Structured logging with Rich (beautiful terminal output) and optional file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console


# Global console for rich output
console = Console()

# Cache loggers to avoid duplicate handlers
_loggers = {}


def get_logger(
    name: str = "brandhealth",
    level: str = "INFO",
    log_to_file: bool = False,
    log_dir: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically module name).
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_to_file: Whether to also log to a file.
        log_dir: Directory for log files.
        use_rich: Whether to use Rich handler for beautiful terminal output.
        
    Returns:
        Configured logging.Logger instance.
        
    Example:
        logger = get_logger(__name__)
        logger.info("Training started", extra={"model": "bilstm"})
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich handler (beautiful terminal output)
    if use_rich:
        rich_handler = RichHandler(
            console=console,
            show_path=True,
            show_time=True,
            rich_tracebacks=True,
            markup=True,
        )
        rich_handler.setLevel(getattr(logging, level.upper()))
        rich_fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        rich_handler.setFormatter(rich_fmt)
        logger.addHandler(rich_handler)
    else:
        # Standard handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(getattr(logging, level.upper()))
        std_fmt = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(std_fmt)
        logger.addHandler(stream_handler)
    
    # File handler (optional)
    if log_to_file:
        log_dir = Path(log_dir or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{name}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


class TrainingLogger:
    """
    Specialized logger for training progress with rich formatting.
    
    Provides methods for logging epoch summaries, metric tables,
    and progress updates in a visually appealing format.
    """

    def __init__(self, name: str = "training"):
        self.logger = get_logger(name)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        self.logger.info(
            f"[bold cyan]━━━ Epoch {epoch}/{total_epochs} ━━━[/bold cyan]"
        )

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict,
        lr: float,
    ):
        metrics_str = " | ".join(
            f"{k}: [bold]{v:.4f}[/bold]" for k, v in val_metrics.items()
        )
        self.logger.info(
            f"  Train Loss: [yellow]{train_loss:.4f}[/yellow] | "
            f"Val Loss: [yellow]{val_loss:.4f}[/yellow] | "
            f"{metrics_str} | "
            f"LR: {lr:.2e}"
        )

    def log_best_model(self, metric_name: str, metric_value: float, path: str):
        self.logger.info(
            f"  [bold green]✓ New best model![/bold green] "
            f"{metric_name}={metric_value:.4f} → Saved to {path}"
        )

    def log_early_stop(self, patience: int):
        self.logger.warning(
            f"  [bold red]✗ Early stopping triggered[/bold red] "
            f"(no improvement for {patience} epochs)"
        )

    def log_training_complete(self, best_metric: float, total_time: float):
        minutes = total_time / 60
        self.logger.info(
            f"\n[bold green]{'═' * 50}[/bold green]\n"
            f"  Training Complete!\n"
            f"  Best F1: [bold]{best_metric:.4f}[/bold]\n"
            f"  Total Time: [bold]{minutes:.1f} minutes[/bold]\n"
            f"[bold green]{'═' * 50}[/bold green]"
        )

"""
Base Model Interface
=====================
Abstract base class that all sentiment models must implement.
Ensures a uniform interface for training, inference, and dashboard integration.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSentimentModel(nn.Module, ABC):
    """
    Abstract base class for all sentiment classification models.
    
    All models (BiLSTM, BiLSTM+Attention, PhoBERT) must implement this 
    interface so the dashboard can swap between them without code changes.
    
    Required methods:
        - forward(): Model forward pass
        - get_name(): Human-readable model name
        - count_parameters(): Total trainable parameters
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Model forward pass.
        
        Returns:
            Dict with at least:
                - 'logits': (batch_size, num_classes) classification logits
                - 'attention_weights': (batch_size, seq_len) attention weights (optional)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return the human-readable model name."""
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        """Count all parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(
        self,
        path: str,
        epoch: int = 0,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        """
        Save model checkpoint with training metadata.
        
        Args:
            path: File path to save the checkpoint.
            epoch: Current epoch number.
            optimizer_state: Optimizer state dict (for resume training).
            metrics: Current metrics dict.
            config: Model/training configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_name": self.get_name(),
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "config": config,
        }
        
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str, device: str = "cpu") -> dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            device: Device to load the checkpoint to.
            
        Returns:
            Checkpoint dict with metadata (epoch, metrics, etc.).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint: {path} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )
        return checkpoint

    def summary(self) -> str:
        """Return a summary string of the model."""
        trainable = self.count_parameters()
        total = self.count_all_parameters()
        frozen = total - trainable
        
        lines = [
            f"Model: {self.get_name()}",
            f"  Trainable params: {trainable:,}",
            f"  Frozen params:    {frozen:,}",
            f"  Total params:     {total:,}",
        ]
        return "\n".join(lines)

"""
Model Registry
===============
Scan, validate, and manage saved model checkpoints.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for saved model checkpoints.
    
    Scans the models/ directory for available checkpoints and provides
    metadata about each (architecture, training date, metrics).
    
    Example:
        registry = ModelRegistry('models')
        available = registry.list_models()
        best_bilstm = registry.get_best_checkpoint('bilstm')
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)

    def list_models(self) -> List[Dict]:
        """
        List all available model checkpoints.
        
        Returns:
            List of dicts with model metadata.
        """
        models = []
        
        for model_type_dir in sorted(self.model_dir.iterdir()):
            if not model_type_dir.is_dir():
                continue
            
            model_type = model_type_dir.name
            
            for ckpt_path in sorted(model_type_dir.glob("*.pt")):
                try:
                    meta = self._get_checkpoint_meta(ckpt_path)
                    meta["model_type"] = model_type
                    meta["path"] = str(ckpt_path)
                    meta["filename"] = ckpt_path.name
                    meta["size_mb"] = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
                    models.append(meta)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
        
        return models

    def get_best_checkpoint(self, model_type: str) -> Optional[str]:
        """
        Get the path to the best checkpoint for a model type.
        
        Args:
            model_type: Model directory name (e.g., 'bilstm').
            
        Returns:
            Path to best_model.pt, or None if not found.
        """
        best_path = self.model_dir / model_type / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        
        # Fallback: look for any checkpoint
        model_dir = self.model_dir / model_type
        if model_dir.exists():
            checkpoints = list(model_dir.glob("*.pt"))
            if checkpoints:
                return str(checkpoints[-1])  # Latest
        
        return None

    def is_available(self, model_type: str) -> bool:
        """Check if any checkpoint exists for a model type."""
        return self.get_best_checkpoint(model_type) is not None

    @staticmethod
    def _get_checkpoint_meta(path: Path) -> dict:
        """Extract metadata from a checkpoint without loading the full model."""
        # Load only metadata (map to CPU, don't load full tensors)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        return {
            "model_name": checkpoint.get("model_name", "unknown"),
            "epoch": checkpoint.get("epoch", "?"),
            "metrics": checkpoint.get("metrics", {}),
        }

    def get_available_model_types(self) -> List[str]:
        """Get list of model types with available checkpoints."""
        available = []
        for name in ["bilstm", "bilstm_attention", "phobert"]:
            if self.is_available(name):
                available.append(name)
        return available

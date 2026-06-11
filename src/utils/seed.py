"""
Reproducibility Utilities
=========================
Set random seeds across all libraries for deterministic behavior.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value.
        
    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU + CUDA)
        - CUDA deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        
        # Deterministic behavior (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

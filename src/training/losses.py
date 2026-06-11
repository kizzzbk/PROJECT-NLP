"""
Custom Loss Functions
======================
Extensible loss functions for sentiment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses training on hard-to-classify examples by down-weighting
    easy (well-classified) examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma=0, equivalent to standard cross-entropy.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) raw logits.
            targets: (batch_size,) integer class labels.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Prevents the model from becoming overconfident by softening
    the target distribution.
    """

    def __init__(self, num_classes: int = 2, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


def get_loss_function(name: str = "cross_entropy", **kwargs) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        name: Loss function name ('cross_entropy', 'focal', 'label_smoothing').
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        PyTorch loss module.
    """
    losses = {
        "cross_entropy": nn.CrossEntropyLoss,
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingLoss,
    }
    
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name](**kwargs)

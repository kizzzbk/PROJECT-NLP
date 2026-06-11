"""
Metrics Calculator
===================
Compute classification metrics for sentiment analysis evaluation.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


class MetricsCalculator:
    """
    Compute standard classification metrics.
    
    Metrics computed:
      - Accuracy
      - F1-score (macro, weighted, per-class)
      - Precision (macro)
      - Recall (macro)
      - Confusion Matrix
    
    Example:
        calc = MetricsCalculator(label_names={0: 'Tiêu cực', 1: 'Tích cực'})
        metrics = calc.compute(y_true, y_pred)
        print(metrics['f1_macro'])
    """

    def __init__(
        self,
        label_names: Optional[Dict[int, str]] = None,
        average: str = "macro",
    ):
        """
        Args:
            label_names: Mapping from label index to human-readable name.
            average: Averaging strategy for multi-class metrics.
        """
        self.label_names = label_names or {0: "Tiêu cực", 1: "Tích cực"}
        self.average = average

    def compute(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_prob: Prediction probabilities (optional).
            
        Returns:
            Dict with metric names and values.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }
        
        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for idx, f1_val in enumerate(per_class_f1):
            label = self.label_names.get(idx, f"class_{idx}")
            metrics[f"f1_{label}"] = f1_val
        
        return metrics

    def confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            2D numpy array (num_classes x num_classes).
        """
        return confusion_matrix(y_true, y_pred)

    def classification_report(
        self,
        y_true: List[int],
        y_pred: List[int],
    ) -> str:
        """
        Generate sklearn classification report string.
        """
        target_names = [self.label_names.get(i, f"class_{i}") for i in sorted(self.label_names.keys())]
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0,
        )

"""Tests for training infrastructure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.metrics import MetricsCalculator
from src.training.losses import get_loss_function, FocalLoss


class TestEarlyStopping:
    """Test early stopping callback."""

    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3, mode="max")
        
        assert es(0.5, epoch=1) == False
        assert es(0.6, epoch=2) == False
        assert es(0.7, epoch=3) == False

    def test_stop_after_patience(self):
        es = EarlyStopping(patience=3, mode="max")
        
        es(0.8, epoch=1)  # Best
        assert es(0.7, epoch=2) == False  # 1 worse
        assert es(0.6, epoch=3) == False  # 2 worse
        assert es(0.5, epoch=4) == True   # 3 worse → stop

    def test_reset(self):
        es = EarlyStopping(patience=2, mode="max")
        es(0.8, epoch=1)
        es(0.7, epoch=2)
        es.reset()
        assert es.counter == 0

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min")
        es(0.5, epoch=1)  # Best
        assert es(0.6, epoch=2) == False  # Worse
        assert es(0.7, epoch=3) == True   # Stop


class TestMetrics:
    """Test metrics calculator."""

    def test_perfect_predictions(self):
        calc = MetricsCalculator()
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        
        metrics = calc.compute(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_all_wrong_predictions(self):
        calc = MetricsCalculator()
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        
        metrics = calc.compute(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        calc = MetricsCalculator()
        cm = calc.confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
        assert cm.shape == (2, 2)

    def test_classification_report(self):
        calc = MetricsCalculator()
        report = calc.classification_report([0, 1, 1, 0], [0, 1, 0, 0])
        assert isinstance(report, str)
        assert "Tích cực" in report or "Tiêu cực" in report


class TestLossFunctions:
    """Test custom loss functions."""

    def test_get_cross_entropy(self):
        loss_fn = get_loss_function("cross_entropy")
        assert loss_fn is not None

    def test_get_focal_loss(self):
        loss_fn = get_loss_function("focal", gamma=2.0)
        assert isinstance(loss_fn, FocalLoss)

    def test_unknown_loss(self):
        with pytest.raises(ValueError):
            get_loss_function("unknown_loss")

    def test_focal_loss_forward(self):
        import torch
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])
        
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

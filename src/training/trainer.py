"""
Unified Trainer
================
Single Trainer class that handles training for all model architectures
(BiLSTM, BiLSTM+Attention, PhoBERT) with integrated experiment tracking.
"""

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.training.metrics import MetricsCalculator
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import get_loss_function
from src.utils.logger import get_logger, TrainingLogger

logger = get_logger(__name__)
train_logger = TrainingLogger()


class Trainer:
    """
    Unified training loop for all sentiment models.
    
    Features:
      - Train/validate/test loop
      - Mixed precision training (AMP)
      - Gradient accumulation
      - Gradient clipping
      - Learning rate scheduling
      - Early stopping
      - Model checkpointing (top-K)
      - Experiment tracking integration
    
    Example:
        trainer = Trainer(model, config, experiment_tracker)
        history = trainer.train(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        experiment_tracker=None,
        device: torch.device = None,
    ):
        """
        Args:
            model: Sentiment model (BiLSTM/BiLSTM+Attention/PhoBERT).
            config: Training configuration.
            experiment_tracker: ExperimentTracker instance (optional).
            device: Compute device.
        """
        self.model = model
        self.config = config
        self.tracker = experiment_tracker
        self.device = device or torch.device("cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Training config
        train_cfg = config.training if hasattr(config, "training") else config
        
        self.epochs = getattr(train_cfg, "epochs", 30)
        self.lr = getattr(train_cfg, "learning_rate", 0.001)
        self.max_grad_norm = getattr(train_cfg, "max_grad_norm", 5.0)
        self.gradient_accumulation_steps = getattr(train_cfg, "gradient_accumulation_steps", 1)
        self.fp16 = getattr(train_cfg, "fp16", False) and torch.cuda.is_available()
        self.log_every_n_steps = getattr(train_cfg, "log_every_n_steps", 50)
        
        # Loss function
        loss_name = getattr(train_cfg, "loss", "cross_entropy")
        self.criterion = get_loss_function(loss_name)
        
        # Optimizer
        self.optimizer = self._create_optimizer(train_cfg)
        
        # Scheduler
        self.scheduler = self._create_scheduler(train_cfg)
        
        # Callbacks
        model_name = model.get_name().lower().replace(" ", "_").replace("+", "")
        model_dir = f"models/{model_name}"
        
        self.early_stopping = None
        if getattr(train_cfg, "early_stopping", True):
            self.early_stopping = EarlyStopping(
                patience=getattr(train_cfg, "patience", 5),
                monitor=getattr(train_cfg, "monitor", "val_f1"),
                mode=getattr(train_cfg, "mode", "max"),
            )
        
        self.checkpoint = ModelCheckpoint(
            save_dir=model_dir,
            monitor=getattr(train_cfg, "monitor", "val_f1"),
            mode=getattr(train_cfg, "mode", "max"),
            save_top_k=getattr(train_cfg, "save_top_k", 3),
            save_last=getattr(train_cfg, "save_last", True),
        )
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.fp16 else None
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        
        # Log model summary
        logger.info(f"\n{model.summary()}")

    def _create_optimizer(self, train_cfg) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_name = getattr(train_cfg, "optimizer", "adam").lower()
        weight_decay = getattr(train_cfg, "weight_decay", 1e-5)
        
        if opt_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        elif opt_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _create_scheduler(self, train_cfg):
        """Create learning rate scheduler from config."""
        sched_name = getattr(train_cfg, "scheduler", None)
        
        if sched_name is None or sched_name == "none":
            return None
        
        if sched_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=getattr(train_cfg, "mode", "max"),
                patience=getattr(train_cfg, "scheduler_patience", 3),
                factor=getattr(train_cfg, "scheduler_factor", 0.5),
                min_lr=getattr(train_cfg, "min_lr", 1e-6),
                verbose=True,
            )
        elif sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        elif sched_name == "linear_warmup":
            # Linear warmup then linear decay
            warmup_ratio = getattr(train_cfg, "warmup_ratio", 0.1)
            # We'll set this properly after we know total steps
            return None  # Will be created in train()
        else:
            logger.warning(f"Unknown scheduler: {sched_name}, using none")
            return None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            
        Returns:
            Training history dict.
        """
        # Create linear warmup scheduler if needed
        train_cfg = self.config.training if hasattr(self.config, "training") else self.config
        sched_name = getattr(train_cfg, "scheduler", None)
        
        if sched_name == "linear_warmup":
            total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
            warmup_steps = int(total_steps * getattr(train_cfg, "warmup_ratio", 0.1))
            
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
                )
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        logger.info(
            f"Starting training: {self.epochs} epochs | "
            f"Train batches: {len(train_loader)} | "
            f"FP16: {self.fp16}"
        )
        
        start_time = time.time()
        global_step = 0
        best_val_f1 = 0.0
        
        for epoch in range(1, self.epochs + 1):
            train_logger.log_epoch_start(epoch, self.epochs)
            
            # Train one epoch
            train_loss, global_step = self._train_one_epoch(
                train_loader, epoch, global_step
            )
            
            # Validate
            val_loss, val_metrics = self._validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log epoch results
            train_logger.log_epoch_end(
                epoch, train_loss, val_loss, val_metrics, current_lr
            )
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_metrics.get("f1_macro", 0))
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0))
            self.history["learning_rate"].append(current_lr)
            
            # Experiment tracking
            if self.tracker:
                self.tracker.log_metrics(
                    step=epoch,
                    metrics={
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    },
                )
            
            # Model checkpointing
            monitor_value = val_metrics.get("f1_macro", 0)
            is_best = self.checkpoint(
                self.model, monitor_value, epoch,
                optimizer_state=self.optimizer.state_dict(),
                config=self.config.to_dict() if hasattr(self.config, "to_dict") else None,
            )
            
            if is_best:
                best_val_f1 = monitor_value
                train_logger.log_best_model("val_f1", monitor_value, self.checkpoint.best_path)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitor_value)
                elif not sched_name == "linear_warmup":
                    # Step-based schedulers step per epoch
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(monitor_value, epoch):
                    train_logger.log_early_stop(self.early_stopping.patience)
                    break
        
        total_time = time.time() - start_time
        train_logger.log_training_complete(best_val_f1, total_time)
        
        # Log final results to tracker
        if self.tracker:
            self.tracker.end_run({
                "best_val_f1": best_val_f1,
                "total_epochs": epoch,
                "total_time_seconds": total_time,
            })
        
        return self.history

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        global_step: int,
    ) -> Tuple[float, int]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, global_step).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass (with optional mixed precision)
            if self.fp16:
                with autocast():
                    outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                    loss = self.criterion(outputs["logits"], batch["labels"])
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = self.criterion(outputs["logits"], batch["labels"])
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                global_step += 1
                
                # Step-level LR scheduling (for warmup schedulers)
                train_cfg = self.config.training if hasattr(self.config, "training") else self.config
                if getattr(train_cfg, "scheduler", None) == "linear_warmup" and self.scheduler:
                    self.scheduler.step()
                
                # Log step-level metrics
                if global_step % self.log_every_n_steps == 0 and self.tracker:
                    self.tracker.log_step(
                        global_step,
                        {"step_loss": loss.item() * self.gradient_accumulation_steps},
                    )
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, global_step

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, dict]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = self.criterion(outputs["logits"], batch["labels"])
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
        
        avg_loss = total_loss / max(len(val_loader), 1)
        metrics = self.metrics_calc.compute(all_labels, all_preds)
        
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate model on test set with full report.
        
        Returns:
            Dict with metrics and classification report.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in test_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
        
        metrics = self.metrics_calc.compute(all_labels, all_preds)
        cm = self.metrics_calc.confusion_matrix(all_labels, all_preds)
        report = self.metrics_calc.classification_report(all_labels, all_preds)
        
        logger.info(f"\n📊 Test Results:\n{report}")
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "predictions": all_preds,
            "probabilities": all_probs,
        }

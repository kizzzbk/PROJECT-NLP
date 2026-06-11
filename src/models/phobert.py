"""
PhoBERT Fine-tuned Model
==========================
PhoBERT (vinai/phobert-base) fine-tuned for Vietnamese sentiment classification.

Architecture:
    PhoBERT Encoder → [CLS] token → Dropout → Linear → Output
    
PhoBERT is pre-trained on Vietnamese text and achieves SOTA on many Vietnamese NLP tasks.
It requires word-segmented input (using VnCoreNLP or underthesea).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.base_model import BaseSentimentModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhoBERTModel(BaseSentimentModel):
    """
    PhoBERT fine-tuned for binary sentiment classification.
    
    Supports:
      - Full fine-tuning or partial freezing of encoder layers
      - Mixed precision training (fp16)
      - Attention weight extraction for visualization
      - Multiple pooling strategies (CLS, mean, max)
    """

    def __init__(
        self,
        pretrained: str = "vinai/phobert-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        classifier_dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
        pooling_strategy: str = "cls",
        output_attentions: bool = True,
    ):
        """
        Args:
            pretrained: HuggingFace model name or path.
            num_labels: Number of classification labels.
            dropout: General dropout rate.
            classifier_dropout: Dropout before classifier.
            freeze_encoder_layers: Number of encoder layers to freeze (0=full fine-tune).
            pooling_strategy: "cls", "mean", or "max".
            output_attentions: Whether to output attention weights.
        """
        super().__init__()
        
        self.pretrained_name = pretrained
        self.pooling_strategy = pooling_strategy
        self.output_attentions = output_attentions
        
        # Load pre-trained PhoBERT
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(
                pretrained,
                output_attentions=output_attentions,
            )
        except Exception as e:
            logger.error(
                f"Failed to load PhoBERT model '{pretrained}'. "
                f"Make sure transformers is installed and you have internet access. "
                f"Error: {e}"
            )
            raise
        
        # Get hidden size from config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder layers if specified
        if freeze_encoder_layers > 0:
            self._freeze_layers(freeze_encoder_layers)
        
        # Classification head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        logger.info(
            f"PhoBERT model initialized: {pretrained} | "
            f"Hidden: {self.hidden_size} | "
            f"Frozen layers: {freeze_encoder_layers} | "
            f"Pooling: {pooling_strategy}"
        )

    def _freeze_layers(self, n_layers: int):
        """
        Freeze the first N encoder layers.
        
        This is useful for reducing memory usage and training time
        when fine-tuning on small datasets.
        """
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified encoder layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        frozen_params = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.encoder.parameters())
        logger.info(f"Frozen {frozen_params}/{total_params} encoder parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs from PhoBERT tokenizer.
            attention_mask: (batch_size, seq_len) attention mask (1=real, 0=pad).
            
        Returns:
            Dict with:
                - 'logits': (batch_size, num_labels)
                - 'attention_weights': (batch_size, num_heads, seq_len, seq_len) or None
        """
        # Encode with PhoBERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
        )
        
        # Pooling
        if self.pooling_strategy == "cls":
            # Use [CLS] token representation (first token)
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            last_hidden = outputs.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = last_hidden.mean(dim=1)
        elif self.pooling_strategy == "max":
            last_hidden = outputs.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).bool()
                last_hidden = last_hidden.masked_fill(~mask, float("-inf"))
            pooled = last_hidden.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        # Extract attention weights (last layer, averaged over heads)
        attention_weights = None
        if self.output_attentions and outputs.attentions:
            # outputs.attentions is a tuple of (num_layers) tensors
            # Each: (batch_size, num_heads, seq_len, seq_len)
            # Use last layer, average over heads, take CLS row
            last_layer_attn = outputs.attentions[-1]  # (batch, heads, seq, seq)
            # Average over heads: (batch, seq, seq)
            avg_attn = last_layer_attn.mean(dim=1)
            # Take CLS token's attention: (batch, seq)
            attention_weights = avg_attn[:, 0, :]
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
        }

    def get_name(self) -> str:
        return "PhoBERT Fine-tuned"

    def save_checkpoint(self, path, epoch=0, optimizer_state=None, metrics=None, config=None):
        """Override to also save tokenizer info."""
        import os
        from pathlib import Path as P
        
        save_dir = P(path).parent
        
        # Save model state
        super().save_checkpoint(path, epoch, optimizer_state, metrics, config)
        
        # Save pretrained name for easy reload
        meta_path = save_dir / "model_meta.json"
        import json
        with open(meta_path, "w") as f:
            json.dump({"pretrained": self.pretrained_name}, f)

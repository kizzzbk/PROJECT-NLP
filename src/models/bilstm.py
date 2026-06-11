"""
BiLSTM Model
==============
Bidirectional LSTM baseline for Vietnamese sentiment classification.

Architecture:
    Embedding → BiLSTM (N layers) → Pooling (max+mean) → FC → Output
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.base_model import BaseSentimentModel
from src.data.vocab import PAD_IDX


class BiLSTMModel(BaseSentimentModel):
    """
    Bidirectional LSTM model for binary sentiment classification.
    
    Supports:
      - Pre-trained word embeddings (Word2Vec, FastText)
      - Multiple pooling strategies (max, mean, max+mean, last_hidden)
      - Configurable depth and width
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        bidirectional: bool = True,
        fc_hidden_dim: int = 128,
        pooling: str = "max_mean",
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embedding: bool = False,
        padding_idx: int = PAD_IDX,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of word embeddings.
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout rate for FC layers.
            lstm_dropout: Dropout between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
            fc_hidden_dim: Hidden dimension of the classifier head.
            pooling: Pooling strategy ('max', 'mean', 'max_mean', 'last_hidden').
            pretrained_embeddings: Pre-trained embedding matrix (optional).
            freeze_embedding: Whether to freeze embeddings during training.
            padding_idx: Index of the padding token.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )
            if freeze_embedding:
                self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0,
        )
        
        # Classifier head
        lstm_output_dim = hidden_dim * self.num_directions
        if pooling == "max_mean":
            classifier_input_dim = lstm_output_dim * 2  # concat max + mean
        else:
            classifier_input_dim = lstm_output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) token index tensor.
            lengths: (batch_size,) actual sequence lengths.
            
        Returns:
            Dict with 'logits' and optionally 'attention_weights' (None for BiLSTM).
        """
        # Embedding
        embedded = self.embedding(input_ids)          # (batch, seq_len, emb_dim)
        embedded = self.embedding_dropout(embedded)
        
        # Pack padded sequences for efficient LSTM computation
        if lengths is not None:
            # Clamp lengths to be at least 1
            lengths_clamped = lengths.clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_clamped, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Pooling
        pooled = self._pool(lstm_out, lengths)
        
        # Classification
        logits = self.classifier(pooled)
        
        return {
            "logits": logits,
            "attention_weights": None,  # No attention for base BiLSTM
        }

    def _pool(
        self,
        lstm_out: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Apply pooling strategy to LSTM outputs.
        
        Args:
            lstm_out: (batch_size, seq_len, lstm_output_dim)
            lengths: (batch_size,) actual lengths
            
        Returns:
            Pooled representation (batch_size, pooled_dim)
        """
        if self.pooling == "last_hidden":
            if self.bidirectional:
                # Concatenate last hidden states from both directions
                # hidden shape: (num_layers * num_directions, batch, hidden_dim)
                # We want the last layer's forward and backward
                pass
            # Use last time step for each sequence
            if lengths is not None:
                idx = (lengths - 1).clamp(min=0).long()
                pooled = lstm_out[torch.arange(lstm_out.size(0)), idx]
            else:
                pooled = lstm_out[:, -1, :]
                
        elif self.pooling == "mean":
            if lengths is not None:
                mask = self._create_mask(lstm_out, lengths)
                pooled = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
            else:
                pooled = lstm_out.mean(dim=1)
                
        elif self.pooling == "max":
            if lengths is not None:
                mask = self._create_mask(lstm_out, lengths)
                lstm_out = lstm_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = lstm_out.max(dim=1)[0]
            
        elif self.pooling == "max_mean":
            # Concatenate max and mean pooling
            if lengths is not None:
                mask = self._create_mask(lstm_out, lengths)
                # Mean
                mean_pool = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
                # Max
                masked_out = lstm_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                max_pool = masked_out.max(dim=1)[0]
            else:
                mean_pool = lstm_out.mean(dim=1)
                max_pool = lstm_out.max(dim=1)[0]
            
            pooled = torch.cat([max_pool, mean_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return pooled

    @staticmethod
    def _create_mask(lstm_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create boolean mask from sequence lengths."""
        batch_size, max_len, _ = lstm_out.size()
        mask = torch.arange(max_len, device=lstm_out.device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask

    def get_name(self) -> str:
        return "BiLSTM"

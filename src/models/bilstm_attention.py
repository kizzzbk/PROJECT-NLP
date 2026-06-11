"""
BiLSTM + Attention Model
=========================
BiLSTM with Attention mechanism for explainable sentiment classification.
Exports attention weights for heatmap visualization (FR-2.2).

Architecture:
    Embedding → BiLSTM → Attention Layer → Context Vector → FC → Output
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.base_model import BaseSentimentModel
from src.models.attention import AdditiveAttention, DotProductAttention
from src.data.vocab import PAD_IDX


class BiLSTMAttentionModel(BaseSentimentModel):
    """
    BiLSTM with Attention mechanism.
    
    Instead of pooling, uses an attention layer to learn which parts
    of the sequence are most relevant for classification.
    
    The attention weights are exported for visualization:
        - Attention Heatmap shows which words the model focuses on
        - Words like "tệ", "chờ lâu" will have higher weights in negative reviews
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
        attention_type: str = "additive",
        attention_dim: int = 128,
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
            dropout: Dropout rate.
            lstm_dropout: Dropout between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
            fc_hidden_dim: Hidden dimension of classifier head.
            attention_type: "additive" (Bahdanau) or "dot".
            attention_dim: Internal attention dimension (additive only).
            pretrained_embeddings: Pre-trained embedding matrix.
            freeze_embedding: Whether to freeze embeddings.
            padding_idx: Padding token index.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
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
        
        # Attention layer
        lstm_output_dim = hidden_dim * self.num_directions
        
        if attention_type == "additive":
            self.attention = AdditiveAttention(lstm_output_dim, attention_dim)
        elif attention_type == "dot":
            self.attention = DotProductAttention(lstm_output_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_hidden_dim),
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
        Forward pass with attention.
        
        Args:
            input_ids: (batch_size, seq_len) token indices.
            lengths: (batch_size,) actual sequence lengths.
            
        Returns:
            Dict with:
                - 'logits': (batch_size, num_classes)
                - 'attention_weights': (batch_size, seq_len) — for heatmap
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        
        # BiLSTM
        if lengths is not None:
            lengths_clamped = lengths.clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_clamped, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(embedded)
        
        # Create attention mask (True for real tokens, False for padding)
        if lengths is not None:
            mask = torch.arange(lstm_out.size(1), device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None
        
        # Attention
        context, attention_weights = self.attention(lstm_out, mask)
        
        # Classification
        logits = self.classifier(context)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
        }

    def get_name(self) -> str:
        return "BiLSTM + Attention"

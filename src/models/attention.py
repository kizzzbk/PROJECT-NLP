"""
Attention Mechanism Module
===========================
Standalone attention implementations for use with LSTM-based models.
Exports attention weights for heatmap visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) Attention Mechanism.
    
    Computes attention weights over LSTM hidden states and returns
    a context vector (weighted sum) plus the attention weight distribution.
    
    Architecture:
        score(h_i) = v^T · tanh(W · h_i + b)
        alpha_i = softmax(score(h_i))
        context = sum(alpha_i * h_i)
    
    The attention weights can be visualized as a heatmap to understand
    which words the model focuses on for its prediction (FR-2.2).
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        """
        Args:
            hidden_dim: Dimension of LSTM hidden states.
                        For BiLSTM: hidden_dim = 2 * lstm_hidden_dim
            attention_dim: Internal dimension of the attention scoring network.
        """
        super().__init__()
        
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Compute attention-weighted context vector.
        
        Args:
            hidden_states: LSTM outputs, shape (batch_size, seq_len, hidden_dim).
            mask: Boolean mask, shape (batch_size, seq_len).
                  True for real tokens, False for padding.
                  
        Returns:
            Tuple of:
                - context: Weighted sum, shape (batch_size, hidden_dim).
                - weights: Attention distribution, shape (batch_size, seq_len).
        """
        # Compute attention scores
        # (batch, seq_len, hidden_dim) → (batch, seq_len, attention_dim)
        energy = torch.tanh(self.W(hidden_states))
        
        # (batch, seq_len, attention_dim) → (batch, seq_len, 1) → (batch, seq_len)
        scores = self.v(energy).squeeze(-1)
        
        # Mask padding positions (set to -inf before softmax)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        # Normalize to get attention weights
        weights = F.softmax(scores, dim=-1)
        
        # Handle edge case: all positions masked → uniform weights
        if mask is not None:
            nan_mask = torch.isnan(weights)
            if nan_mask.any():
                weights = weights.masked_fill(nan_mask, 0.0)
        
        # Compute weighted sum (context vector)
        # (batch, 1, seq_len) × (batch, seq_len, hidden_dim) → (batch, 1, hidden_dim) → (batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return context, weights


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (simplified version for LSTM outputs).
    
    Uses a learnable query vector to compute attention over hidden states.
    
    Architecture:
        score(h_i) = q^T · h_i / sqrt(d)
        alpha_i = softmax(score(h_i))
        context = sum(alpha_i * h_i)
    """

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of LSTM hidden states.
        """
        super().__init__()
        
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len), True for real tokens
            
        Returns:
            context: (batch_size, hidden_dim)
            weights: (batch_size, seq_len)
        """
        # (batch, seq_len, hidden_dim) · (hidden_dim,) → (batch, seq_len)
        scores = torch.matmul(hidden_states, self.query) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        
        if mask is not None:
            nan_mask = torch.isnan(weights)
            if nan_mask.any():
                weights = weights.masked_fill(nan_mask, 0.0)
        
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return context, weights

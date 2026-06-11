"""Tests for model architectures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from src.models.bilstm import BiLSTMModel
from src.models.bilstm_attention import BiLSTMAttentionModel
from src.models.attention import AdditiveAttention, DotProductAttention


VOCAB_SIZE = 1000
BATCH_SIZE = 4
SEQ_LEN = 20


class TestBiLSTM:
    """Test BiLSTM model."""

    def test_forward_shape(self):
        model = BiLSTMModel(vocab_size=VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE)
        
        outputs = model(input_ids=input_ids, lengths=lengths)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (BATCH_SIZE, 2)
        assert outputs["attention_weights"] is None

    def test_variable_lengths(self):
        model = BiLSTMModel(vocab_size=VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        lengths = torch.tensor([10, 15, 5, 20])
        
        outputs = model(input_ids=input_ids, lengths=lengths)
        assert outputs["logits"].shape == (BATCH_SIZE, 2)

    def test_different_pooling(self):
        for pooling in ["max", "mean", "max_mean", "last_hidden"]:
            model = BiLSTMModel(vocab_size=VOCAB_SIZE, pooling=pooling)
            input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
            lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE)
            
            outputs = model(input_ids=input_ids, lengths=lengths)
            assert outputs["logits"].shape == (BATCH_SIZE, 2), f"Failed for pooling={pooling}"

    def test_model_name(self):
        model = BiLSTMModel(vocab_size=VOCAB_SIZE)
        assert model.get_name() == "BiLSTM"

    def test_parameter_count(self):
        model = BiLSTMModel(vocab_size=VOCAB_SIZE)
        assert model.count_parameters() > 0


class TestBiLSTMAttention:
    """Test BiLSTM + Attention model."""

    def test_forward_with_attention_weights(self):
        model = BiLSTMAttentionModel(vocab_size=VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE)
        
        outputs = model(input_ids=input_ids, lengths=lengths)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (BATCH_SIZE, 2)
        
        # Attention weights must exist
        assert "attention_weights" in outputs
        assert outputs["attention_weights"] is not None
        assert outputs["attention_weights"].shape == (BATCH_SIZE, SEQ_LEN)

    def test_attention_weights_sum_to_one(self):
        model = BiLSTMAttentionModel(vocab_size=VOCAB_SIZE)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE)
        
        outputs = model(input_ids=input_ids, lengths=lengths)
        weights = outputs["attention_weights"]
        
        # Attention weights should sum to approximately 1.0
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_model_name(self):
        model = BiLSTMAttentionModel(vocab_size=VOCAB_SIZE)
        assert model.get_name() == "BiLSTM + Attention"


class TestAttentionMechanisms:
    """Test standalone attention modules."""

    def test_additive_attention(self):
        hidden_dim = 512  # BiLSTM: 2 * 256
        attn = AdditiveAttention(hidden_dim, attention_dim=128)
        
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_dim)
        mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        
        context, weights = attn(hidden, mask)
        
        assert context.shape == (BATCH_SIZE, hidden_dim)
        assert weights.shape == (BATCH_SIZE, SEQ_LEN)

    def test_dot_product_attention(self):
        hidden_dim = 512
        attn = DotProductAttention(hidden_dim)
        
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_dim)
        
        context, weights = attn(hidden)
        
        assert context.shape == (BATCH_SIZE, hidden_dim)
        assert weights.shape == (BATCH_SIZE, SEQ_LEN)

    def test_attention_with_mask(self):
        hidden_dim = 512
        attn = AdditiveAttention(hidden_dim)
        
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_dim)
        mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, 10:] = False  # Mask last 10 positions
        
        context, weights = attn(hidden, mask)
        
        # Masked positions should have near-zero weights
        assert weights[:, 10:].max() < 1e-5


class TestModelCheckpoint:
    """Test model save/load."""

    def test_save_load(self, tmp_path):
        model = BiLSTMModel(vocab_size=VOCAB_SIZE)
        save_path = str(tmp_path / "test_model.pt")
        
        # Save
        model.save_checkpoint(save_path, epoch=5, metrics={"val_f1": 0.85})
        
        # Load
        new_model = BiLSTMModel(vocab_size=VOCAB_SIZE)
        checkpoint = new_model.load_checkpoint(save_path)
        
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["val_f1"] == 0.85
        
        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1.data, p2.data)

"""Tests for inference pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.vocab import Vocabulary, PAD_IDX, UNK_IDX
from src.data.dataset import SentimentDataset


class TestVocabulary:
    """Test vocabulary builder."""

    def test_build_from_texts(self):
        texts = ["hello world", "hello python", "world python"]
        vocab = Vocabulary()
        vocab.build_from_texts(texts, min_freq=1)
        
        assert vocab.size > 2  # At least PAD + UNK + words
        assert vocab.word_to_index("<PAD>") == PAD_IDX
        assert vocab.word_to_index("<UNK>") == UNK_IDX

    def test_min_freq_filtering(self):
        texts = ["hello hello hello", "rare_word", "hello"]
        vocab = Vocabulary()
        vocab.build_from_texts(texts, min_freq=2)
        
        assert vocab.word_to_index("hello") != UNK_IDX
        assert vocab.word_to_index("rare_word") == UNK_IDX

    def test_text_to_indices(self):
        texts = ["hello world"]
        vocab = Vocabulary()
        vocab.build_from_texts(texts)
        
        indices = vocab.text_to_indices("hello world unknown")
        assert len(indices) == 3
        assert indices[2] == UNK_IDX  # unknown word → UNK

    def test_save_load(self, tmp_path):
        texts = ["hello world", "test text"]
        vocab = Vocabulary()
        vocab.build_from_texts(texts)
        
        save_path = str(tmp_path / "vocab.json")
        vocab.save(save_path)
        
        loaded = Vocabulary.load(save_path)
        assert loaded.size == vocab.size
        assert loaded.word_to_index("hello") == vocab.word_to_index("hello")


class TestSentimentDataset:
    """Test PyTorch dataset."""

    @pytest.fixture
    def dataset(self):
        texts = ["hello world", "test text", "another sentence"]
        labels = [1, 0, 1]
        vocab = Vocabulary()
        vocab.build_from_texts(texts)
        return SentimentDataset(texts, labels, vocab)

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem(self, dataset):
        item = dataset[0]
        assert "input_ids" in item
        assert "length" in item
        assert "label" in item

    def test_collate_fn(self, dataset):
        batch = [dataset[0], dataset[1]]
        result = SentimentDataset.collate_fn(batch)
        
        assert "input_ids" in result
        assert "lengths" in result
        assert "labels" in result
        assert result["input_ids"].shape[0] == 2  # batch_size

"""
PyTorch Dataset Classes
========================
Dataset implementations for different model architectures:
  - SentimentDataset: For BiLSTM models (index-based encoding)
  - PhoBERTSentimentDataset: For PhoBERT (transformer tokenizer)
"""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.data.vocab import Vocabulary, PAD_IDX


class SentimentDataset(Dataset):
    """
    Dataset for BiLSTM-based models.
    
    Converts tokenized text to index sequences using a Vocabulary.
    Handles variable-length sequences with padding in the collate function.
    
    Example:
        dataset = SentimentDataset(texts, labels, vocab, max_len=256)
        loader = DataLoader(dataset, collate_fn=SentimentDataset.collate_fn)
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        vocab: Optional[Vocabulary] = None,
        max_len: int = 256,
    ):
        """
        Args:
            texts: List of preprocessed (word-segmented) text strings.
            labels: List of integer labels (0=Negative, 1=Positive). None for inference.
            vocab: Vocabulary instance for word-to-index conversion.
            max_len: Maximum sequence length (truncate longer sequences).
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        
        # Convert to indices
        indices = self.vocab.text_to_indices(text)
        
        # Truncate if needed
        if len(indices) > self.max_len:
            indices = indices[: self.max_len]
        
        # Convert to tensor
        input_ids = torch.tensor(indices, dtype=torch.long)
        length = torch.tensor(len(indices), dtype=torch.long)
        
        item = {
            "input_ids": input_ids,
            "length": length,
        }
        
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Custom collate function for DataLoader.
        Pads sequences to the max length in the batch.
        
        Args:
            batch: List of dicts from __getitem__.
            
        Returns:
            Dict with padded tensors and lengths.
        """
        input_ids = [item["input_ids"] for item in batch]
        lengths = torch.stack([item["length"] for item in batch])
        
        # Pad sequences
        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_IDX)
        
        result = {
            "input_ids": padded_ids,
            "lengths": lengths,
        }
        
        if "label" in batch[0]:
            labels = torch.stack([item["label"] for item in batch])
            result["labels"] = labels
        
        return result


class PhoBERTSentimentDataset(Dataset):
    """
    Dataset for PhoBERT model.
    
    Uses HuggingFace tokenizer for encoding.
    Text should already be word-segmented before passing to this dataset.
    
    Example:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        dataset = PhoBERTSentimentDataset(texts, labels, tokenizer, max_len=256)
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_len: int = 256,
    ):
        """
        Args:
            texts: List of word-segmented text strings.
            labels: List of integer labels. None for inference.
            tokenizer: HuggingFace tokenizer instance.
            max_len: Maximum token length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        
        # Tokenize with HuggingFace tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Collate function for PhoBERT dataset.
        Since we use max_length padding, tensors are already aligned.
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if "label" in batch[0]:
            labels = torch.stack([item["label"] for item in batch])
            result["labels"] = labels
        
        return result

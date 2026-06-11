"""
Vocabulary Builder
==================
Build and manage vocabulary for BiLSTM models.
Handles word-to-index mapping, pre-trained embedding loading, 
and serialization for inference.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    """
    Word-to-index vocabulary for BiLSTM models.
    
    Provides:
      - Build vocab from corpus with frequency threshold
      - Load pre-trained word embeddings (Word2Vec, FastText)
      - Serialize/deserialize for inference
    
    Example:
        vocab = Vocabulary()
        vocab.build_from_texts(train_texts, min_freq=2)
        indices = vocab.text_to_indices("sản_phẩm tốt lắm")
        # => [45, 123, 67]
    """

    def __init__(self):
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_freq: Counter = Counter()
        self._frozen = False

    @property
    def size(self) -> int:
        """Vocabulary size (including special tokens)."""
        return len(self.word2idx)

    def build_from_texts(
        self,
        texts: List[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ):
        """
        Build vocabulary from a list of (already tokenized) texts.
        
        Args:
            texts: List of space-separated tokenized text strings.
            min_freq: Minimum word frequency to include.
            max_vocab_size: Maximum vocabulary size (None = unlimited).
        """
        # Count word frequencies
        self.word_freq = Counter()
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        total_words = len(self.word_freq)
        
        # Filter by frequency
        filtered = {
            word: freq
            for word, freq in self.word_freq.items()
            if freq >= min_freq
        }
        
        # Sort by frequency (most common first) and optionally limit size
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        if max_vocab_size is not None:
            sorted_words = sorted_words[: max_vocab_size - 2]  # Reserve for PAD, UNK
        
        # Build mappings
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        
        for word, _ in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self._frozen = True
        
        logger.info(
            f"Vocabulary built: {self.size} words "
            f"(from {total_words} unique, min_freq={min_freq})"
        )

    def word_to_index(self, word: str) -> int:
        """Convert a word to its index (UNK_IDX if not found)."""
        return self.word2idx.get(word, UNK_IDX)

    def index_to_word(self, idx: int) -> str:
        """Convert an index to its word (UNK_TOKEN if not found)."""
        return self.idx2word.get(idx, UNK_TOKEN)

    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert a space-separated text to a list of indices.
        
        Args:
            text: Space-separated tokenized text.
            
        Returns:
            List of integer indices.
        """
        return [self.word_to_index(w) for w in text.split()]

    def indices_to_text(self, indices: List[int]) -> str:
        """Convert a list of indices back to text."""
        words = [self.index_to_word(idx) for idx in indices if idx != PAD_IDX]
        return " ".join(words)

    def load_pretrained_embeddings(
        self,
        embedding_path: str,
        embedding_dim: int = 300,
    ) -> np.ndarray:
        """
        Load pre-trained word embeddings and create embedding matrix.
        
        Supports Word2Vec text format and FastText .vec format.
        
        Args:
            embedding_path: Path to embedding file.
            embedding_dim: Dimension of embeddings.
            
        Returns:
            NumPy array of shape (vocab_size, embedding_dim).
            Words not in pre-trained are initialized randomly.
        """
        logger.info(f"Loading pre-trained embeddings from {embedding_path}...")
        
        # Initialize random embeddings
        embedding_matrix = np.random.normal(
            scale=0.6, size=(self.size, embedding_dim)
        ).astype(np.float32)
        
        # PAD token should be zero
        embedding_matrix[PAD_IDX] = np.zeros(embedding_dim)
        
        # Load pre-trained vectors
        found = 0
        with open(embedding_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in self.word2idx:
                    try:
                        vector = np.array(parts[1:], dtype=np.float32)
                        if len(vector) == embedding_dim:
                            embedding_matrix[self.word2idx[word]] = vector
                            found += 1
                    except ValueError:
                        continue
        
        coverage = found / (self.size - 2) * 100  # Exclude PAD, UNK
        logger.info(
            f"Embedding coverage: {found}/{self.size - 2} words ({coverage:.1f}%)"
        )
        
        return embedding_matrix

    def save(self, path: str):
        """
        Save vocabulary to JSON file.
        
        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "word2idx": self.word2idx,
            "word_freq": dict(self.word_freq.most_common()),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vocabulary saved to {path} ({self.size} words)")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """
        Load vocabulary from JSON file.
        
        Args:
            path: Path to saved vocabulary file.
            
        Returns:
            Loaded Vocabulary instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.word_freq = Counter(data.get("word_freq", {}))
        vocab._frozen = True
        
        logger.info(f"Vocabulary loaded from {path} ({vocab.size} words)")
        return vocab

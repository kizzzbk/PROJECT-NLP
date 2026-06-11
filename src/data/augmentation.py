"""
Data Augmentation (Extensible)
==============================
Placeholder for text augmentation strategies.
Can be expanded with synonym replacement, back-translation, etc.
"""

import random
from typing import List, Optional


class TextAugmenter:
    """
    Text augmentation for sentiment analysis.
    
    Currently supports:
      - Random word deletion
      - Random word swap
      
    Easily extensible with:
      - Synonym replacement (using Vietnamese WordNet)
      - Back-translation (vi → en → vi)
      - Contextual augmentation (using language models)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.
        
        Args:
            text: Input text (space-separated words).
            p: Probability of deleting each word.
            
        Returns:
            Augmented text.
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        remaining = [w for w in words if self.rng.random() > p]
        
        if not remaining:
            return self.rng.choice(words)
        
        return " ".join(remaining)

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words.
        
        Args:
            text: Input text (space-separated words).
            n: Number of swaps.
            
        Returns:
            Augmented text.
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            i = self.rng.randint(0, len(new_words) - 1)
            j = self.rng.randint(0, len(new_words) - 1)
            new_words[i], new_words[j] = new_words[j], new_words[i]
        
        return " ".join(new_words)

    def augment(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        n_augments: int = 1,
    ) -> List[str]:
        """
        Apply augmentation and return multiple variants.
        
        Args:
            text: Original text.
            methods: List of method names to apply. Default: ["random_deletion"].
            n_augments: Number of augmented versions to generate.
            
        Returns:
            List of augmented texts (not including original).
        """
        if methods is None:
            methods = ["random_deletion"]
        
        results = []
        for _ in range(n_augments):
            augmented = text
            for method in methods:
                fn = getattr(self, method, None)
                if fn:
                    augmented = fn(augmented)
            results.append(augmented)
        
        return results

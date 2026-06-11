"""
Text Preprocessing Pipeline
============================
Modular Vietnamese text cleaning pipeline with configurable steps.
Each step is a standalone function that can be toggled on/off via config.
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import List, Optional, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Vietnamese text preprocessing pipeline.
    
    Each step can be enabled/disabled via the preprocessing config.
    Steps are applied in a deterministic order for reproducibility.
    
    Example:
        preprocessor = TextPreprocessor(config)
        clean_text = preprocessor.process("sp quá tệ, ship chậm lắm :((")
        # => "sản phẩm quá tệ giao hàng chậm lắm buồn"
        
        # Batch processing
        results = preprocessor.process_batch(["text1", "text2", ...])
    """

    def __init__(self, config=None, teencode_dict_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing config (from configs/preprocessing.yaml).
            teencode_dict_path: Path to teencode dictionary JSON.
        """
        self.config = config
        
        # Load teencode dictionary
        self.teencode_dict = {}
        tc_path = teencode_dict_path
        if tc_path is None and config:
            tc_path = getattr(config, "teencode_dict_path", None) if config else None
        if tc_path is None:
            tc_path = "data/teencode_dict.json"
        
        self._load_teencode_dict(tc_path)
        
        # Load step configuration
        self.steps = {}
        if config and hasattr(config, "steps"):
            self.steps = dict(config.steps)
        else:
            # Default: all basic steps enabled
            self.steps = {
                "lowercase": True,
                "normalize_unicode": True,
                "remove_urls": True,
                "remove_emails": True,
                "remove_phone_numbers": True,
                "remove_html_tags": True,
                "remove_special_chars": True,
                "remove_extra_whitespace": True,
                "remove_emojis": False,
                "remove_numbers": False,
                "remove_stopwords": False,
                "replace_teencode": True,
                "word_segmentation": True,
            }
        
        # Word segmenter (lazy-loaded)
        self._segmenter = None
        
        logger.info(
            f"TextPreprocessor initialized | "
            f"Teencode entries: {len(self.teencode_dict)} | "
            f"Active steps: {sum(1 for v in self.steps.values() if v)}"
        )

    def _load_teencode_dict(self, path: str):
        """Load teencode dictionary from JSON file."""
        path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Support both flat dict and nested {"dictionary": {...}} format
                if "dictionary" in data:
                    self.teencode_dict = data["dictionary"]
                else:
                    self.teencode_dict = data
            logger.info(f"Loaded {len(self.teencode_dict)} teencode entries from {path}")
        else:
            logger.warning(f"Teencode dict not found at {path}, skipping teencode replacement")

    # ================================================================
    # Individual preprocessing steps
    # ================================================================

    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Vietnamese unicode (NFC normalization)."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs (http, https, www)."""
        return re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
        )

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses."""
        return re.sub(r"\S+@\S+\.\S+", " ", text)

    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove Vietnamese phone numbers."""
        return re.sub(
            r"(\+84|0)\d{9,10}",
            " ",
            text,
        )

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags."""
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def remove_special_chars(text: str) -> str:
        """
        Remove special characters, keeping Vietnamese letters, 
        digits, and basic punctuation.
        """
        # Keep Vietnamese characters, digits, spaces, and basic punctuation
        return re.sub(
            r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,!?;:\-]",
            " ",
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emoji characters."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(" ", text)

    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove standalone numbers (keep numbers attached to words)."""
        return re.sub(r"\b\d+\b", " ", text)

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Collapse multiple whitespace into single space."""
        return re.sub(r"\s+", " ", text).strip()

    def replace_teencode(self, text: str) -> str:
        """
        Replace teencode/slang with standard Vietnamese.
        Uses word-boundary matching to avoid partial replacements.
        """
        if not self.teencode_dict:
            return text
        
        words = text.split()
        result = []
        for word in words:
            # Check exact match (case-insensitive)
            replacement = self.teencode_dict.get(word.lower(), word)
            result.append(replacement)
        
        return " ".join(result)

    def word_segmentation(self, text: str) -> str:
        """
        Perform Vietnamese word segmentation.
        Uses underthesea by default, falls back gracefully.
        """
        if self._segmenter is None:
            self._init_segmenter()
        
        if self._segmenter is not None:
            try:
                return self._segmenter.segment(text)
            except Exception as e:
                logger.warning(f"Segmentation failed: {e}, returning original text")
                return text
        return text

    def _init_segmenter(self):
        """Initialize word segmenter (lazy loading)."""
        from src.data.tokenizer import VietnameseTokenizer
        
        backend = "underthesea"
        if self.config and hasattr(self.config, "segmenter"):
            backend = getattr(self.config.segmenter, "backend", "underthesea")
        
        try:
            self._segmenter = VietnameseTokenizer(backend=backend)
            logger.info(f"Word segmenter initialized: {backend}")
        except Exception as e:
            logger.warning(f"Failed to init segmenter ({backend}): {e}")
            self._segmenter = None

    # ================================================================
    # Main pipeline
    # ================================================================

    def process(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on a single text.
        
        Steps are applied in order:
        1. Normalize unicode
        2. Remove HTML tags
        3. Remove URLs
        4. Remove emails
        5. Remove phone numbers
        6. Lowercase
        7. Remove emojis (optional)
        8. Replace teencode
        9. Remove special chars
        10. Remove numbers (optional)
        11. Word segmentation
        12. Remove extra whitespace
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        if not text:
            return ""

        # Apply steps in order
        pipeline = [
            ("normalize_unicode", self.normalize_unicode),
            ("remove_html_tags", self.remove_html_tags),
            ("remove_urls", self.remove_urls),
            ("remove_emails", self.remove_emails),
            ("remove_phone_numbers", self.remove_phone_numbers),
            ("lowercase", self.lowercase),
            ("remove_emojis", self.remove_emojis),
            ("replace_teencode", self.replace_teencode),
            ("remove_special_chars", self.remove_special_chars),
            ("remove_numbers", self.remove_numbers),
            ("word_segmentation", self.word_segmentation),
            ("remove_extra_whitespace", self.remove_extra_whitespace),
        ]

        for step_name, step_fn in pipeline:
            if self.steps.get(step_name, False):
                text = step_fn(text)
        
        return text

    def process_batch(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of raw text strings.
            show_progress: Whether to show a progress bar.
            
        Returns:
            List of cleaned text strings.
        """
        if show_progress:
            try:
                from rich.progress import track
                return [self.process(t) for t in track(texts, description="Preprocessing...")]
            except ImportError:
                pass
        
        return [self.process(t) for t in texts]

    def get_active_steps(self) -> List[str]:
        """Return list of currently active preprocessing steps."""
        return [name for name, active in self.steps.items() if active]
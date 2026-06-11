"""
Vietnamese Word Segmentation (Tokenizer)
=========================================
Unified interface for Vietnamese word segmentation backends:
  - underthesea (Python-native, no Java dependency)
  - VnCoreNLP (Java-based, more accurate for formal text)
"""

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VietnameseTokenizer:
    """
    Vietnamese word segmentation wrapper.
    
    Provides a unified `.segment()` interface regardless of backend.
    
    Example:
        tok = VietnameseTokenizer(backend="underthesea")
        result = tok.segment("Hôm nay trời đẹp quá")
        # => "Hôm_nay trời đẹp quá"
    """

    def __init__(self, backend: str = "underthesea", vncorenlp_dir: Optional[str] = None):
        """
        Initialize the tokenizer.
        
        Args:
            backend: "underthesea" or "vncorenlp".
            vncorenlp_dir: Path to VnCoreNLP jar (only for vncorenlp backend).
        """
        self.backend = backend
        self._segmenter = None
        
        if backend == "underthesea":
            self._init_underthesea()
        elif backend == "vncorenlp":
            self._init_vncorenlp(vncorenlp_dir)
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}. Use 'underthesea' or 'vncorenlp'.")

    def _init_underthesea(self):
        """Initialize underthesea segmenter."""
        try:
            from underthesea import word_tokenize
            self._segment_fn = word_tokenize
            self.backend = "underthesea"
            logger.info("Initialized underthesea word segmenter")
        except ImportError:
            raise ImportError(
                "underthesea is not installed. "
                "Install it with: pip install underthesea"
            )

    def _init_vncorenlp(self, vncorenlp_dir: Optional[str]):
        """Initialize VnCoreNLP segmenter (requires Java)."""
        try:
            from vncorenlp import VnCoreNLP
            
            jar_path = vncorenlp_dir or "vncorenlp/VnCoreNLP-1.1.1.jar"
            self._segmenter = VnCoreNLP(jar_path, annotators="wseg", max_heap_size="-Xmx500m")
            self.backend = "vncorenlp"
            logger.info(f"Initialized VnCoreNLP segmenter from {jar_path}")
        except ImportError:
            logger.warning(
                "vncorenlp not installed, falling back to underthesea. "
                "Install with: pip install vncorenlp"
            )
            self._init_underthesea()
        except Exception as e:
            logger.warning(f"VnCoreNLP init failed ({e}), falling back to underthesea")
            self._init_underthesea()

    def segment(self, text: str) -> str:
        """
        Segment Vietnamese text into words.
        
        Compound words are joined with underscores:
            "Hôm nay" → "Hôm_nay"
        
        Args:
            text: Raw Vietnamese text.
            
        Returns:
            Word-segmented text string.
        """
        if not text or not text.strip():
            return ""
        
        if self.backend == "underthesea":
            return self._segment_underthesea(text)
        elif self.backend == "vncorenlp":
            return self._segment_vncorenlp(text)
        
        return text

    def _segment_underthesea(self, text: str) -> str:
        """Segment using underthesea."""
        # underthesea.word_tokenize returns segmented text
        result = self._segment_fn(text, format="text")
        return result

    def _segment_vncorenlp(self, text: str) -> str:
        """Segment using VnCoreNLP."""
        sentences = self._segmenter.tokenize(text)
        # VnCoreNLP returns list of lists of words
        words = []
        for sentence in sentences:
            words.extend(sentence)
        return " ".join(words)

    def tokenize(self, text: str) -> list:
        """
        Segment and split into word tokens.
        
        Args:
            text: Raw Vietnamese text.
            
        Returns:
            List of word tokens.
        """
        segmented = self.segment(text)
        return segmented.split()

    def __del__(self):
        """Clean up VnCoreNLP process if running."""
        if self.backend == "vncorenlp" and self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass

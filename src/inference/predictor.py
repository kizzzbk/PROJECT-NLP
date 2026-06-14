"""
Sentiment Predictor
====================
Unified inference interface for all model architectures.
Handles text preprocessing, model inference, and result formatting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import pandas as pd

from src.data.preprocessing import TextPreprocessor
from src.data.vocab import Vocabulary
from src.models import get_model_class
from src.utils.logger import get_logger
from src.utils.config import load_config, merge_configs

logger = get_logger(__name__)

# Label mapping
LABEL_NAMES = {0: "Tiêu cực", 1: "Tích cực"}


class SentimentPredictor:
    """
    Unified inference interface for sentiment prediction.
    
    Wraps model loading, preprocessing, and prediction into a single API.
    Works with BiLSTM, BiLSTM+Attention, and PhoBERT models.
    
    Example:
        predictor = SentimentPredictor('bilstm', 'models/bilstm/best_model.pt')
        result = predictor.predict_single("sản phẩm quá tệ, ship chậm")
        # => {'label': 'Tiêu cực', 'confidence': 0.92, 'attention_weights': None}
        
        # Batch prediction
        df = predictor.predict_csv('data/reviews.csv', text_column='comment')
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_name: Model type ('bilstm', 'bilstm_attention', 'phobert').
            checkpoint_path: Path to saved model checkpoint.
            config_path: Path to model config YAML (optional, loaded from checkpoint).
            device: Device to use ('auto', 'cuda', 'cpu').
        """
        self.model_name = model_name.replace("bilstm__attention", "bilstm_attention")
        self.checkpoint_path = checkpoint_path

        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load config
        self.config = None
        if config_path:
            self.config = load_config(config_path)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Load model and auxiliary data
        self.model = None
        self.vocab = None
        self.tokenizer = None  # For PhoBERT
        
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        
        config = checkpoint.get("config", {})
        model_cfg = config.get("model", {}) if config else {}
        
        if self.model_name in ["bilstm", "bilstm_attention"]:
            self._load_lstm_model(checkpoint, model_cfg)
        elif self.model_name == "phobert":
            self._load_phobert_model(checkpoint, model_cfg)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(
            f"Loaded {self.model_name} from {self.checkpoint_path} "
            f"(device: {self.device})"
        )

    def _load_lstm_model(self, checkpoint: dict, model_cfg: dict):
        """Load BiLSTM or BiLSTM+Attention model."""
        # Load vocabulary
        vocab_path = Path("data/vocab/vocab.json")
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_path}. "
                f"Run `python scripts/prepare_data.py` first."
            )
        self.vocab = Vocabulary.load(str(vocab_path))
        
        # Create model
        ModelClass = get_model_class(self.model_name)
        
        kwargs = {
            "vocab_size": self.vocab.size,
            "embedding_dim": model_cfg.get("embedding_dim", 300),
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "num_layers": model_cfg.get("num_layers", 2),
            "num_classes": model_cfg.get("num_classes", 2),
            "dropout": model_cfg.get("dropout", 0.3),
            "bidirectional": model_cfg.get("bidirectional", True),
            "fc_hidden_dim": model_cfg.get("fc_hidden_dim", 128),
        }
        
        if self.model_name == "bilstm":
            kwargs["pooling"] = model_cfg.get("pooling", "max_mean")
        elif self.model_name == "bilstm_attention":
            attn_cfg = model_cfg.get("attention", {})
            kwargs["attention_type"] = attn_cfg.get("type", "additive") if isinstance(attn_cfg, dict) else "additive"
            kwargs["attention_dim"] = attn_cfg.get("attention_dim", 128) if isinstance(attn_cfg, dict) else 128
        
        self.model = ModelClass(**kwargs)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _load_phobert_model(self, checkpoint: dict, model_cfg: dict):
        """Load PhoBERT model."""
        from transformers import AutoTokenizer
        
        pretrained = model_cfg.get("pretrained", "vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        
        ModelClass = get_model_class("phobert")
        self.model = ModelClass(
            pretrained=pretrained,
            num_labels=model_cfg.get("num_labels", 2),
            dropout=model_cfg.get("dropout", 0.1),
            output_attentions=model_cfg.get("output_attentions", True),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    @torch.no_grad()
    def predict_single(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Raw input text.
            
        Returns:
            Dict with:
                - original_text: Original input
                - cleaned_text: After preprocessing
                - label: 'Tích cực' or 'Tiêu cực'
                - label_id: 0 or 1
                - confidence: Float 0-1
                - probabilities: {'Tích cực': float, 'Tiêu cực': float}
                - attention_weights: List[float] or None
                - tokens: List[str] (for attention visualization)
        """
        # Preprocess
        cleaned = self.preprocessor.process(text)
        
        if not cleaned.strip():
            return {
                "original_text": text,
                "cleaned_text": "",
                "label": "Không xác định",
                "label_id": -1,
                "confidence": 0.0,
                "probabilities": {},
                "attention_weights": None,
                "tokens": [],
            }
        
        # Prepare input based on model type
        if self.model_name in ["bilstm", "bilstm_attention"]:
            result = self._predict_lstm(text, cleaned)
        else:
            result = self._predict_phobert(text, cleaned)
        
        return result

    def _predict_lstm(self, original_text: str, cleaned_text: str) -> Dict:
        """Run inference with BiLSTM model."""
        tokens = cleaned_text.split()
        indices = self.vocab.text_to_indices(cleaned_text)
        
        input_ids = torch.tensor([indices], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(indices)], dtype=torch.long).to(self.device)
        
        outputs = self.model(input_ids=input_ids, lengths=lengths)
        
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
        
        # Attention weights
        attention_weights = None
        if outputs.get("attention_weights") is not None:
            attention_weights = outputs["attention_weights"].squeeze(0).cpu().tolist()
            # Truncate to match token length
            attention_weights = attention_weights[:len(tokens)]
        
        return {
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "label": LABEL_NAMES.get(pred_id, "Unknown"),
            "label_id": pred_id,
            "confidence": confidence,
            "probabilities": {
                LABEL_NAMES[0]: probs[0].item(),
                LABEL_NAMES[1]: probs[1].item(),
            },
            "attention_weights": attention_weights,
            "tokens": tokens,
        }

    def _predict_phobert(self, original_text: str, cleaned_text: str) -> Dict:
        """Run inference with PhoBERT model."""
        encoding = self.tokenizer(
            cleaned_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
        
        # Get tokens for attention visualization
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).cpu().tolist()
        )
        # Filter out padding tokens
        real_len = attention_mask.sum().item()
        tokens = tokens[:int(real_len)]
        
        # Attention weights
        attention_weights = None
        if outputs.get("attention_weights") is not None:
            attention_weights = outputs["attention_weights"].squeeze(0).cpu().tolist()
            attention_weights = attention_weights[:int(real_len)]
        
        return {
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "label": LABEL_NAMES.get(pred_id, "Unknown"),
            "label_id": pred_id,
            "confidence": confidence,
            "probabilities": {
                LABEL_NAMES[0]: probs[0].item(),
                LABEL_NAMES[1]: probs[1].item(),
            },
            "attention_weights": attention_weights,
            "tokens": tokens,
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of raw text strings.
            
        Returns:
            DataFrame with predictions for each text.
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append({
                "text": result["original_text"],
                "cleaned_text": result["cleaned_text"],
                "label": result["label"],
                "confidence": result["confidence"],
            })
        
        return pd.DataFrame(results)

    def predict_csv(
        self,
        csv_path: str,
        text_column: str = "text",
    ) -> pd.DataFrame:
        """
        Process a CSV file and predict sentiment for all rows.
        
        Args:
            csv_path: Path to CSV file.
            text_column: Name of the text column.
            
        Returns:
            DataFrame with original data + prediction columns.
        """
        df = pd.read_csv(csv_path, encoding="utf-8")
        
        if text_column not in df.columns:
            # Try to auto-detect
            candidates = ["text", "comment", "review", "content", "bình luận"]
            for col in df.columns:
                if col.lower().strip() in candidates:
                    text_column = col
                    break
            else:
                raise ValueError(
                    f"Column '{text_column}' not found. Available: {list(df.columns)}"
                )
        
        texts = df[text_column].astype(str).tolist()
        predictions = self.predict_batch(texts)
        
        df["predicted_label"] = predictions["label"].values
        df["confidence"] = predictions["confidence"].values
        
        logger.info(f"Predicted {len(df)} rows from {csv_path}")
        
        return df

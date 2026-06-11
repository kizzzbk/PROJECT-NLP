"""
CLI Prediction Script
======================
Quick single-text prediction from the command line.

Usage:
    python scripts/predict.py --model bilstm --text "sản phẩm quá tệ, ship chậm"
    python scripts/predict.py --model phobert --text "hàng đẹp lắm, sẽ mua tiếp"
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import SentimentPredictor
from src.inference.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger("predict")


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a text")
    parser.add_argument("--model", type=str, required=True,
                        choices=["bilstm", "bilstm_attention", "phobert"])
    parser.add_argument("--text", type=str, required=True,
                        help="Text to predict sentiment for")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        registry = ModelRegistry("models")
        checkpoint_path = registry.get_best_checkpoint(args.model)
        if checkpoint_path is None:
            logger.error(f"No checkpoint found for {args.model}. Train the model first.")
            sys.exit(1)
    
    # Create predictor
    predictor = SentimentPredictor(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    
    # Predict
    result = predictor.predict_single(args.text)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"📝 Input:    {result['original_text']}")
    print(f"🧹 Cleaned:  {result['cleaned_text']}")
    print(f"🏷️  Label:    {result['label']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"📈 Probabilities:")
    for label, prob in result['probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"   {label}: {prob:.4f} {bar}")
    
    if result.get('attention_weights') and result.get('tokens'):
        print(f"\n🔍 Attention Weights (top tokens):")
        token_weights = list(zip(result['tokens'], result['attention_weights']))
        token_weights.sort(key=lambda x: x[1], reverse=True)
        for token, weight in token_weights[:10]:
            bar = "█" * int(weight * 50)
            print(f"   {token:20s} {weight:.4f} {bar}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

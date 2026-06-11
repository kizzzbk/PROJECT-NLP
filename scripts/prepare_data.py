"""
Data Preparation Script
========================
Download, preprocess, and split the NTC-SCV dataset.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data_path data/raw/my_data.csv --text_col review
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import merge_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.preprocessing import TextPreprocessor
from src.data.vocab import Vocabulary

logger = get_logger("prepare_data")


def load_raw_data(data_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load raw dataset from CSV.
    
    Expected format:
        text,label
        "sản phẩm tốt lắm",1
        "dở quá, không mua nữa",0
    
    Args:
        data_path: Path to CSV file.
        text_col: Name of the text column.
        label_col: Name of the label column.
        
    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    path = Path(data_path)
    
    if not path.exists():
        logger.warning(f"Data file not found: {path}")
        logger.info("Creating sample dataset for demonstration...")
        return create_sample_data()
    
    # Try different separators
    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            if text_col in df.columns and label_col in df.columns:
                df = df[[text_col, label_col]].rename(
                    columns={text_col: "text", label_col: "label"}
                )
                break
        except Exception:
            continue
    else:
        # Try auto-detect columns
        df = pd.read_csv(path, encoding="utf-8")
        # Look for text-like and label-like columns
        text_candidates = ["text", "comment", "review", "content", "sentence", "bình luận"]
        label_candidates = ["label", "sentiment", "target", "nhãn"]
        
        found_text = None
        found_label = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in text_candidates:
                found_text = col
            if col_lower in label_candidates:
                found_label = col
        
        if found_text and found_label:
            df = df[[found_text, found_label]].rename(
                columns={found_text: "text", found_label: "label"}
            )
        else:
            raise ValueError(
                f"Cannot auto-detect columns. Found: {list(df.columns)}. "
                f"Please specify --text_col and --label_col."
            )
    
    # Clean up
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    
    # Validate labels are 0 or 1
    valid_labels = df["label"].isin([0, 1])
    if not valid_labels.all():
        logger.warning(
            f"Found {(~valid_labels).sum()} rows with invalid labels "
            f"(not 0 or 1). Removing them."
        )
        df = df[valid_labels]
    
    logger.info(f"Loaded {len(df)} samples from {path}")
    logger.info(f"  Label distribution: {dict(df['label'].value_counts())}")
    
    return df


def create_sample_data() -> pd.DataFrame:
    """Create a sample dataset for demonstration/testing purposes."""
    samples = [
        # Positive
        ("Sản phẩm rất tốt, mình rất hài lòng", 1),
        ("Chất lượng tuyệt vời, giá cả hợp lý", 1),
        ("Giao hàng nhanh, đóng gói cẩn thận", 1),
        ("Shop nhiệt tình, tư vấn tận tâm lắm", 1),
        ("Sản phẩm đúng mô tả, rất đẹp", 1),
        ("Mua lần 2 rồi, lần nào cũng ok", 1),
        ("Hàng xịn, giá tốt, sẽ ủng hộ tiếp", 1),
        ("Sp quá đẹp, vượt ngoài mong đợi", 1),
        ("Nhân viên phục vụ rất chuyên nghiệp", 1),
        ("Đồ ăn ngon, không gian thoải mái", 1),
        ("Chất liệu mềm mại, mặc rất thoáng", 1),
        ("Đáng đồng tiền bát gạo, recommend cho mọi người", 1),
        ("Tks shop, hàng đẹp lắm ạ", 1),
        ("Sp chính hãng, yên tâm sử dụng", 1),
        ("Mình rất thích, sẽ giới thiệu bạn bè", 1),
        ("Dịch vụ 5 sao, không có gì phải chê", 1),
        ("Hàng về nhanh hơn dự kiến, cảm ơn shop", 1),
        ("Chất lượng xứng đáng với giá tiền", 1),
        ("Rất hài lòng với trải nghiệm mua sắm", 1),
        ("Sản phẩm tốt, giao hàng đúng hẹn", 1),
        # Negative
        ("Sản phẩm quá tệ, không đáng tiền", 0),
        ("Giao hàng chậm, thái độ nhân viên kém", 0),
        ("Hàng bị lỗi, liên hệ shop ko trả lời", 0),
        ("Chất lượng không như quảng cáo", 0),
        ("Mua về bị hỏng, đòi đổi trả k dc", 0),
        ("Dịch vụ tệ, không bao giờ mua lại", 0),
        ("Ship chậm lắm, đợi cả tuần", 0),
        ("Hàng fake, không giống hình", 0),
        ("Sản phẩm kém chất lượng, rất thất vọng", 0),
        ("Tư vấn sai, mua về không dùng được", 0),
        ("Đắt mà chất lượng như hàng chợ", 0),
        ("Bao bì bị rách, hàng bên trong bẩn", 0),
        ("Không đáng mua, phí tiền", 0),
        ("Sp lởm, ko giống mô tả", 0),
        ("Giao nhầm hàng, liên hệ mãi ko dc", 0),
        ("Thất vọng hoàn toàn, chất lượng quá tệ", 0),
        ("Hàng nhận được khác xa với hình ảnh", 0),
        ("Đóng gói sơ sài, hàng bị méo", 0),
        ("Không recommend, chất lượng quá kém", 0),
        ("Mua lần đầu và cũng là lần cuối", 0),
    ]
    
    df = pd.DataFrame(samples, columns=["text", "label"])
    logger.info(f"Created sample dataset with {len(df)} entries")
    return df


def preprocess_and_split(
    df: pd.DataFrame,
    preprocessor: TextPreprocessor,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Preprocess texts and split into train/val/test sets.
    
    Args:
        df: DataFrame with 'text' and 'label' columns.
        preprocessor: TextPreprocessor instance.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test).
        seed: Random seed.
        
    Returns:
        Dict with train/val/test DataFrames.
    """
    logger.info("Preprocessing texts...")
    df["text_clean"] = preprocessor.process_batch(df["text"].tolist(), show_progress=True)
    
    # Remove empty texts after preprocessing
    before_len = len(df)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    if len(df) < before_len:
        logger.warning(f"Removed {before_len - len(df)} empty texts after preprocessing")
    
    # Split: first take out test set, then split remaining into train/val
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    
    # Adjust val_size relative to train_val
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=seed, stratify=train_val_df["label"]
    )
    
    logger.info(
        f"Split sizes — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )
    
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = dict(split_df["label"].value_counts())
        logger.info(f"  {name}: {dist}")
    
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--data_path", type=str, default="data/raw/ntc_scv.csv",
                        help="Path to raw dataset CSV")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Name of text column in CSV")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Name of label column in CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--vocab_dir", type=str, default="data/vocab",
                        help="Output directory for vocabulary")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Minimum word frequency for vocabulary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load config
    try:
        config = merge_configs(
            PROJECT_ROOT / "configs" / "base.yaml",
            PROJECT_ROOT / "configs" / "preprocessing.yaml",
        )
        preprocess_config = config.preprocessing if hasattr(config, "preprocessing") else None
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        preprocess_config = None
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        config=preprocess_config,
        teencode_dict_path=str(PROJECT_ROOT / "data" / "teencode_dict.json"),
    )
    
    # Load data
    df = load_raw_data(args.data_path, args.text_col, args.label_col)
    
    # Preprocess and split
    splits = preprocess_and_split(df, preprocessor, seed=args.seed)
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, split_df in splits.items():
        save_path = output_dir / f"{name}.csv"
        split_df[["text_clean", "label"]].to_csv(save_path, index=False, encoding="utf-8")
        logger.info(f"Saved {name} set → {save_path}")
    
    # Build vocabulary (for BiLSTM models)
    vocab = Vocabulary()
    vocab.build_from_texts(
        splits["train"]["text_clean"].tolist(),
        min_freq=args.min_freq,
    )
    
    vocab_dir = Path(args.vocab_dir)
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_dir / "vocab.json")
    
    logger.info("\n✅ Data preparation complete!")
    logger.info(f"  Processed data: {output_dir}")
    logger.info(f"  Vocabulary: {vocab_dir / 'vocab.json'} ({vocab.size} words)")


if __name__ == "__main__":
    main()

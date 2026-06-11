import os
import sys
from pathlib import Path

# Standalone download script to avoid importing project files before torch/etc are installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    try:
        import pandas as pd
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: Missing dependencies. Please install 'datasets', 'pyarrow', and 'pandas' by running:\n"
            "  pip install datasets pyarrow pandas"
        )
        sys.exit(1)

    print("Starting NTC-SCV dataset download from Hugging Face (thainq107/ntc-scv)...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("thainq107/ntc-scv")
        
        # Convert splits to pandas dataframes
        print("Converting Hugging Face dataset splits to Pandas DataFrames...")
        splits = []
        for split_name in dataset.keys():
            df_split = dataset[split_name].to_pandas()
            print(f"  Loaded split '{split_name}' with {len(df_split)} rows.")
            splits.append(df_split)
        
        # Combine all splits
        full_df = pd.concat(splits, ignore_index=True)
        print(f"Combined all splits into a single DataFrame with {len(full_df)} rows.")
        
        # Identify text column (usually 'sentence' or 'text')
        text_col = "sentence" if "sentence" in full_df.columns else "text"
        if text_col not in full_df.columns:
            # Fallback to check if another column exists
            cols = [c for c in full_df.columns if c != "label"]
            if cols:
                text_col = cols[0]
            else:
                raise ValueError(f"Could not find a text column in dataset columns: {list(full_df.columns)}")
        
        # Keep only text and label columns, rename text to 'text'
        print(f"Using column '{text_col}' as text and 'label' as label.")
        output_df = full_df[[text_col, "label"]].rename(columns={text_col: "text"})
        
        # Ensure raw data directory exists
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = raw_dir / "ntc_scv.csv"
        
        # Save to CSV
        output_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n[SUCCESS] Successfully downloaded and saved NTC-SCV raw dataset to: {output_path}")
        print(f"   Shape: {output_df.shape}")
        print(f"   Label distribution:\n{output_df['label'].value_counts()}")
        
    except Exception as e:
        print(f"Error: Failed to download/process the dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

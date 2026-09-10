"""
Dataset Downloader & Local Cache Manager
Downloads and verifies:
1. Customer Support on Twitter (Kaggle: thoughtvector/customer-support-on-twitter) -> data/raw/twcs/
2. Banking77 (Hugging Face: PolyAI/banking77) -> data/raw/banking77/
"""

import os
import sys
import json
import zipfile
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DatasetDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TWCS_DIR = RAW_DIR / "twcs"
BANKING_DIR = RAW_DIR / "banking77"


def download_banking77():
    """Download Banking77 dataset from Hugging Face and save as CSV + metadata."""
    logger.info("=== 1/2: Fetching Banking77 Dataset from Hugging Face ===")
    BANKING_DIR.mkdir(parents=True, exist_ok=True)
    
    train_file = BANKING_DIR / "train.csv"
    test_file = BANKING_DIR / "test.csv"
    meta_file = BANKING_DIR / "categories.json"

    if train_file.exists() and test_file.exists() and meta_file.exists():
        logger.info(f"Banking77 already exists in {BANKING_DIR}. Validating...")
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        logger.info(f"Banking77 validated: {len(df_train)} train rows, {len(df_test)} test rows.")
        return True

    try:
        from datasets import load_dataset
        logger.info("Loading 'PolyAI/banking77' via Hugging Face datasets library...")
        ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load via 'PolyAI/banking77' ({e}), trying fallback 'banking77'...")
        from datasets import load_dataset
        ds = load_dataset("banking77", trust_remote_code=True)

    # Extract label names
    label_names = ds["train"].features["label"].names
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"intents": label_names, "num_intents": len(label_names)}, f, indent=2)
    logger.info(f"Saved {len(label_names)} intent category definitions to {meta_file}")

    # Convert to DataFrames and map labels
    df_train = pd.DataFrame(ds["train"])
    df_train["intent_name"] = df_train["label"].apply(lambda i: label_names[i])
    df_train.to_csv(train_file, index=False)
    logger.info(f"Saved Banking77 train split ({len(df_train)} rows) -> {train_file}")

    df_test = pd.DataFrame(ds["test"])
    df_test["intent_name"] = df_test["label"].apply(lambda i: label_names[i])
    df_test.to_csv(test_file, index=False)
    logger.info(f"Saved Banking77 test split ({len(df_test)} rows) -> {test_file}")

    return True


def download_twcs():
    """Download Customer Support on Twitter from Kaggle API and extract twcs.csv."""
    logger.info("=== 2/2: Fetching Customer Support on Twitter (twcs.csv) from Kaggle ===")
    TWCS_DIR.mkdir(parents=True, exist_ok=True)
    
    target_csv = TWCS_DIR / "twcs.csv"
    if target_csv.exists() and target_csv.stat().st_size > 100_000_000:
        logger.info(f"twcs.csv already exists at {target_csv} ({target_csv.stat().st_size / (1024*1024):.1f} MB).")
        return True

    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset 'thoughtvector/customer-support-on-twitter' via Kaggle API...")
    api.dataset_download_files(
        dataset="thoughtvector/customer-support-on-twitter",
        path=str(TWCS_DIR),
        unzip=True,
        quiet=False
    )

    # Check if target_csv or nested twcs.csv exists
    nested_csv = TWCS_DIR / "twcs" / "twcs.csv"
    if nested_csv.exists():
        logger.info(f"Moving {nested_csv} -> {target_csv}")
        nested_csv.replace(target_csv)
        # Remove empty nested folder
        try:
            (TWCS_DIR / "twcs").rmdir()
        except Exception:
            pass

    if not target_csv.exists():
        # Check if zip exists and unzip manually
        zip_files = list(TWCS_DIR.glob("*.zip"))
        if zip_files:
            logger.info(f"Extracting {zip_files[0]}...")
            with zipfile.ZipFile(zip_files[0], "r") as zf:
                zf.extractall(TWCS_DIR)
            zip_files[0].unlink()  # cleanup zip

    # Final check
    if not target_csv.exists() and (TWCS_DIR / "twcs" / "twcs.csv").exists():
        (TWCS_DIR / "twcs" / "twcs.csv").replace(target_csv)
        try:
            (TWCS_DIR / "twcs").rmdir()
        except Exception:
            pass

    if target_csv.exists():
        size_mb = target_csv.stat().st_size / (1024 * 1024)
        logger.info(f"Successfully located twcs.csv: {size_mb:.1f} MB at {target_csv}")
        
        # Save a quick metadata summary
        df_head = pd.read_csv(target_csv, nrows=100)
        meta = {
            "file": "twcs.csv",
            "size_mb": round(size_mb, 2),
            "columns": list(df_head.columns),
            "sample_head": df_head.to_dict(orient="records")[:3]
        }
        with open(TWCS_DIR / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return True
    else:
        raise FileNotFoundError(f"twcs.csv was not found in {TWCS_DIR} after download.")


def main():
    logger.info(f"Root data directory: {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    download_banking77()
    download_twcs()
    
    logger.info("All datasets downloaded, verified, and structured successfully in data/raw/")


if __name__ == "__main__":
    main()

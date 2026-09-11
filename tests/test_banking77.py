"""
Unit Tests for Banking77 Dataset Augmentation & Intent Taxonomy Mapping.
"""

from src.banking77_augment import (
    BANKING77_TO_AMAZONHELP,
    load_banking77_split,
    get_augmented_dataset,
    evaluate_classifier_on_banking77
)
from src.config import get_app_config
import pandas as pd


def test_banking77_mapping_coverage():
    """Verify that all 77 Banking77 classes are mapped to valid support intents."""
    app_config = get_app_config("amazonhelp")
    valid_intents = set(app_config.brand.intent_ids)

    assert len(BANKING77_TO_AMAZONHELP) == 77
    for b77_label, support_intent in BANKING77_TO_AMAZONHELP.items():
        assert support_intent in valid_intents, f"Invalid intent mapping: {support_intent} for {b77_label}"


def test_banking77_dataset_loading():
    """Verify loading and intent column synthesis on Banking77 dataset."""
    df_train = load_banking77_split(split="train", max_per_intent=10)
    assert len(df_train) > 0
    assert "text" in df_train.columns
    assert "target_intent" in df_train.columns


def test_banking77_augmentation():
    """Verify augmented dataset generation combining Golden and Banking77."""
    train_texts, val_texts, train_labels, val_labels, label2id, id2label = get_augmented_dataset(
        brand="amazonhelp",
        include_golden=True,
        max_banking_per_class=20,
        val_split_ratio=0.2
    )
    assert len(train_texts) > 0
    assert len(val_texts) > 0
    assert len(train_labels) == len(train_texts)
    assert len(val_labels) == len(val_texts)
    assert len(label2id) == 7


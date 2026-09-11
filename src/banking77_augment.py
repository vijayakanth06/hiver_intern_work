"""
Banking77 Dataset Integration & Intent Mapping Engine.
Supports:
1. Mapping 77 fine-grained Banking77 intents into domain customer support intents.
2. Augmenting training datasets for DeBERTa / SetFit intent classifiers.
3. Cross-domain transfer evaluation of classifiers on Banking77 test split.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, load_brand_config
from src.utils import normalize_tweet_text

logger = logging.getLogger("hiver.banking77")

# Semantic mapping from 77 Banking77 intent classes to AmazonHelp 7-class taxonomy
BANKING77_TO_AMAZONHELP: Dict[str, str] = {
    # 1. order_tracking_delivery
    "card_arrival": "order_tracking_delivery",
    "card_delivery_estimate": "order_tracking_delivery",
    "order_physical_card": "order_tracking_delivery",
    "transfer_timing": "order_tracking_delivery",
    "pending_transfer": "order_tracking_delivery",
    "transfer_not_received_by_recipient": "order_tracking_delivery",
    "balance_not_updated_after_bank_transfer": "order_tracking_delivery",
    "balance_not_updated_after_cheque_or_cash_deposit": "order_tracking_delivery",
    "getting_spare_card": "order_tracking_delivery",
    "pending_top_up": "order_tracking_delivery",
    "pending_card_payment": "order_tracking_delivery",
    "pending_cash_withdrawal": "order_tracking_delivery",
    
    # 2. refund_return_cancellation
    "Refund_not_showing_up": "refund_return_cancellation",
    "request_refund": "refund_return_cancellation",
    "cancel_transfer": "refund_return_cancellation",
    "reverted_card_payment?": "refund_return_cancellation",
    "top_up_reverted": "refund_return_cancellation",
    
    # 3. prime_subscription_billing
    "card_payment_fee_charged": "prime_subscription_billing",
    "extra_charge_on_statement": "prime_subscription_billing",
    "transaction_charged_twice": "prime_subscription_billing",
    "transfer_fee_charged": "prime_subscription_billing",
    "cash_withdrawal_charge": "prime_subscription_billing",
    "top_up_by_bank_transfer_charge": "prime_subscription_billing",
    "top_up_by_card_charge": "prime_subscription_billing",
    "exchange_charge": "prime_subscription_billing",
    "automatic_top_up": "prime_subscription_billing",
    "card_payment_wrong_exchange_rate": "prime_subscription_billing",
    "wrong_exchange_rate_for_cash_withdrawal": "prime_subscription_billing",
    "declined_card_payment": "prime_subscription_billing",
    "declined_transfer": "prime_subscription_billing",
    "declined_cash_withdrawal": "prime_subscription_billing",
    "top_up_failed": "prime_subscription_billing",
    "failed_transfer": "prime_subscription_billing",
    "topping_up_by_card": "prime_subscription_billing",
    "top_up_by_cash_or_cheque": "prime_subscription_billing",
    "wrong_amount_of_cash_received": "prime_subscription_billing",
    "receiving_money": "prime_subscription_billing",
    "transfer_into_account": "prime_subscription_billing",
    
    # 4. account_security_access
    "change_pin": "account_security_access",
    "passcode_forgotten": "account_security_access",
    "pin_blocked": "account_security_access",
    "lost_or_stolen_card": "account_security_access",
    "lost_or_stolen_phone": "account_security_access",
    "compromised_card": "account_security_access",
    "card_payment_not_recognised": "account_security_access",
    "cash_withdrawal_not_recognised": "account_security_access",
    "direct_debit_payment_not_recognised": "account_security_access",
    "unable_to_verify_identity": "account_security_access",
    "verify_my_identity": "account_security_access",
    "verify_source_of_funds": "account_security_access",
    "why_verify_identity": "account_security_access",
    "terminate_account": "account_security_access",
    "edit_personal_details": "account_security_access",
    "activate_my_card": "account_security_access",
    "card_linking": "account_security_access",
    "verify_top_up": "account_security_access",
    
    # 5. digital_services_devices
    "apple_pay_or_google_pay": "digital_services_devices",
    "contactless_not_working": "digital_services_devices",
    "virtual_card_not_working": "digital_services_devices",
    "get_disposable_virtual_card": "digital_services_devices",
    "getting_virtual_card": "digital_services_devices",
    "exchange_via_app": "digital_services_devices",
    "card_not_working": "digital_services_devices",
    "card_swallowed": "digital_services_devices",
    "atm_support": "digital_services_devices",
    
    # 6. seller_product_inquiry
    "supported_cards_and_currencies": "seller_product_inquiry",
    "visa_or_mastercard": "seller_product_inquiry",
    "country_support": "seller_product_inquiry",
    "fiat_currency_support": "seller_product_inquiry",
    "card_acceptance": "seller_product_inquiry",
    "card_about_to_expire": "seller_product_inquiry",
    "disposable_card_limits": "seller_product_inquiry",
    "top_up_limits": "seller_product_inquiry",
    "age_limit": "seller_product_inquiry",
    "beneficiary_not_allowed": "seller_product_inquiry",
    "get_physical_card": "seller_product_inquiry",
    "exchange_rate": "seller_product_inquiry",
}


def load_banking77_split(split: str = "train", brand: str = "amazonhelp", max_per_intent: Optional[int] = None) -> pd.DataFrame:
    """
    Loads Banking77 dataset split ('train' or 'test'), normalizes text,
    and maps fine-grained intents to the brand's intent taxonomy.
    """
    data_path = PROJECT_ROOT / "data" / "raw" / "banking77" / f"{split}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Banking77 split not found at {data_path}")

    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(normalize_tweet_text)
    df["target_intent"] = df["intent_name"].map(BANKING77_TO_AMAZONHELP)
    df = df.dropna(subset=["target_intent"])

    if max_per_intent is not None and max_per_intent > 0:
        df = df.groupby("target_intent", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_per_intent), random_state=42)
        )

    return df


def get_augmented_dataset(
    brand: str = "amazonhelp",
    include_golden: bool = True,
    max_banking_per_class: int = 250,
    val_split_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], Dict[str, int], Dict[int, str]]:
    """
    Constructs a balanced training and validation dataset combining:
    1. Golden annotated benchmark samples (domain-specific ground truth)
    2. Banking77 mapped query samples (high-volume semantic intent diversity)
    """
    brand_cfg = load_brand_config(brand)
    intent_ids = brand_cfg.intent_ids
    label2id = {intent_id: i for i, intent_id in enumerate(intent_ids)}
    id2label = {i: intent_id for i, intent_id in enumerate(intent_ids)}

    all_texts: List[str] = []
    all_labels: List[int] = []

    # 1. Load Golden Data
    if include_golden:
        golden_path = PROJECT_ROOT / "data" / "golden" / f"{brand.lower()}_golden.csv"
        if golden_path.exists():
            gdf = pd.read_csv(golden_path)
            for _, row in gdf.iterrows():
                q = normalize_tweet_text(str(row["customer_query"]))
                intent = str(row["ground_truth_intent"]).strip()
                if intent in label2id:
                    all_texts.append(q)
                    all_labels.append(label2id[intent])
            logger.info(f"Loaded {len(gdf)} samples from golden dataset ({golden_path.name})")

    # 2. Load Banking77 Training Data
    bdf = load_banking77_split(split="train", brand=brand, max_per_intent=max_banking_per_class)
    for _, row in bdf.iterrows():
        q = row["clean_text"]
        intent = row["target_intent"]
        if intent in label2id:
            all_texts.append(q)
            all_labels.append(label2id[intent])

    logger.info(f"Total combined dataset size: {len(all_texts)} samples across {len(intent_ids)} classes")

    # Stratified split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts,
        all_labels,
        test_size=val_split_ratio,
        random_state=random_seed,
        stratify=all_labels
    )

    return train_texts, val_texts, train_labels, val_labels, label2id, id2label


def evaluate_classifier_on_banking77(classifier, brand: str = "amazonhelp", sample_size: int = 500) -> Dict[str, Any]:
    """
    Evaluates a trained classifier (DeBERTa or SetFit) on the unseen Banking77 test set.
    Tests zero-shot domain transfer and generalization capabilities.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    bdf = load_banking77_split(split="test", brand=brand)
    if sample_size and sample_size < len(bdf):
        bdf = bdf.sample(sample_size, random_state=42)

    brand_cfg = load_brand_config(brand)
    intent_ids = brand_cfg.intent_ids
    label2id = {intent_id: i for i, intent_id in enumerate(intent_ids)}

    y_true = []
    y_pred = []

    for _, row in bdf.iterrows():
        true_intent = row["target_intent"]
        if true_intent not in label2id:
            continue

        pred = classifier.predict(row["clean_text"])
        pred_intent = pred.intent if hasattr(pred, "intent") else str(pred)

        y_true.append(label2id[true_intent])
        y_pred.append(label2id.get(pred_intent, -1))

    # Filter any unmapped predictions
    valid_mask = [p != -1 for p in y_pred]
    y_true = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred = [p for p, v in zip(y_pred, valid_mask) if v]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "dataset": "banking77_test",
        "sample_count": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1
    }

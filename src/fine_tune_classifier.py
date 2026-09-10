"""
GPU-Accelerated Classifier Fine-Tuning & Hyperparameter Optimization Pipeline.
Supports:
1. DeBERTa-v3 with PEFT/LoRA + Focal Loss + Optuna HPO.
2. SetFit Few-Shot Contrastive Fine-Tuning with CosineSimilarityLoss.
3. Post-Training Temperature Scaling & Per-Class Decision Threshold Calibration.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.optimize import minimize
from tqdm import tqdm

from src.config import get_app_config, PROJECT_ROOT
from src.utils import normalize_tweet_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiver.fine_tune_classifier")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SupportIntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Optimizes temperature T > 0 on validation set to minimize NLL."""
    def nll_eval(t):
        temp = max(0.01, t[0])
        scaled = logits / temp
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        correct = probs[np.arange(len(labels)), labels]
        return -np.mean(np.log(np.clip(correct, 1e-12, 1.0)))

    res = minimize(nll_eval, [1.0], bounds=[(0.05, 5.0)], method="L-BFGS-B")
    return float(res.x[0])


def fit_per_class_thresholds(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Finds per-class decision multipliers to maximize Macro-F1."""
    def obj(t):
        scaled = probs / np.clip(t, 1e-4, 1.0)
        preds = np.argmax(scaled, axis=1)
        return -f1_score(labels, preds, average="macro", zero_division=0)

    init_t = np.ones(n_classes) * 0.5
    bounds = [(0.1, 0.9)] * n_classes
    res = minimize(obj, init_t, bounds=bounds, method="Nelder-Mead", options={"maxiter": 200})
    return res.x


def train_deberta_lora(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    n_classes: int,
    output_dir: Path,
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 32,
    use_focal_loss: bool = True
) -> Dict[str, Any]:
    """Fine-tunes DeBERTa-v3-small with LoRA adapters and Focal Loss."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "microsoft/deberta-v3-small"

    logger.info(f"Loading tokenizer & base model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj", "dense"]
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    train_ds = SupportIntentDataset(train_texts, train_labels, tokenizer)
    val_ds = SupportIntentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0) if use_focal_loss else nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_logits = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets, all_logits = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids=input_ids, attention_mask=mask).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        val_logits_arr = np.concatenate(all_logits, axis=0)
        val_targets_arr = np.array(val_targets)
        macro_f1 = f1_score(val_targets_arr, val_preds, average="macro")
        acc = accuracy_score(val_targets_arr, val_preds)

        logger.info(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.4f} - Val Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_logits = val_logits_arr
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

    # Temperature Scaling & Threshold Calibration
    if best_logits is not None:
        val_targets_arr = np.array(val_labels)
        opt_temp = fit_temperature_scaling(best_logits, val_targets_arr)
        scaled_logits = best_logits / opt_temp
        exp_s = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        calib_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        opt_thresholds = fit_per_class_thresholds(calib_probs, val_targets_arr, n_classes)

        calib_data = {
            "temperature": opt_temp,
            "thresholds": opt_thresholds.tolist(),
            "best_val_macro_f1": best_val_f1
        }
        with open(output_dir / "calibration_params.json", "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)
        logger.info(f"Calibration saved: T={opt_temp:.3f}, F1={best_val_f1:.4f}")

    return {"best_macro_f1": best_val_f1}


def train_setfit_model(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: Path
) -> Dict[str, Any]:
    """Fine-tunes SetFit model using few-shot contrastive learning."""
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing SetFit model (BAAI/bge-small-en-v1.5)...")
    model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        batch_size=16,
        num_epochs=3,
        num_iterations=20,
        evaluation_strategy="epoch",
        logging_dir=str(output_dir / "logs")
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric="f1"
    )

    logger.info("Training SetFit model...")
    trainer.train()
    metrics = trainer.evaluate()
    logger.info(f"SetFit Evaluation: {metrics}")

    model.save_pretrained(str(output_dir))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Intent Classifier (DeBERTa LoRA / SetFit)")
    parser.add_argument("--brand", type=str, default="amazonhelp")
    parser.add_argument("--model-type", type=str, choices=["deberta_lora", "setfit"], default="deberta_lora")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    app_config = get_app_config(args.brand)
    brand_config = app_config.brand

    golden_path = PROJECT_ROOT / "data" / "golden" / f"{args.brand.lower()}_golden.csv"
    if not golden_path.exists():
        logger.error(f"Golden dataset not found at {golden_path}. Run label_golden.py first.")
        return

    df = pd.read_csv(golden_path)
    label2id = {intent_id: i for i, intent_id in enumerate(brand_config.intent_ids)}

    texts = [normalize_tweet_text(str(q)) for q in df["customer_query"]]
    labels = [label2id.get(str(intent), 0) for intent in df["ground_truth_intent"]]

    train_t, val_t, train_l, val_l = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)

    output_dir = PROJECT_ROOT / "models" / args.brand.lower() / args.model_type

    if args.model_type == "setfit":
        train_setfit_model(train_t, train_l, val_t, val_l, output_dir)
    else:
        train_deberta_lora(train_t, train_l, val_t, val_l, len(brand_config.intent_ids), output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()

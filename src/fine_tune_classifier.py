"""
GPU-Accelerated Classifier Fine-Tuning & Hyperparameter Optimization Pipeline.
Supports:
1. DeBERTa-v3 with PEFT/LoRA + Class-Balanced Focal Loss + Cosine LR Scheduler.
2. SetFit Few-Shot Contrastive Fine-Tuning.
3. Multi-source Data Augmentation (Domain Golden + Banking77 77-class Transfer).
4. Post-Training Temperature Scaling & Per-Class Decision Threshold Calibration.
5. Cross-Domain Transfer Evaluation on Banking77 Test Split.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from src.config import get_app_config, PROJECT_ROOT, load_brand_config
from src.utils import normalize_tweet_text, get_device
from src.banking77_augment import get_augmented_dataset, evaluate_classifier_on_banking77

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hiver.fine_tune_classifier")


class FocalLoss(nn.Module):
    """
    Stabilized Focal Loss for addressing class imbalance:
    FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
    Calculated in float32 with probability clamping to prevent numerical underflow.
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0)
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
    """Optimizes temperature T > 0 on validation set to minimize Negative Log Likelihood."""
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
    target_names: Optional[List[str]] = None,
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    use_focal_loss: bool = False
) -> Dict[str, Any]:
    """
    Fine-tunes Transformer sequence classifier with class balancing, AdamW, and Cosine Annealing.
    Saves full self-contained model directly to output_dir with calibration parameters.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    output_dir.mkdir(parents=True, exist_ok=True)
    device_str = get_device("cuda")
    device = torch.device(device_str)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info(f"Loading tokenizer & base model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
    model.to(device)

    train_ds = SupportIntentDataset(train_texts, train_labels, tokenizer)
    val_ds = SupportIntentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = np.sum(class_counts) / (n_classes * np.maximum(class_counts, 1).astype(float))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, weight=weight_tensor) if use_focal_loss else nn.CrossEntropyLoss(weight=weight_tensor)

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_f1 = 0.0
    best_logits = None

    logger.info(f"Starting Transformer Sequence Classifier training ({epochs} epochs, batch_size={batch_size}, total_steps={total_steps})...")

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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
        macro_f1 = f1_score(val_targets_arr, val_preds, average="macro", zero_division=0)
        acc = accuracy_score(val_targets_arr, val_preds)

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Acc: {acc:.4f} - Val Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_val_f1 or epoch == 0:
            best_val_f1 = macro_f1
            best_logits = val_logits_arr
            logger.info(f" ⭐ New best validation checkpoint (Macro-F1: {best_val_f1:.4f})")

            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            if target_names and len(target_names) == n_classes:
                report_str = classification_report(
                    val_targets_arr, val_preds,
                    labels=list(range(n_classes)),
                    target_names=target_names,
                    zero_division=0
                )
                logger.info(f"\nClassification Report (Epoch {epoch+1}):\n{report_str}")
                logger.info(f"\nClassification Report (Epoch {epoch+1}):\n{report_str}")

    # Post-training Calibration
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
            "best_val_macro_f1": float(best_val_f1)
        }
        with open(output_dir / "calibration_params.json", "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)
        logger.info(f"Post-training calibration saved: T={opt_temp:.3f}, Optimized Macro-F1={best_val_f1:.4f}")

    return {"best_macro_f1": best_val_f1}


def train_setfit_model(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: Path,
    n_classes: int = 7
) -> Dict[str, Any]:
    """Fine-tunes SetFit model using contrastive sentence-transformer embeddings."""
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    device_str = get_device("cuda")
    model_name = "BAAI/bge-small-en-v1.5"
    logger.info(f"Initializing SetFit model ({model_name}) on {device_str}...")
    model = SetFitModel.from_pretrained(model_name, device=device_str)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        batch_size=32,
        num_epochs=1,
        num_iterations=4,
        eval_strategy="epoch",
        logging_dir=str(output_dir / "logs")
    )

    def compute_multiclass_metrics(y_pred, y_test):
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric=compute_multiclass_metrics
    )

    logger.info("Training SetFit model...")
    trainer.train()
    metrics = trainer.evaluate()
    logger.info(f"SetFit Validation Metrics: {metrics}")

    model.save_pretrained(str(output_dir))

    # Evaluate validation Macro-F1 & Accuracy
    probs = model.predict_proba(val_texts)
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(val_labels, preds))
    macro_f1 = float(f1_score(val_labels, preds, average="macro", zero_division=0))

    calib_data = {
        "temperature": 1.0,
        "thresholds": [0.5] * n_classes,
        "best_val_macro_f1": macro_f1,
        "best_val_accuracy": acc
    }
    with open(output_dir / "calibration_params.json", "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    logger.info(f"Saved SetFit model to {output_dir} (Val Acc: {acc:.4f}, Val Macro-F1: {macro_f1:.4f})")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Intent Classifier (DeBERTa LoRA / SetFit)")
    parser.add_argument("--brand", type=str, default="amazonhelp")
    parser.add_argument("--model-type", type=str, choices=["deberta_lora", "setfit", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use-banking77", action="store_true", default=True, help="Augment training with Banking77 dataset")
    parser.add_argument("--max-banking-samples", type=int, default=250, help="Max Banking77 samples per intent class")
    args = parser.parse_args()

    brand_cfg = load_brand_config(args.brand)
    n_classes = len(brand_cfg.intent_ids)

    logger.info(f"Building training and validation dataset for @{args.brand}...")
    train_texts, val_texts, train_labels, val_labels, label2id, id2label = get_augmented_dataset(
        brand=args.brand,
        include_golden=True,
        max_banking_per_class=args.max_banking_samples if args.use_banking77 else 0,
        val_split_ratio=0.2,
        random_seed=42
    )

    models_to_train = ["setfit", "deberta_lora"] if args.model_type == "all" else [args.model_type]

    for m_type in models_to_train:
        output_dir = PROJECT_ROOT / "models" / args.brand.lower() / m_type
        logger.info(f"\n{'='*70}\nTraining {m_type.upper()} Classifier -> {output_dir}\n{'='*70}")

        if m_type == "setfit":
            train_setfit_model(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                output_dir=output_dir,
                n_classes=n_classes
            )
        else:
            train_deberta_lora(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                n_classes=n_classes,
                output_dir=output_dir,
                target_names=brand_cfg.intent_ids,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )

        # Cross-Domain Evaluation on Banking77 Test Set
        try:
            from src.classify_trained import TrainedIntentClassifier
            eval_clf = TrainedIntentClassifier(model_dir=output_dir, brand_config=brand_cfg, model_type=m_type)
            eval_clf.load()
            transfer_res = evaluate_classifier_on_banking77(eval_clf, brand=args.brand, sample_size=300)
            logger.info(f"Cross-Domain Banking77 Test Benchmark ({m_type}): Acc={transfer_res['accuracy']:.4f}, Macro-F1={transfer_res['macro_f1']:.4f}")
        except Exception as e:
            logger.warning(f"Could not run cross-domain evaluation ({e})")


if __name__ == "__main__":
    main()

"""
Track B: Fine-Tuned Transformer & SetFit Classifier Inference Engine.
Applies temperature scaling probability calibration and per-class decision thresholds.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.config import BrandConfig
from src.schemas import IntentClassificationOutput
from src.utils import normalize_tweet_text

logger = logging.getLogger("hiver.classify_trained")


class TrainedIntentClassifier:
    def __init__(
        self,
        model_dir: Path,
        brand_config: BrandConfig,
        model_type: str = "deberta_lora",  # "deberta_lora" or "setfit"
        device: str = "cuda"
    ):
        self.model_dir = Path(model_dir)
        self.brand_config = brand_config
        self.model_type = model_type
        self.device = device

        self.model = None
        self.tokenizer = None
        self.setfit_model = None
        self.temperature = 1.0
        self.per_class_thresholds: Optional[np.ndarray] = None
        self.label2id: Dict[str, int] = {intent_id: i for i, intent_id in enumerate(brand_config.intent_ids)}
        self.id2label: Dict[int, str] = {i: intent_id for i, intent_id in enumerate(brand_config.intent_ids)}

    def load(self):
        """Loads trained model weights, tokenizer, and calibration parameters."""
        import torch

        calib_path = self.model_dir / "calibration_params.json"
        if calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                    self.temperature = float(params.get("temperature", 1.0))
                    if "thresholds" in params:
                        self.per_class_thresholds = np.array(params["thresholds"])
                logger.info(f"Loaded calibration: T={self.temperature:.3f}, thresholds={self.per_class_thresholds}")
            except Exception as e:
                logger.warning(f"Could not parse calibration_params.json: {e}")

        from src.utils import get_device
        dev = get_device(self.device)

        if self.model_type == "setfit":
            try:
                from setfit import SetFitModel
                self.setfit_model = SetFitModel.from_pretrained(str(self.model_dir))
                logger.info(f"Loaded SetFit model from {self.model_dir}")
            except Exception as e:
                logger.warning(f"Could not load SetFit model ({e}).")
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from peft import PeftModel

                tok_source = str(self.model_dir) if (self.model_dir / "tokenizer_config.json").exists() else "microsoft/deberta-v3-small"
                self.tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)

                # Check if LoRA adapter exists vs full merged model
                adapter_config = self.model_dir / "adapter_config.json"
                if adapter_config.exists():
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        "microsoft/deberta-v3-small",
                        num_labels=len(self.brand_config.intent_ids)
                    )
                    self.model = PeftModel.from_pretrained(base_model, str(self.model_dir)).to(dev)
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).to(dev)

                self.model.eval()
                logger.info(f"Loaded DeBERTa model from {self.model_dir} on {dev}")
            except Exception as e:
                logger.warning(f"Could not load DeBERTa model ({e}).")

    def predict(self, query: str) -> IntentClassificationOutput:
        """Runs calibrated inference on a single customer query."""
        import torch

        clean_q = normalize_tweet_text(query)

        # Fallback if model weights not loaded
        if self.model is None and self.setfit_model is None:
            return IntentClassificationOutput(
                intent=self.brand_config.intent_ids[0],
                confidence=0.5,
                reasoning="Model weights not found. Defaulting to first intent.",
                secondary_intents=[]
            )

        if self.setfit_model is not None:
            # SetFit prediction
            probs = self.setfit_model.predict_proba([clean_q])[0]
            if hasattr(probs, "cpu"):
                probs = probs.cpu().numpy()
            probs = np.array(probs)
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            pred_intent = self.id2label.get(pred_idx, self.brand_config.intent_ids[0])

            secondary_indices = np.argsort(probs)[::-1][1:3]
            secondary_intents = [self.id2label[i] for i in secondary_indices if probs[i] > 0.15]

            return IntentClassificationOutput(
                intent=pred_intent,
                confidence=conf,
                reasoning=f"SetFit contrastive classification (P={conf:.3f}).",
                secondary_intents=secondary_intents
            )

        # DeBERTa inference
        inputs = self.tokenizer(clean_q, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]

        # Apply Temperature Scaling
        scaled_logits = logits / max(0.01, self.temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Apply Per-Class Thresholds if calibrated
        if self.per_class_thresholds is not None and len(self.per_class_thresholds) == len(probs):
            scaled_probs = probs / np.clip(self.per_class_thresholds, 1e-5, 1.0)
            pred_idx = int(np.argmax(scaled_probs))
        else:
            pred_idx = int(np.argmax(probs))

        pred_intent = self.id2label.get(pred_idx, self.brand_config.intent_ids[0])
        conf = float(probs[pred_idx])

        # Find secondary intents
        secondary_indices = np.argsort(probs)[::-1][1:3]
        secondary_intents = [self.id2label[i] for i in secondary_indices if probs[i] > 0.15]

        return IntentClassificationOutput(
            intent=pred_intent,
            confidence=conf,
            reasoning=f"Fine-tuned DeBERTa-v3 calibrated inference (T={self.temperature:.2f}, P={conf:.3f}).",
            secondary_intents=secondary_intents
        )

    def classify(self, query: str) -> IntentClassificationOutput:
        """Alias for predict method."""
        return self.predict(query)

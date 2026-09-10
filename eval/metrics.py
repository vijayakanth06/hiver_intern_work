"""
Comprehensive Evaluation Metrics Suite.
Calculates Intent Macro/Weighted F1, Escalation Precision/Recall, Asymmetric Business Cost,
and Expected Calibration Error (ECE).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.schemas import EvaluationMetricsSummary


def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    """Computes standard multi-class classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    }


def compute_escalation_metrics(
    y_true_esc: List[bool],
    y_pred_esc: List[bool],
    cost_fa: float = 10.0,
    cost_fe: float = 1.50
) -> Dict[str, Any]:
    """
    Computes binary escalation classification metrics and business loss cost.
    y_true_esc: True if should escalate, False if auto-handle.
    """
    y_true_arr = np.array(y_true_esc, dtype=bool)
    y_pred_arr = np.array(y_pred_esc, dtype=bool)

    # False Auto-Handle (True=1, Pred=0) -> Dangerous
    fa_count = int(np.sum((y_true_arr == True) & (y_pred_arr == False)))
    # False Escalation (True=0, Pred=1) -> Labor cost
    fe_count = int(np.sum((y_true_arr == False) & (y_pred_arr == True)))
    # True Auto-Handle
    ta_count = int(np.sum((y_true_arr == False) & (y_pred_arr == False)))
    # True Escalation
    te_count = int(np.sum((y_true_arr == True) & (y_pred_arr == True)))

    total_cost = (fa_count * cost_fa) + (fe_count * cost_fe)
    avg_cost = total_cost / max(1, len(y_true_esc))

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "false_auto_handle_count": fa_count,
        "false_escalation_count": fe_count,
        "true_auto_handle_count": ta_count,
        "true_escalation_count": te_count,
        "total_business_cost": total_cost,
        "average_cost_per_ticket": avg_cost
    }


def compute_expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across probability bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / len(labels)) * np.abs(bin_acc - bin_conf)

    return float(ece)

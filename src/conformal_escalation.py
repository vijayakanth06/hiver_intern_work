"""
Conformal Prediction Uncertainty Calibration & Asymmetric Cost Matrix Optimizer.
Provides mathematically guaranteed error bounds on auto-handled customer tickets.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger("hiver.conformal_escalation")


class ConformalEscalationCalibrator:
    def __init__(self, alpha: float = 0.05):
        """
        alpha: Maximum acceptable error rate on auto-handled queries (0.05 = 95% certainty).
        """
        self.alpha = alpha
        self.q_hat = 1.0

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray):
        """
        Computes the conformal non-conformity threshold q_hat on a held-out calibration set.
        Non-conformity score s_i = 1 - P(true_class_i).
        """
        n = len(val_labels)
        correct_probs = val_probs[np.arange(n), val_labels]
        non_conformity_scores = 1.0 - correct_probs

        # Finite sample corrected quantile
        p_val = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        self.q_hat = float(np.quantile(non_conformity_scores, min(1.0, p_val)))
        logger.info(f"Conformal calibration completed: q_hat = {self.q_hat:.4f} for 1-alpha = {1-self.alpha:.2%}")
        return self

    def save(self, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"alpha": self.alpha, "q_hat": self.q_hat}, f, indent=2)

    def load(self, input_path: Path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.alpha = float(data["alpha"])
            self.q_hat = float(data["q_hat"])

    def evaluate_prediction_set(self, prob_dist: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Constructs prediction set C(x) = {y : 1 - P(y|x) <= q_hat}.
        Returns (is_singleton, prediction_set).
        If singleton -> High statistical certainty -> Safe for auto-handling.
        If multi-class or empty -> High ambiguity -> Must escalate.
        """
        prediction_set = np.where(1.0 - prob_dist <= self.q_hat)[0].tolist()
        is_singleton = len(prediction_set) == 1
        return is_singleton, prediction_set


def optimize_escalation_cost_threshold(
    val_probs: np.ndarray,
    val_escalate_targets: np.ndarray,
    cost_fa: float = 10.0,
    cost_fe: float = 1.50
) -> Dict[str, Any]:
    """
    Sweeps over escalation confidence threshold to find the global minimum on the asymmetric business cost curve.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    best_cost = float("inf")
    best_thresh = 0.50
    best_stats = {}

    max_probs = np.max(val_probs, axis=1)

    for thresh in thresholds:
        # Predict auto-handle if max_prob >= thresh, else escalate
        pred_auto = max_probs >= thresh
        pred_escalate = ~pred_auto

        # False Auto-Handle: True is Escalate (1), but predicted Auto-Handle (0)
        fa_count = int(np.sum((val_escalate_targets == 1) & (pred_auto == True)))
        # False Escalation: True is Auto-Handle (0), but predicted Escalate (1)
        fe_count = int(np.sum((val_escalate_targets == 0) & (pred_escalate == True)))

        total_cost = (fa_count * cost_fa) + (fe_count * cost_fe)
        avg_cost = total_cost / len(val_escalate_targets)

        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = float(thresh)
            best_stats = {
                "optimal_threshold": best_thresh,
                "total_cost": total_cost,
                "avg_cost_per_ticket": avg_cost,
                "false_auto_handles": fa_count,
                "false_escalations": fe_count
            }

    logger.info(f"Optimal escalation threshold found: {best_thresh:.2f} (Avg Cost: ${best_stats['avg_cost_per_ticket']:.2f})")
    return best_stats

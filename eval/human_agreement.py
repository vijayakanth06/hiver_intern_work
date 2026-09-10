"""
Human Agreement & Annotation Quality Analysis.
Calculates Cohen's Kappa, percent agreement, and confusion distribution
between LLM pre-labeling and verified human annotations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score


def compute_annotation_agreement(df_golden: pd.DataFrame, df_prelabels: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes agreement statistics between raw LLM prelabels and verified golden annotations.
    """
    merged = pd.merge(df_golden, df_prelabels, on="example_id", suffixes=("_gold", "_pre"))

    y_gold_intent = merged["ground_truth_intent"].tolist()
    y_pre_intent = merged["llm_predicted_intent"].tolist()

    y_gold_esc = merged["ground_truth_escalate"].astype(bool).tolist()
    y_pre_esc = merged["rule_escalate"].astype(bool).tolist()

    intent_kappa = float(cohen_kappa_score(y_gold_intent, y_pre_intent))
    intent_agreement = float(accuracy_score(y_gold_intent, y_pre_intent))

    esc_kappa = float(cohen_kappa_score(y_gold_esc, y_pre_esc))
    esc_agreement = float(accuracy_score(y_gold_esc, y_pre_esc))

    return {
        "total_samples": len(merged),
        "intent_cohens_kappa": intent_kappa,
        "intent_percent_agreement": intent_agreement,
        "escalation_cohens_kappa": esc_kappa,
        "escalation_percent_agreement": esc_agreement,
        "human_corrections_count": int(np.sum(np.array(y_gold_intent) != np.array(y_pre_intent)))
    }

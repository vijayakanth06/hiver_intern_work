"""
Unit Tests for Escalation Engine and Conformal Logic.
"""

from src.config import get_app_config
from src.escalate import EscalationEngine
from src.conformal_escalation import ConformalEscalationCalibrator, optimize_escalation_cost_threshold
import numpy as np


def test_escalation_security_intent():
    app_config = get_app_config("amazonhelp")
    engine = EscalationEngine(app_config)

    # Intent account_security_access is non-auto-handle
    out = engine.evaluate(
        query="I cannot log in to my account, password reset failed.",
        predicted_intent="account_security_access",
        confidence=0.95
    )
    assert out.should_escalate is True
    assert "Security Policy" in out.reason


def test_escalation_legal_keyword():
    app_config = get_app_config("amazonhelp")
    engine = EscalationEngine(app_config)

    out = engine.evaluate(
        query="This is fraud and my lawyer will be contacting you!",
        predicted_intent="order_tracking_delivery",
        confidence=0.95
    )
    assert out.should_escalate is True
    assert "lawyer" in out.reason


def test_escalation_auto_handle():
    app_config = get_app_config("amazonhelp")
    engine = EscalationEngine(app_config)

    out = engine.evaluate(
        query="Can you help me check tracking for my package please?",
        predicted_intent="order_tracking_delivery",
        confidence=0.92,
        turn_count=1,
        rag_similarity=0.88
    )
    assert out.should_escalate is False
    assert out.risk_level == "LOW"


def test_conformal_calibrator():
    calibrator = ConformalEscalationCalibrator(alpha=0.05)
    probs = np.array([
        [0.9, 0.1],
        [0.85, 0.15],
        [0.95, 0.05],
        [0.8, 0.2]
    ])
    labels = np.array([0, 0, 0, 0])
    calibrator.calibrate(probs, labels)
    assert calibrator.q_hat <= 0.25

    # Singleton test
    is_singleton, pred_set = calibrator.evaluate_prediction_set(np.array([0.95, 0.05]))
    assert is_singleton is True

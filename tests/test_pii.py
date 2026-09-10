"""
Unit Tests for PII Sanitization and Normalization.
"""

from src.utils import sanitize_pii, normalize_tweet_text, is_english_text


def test_sanitize_email():
    text = "My order is lost, email me at customer.test@example.com please"
    cleaned = sanitize_pii(text)
    assert "[EMAIL]" in cleaned
    assert "customer.test@example.com" not in cleaned


def test_sanitize_phone():
    text = "Call me at (555) 123-4567 or 555-987-6543 regarding my shipment"
    cleaned = sanitize_pii(text)
    assert "[PHONE]" in cleaned
    assert "555-123-4567" not in cleaned


def test_sanitize_order_id():
    text = "My Amazon order 123-4567890-1234567 has not arrived yet!"
    cleaned = sanitize_pii(text)
    assert "[ORDER_ID]" in cleaned
    assert "123-4567890-1234567" not in cleaned


def test_japanese_filtering():
    japanese_text = "注文した商品がまだ届きません。確認してください。"
    english_text = "Where is my package? It has not arrived."
    assert is_english_text(japanese_text) is False
    assert is_english_text(english_text) is True

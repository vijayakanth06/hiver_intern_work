"""
Text Utilities, PII Sanitization, Sentiment Analysis, and Language Detection.
Engineered for high-throughput regex compilation and robust NLP preprocessing.
"""

import re
import html
from typing import Tuple, Dict, Any, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER_ANALYZER = None

try:
    from langdetect import detect as _lang_detect
except ImportError:
    _lang_detect = None


# Precompiled PII regex patterns for ultra-fast matching
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
RE_PHONE = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
RE_ORDER_ID = re.compile(r'\b\d{3}-\d{7}-\d{7}\b|\b\d{17}\b|\b[A-Z0-9]{10,12}\b')
RE_CREDIT_CARD = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
RE_TWITTER_HANDLE = re.compile(r'@\w+')
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_MULTIPLE_SPACES = re.compile(r'\s+')


def sanitize_pii(text: str) -> str:
    """
    Redacts personal identifiable information (PII) including email addresses,
    phone numbers, Amazon order numbers, and credit card patterns with standard tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities
    cleaned = html.unescape(text)

    # Redact sensitive data
    cleaned = RE_EMAIL.sub("[EMAIL]", cleaned)
    cleaned = RE_CREDIT_CARD.sub("[CREDIT_CARD]", cleaned)
    cleaned = RE_ORDER_ID.sub("[ORDER_ID]", cleaned)
    cleaned = RE_PHONE.sub("[PHONE]", cleaned)

    return cleaned.strip()


def normalize_tweet_text(text: str, remove_handles: bool = False) -> str:
    """
    Cleans tweet noise, unescapes characters, and collapses whitespace while preserving
    contextual punctuation and casing.
    """
    if not text:
        return ""

    cleaned = html.unescape(text)
    if remove_handles:
        cleaned = RE_TWITTER_HANDLE.sub("", cleaned)

    cleaned = RE_MULTIPLE_SPACES.sub(" ", cleaned)
    return cleaned.strip()


def compute_sentiment_score(text: str) -> float:
    """
    Returns the VADER compound sentiment score in [-1.0, 1.0].
    -1.0 = extremely negative/angry, +1.0 = extremely positive.
    """
    if not text or _VADER_ANALYZER is None:
        return 0.0
    scores = _VADER_ANALYZER.polarity_scores(text)
    return float(scores.get("compound", 0.0))


def is_english_text(text: str) -> bool:
    """
    Detects if the given text is English.
    Returns True if detected as English or if detection fails/is unavailable.
    """
    if not text or len(text.strip()) < 5:
        return True

    # Fast heuristic check for Asian script / Japanese characters
    # (AmazonHelp contains many Japanese tweets in TWCS)
    for ch in text:
        # Check for Hiragana, Katakana, CJK Unified Ideographs
        if '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff':
            return False

    if _lang_detect is None:
        return True

    try:
        lang = _lang_detect(text)
        return lang == "en"
    except Exception:
        return True


def get_device(preferred: str = "auto") -> str:
    """
    Returns the optimal compute device across Linux, macOS, and Windows:
    - 'cuda' if NVIDIA GPU is present and supported
    - 'mps' if Apple Silicon (macOS Metal Performance Shaders) is available
    - 'cpu' fallback for universal cross-platform execution
    """
    import torch
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


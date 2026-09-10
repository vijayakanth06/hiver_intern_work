"""
Configuration loader with Pydantic V2 validation.
Combines base.yaml, brand configs, and .env environment variables.
"""

import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.schemas import IntentDefinition, BrandPersona

# Silence noisy third-party loader logs
for noisy_logger in ["faiss", "faiss.loader", "urllib3", "filelock", "huggingface_hub", "transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class BrandConfig(BaseModel):
    id: str
    name: str
    display_name: str
    domain: str
    language: str = "en"
    filter_non_english: bool = True
    intents: List[IntentDefinition]
    persona: BrandPersona
    sampling: Dict[str, Any] = Field(default_factory=dict)

    @property
    def intent_ids(self) -> List[str]:
        return [intent.id for intent in self.intents]

    @property
    def intent_dict(self) -> Dict[str, IntentDefinition]:
        return {intent.id: intent for intent in self.intents}

    def get_intent(self, intent_id: str) -> Optional[IntentDefinition]:
        return self.intent_dict.get(intent_id)


class BasePipelineConfig(BaseModel):
    pipeline: Dict[str, Any]
    llm: Dict[str, Any]
    embeddings: Dict[str, Any]
    retrieval: Dict[str, Any]
    classification: Dict[str, Any]
    escalation: Dict[str, Any]
    semantic_cache: Dict[str, Any]
    eval: Dict[str, Any]


class AppConfig(BaseModel):
    base: BasePipelineConfig
    brand: BrandConfig
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    project_root: Path = PROJECT_ROOT


def load_brand_config(brand_name: str) -> BrandConfig:
    brand_path = PROJECT_ROOT / "configs" / "brands" / f"{brand_name.lower()}.yaml"
    if not brand_path.exists():
        raise FileNotFoundError(f"Brand configuration file not found at: {brand_path}")

    with open(brand_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return BrandConfig(
        id=data["brand"]["id"],
        name=data["brand"]["name"],
        display_name=data["brand"]["display_name"],
        domain=data["brand"]["domain"],
        language=data["brand"].get("language", "en"),
        filter_non_english=data["brand"].get("filter_non_english", True),
        intents=[IntentDefinition(**item) for item in data["intents"]],
        persona=BrandPersona(**data["persona"]),
        sampling=data.get("sampling", {})
    )


def load_base_config() -> BasePipelineConfig:
    base_path = PROJECT_ROOT / "configs" / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return BasePipelineConfig(**data)


def get_app_config(brand_name: str = "amazonhelp") -> AppConfig:
    base = load_base_config()
    brand = load_brand_config(brand_name)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    return AppConfig(
        base=base,
        brand=brand,
        groq_api_key=groq_key if groq_key else None,
        openrouter_api_key=openrouter_key if openrouter_key else None,
        project_root=PROJECT_ROOT
    )

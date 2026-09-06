"""
Multilingual interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class LanguageCode(str, Enum):
    """Supported languages (ISO 639-1)."""
    EN = "en"
    HI = "hi"
    TA = "ta"
    BN = "bn"
    TE = "te"
    MR = "mr"
    GU = "gu"
    KN = "kn"
    ML = "ml"
    PA = "pa"
    OR = "or"
    AS = "as"
    UR = "ur"


QueryLanguage = LanguageCode


class TranslationRequest(BaseModel):
    """Request for translation."""
    text: str
    source_language: LanguageCode
    target_language: LanguageCode
    domain: str = "legal"


class TranslationResult(BaseModel):
    """Result from translation."""
    translated_text: str
    source_language: LanguageCode
    target_language: LanguageCode
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ITranslator(ABC):
    """Interface for translation components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Translator name."""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[LanguageCode]:
        """Supported languages."""
        pass
    
    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text."""
        pass
    
    @abstractmethod
    async def translate_batch(self, requests: List[TranslationRequest]) -> List[TranslationResult]:
        """Batch translation."""
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> LanguageCode:
        """Detect language of text."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        pass


class TranslatorMetrics(BaseModel):
    """Metrics for translator evaluation."""
    latency_ms: float = 0.0
    throughput_chars_per_sec: float = 0.0
    bleu_score: float = 0.0
    domain_accuracy: float = 0.0
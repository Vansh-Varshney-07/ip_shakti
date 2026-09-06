"""
Authority system interfaces for IP-SAKTI.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: str = ""
    document_type: str = "legal"
    jurisdiction: str = "INDIA"
    language: str = "en"
    mime_type: Optional[str] = None
    source_path: Optional[str] = None
    version: str = "1.0"
    effective_date: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Chunk model for authority."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    authority_score: float = 0.0
    jurisdiction: Optional[str] = None
    document_type: Optional[str] = None
    embedding: Optional[List[float]] = None


class AuthorityTier(int, Enum):
    """Authority tiers (1=highest)."""
    TIER_1_SUPREME = 1      # Supreme Court, Constitution, Primary Acts
    TIER_2_HIGH_COURT = 2   # High Courts, Major Regulations
    TIER_3_TRIBUNAL = 3     # Tribunals, Rules, Guidelines
    TIER_4_SECONDARY = 4    # Secondary sources, Commentary
    TIER_5_TERTIARY = 5     # News, Blogs, Unverified


class AuthoritySourceType(str, Enum):
    """Types of authority sources."""
    CONSTITUTION = "constitution"
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    TRIBUNAL = "tribunal"
    STATUTE = "statute"
    REGULATION = "regulation"
    RULE = "rule"
    GUIDELINE = "guideline"
    NOTIFICATION = "notification"
    CIRCULAR = "circular"
    INTERNATIONAL_TREATY = "international_treaty"
    CASE_LAW = "case_law"
    LEGAL_COMMENTARY = "legal_commentary"
    GOVERNMENT_REPORT = "government_report"
    ACADEMIC = "academic"
    NEWS = "news"
    OTHER = "other"


class AuthorityScore(BaseModel):
    """Authority score for a document or chunk."""
    tier: AuthorityTier
    source_type: AuthoritySourceType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    court_level: Optional[str] = None
    jurisdiction: Optional[str] = None
    year: Optional[int] = None
    is_binding: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthorityContext(BaseModel):
    """Context for authority evaluation."""
    query: str
    jurisdiction: Optional[str] = None
    practice_area: Optional[str] = None
    prefer_recent: bool = True
    min_tier: Optional[AuthorityTier] = None


class IAuthoritySystem(ABC):
    """Interface for authority system."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Authority system name."""
        pass
    
    @abstractmethod
    async def evaluate_document(self, metadata: DocumentMetadata) -> AuthorityScore:
        """Evaluate authority of a document."""
        pass
    
    @abstractmethod
    async def evaluate_chunk(self, chunk: Chunk) -> AuthorityScore:
        """Evaluate authority of a chunk."""
        pass
    
    @abstractmethod
    async def evaluate_batch(self, chunks: List[Chunk]) -> List[AuthorityScore]:
        """Batch evaluate authority."""
        pass
    
    @abstractmethod
    async def get_authority_weight(self, score: AuthorityScore, context: Optional[AuthorityContext] = None) -> float:
        """Get authority weight for fusion."""
        pass
    
    @abstractmethod
    async def rank_by_authority(self, chunks: List[Chunk], scores: List[AuthorityScore]) -> List[int]:
        """Return indices sorted by authority (highest first)."""
        pass
    
    @abstractmethod
    async def get_tier_distribution(self, chunks: List[Chunk]) -> Dict[AuthorityTier, int]:
        """Get distribution of authority tiers."""
        pass


class AuthorityMetrics(BaseModel):
    """Metrics for authority system evaluation."""
    tier_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_score: float = 0.0
    evaluation_latency_ms: float = 0.0
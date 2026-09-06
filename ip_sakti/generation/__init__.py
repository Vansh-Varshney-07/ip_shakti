"""
IP-SAKTI Generation Package
"""

from ip_sakti.generation.citation_first import (
    CitationFirstGenerator,
    LegalCitationGenerator,
    PatentCitationGenerator,
    StreamingCitationGenerator,
    CitationConfig,
    CitationStyle,
    CitationRequirement,
    Citation,
    CitedAnswer,
    CitationFormatter,
    CitationVerifier,
    ClaimExtractor,
    create_citation_generator,
)

from ip_sakti.generation.strategies.citation_generator import (
    GenerationStrategy,
    GenerationConfig,
    GeneratedAnswer,
    BaseGenerator,
    StructuredGenerator,
    GeneratorFactory,
)

__all__ = [
    # Original citation_first exports
    "CitationFirstGenerator",
    "LegalCitationGenerator",
    "PatentCitationGenerator",
    "StreamingCitationGenerator",
    "CitationConfig",
    "CitationStyle",
    "CitationRequirement",
    "Citation",
    "CitedAnswer",
    "CitationFormatter",
    "CitationVerifier",
    "ClaimExtractor",
    "create_citation_generator",
    # New strategy exports
    "GenerationStrategy",
    "GenerationConfig",
    "GeneratedAnswer",
    "BaseGenerator",
    "StructuredGenerator",
    "GeneratorFactory",
]
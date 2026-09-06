"""
IP-SAKTI Chunking Strategy - Fast built-in recursive text splitter.
No external dependencies - uses only Python standard library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import uuid
import re
import logging

from ip_sakti.config.loader import get_settings
from ip_sakti.core.models import Document, DocumentChunk, DocumentType

logger = logging.getLogger(__name__)


class ChunkingStrategyType(str, Enum):
    """Types of chunking strategies."""
    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    strategy: ChunkingStrategyType = ChunkingStrategyType.RECURSIVE
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    length_function: Callable[[str], int] = len
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


class BaseChunker(ABC):
    """Abstract base class for chunkers."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    async def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text into DocumentChunks."""
        pass

    def _create_chunk(
        self,
        text: str,
        index: int,
        document_id: str,
        start_char: int,
        end_char: int,
        metadata: Dict[str, Any],
    ) -> DocumentChunk:
        """Create a DocumentChunk with standard fields."""
        return DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            content=text,
            chunk_index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata.copy(),
        )


class RecursiveChunker(BaseChunker):
    """
    Fast recursive character text splitter using Python standard library.
    
    Mimics LangChain's RecursiveCharacterTextSplitter behavior:
    1. Tries to split by separators in order (paragraphs, sentences, words, characters)
    2. Recursively splits oversized chunks using the next separator
    3. Merges splits back together respecting chunk_size and chunk_overlap
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._separators = config.separators

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text using the given separators."""
        if not separators:
            # Character-level splitting as last resort
            return [text[i:i+self.config.chunk_size] for i in range(0, len(text), self.config.chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator == "":
            # Character level - split into individual characters
            splits = list(text)
        else:
            splits = text.split(separator)
        
        # Rejoin with separator (except for empty separator)
        result = []
        for split in splits:
            if len(split) <= self.config.chunk_size:
                result.append(split)
            elif remaining_separators:
                # Recursively split oversized chunks
                sub_splits = self._split_text(split, remaining_separators)
                result.extend(sub_splits)
            else:
                # No more separators, force split by chunk_size
                for i in range(0, len(split), self.config.chunk_size):
                    result.append(split[i:i+self.config.chunk_size])
        
        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits respecting chunk_size and chunk_overlap."""
        if not splits:
            return []
        
        merged = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk_size, save current and start new
            if current_chunk and len(current_chunk) + len(split) > self.config.chunk_size:
                merged.append(current_chunk)
                # Start new chunk with overlap from end of previous
                overlap_start = max(0, len(current_chunk) - self.config.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + split
            else:
                if current_chunk:
                    current_chunk += split
                else:
                    current_chunk = split
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged

    async def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text using fast recursive splitting strategy."""
        if not text or not text.strip():
            return []

        # Fast split using regex-based separators
        initial_splits = self._split_text(text, self._separators)
        merged_splits = self._merge_splits(initial_splits)

        # Create DocumentChunk objects
        chunks = []
        document_id = metadata.get('document_id', '')
        char_offset = 0

        for i, chunk_text in enumerate(merged_splits):
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    text=chunk_text,
                    index=i,
                    document_id=document_id,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    metadata=metadata,
                )
                chunks.append(chunk)
                char_offset += len(chunk_text)

        return chunks


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking using Python standard library."""

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)

    async def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text using fixed-size splitting."""
        if not text or not text.strip():
            return []

        chunks = []
        document_id = metadata.get('document_id', '')
        char_offset = 0
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        i = 0
        while char_offset < len(text):
            end = min(char_offset + chunk_size, len(text))
            chunk_text = text[char_offset:end]

            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    text=chunk_text,
                    index=i,
                    document_id=document_id,
                    start_char=char_offset,
                    end_char=end,
                    metadata=metadata,
                )
                chunks.append(chunk)
                i += 1

            char_offset += chunk_size - overlap

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking using embeddings similarity (placeholder - falls back to recursive)."""

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._recursive = RecursiveChunker(config)

    async def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        # Fall back to recursive for now
        return await self._recursive.chunk(text, metadata)


class ChunkingStrategy:
    """Main chunking strategy orchestrator."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._chunker = self._create_chunker()

    def _create_chunker(self) -> BaseChunker:
        strategy_map = {
            ChunkingStrategyType.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategyType.RECURSIVE: RecursiveChunker,
            ChunkingStrategyType.SEMANTIC: SemanticChunker,
        }

        chunker_class = strategy_map.get(self.config.strategy, RecursiveChunker)
        return chunker_class(self.config)

    async def chunk_document(
        self,
        document: Document,
        text: str,
    ) -> List[DocumentChunk]:
        metadata = {
            'document_id': document.id,
            'document_type': document.metadata.document_type,
            'jurisdiction': document.metadata.jurisdiction,
            'legal_document': True,
        }

        if document.source.authority_tier:
            metadata['authority_tier'] = document.source.authority_tier.value

        return await self._chunker.chunk(text, metadata)

    async def chunk_text(
        self,
        text: str,
        document_id: str,
        document_type: DocumentType = DocumentType.PATENT,
        jurisdiction=None,
    ) -> List[DocumentChunk]:
        metadata = {
            'document_id': document_id,
            'document_type': document_type,
            'jurisdiction': jurisdiction,
            'legal_document': True,
        }
        return await self._chunker.chunk(text, metadata)


def create_chunker(
    strategy: ChunkingStrategyType = ChunkingStrategyType.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs
) -> ChunkingStrategy:
    """Factory function to create a chunking strategy."""
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    return ChunkingStrategy(config)
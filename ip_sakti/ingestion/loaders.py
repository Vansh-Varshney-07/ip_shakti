"""
IP-SAKTI Document Loaders - Wrapper around LangChain's document loaders.
Uses langchain-community for loading various document formats.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import mimetypes

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logger = logging.getLogger(__name__)


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    XML = "xml"
    UNKNOWN = "unknown"


@dataclass
class LoadedDocument:
    """Standardized loaded document."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    format: DocumentFormat = DocumentFormat.UNKNOWN


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    async def load(self, file_path: str) -> List[LoadedDocument]:
        """Load document(s) from file path."""
        pass
    
    @abstractmethod
    def supports_format(self, format: DocumentFormat) -> bool:
        """Check if loader supports a format."""
        pass


class TextLoaderWrapper(BaseDocumentLoader):
    """Wrapper around LangChain's TextLoader."""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
        self._loader_class = None
    
    def _get_loader(self):
        if self._loader_class is None:
            from langchain_community.document_loaders import TextLoader
            self._loader_class = TextLoader
        return self._loader_class
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        try:
            loader_class = self._get_loader()
            loader = loader_class(file_path, encoding=self.encoding)
            docs = loader.load()
        except ImportError:
            logger.info("langchain-community unavailable; using native text loader for %s", file_path)
            with open(file_path, "r", encoding=self.encoding, errors="ignore") as handle:
                docs = [type("TextDocument", (), {"page_content": handle.read(), "metadata": {"source": file_path}})()]
        
        return [
            LoadedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                source_path=file_path,
                format=DocumentFormat.TXT,
            )
            for doc in docs
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return format == DocumentFormat.TXT


class PDFLoaderWrapper(BaseDocumentLoader):
    """Wrapper around LangChain's PyPDFLoader with HTML fallback."""
    
    def __init__(self):
        self._loader_class = None
    
    def _get_loader(self):
        if self._loader_class is None:
            from langchain_community.document_loaders import PyPDFLoader
            self._loader_class = PyPDFLoader
        return self._loader_class
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        try:
            loader_class = self._get_loader()
            loader = loader_class(file_path)
            docs = loader.load()
        except ImportError:
            logger.info("langchain-community unavailable; using pdfplumber loader for %s", file_path)
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                docs = [
                    type("PDFDocument", (), {"page_content": page.extract_text() or "", "metadata": {"source": file_path}})()
                    for page in pdf.pages
                ]
        except Exception as e:
            # Fallback to HTML loader for files with .pdf extension that are actually HTML
            logger.warning(f"PyPDFLoader failed for {file_path}: {e}, trying HTML loader")
            if not HAS_BS4:
                raise ImportError("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style", "meta", "link", "noscript"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)
            
            return [
                LoadedDocument(
                    content=text,
                    metadata={'source': file_path},
                    source_path=file_path,
                    format=DocumentFormat.PDF,
                )
            ]
        
        return [
            LoadedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                source_path=file_path,
                format=DocumentFormat.PDF,
            )
            for doc in docs
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return format == DocumentFormat.PDF


class DocxLoaderWrapper(BaseDocumentLoader):
    """Wrapper around LangChain's UnstructuredWordDocumentLoader."""
    
    def __init__(self):
        self._loader_class = None
    
    def _get_loader(self):
        if self._loader_class is None:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            self._loader_class = UnstructuredWordDocumentLoader
        return self._loader_class
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        loader_class = self._get_loader()
        loader = loader_class(file_path)
        docs = loader.load()
        
        return [
            LoadedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                source_path=file_path,
                format=DocumentFormat.DOCX,
            )
            for doc in docs
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return format == DocumentFormat.DOCX


class HTMLLoaderWrapper(BaseDocumentLoader):
    """Wrapper for HTML documents using BeautifulSoup (no unstructured required)."""
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        if not HAS_BS4:
            raise ImportError("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link", "noscript"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        return [
            LoadedDocument(
                content=text,
                metadata={'source': file_path},
                source_path=file_path,
                format=DocumentFormat.HTML,
            )
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return format == DocumentFormat.HTML


class MarkdownLoaderWrapper(BaseDocumentLoader):
    """Wrapper for Markdown documents using simple text parsing."""
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Simple markdown cleaning - just return as text
        return [
            LoadedDocument(
                content=content,
                metadata={'source': file_path},
                source_path=file_path,
                format=DocumentFormat.MARKDOWN,
            )
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return format == DocumentFormat.MARKDOWN


class XMLLoaderWrapper(BaseDocumentLoader):
    """Wrapper for XML documents using simple text parsing."""
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        if not HAS_BS4:
            raise ImportError("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = f.read()
        
        soup = BeautifulSoup(xml_content, 'xml')
        text = soup.get_text(separator='\n', strip=True)
        
        return [
            LoadedDocument(
                content=text,
                metadata={'source': file_path},
                source_path=file_path,
                format=DocumentFormat.XML,
            )
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return format == DocumentFormat.XML


class FallbackLoaderWrapper(BaseDocumentLoader):
    """Fallback loader that reads files as plain text."""
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return [
            LoadedDocument(
                content=content,
                metadata={'source': file_path},
                source_path=file_path,
                format=DocumentFormat.UNKNOWN,
            )
        ]
    
    def supports_format(self, format: DocumentFormat) -> bool:
        return True  # Supports everything as fallback


class DocumentLoaderRegistry:
    """Registry for document loaders."""
    
    def __init__(self):
        self._loaders: List[BaseDocumentLoader] = []
        self._format_map: Dict[DocumentFormat, BaseDocumentLoader] = {}
        self._extension_map: Dict[str, DocumentFormat] = {
            '.pdf': DocumentFormat.PDF,
            '.txt': DocumentFormat.TXT,
            '.docx': DocumentFormat.DOCX,
            '.html': DocumentFormat.HTML,
            '.htm': DocumentFormat.HTML,
            '.md': DocumentFormat.MARKDOWN,
            '.markdown': DocumentFormat.MARKDOWN,
            '.xml': DocumentFormat.XML,
        }
        
        # Register default loaders
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register all default loaders."""
        loaders = [
            TextLoaderWrapper(),
            PDFLoaderWrapper(),
            DocxLoaderWrapper(),
            HTMLLoaderWrapper(),
            MarkdownLoaderWrapper(),
            XMLLoaderWrapper(),
            FallbackLoaderWrapper(),  # Fallback
        ]
        
        for loader in loaders:
            self.register_loader(loader)
    
    def register_loader(self, loader: BaseDocumentLoader):
        """Register a loader."""
        self._loaders.append(loader)
        
        # Map supported formats
        for fmt in DocumentFormat:
            if loader.supports_format(fmt) and fmt not in self._format_map:
                self._format_map[fmt] = loader
    
    def get_loader_for_format(self, format: DocumentFormat) -> Optional[BaseDocumentLoader]:
        """Get loader for a specific format."""
        return self._format_map.get(format)
    
    def get_loader_for_extension(self, extension: str) -> Optional[BaseDocumentLoader]:
        """Get loader for a file extension."""
        format = self._extension_map.get(extension.lower())
        if format:
            return self.get_loader_for_format(format)
        # Try fallback
        return self.get_loader_for_format(DocumentFormat.UNKNOWN)
    
    def detect_format(self, file_path: str) -> DocumentFormat:
        """Detect document format from file path."""
        ext = Path(file_path).suffix.lower()
        return self._extension_map.get(ext, DocumentFormat.UNKNOWN)
    
    async def load(self, file_path: str) -> List[LoadedDocument]:
        """Load document using appropriate loader."""
        format = self.detect_format(file_path)
        loader = self.get_loader_for_format(format)
        
        if not loader:
            logger.warning(f"No specific loader for {format}, using fallback")
            loader = self.get_loader_for_format(DocumentFormat.UNKNOWN)
        
        if not loader:
            raise ValueError(f"No loader available for {file_path}")
        
        try:
            docs = await loader.load(file_path)
            logger.info(f"Loaded {len(docs)} document(s) from {file_path} using {type(loader).__name__}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load {file_path} with {type(loader).__name__}: {e}")
            # Try fallback
            fallback = self.get_loader_for_format(DocumentFormat.UNKNOWN)
            if fallback and fallback != loader:
                logger.info(f"Trying fallback loader...")
                return await fallback.load(file_path)
            raise


# Global registry instance
_registry = DocumentLoaderRegistry()


def get_loader_registry() -> DocumentLoaderRegistry:
    """Get the global loader registry."""
    return _registry


async def load_document(file_path: str) -> List[LoadedDocument]:
    """Convenience function to load a document."""
    return await _registry.load(file_path)


def detect_format(file_path: str) -> DocumentFormat:
    """Convenience function to detect format."""
    return _registry.detect_format(file_path)

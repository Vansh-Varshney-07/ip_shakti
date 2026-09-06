"""
Phase 8: Multimodal Document Processing Module
"""

import asyncio
import base64
import io
import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Types of document modalities."""
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    TABLE = "table"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"  # JSON, XML, CSV


@dataclass
class ProcessedContent:
    """Content extracted from a document."""
    modality: ModalityType
    text: str = ""
    images: List[Dict[str, Any]] = field(default_factory=list)  # base64 encoded
    tables: List[Dict[str, Any]] = field(default_factory=list)  # structured table data
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: int = 0
    confidence: float = 1.0


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    # OCR settings
    enable_ocr: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["eng", "hin"])
    ocr_dpi: int = 300
    # Table extraction
    extract_tables: bool = True
    table_format: str = "markdown"  # markdown, csv, json
    # Image processing
    extract_images: bool = True
    image_format: str = "png"
    image_max_size: Tuple[int, int] = (1920, 1920)
    # PDF settings
    pdf_extract_images: bool = True
    pdf_ocr_threshold: int = 50  # min text chars per page before OCR
    # Chunking
    chunk_by_page: bool = True
    max_chunk_size: int = 2000
    chunk_overlap: int = 200


class BaseProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the processor."""
        pass
    
    @abstractmethod
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        **kwargs,
    ) -> ProcessedContent:
        """Process a document and extract content."""
        pass
    
    @abstractmethod
    def supported_mimetypes(self) -> List[str]:
        """Return list of supported MIME types."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass


class TextProcessor(BaseProcessor):
    """Processor for plain text files."""
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        **kwargs,
    ) -> ProcessedContent:
        if isinstance(file_path, (str, Path)):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        else:
            file_path.seek(0)
            text = file_path.read().decode('utf-8', errors='replace')
        
        return ProcessedContent(
            modality=ModalityType.TEXT,
            text=text,
            metadata={"source_type": "text"},
        )
    
    def supported_mimetypes(self) -> List[str]:
        return ["text/plain", "text/markdown", "text/csv", "application/json"]
    
    async def close(self) -> None:
        self._initialized = False


class PDFProcessor(BaseProcessor):
    """Processor for PDF documents with OCR and table extraction."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self._pdfplumber = None
        self._pymupdf = None
        self._ocr_reader = None
    
    async def initialize(self) -> None:
        try:
            import pdfplumber
            self._pdfplumber = pdfplumber
            logger.info("pdfplumber loaded")
        except ImportError:
            logger.warning("pdfplumber not available")
        
        try:
            import fitz
            self._pymupdf = fitz
            logger.info("PyMuPDF loaded")
        except ImportError:
            logger.warning("PyMuPDF not available")
        
        if self.config.enable_ocr:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(self.config.ocr_languages)
                logger.info(f"EasyOCR initialized with languages: {self.config.ocr_languages}")
            except ImportError:
                logger.warning("EasyOCR not available, OCR disabled")
                self.config.enable_ocr = False
        
        self._initialized = True
    
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        **kwargs,
    ) -> ProcessedContent:
        if not self._initialized:
            await self.initialize()
        
        text_parts = []
        images = []
        tables = []
        metadata = {"source_type": "pdf"}
        
        if isinstance(file_path, BinaryIO):
            file_path.seek(0)
            pdf_bytes = file_path.read()
            doc = self._pymupdf.open(stream=pdf_bytes, filetype="pdf") if self._pymupdf else None
        else:
            doc = self._pymupdf.open(file_path) if self._pymupdf else None
        
        if doc:
            metadata["page_count"] = len(doc)
            
            for page_num, page in enumerate(doc):
                # Extract text
                page_text = page.get_text()
                text_parts.append(f"[PAGE {page_num + 1}]\n{page_text}")
                
                # Check if OCR needed
                if self.config.enable_ocr and len(page_text.strip()) < self.config.pdf_ocr_threshold:
                    # Render page as image and OCR
                    pix = page.get_pixmap(dpi=self.config.ocr_dpi)
                    img_bytes = pix.tobytes("png")
                    ocr_text = await self._ocr_image(img_bytes)
                    if ocr_text:
                        text_parts.append(f"[PAGE {page_num + 1} OCR]\n{ocr_text}")
                
                # Extract images
                if self.config.pdf_extract_images:
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_ext = base_image["ext"]
                        
                        # Resize if needed
                        if self._pymupdf:
                            img_doc = self._pymupdf.open(stream=img_bytes, filetype=img_ext)
                            if img_doc:
                                pix = img_doc[0].get_pixmap()
                                if pix.width > self.config.image_max_size[0] or pix.height > self.config.image_max_size[1]:
                                    # Resize
                                    scale = min(
                                        self.config.image_max_size[0] / pix.width,
                                        self.config.image_max_size[1] / pix.height,
                                    )
                                    pix = img_doc[0].get_pixmap(matrix=self._pymupdf.Matrix(scale, scale))
                                img_bytes = pix.tobytes("png")
                                img_ext = "png"
                        
                        images.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "format": img_ext,
                            "data": base64.b64encode(img_bytes).decode(),
                            "width": pix.width if 'pix' in locals() else 0,
                            "height": pix.height if 'pix' in locals() else 0,
                        })
            
            doc.close()
        
        # Use pdfplumber for table extraction
        if self.config.extract_tables and self._pdfplumber:
            tables = await self._extract_tables(file_path)
        
        full_text = "\n\n".join(text_parts)
        
        return ProcessedContent(
            modality=ModalityType.PDF,
            text=full_text,
            images=images,
            tables=tables,
            metadata=metadata,
            page_count=metadata.get("page_count", 0),
        )
    
    async def _ocr_image(self, img_bytes: bytes) -> str:
        """Run OCR on image bytes."""
        if not self._ocr_reader:
            return ""
        
        import numpy as np
        from PIL import Image
        
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._ocr_reader.readtext(img_array, detail=0)
            )
            
            return "\n".join(result)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
    
    async def _extract_tables(self, file_path: Union[str, Path, BinaryIO]) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        
        def extract():
            extracted = []
            if isinstance(file_path, BinaryIO):
                file_path.seek(0)
                with self._pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_tables = page.extract_tables()
                        for table_idx, table in enumerate(page_tables):
                            if table:
                                extracted.append({
                                    "page": page_num + 1,
                                    "index": table_idx,
                                    "data": table,
                                    "format": self.config.table_format,
                                })
            else:
                with self._pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_tables = page.extract_tables()
                        for table_idx, table in enumerate(page_tables):
                            if table:
                                extracted.append({
                                    "page": page_num + 1,
                                    "index": table_idx,
                                    "data": table,
                                    "format": self.config.table_format,
                                })
            return extracted
        
        loop = asyncio.get_event_loop()
        try:
            tables = await loop.run_in_executor(None, extract)
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
        
        return tables
    
    def supported_mimetypes(self) -> List[str]:
        return ["application/pdf"]
    
    async def close(self) -> None:
        self._pdfplumber = None
        self._pymupdf = None
        self._ocr_reader = None
        self._initialized = False


class ImageProcessor(BaseProcessor):
    """Processor for image files with OCR."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self._ocr_reader = None
        self._pil = None
    
    async def initialize(self) -> None:
        try:
            from PIL import Image
            self._pil = Image
        except ImportError:
            logger.warning("PIL not available")
        
        if self.config.enable_ocr:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(self.config.ocr_languages)
            except ImportError:
                logger.warning("EasyOCR not available")
                self.config.enable_ocr = False
        
        self._initialized = True
    
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        **kwargs,
    ) -> ProcessedContent:
        if not self._initialized:
            await self.initialize()
        
        if isinstance(file_path, BinaryIO):
            file_path.seek(0)
            img_bytes = file_path.read()
        else:
            with open(file_path, 'rb') as f:
                img_bytes = f.read()
        
        text = ""
        if self.config.enable_ocr and self._ocr_reader:
            import numpy as np
            img = self._pil.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._ocr_reader.readtext(img_array, detail=0)
            )
            text = "\n".join(result)
        
        # Resize image if needed
        if self._pil:
            img = self._pil.open(io.BytesIO(img_bytes))
            if img.width > self.config.image_max_size[0] or img.height > self.config.image_max_size[1]:
                img.thumbnail(self.config.image_max_size, self._pil.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format=self.config.image_format.upper())
                img_bytes = buffer.getvalue()
        
        return ProcessedContent(
            modality=ModalityType.IMAGE,
            text=text,
            images=[{
                "format": self.config.image_format,
                "data": base64.b64encode(img_bytes).decode(),
                "width": img.width if 'img' in locals() else 0,
                "height": img.height if 'img' in locals() else 0,
            }],
            metadata={"source_type": "image", "ocr_enabled": self.config.enable_ocr},
        )
    
    def supported_mimetypes(self) -> List[str]:
        return ["image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp"]
    
    async def close(self) -> None:
        self._ocr_reader = None
        self._pil = None
        self._initialized = False


class StructuredProcessor(BaseProcessor):
    """Processor for structured documents (JSON, XML, CSV)."""
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        **kwargs,
    ) -> ProcessedContent:
        import json
        import csv
        import xml.etree.ElementTree as ET
        
        if isinstance(file_path, BinaryIO):
            file_path.seek(0)
            content = file_path.read().decode('utf-8', errors='replace')
            mime_type = kwargs.get('mime_type', 'application/json')
        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            mime_type = mimetypes.guess_type(str(file_path))[0] or 'application/json'
        
        text = ""
        tables = []
        
        if mime_type == 'application/json':
            try:
                data = json.loads(content)
                text = json.dumps(data, indent=2)
                if isinstance(data, list) and data:
                    tables.append({
                        "data": data,
                        "format": "json",
                    })
            except json.JSONDecodeError:
                text = content
        
        elif mime_type in ('text/csv', 'application/csv'):
            try:
                reader = csv.reader(io.StringIO(content))
                rows = list(reader)
                text = "\n".join([",".join(row) for row in rows])
                tables.append({
                    "data": rows,
                    "format": "csv",
                })
            except Exception:
                text = content
        
        elif mime_type in ('application/xml', 'text/xml'):
            try:
                root = ET.fromstring(content)
                text = ET.tostring(root, encoding='unicode')
            except ET.ParseError:
                text = content
        
        else:
            text = content
        
        return ProcessedContent(
            modality=ModalityType.STRUCTURED,
            text=text,
            tables=tables,
            metadata={"source_type": "structured", "mime_type": mime_type},
        )
    
    def supported_mimetypes(self) -> List[str]:
        return ["application/json", "text/csv", "application/xml", "text/xml", "application/yaml"]
    
    async def close(self) -> None:
        self._initialized = False


class AudioProcessor(BaseProcessor):
    """Processor for audio files with transcription."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self._whisper = None
    
    async def initialize(self) -> None:
        try:
            import whisper
            self._whisper = whisper.load_model("base")
            logger.info("Whisper model loaded")
        except ImportError:
            logger.warning("Whisper not available")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
        
        self._initialized = True
    
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        **kwargs,
    ) -> ProcessedContent:
        if not self._initialized:
            await self.initialize()
        
        text = ""
        if self._whisper:
            # Save to temp file if BinaryIO
            if isinstance(file_path, BinaryIO):
                import tempfile
                file_path.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(file_path.read())
                    tmp_path = tmp.name
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._whisper.transcribe(tmp_path)
                    )
                    text = result.get("text", "")
                finally:
                    os.unlink(tmp_path)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._whisper.transcribe(str(file_path))
                )
                text = result.get("text", "")
        
        return ProcessedContent(
            modality=ModalityType.AUDIO,
            text=text,
            metadata={"source_type": "audio", "transcribed": bool(text)},
        )
    
    def supported_mimetypes(self) -> List[str]:
        return ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/flac"]
    
    async def close(self) -> None:
        self._whisper = None
        self._initialized = False


class MultiModalProcessor:
    """Main multimodal document processor."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._processors: Dict[str, BaseProcessor] = {}
        self._mime_to_processor: Dict[str, BaseProcessor] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all processors."""
        # Register default processors
        self.register_processor(TextProcessor(self.config))
        self.register_processor(PDFProcessor(self.config))
        self.register_processor(ImageProcessor(self.config))
        self.register_processor(StructuredProcessor(self.config))
        self.register_processor(AudioProcessor(self.config))
        
        # Initialize all
        for processor in self._processors.values():
            await processor.initialize()
        
        self._initialized = True
        logger.info("Multimodal processor initialized")
    
    def register_processor(self, processor: BaseProcessor) -> None:
        """Register a processor."""
        name = processor.__class__.__name__
        self._processors[name] = processor
        for mime in processor.supported_mimetypes():
            self._mime_to_processor[mime] = processor
    
    def get_processor(self, mime_type: str) -> Optional[BaseProcessor]:
        """Get processor for MIME type."""
        return self._mime_to_processor.get(mime_type)
    
    async def process(
        self,
        file_path: Union[str, Path, BinaryIO],
        mime_type: Optional[str] = None,
        **kwargs,
    ) -> ProcessedContent:
        """Process a document."""
        if not self._initialized:
            await self.initialize()
        
        # Detect MIME type if not provided
        if mime_type is None:
            if isinstance(file_path, (str, Path)):
                mime_type = mimetypes.guess_type(str(file_path))[0]
            else:
                mime_type = kwargs.get('mime_type', 'application/octet-stream')
        
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        # Get processor
        processor = self.get_processor(mime_type)
        
        if processor is None:
            # Fallback to text processor
            processor = self.get_processor('text/plain')
            if processor is None:
                raise ValueError(f"No processor for MIME type: {mime_type}")
        
        return await processor.process(file_path, mime_type=mime_type, **kwargs)
    
    async def process_batch(
        self,
        files: List[Tuple[Union[str, Path, BinaryIO], Optional[str]]],
        **kwargs,
    ) -> List[ProcessedContent]:
        """Process multiple files in parallel."""
        tasks = [
            self.process(file_path, mime_type, **kwargs)
            for file_path, mime_type in files
        ]
        return await asyncio.gather(*tasks)
    
    async def close(self) -> None:
        """Close all processors."""
        for processor in self._processors.values():
            await processor.close()
        self._initialized = False


async def create_multimodal_processor(
    config: Optional[ProcessingConfig] = None,
) -> MultiModalProcessor:
    """Factory to create multimodal processor."""
    processor = MultiModalProcessor(config)
    await processor.initialize()
    return processor
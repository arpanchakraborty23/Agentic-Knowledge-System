"""
Unit tests for data_search module
"""

import pytest
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nodes.data_search import (
    BaseLoader,
    TavilyWebLoader,
    PDFLoader,
    EPUBLoader,
    get_loader,
    LOADER_MAP
)


class TestBaseLoader:
    """Test base loader functionality"""
    
    def test_loader_map_structure(self):
        """Test that LOADER_MAP has expected structure"""
        assert "tavily" in LOADER_MAP
        assert "web" in LOADER_MAP
        assert "pdf" in LOADER_MAP
        assert "epub" in LOADER_MAP
        assert LOADER_MAP["tavily"] == TavilyWebLoader
        assert LOADER_MAP["pdf"] == PDFLoader
    
    def test_get_loader_factory(self):
        """Test factory function returns correct loaders"""
        # Test getting each loader type
        with pytest.raises(ValueError):  # No API key
            get_loader("tavily")
        
        pdf_loader = get_loader("pdf")
        assert isinstance(pdf_loader, PDFLoader)
        
        epub_loader = get_loader("epub")
        assert isinstance(epub_loader, EPUBLoader)
    
    def test_get_loader_invalid(self):
        """Test factory raises error for invalid loader type"""
        with pytest.raises(ValueError) as excinfo:
            get_loader("invalid_type")
        assert "Unknown loader type" in str(excinfo.value)


class TestPDFLoader:
    """Test PDF loader functionality"""
    
    def test_pdf_loader_initialization(self):
        """Test PDFLoader initialization"""
        loader = PDFLoader(extract_tables=True, use_ocr=True)
        assert loader.extract_tables is True
        assert loader.use_ocr is True
        assert loader.ocr_languages == ["eng"]
    
    def test_pdf_load_bytes(self):
        """Test loading PDF from bytes"""
        try:
            import fitz
            
            # Create simple PDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test content")
            pdf_bytes = doc.write()
            
            loader = PDFLoader()
            results = list(loader.load(pdf_bytes))
            
            assert len(results) == 1
            assert "Test content" in results[0].page_content
            assert results[0].metadata["type"] == "pdf"
            
            doc.close()
        except ImportError:
            pytest.skip("PyMuPDF not available")
    
    def test_pdf_load_and_chunk(self):
        """Test PDF loading with chunking"""
        try:
            import fitz
            
            # Create PDF with substantial content
            doc = fitz.open()
            for i in range(3):
                page = doc.new_page()
                content = f"""This is page {i+1} with substantial content.
                Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."""
                page.insert_text((50, 50), content)
            
            pdf_bytes = doc.write()
            
            loader = PDFLoader()
            chunks = loader.load_and_chunk(
                pdf_bytes,
                chunk_size=200,
                chunk_overlap=20
            )
            
            assert len(chunks) > 0
            assert all(hasattr(c, "page_content") for c in chunks)
            assert all(hasattr(c, "metadata") for c in chunks)
            
            doc.close()
        except ImportError:
            pytest.skip("PyMuPDF not available")
    
    def test_pdf_validation(self):
        """Test PDF source validation"""
        loader = PDFLoader()
        
        # Invalid paths
        assert loader.validate_source("nonexistent.pdf") is False
        assert loader.validate_source("/tmp/not_a_pdf.txt") is False
        assert loader.validate_source("document") is False


class TestEPUBLoader:
    """Test EPUB loader functionality"""
    
    def test_epub_loader_initialization(self):
        """Test EPUBLoader initialization"""
        loader = EPUBLoader()
        assert loader is not None
    
    def test_epub_validation(self):
        """Test EPUB source validation"""
        loader = EPUBLoader()
        
        # Invalid paths
        assert loader.validate_source("nonexistent.epub") is False
        assert loader.validate_source("file.txt") is False


class TestTavilyWebLoader:
    """Test Tavily web loader functionality"""
    
    def test_web_loader_initialization(self):
        """Test Tavily loader initialization"""
        # Test without API key
        with pytest.raises(ValueError) as excinfo:
            TavilyWebLoader()
        assert "API key required" in str(excinfo.value)
        
        # Test with dummy key
        loader = TavilyWebLoader(api_key="dummy-key")
        assert loader.api_key == "dummy-key"
        assert loader.search_depth == "advanced"
    
    def test_web_loader_with_key(self, monkeypatch):
        """Test Tavily loader with API key from env"""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        
        loader = TavilyWebLoader()
        assert loader.api_key == "test-key"
    
    def test_web_validation(self):
        """Test web source validation"""
        # Note: Actual Tavily validation requires network
        # This tests the validation method exists
        loader = TavilyWebLoader(api_key="test-key")
        result = loader.validate_source("test-source")
        assert isinstance(result, bool)


class TestIntegration:
    """Test integration scenarios"""
    
    def test_multiple_loaders(self):
        """Test using multiple loader types in sequence"""
        # Initialize all loaders
        loaders = []
        
        # PDF loader should work
        try:
            pdf_loader = get_loader("pdf")
            loaders.append(("pdf", pdf_loader))
        except:
            pass
        
        # EPUB loader should work
        try:
            epub_loader = get_loader("epub")
            loaders.append(("epub", epub_loader))
        except:
            pass
        
        assert len(loaders) >= 1, "At least one loader should initialize"
    
    def test_document_content_type(self):
        """Test that documents have correct structure"""
        try:
            import fitz
            
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test")
            pdf_bytes = doc.write()
            
            loader = PDFLoader()
            documents = list(loader.load(pdf_bytes))
            
            for doc in documents:
                assert hasattr(doc, "page_content")
                assert hasattr(doc, "metadata")
                assert isinstance(doc.page_content, str)
                assert isinstance(doc.metadata, dict)
            
            doc.close()
        except ImportError:
            pytest.skip("PyMuPDF not available")


class TestChunking:
    """Test document chunking functionality"""
    
    def test_chunking_preserves_content(self):
        """Test that chunking doesn't lose content"""
        try:
            import fitz
            
            # Create PDF with known content
            test_text = "A" * 5000  # 5000 characters
            
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), test_text)
            pdf_bytes = doc.write()
            
            loader = PDFLoader()
            
            # Load original
            original_docs = list(loader.load(pdf_bytes))
            original_text = "\n".join([doc.page_content for doc in original_docs])
            
            # Load with chunking
            chunks = loader.load_and_chunk(
                pdf_bytes,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            chunked_text = "".join([chunk.page_content for chunk in chunks])
            
            # Content should be preserved
            assert len(chunked_text) >= len(test_text)
            
            doc.close()
        except ImportError:
            pytest.skip("PyMuPDF not available")
    
    def test_chunking_produces_valid_documents(self):
        """Test that chunked documents are valid"""
        try:
            import fitz
            
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "X" * 2000)
            pdf_bytes = doc.write()
            
            loader = PDFLoader()
            chunks = loader.load_and_chunk(pdf_bytes, chunk_size=500)
            
            assert len(chunks) > 0
            assert all(hasattr(chunk, "page_content") for chunk in chunks)
            assert all(hasattr(chunk, "metadata") for chunk in chunks)
            
            # All chunks should have reasonable size
            for chunk in chunks:
                assert 100 <= len(chunk.page_content) <= 600
            
            doc.close()
        except ImportError:
            pytest.skip("PyMuPDF not available")


# Mock tests for missing dependencies
@pytest.fixture
def mock_fitz(monkeypatch):
    """Mock PyMuPDF when not available"""
    monkeypatch.setitem(sys.modules, "fitz", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
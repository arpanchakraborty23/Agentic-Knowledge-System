import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun, ArxivQueryRun
from src.constants import ProviderConfig

from src.constants import GraphState



class TavilyWebLoader:
    """Web loader using Tavily API for search and content extraction"""
    
    

class PDFLoader(BaseLoader):
    """Loader for PDF documents with text extraction"""
    
    def __init__(self, extract_tables: bool = False, 
                 use_ocr: bool = False,
                 ocr_languages: List[str] = ["eng"]):
        super().__init__()
        self.extract_tables = extract_tables
        self.use_ocr = use_ocr
        self.ocr_languages = ocr_languages
    
    def load(self, source: Union[str, bytes], **kwargs) -> Iterator[Document]:
        """Load PDF document and extract text
        
        Args:
            source: File path to PDF (str) or PDF content (bytes)
            **kwargs: Additional options
                - password: Password for encrypted PDFs
                - page_nums: Specific pages to extract
        """
        try:
            # Handle file path string
            if isinstance(source, str):
                file_path = Path(source)
                if not file_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {source}")
                
                # Use PyMuPDF loader
                loader = PyMuPDFLoader(str(file_path), **kwargs)
                documents = loader.load()
                
            # Handle bytes content
            elif isinstance(source, bytes):
                import fitz
                doc = fitz.open(stream=source, filetype="pdf")
                documents = self._parse_fitz_document(doc, source)
            else:
                raise ValueError("Source must be file path (str) or PDF content (bytes)")
            
            # Enhance metadata for each document
            for doc in documents:
                doc.metadata.update({
                    "type": "pdf",
                    "extracted_by": "pymupdf"
                })
                yield doc
                
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            yield Document(
                page_content=f"Error loading PDF: {str(e)}",
                metadata={
                    "source": str(source),
                    "type": "pdf",
                    "error": str(e)
                }
            )
    
    def _parse_fitz_document(self, doc, source) -> List[Document]:
        """Parse PyMuPDF document object"""
        documents = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            metadata = {
                "source": f"pdf_stream_page_{page_num + 1}",
                "page": page_num + 1,
                "type": "pdf"
            }
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        return documents
    
    def validate_source(self, source: str) -> bool:
        """Validate PDF file"""
        try:
            if isinstance(source, str):
                file_path = Path(source)
                return file_path.exists() and file_path.suffix.lower() == ".pdf"
            return False
        except Exception:
            return False


class EPUBLoader(BaseLoader):
    """Loader for EPUB e-book documents"""
    
    def __init__(self):
        super().__init__()
    
    def load(self, source: str, **kwargs) -> Iterator[Document]:
        """Load EPUB document
        
        Args:
            source: Path to .epub file
            **kwargs: Additional options for UnstructuredEPubLoader
        """
        try:
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"EPUB file not found: {source}")
            
            # Use Unstructured EPUB loader
            loader = UnstructuredEPubLoader(str(file_path), **kwargs)
            documents = loader.load()
            
            # Enhanced metadata
            for doc in documents:
                doc.metadata.update({
                    "type": "epub",
                    "filename": file_path.name,
                    "file_size": file_path.stat().st_size
                })
                yield doc
                
        except Exception as e:
            print(f"Error loading EPUB: {str(e)}")
            yield Document(
                page_content=f"Error loading EPUB: {str(e)}",
                metadata={
                    "source": str(source),
                    "type": "epub",
                    "error": str(e)
                }
            )
    
    def validate_source(self, source: str) -> bool:
        """Validate EPUB file"""
        try:
            file_path = Path(source)
            return file_path.exists() and file_path.suffix.lower() == ".epub"
        except Exception:
            return False


# Loader Factory
LOADER_MAP = {
    "tavily": TavilyWebLoader,
    "web": TavilyWebLoader,  # Alias
    "pdf": PDFLoader,
    "epub": EPUBLoader,
}

def get_loader(loader_type: str, **config) -> BaseLoader:
    """Factory function to get loader by type
    
    Args:
        loader_type: Type of loader ("tavily", "web", "pdf", "epub")
        **config: Configuration for the loader
    
    Returns:
        Configured loader instance
    """
    loader_class = LOADER_MAP.get(loader_type.lower())
    
    if not loader_class:
        available = list(LOADER_MAP.keys())
        raise ValueError(f"Unknown loader type: {loader_type}. Available: {available}")
    
    return loader_class(**config)


# Convenience functions for common use cases
def load_web_content(query: str, api_key: Optional[str] = None, 
                    max_results: int = 5) -> List[Document]:
    """Convenience function to load web search results"""
    loader = TavilyWebLoader(api_key=api_key)
    return list(loader.load(query, max_results=max_results))


def load_pdf_document(file_path: str, **kwargs) -> List[Document]:
    """Convenience function to load PDF document"""
    loader = PDFLoader()
    return list(loader.load(file_path, **kwargs))


def load_book_document(file_path: str, **kwargs) -> List[Document]:
    """Convenience function to load ebook document"""
    ext = Path(file_path).suffix.lower()
    
    if ext == ".epub":
        loader = EPUBLoader()
    else:
        raise ValueError(f"Unsupported book format: {ext}")
    
    return list(loader.load(file_path, **kwargs))


# Example usage and self-test
if __name__ == "__main__":
    print("Testing Data Search Loaders...")
    
    # Test Tavily loader (requires API key)
    try:
        print("\n1. Testing Tavily Web Loader:")
        web_loader = TavilyWebLoader()
        docs = web_loader.load("Python programming basics", max_results=2)
        print(f"✓ Loaded {len(docs)} web documents")
        for i, doc in enumerate(docs[:2]):
            print(f"  Doc {i+1}: {doc.metadata.get('source', 'N/A')}")
    except Exception as e:
        print(f"✗ Tavily test failed: {e}")
    
    # Test PDF loader with example
    print("\n2. Testing PDF Loader:")
    try:
        # Create a simple test PDF in memory
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF content for loader")
        pdf_bytes = doc.write()
        doc.close()
        
        pdf_loader = PDFLoader()
        docs = pdf_loader.load(pdf_bytes)
        print(f"✓ Loaded {len(docs)} PDF pages")
        for i, doc in enumerate(docs):
            print(f"  Page {i+1}: {len(doc.page_content)} characters")
    except Exception as e:
        print(f"✗ PDF test failed: {e}")
    
    print("\n✓ All loader tests completed!")
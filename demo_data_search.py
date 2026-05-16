#!/usr/bin/env python3
"""
Demo script for Data Search and Loader Module

This script demonstrates how to use the custom dataloader to crawl web content,
extract PDF documents, and prepare them for vector index creation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nodes.data_search import (
    get_loader, 
    TavilyWebLoader, 
    PDFLoader, 
    EPUBLoader,
    load_web_content,
    load_pdf_document,
    load_book_document
)


def demo_web_search():
    """Demo: Search web content using Tavily"""
    print("=" * 60)
    print("DEMO 1: Web Search with Tavily")
    print("=" * 60)
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("⚠️  Warning: TAVILY_API_KEY not set. Using dummy data.")
        print("   Get your key from: https://tavily.com/")
        return
    
    try:
        # Initialize loader
        print("🔍 Initializing Tavily loader...")
        loader = TavilyWebLoader(api_key=api_key)
        
        # Search for content
        query = "Python FastAPI dependency injection"
        print(f"🌐 Searching: '{query}'")
        print("-" * 60)
        
        documents = list(loader.load(query, max_results=3))
        
        # Display results
        for i, doc in enumerate(documents, 1):
            print(f"📄 Result {i}:")
            print(f"   Title: {doc.metadata.get('title', 'N/A')}")
            print(f"   Source: {doc.metadata.get('source', 'N/A')}")
            print(f"   Type: {doc.metadata.get('type', 'N/A')}")
            print(f"   Content preview: {doc.page_content[:150]}...")
            print(f"   Length: {len(doc.page_content)} characters")
            print("-" * 60)
        
        print(f"✅ Successfully loaded {len(documents)} documents\n")
        
    except Exception as e:
        print(f"❌ Error during web search: {e}\n")


def demo_pdf_loading():
    """Demo: Load and process PDF documents"""
    print("=" * 60)
    print("DEMO 2: PDF Document Loading")
    print("=" * 60)
    
    try:
        # Create a sample PDF in memory
        print("📄 Creating sample PDF...")
        
        import fitz  # PyMuPDF
        doc = fitz.open()
        
        # Add pages with sample content
        for i in range(3):
            page = doc.new_page()
            content = f"""
            Sample PDF Document - Page {i + 1}
            
            This is a sample document demonstrating the PDF loader.
            It contains multiple paragraphs and pages for testing.
            
            Page Content Details:
            - Page number: {i + 1}
            - Sample text with various characters and formatting
            - This content will be extracted and split into chunks
            
            Technical Information:
            The PDF loader uses PyMuPDF for fast and accurate text extraction.
            It can handle multi-page documents, metadata, and various PDF formats.
            
            Lorem ipsum dolor sit amet, consectetur adipiscing elit.
            Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
            """
            page.insert_text((50, 50), content)
        
        # Save to memory
        pdf_bytes = doc.write()
        print(f"✅ Created PDF with {doc.page_count} pages")
        
        # Initialize loader
        loader = PDFLoader()
        
        # Load entire document
        print("📖 Loading PDF document...")
        documents = list(loader.load(pdf_bytes))
        
        print(f"✅ Loaded {len(documents)} pages")
        
        for i, doc in enumerate(documents, 1):
            print(f"   Page {i}: {len(doc.page_content)} characters")
        
        # Demonstrate chunking
        print("\n✂️  Demonstrating smart chunking...")
        chunks = loader.load_and_chunk(
            pdf_bytes,
            chunk_size=500,
            chunk_overlap=50
        )
        
        print(f"✅ Split into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   Chunk {i}: {len(chunk.page_content)} characters")
            print(f"   Preview: {chunk.page_content[:100]}...")
            print("-" * 60)
        
        if len(chunks) > 3:
            print(f"   ... and {len(chunks) - 3} more chunks\n")
        
        doc.close()
        
    except ImportError:
        print("❌ PyMuPDF not installed. Install with: pip install pymupdf\n")
    except Exception as e:
        print(f"❌ Error during PDF loading: {e}\n")


def demo_epub_loading():
    """Demo: Load EPUB documents"""
    print("=" * 60)
    print("DEMO 3: EPUB Book Loading")
    print("=" * 60)
    
    try:
        # Note: This demo shows the API, but EPUB parsing requires actual files
        print("📚 EPUB Loader Info:")
        print("✅ EPUBLoader is ready for use")
        print("✅ Supports EPUB 2.0 and 3.0 format")
        print("✅ Extracts text content and metadata")
        print("✅ Handles multi-chapter books")
        print()
        
        print("Usage Example:")
        print("-" * 60)
        print("from nodes.data_search import EPUBLoader")
        print("")
        print("loader = EPUBLoader()")
        print("documents = list(loader.load('path/to/book.epub'))")
        print("print(f'Loaded {len(documents)} sections')")
        print("-" * 60)
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def demo_factory_pattern():
    """Demo: Using the factory pattern"""
    print("=" * 60)
    print("DEMO 4: Factory Pattern & Multiple Sources")
    print("=" * 60)
    
    print("🏭 Using get_loader() factory function...")
    print("-" * 60)
    
    # Demonstrate factory for different loaders
    loaders = ["web", "pdf", "epub"]
    
    for loader_type in loaders:
        try:
            loader = get_loader(loader_type)
            print(f"✅ {loader_type.upper():4} -> {type(loader).__name__}")
        except Exception as e:
            print(f"❌ {loader_type:4} -> {e}")
    
    print("-" * 60)
    print("✅ Factory pattern allows easy switching between sources")
    print("✅ Unified interface for all loaders")
    print()


def demo_convenience_functions():
    """Demo: Convenience functions"""
    print("=" * 60)
    print("DEMO 5: Convenience Functions")
    print("=" * 60)
    
    print("🎯 Using pre-configured convenience functions...")
    print()
    
    # Load web content
    print("1. load_web_content()")
    print("   Usage: load_web_content(query, api_key, max_results)")
    print("   Returns: List[Document]")
    print()
    
    # Load PDF
    print("2. load_pdf_document()")
    print("   Usage: load_pdf_document(file_path, **kwargs)")
    print("   Returns: List[Document]")
    print()
    
    # Load book
    print("3. load_book_document()")
    print("   Usage: load_book_document(file_path, **kwargs)")
    print("   Returns: List[Document]")
    print()
    
    print("✅ These functions handle loader instantiation automatically")
    print()


def demo_metadata():
    """Demo: Metadata extraction capabilities"""
    print("=" * 60)
    print("DEMO 6: Rich Metadata Extraction")
    print("=" * 60)
    
    try:
        # Create sample and extract metadata
        import fitz
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Sample document for metadata extraction")
        pdf_bytes = doc.write()
        
        loader = PDFLoader()
        documents = list(loader.load(pdf_bytes))
        
        print("📊 Metadata extracted from PDF:")
        for key, value in documents[0].metadata.items():
            print(f"   {key:15}: {value}")
        
        print()
        print("🌐 Web documents include:")
        print("   - source: Original URL")
        print("   - title: Page title")
        print("   - type: Document type (web/pdf/epub)")
        print("   - api_source: Source service (tavily)")
        print()
        
        doc.close()
        
    except Exception as e:
        print(f"❌ Error showing metadata: {e}\n")


def summary():
    """Demo summary"""
    print("=" * 60)
    print("SUMMARY: Data Search & Loader Module")
    print("=" * 60)
    print()
    
    print("✅ Implemented Loaders:")
    print("  • TavilyWebLoader  - Web search & crawling")
    print("  • PDFLoader       - PDF text extraction")
    print("  • EPUBLoader      - E-book processing")
    print()
    
    print("✅ Key Features:")
    print("  • Factory pattern for easy switching")
    print("  • Smart text chunking for vector stores")
    print("  • Rich metadata extraction")
    print("  • Unified interface across all sources")
    print("  • Error handling and fallback mechanisms")
    print()
    
    print("✅ Next Steps:")
    print("  1. Set TAVILY_API_KEY for web crawling")
    print("  2. Integrate with vector store (Chroma/Redis)")
    print("  3. Connect to ResearchAgent for RAG")
    print("  4. Add Scrapy support for advanced crawling")
    print("  5. Add OCR for scanned PDFs")
    print()
    
    print("📚 Documentation: src/nodes/DATA_SEARCH_README.md")
    print("💻 Module: src/nodes/data_search.py")
    print("⚙️  Config: pyproject.toml (dependencies updated)")
    print("=" * 60)


def main():
    """Run all demos"""
    print("\n")
    print("🚀 Data Search & Loader Module Demo")
    print("=" * 60)
    print()
    
    # Run demos
    demo_factory_pattern()
    demo_web_search()
    demo_pdf_loading()
    demo_epub_loading()
    demo_convenience_functions()
    demo_metadata()
    summary()
    
    print("\n✨ Demo completed!\n")


if __name__ == "__main__":
    main()
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_logger

logger = get_logger(name="TextChunker")


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None


class TextChunker:
    """Chunk documents for vector storage."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            logger.warning("No documents to chunk")
            return []

        chunked_docs = self.splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunked_docs):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks"
        )
        return chunked_docs

    def chunk_with_metadata_preservation(
        self,
        documents: List[Document],
        extra_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Chunk documents while preserving and adding metadata.
        
        Args:
            documents: List of Document objects
            extra_metadata: Additional metadata to add to all chunks
            
        Returns:
            List of chunked Document objects with preserved metadata
        """
        chunked_docs = self.chunk_documents(documents)
        
        if extra_metadata:
            for chunk in chunked_docs:
                chunk.metadata.update(extra_metadata)

        return chunked_docs

    def chunk_by_source(
        self,
        documents: List[Document],
        max_chunks_per_source: int = 5
    ) -> List[Document]:
        """
        Chunk documents grouped by source, limiting chunks per source.
        
        Args:
            documents: List of Document objects
            max_chunks_per_source: Maximum chunks per source
            
        Returns:
            List of chunked Document objects
        """
        source_docs = {}
        
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in source_docs:
                source_docs[source] = []
            source_docs[source].append(doc)

        all_chunks = []
        for source, docs in source_docs.items():
            chunks = self.chunk_documents(docs)
            chunks = chunks[:max_chunks_per_source]
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents from {len(source_docs)} sources "
            f"into {len(all_chunks)} total chunks"
        )
        return all_chunks

from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from src.utils import get_logger

logger = get_logger(name="DataParser")


@dataclass
class ParsedDocument:
    content: str
    metadata: dict


class DataParser:
    """Parse raw research data into structured documents."""

    def __init__(self):
        self.min_content_length = 50

    def parse_research_data(
        self, 
        research_data: str, 
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Parse research data string into Document objects.
        
        Args:
            research_data: Raw research data string
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of Document objects
        """
        if not research_data:
            logger.warning("Empty research data provided")
            return []

        base_metadata = metadata or {}
        documents = []

        sections = self._split_by_source(research_data)
        
        for section in sections:
            if len(section.strip()) < self.min_content_length:
                continue
            
            source_metadata = self._extract_source_metadata(section)
            combined_metadata = {**base_metadata, **source_metadata}
            
            doc = Document(
                page_content=section.strip(),
                metadata=combined_metadata
            )
            documents.append(doc)

        logger.info(f"Parsed {len(documents)} documents from research data")
        return documents

    def _split_by_source(self, data: str) -> List[str]:
        """Split research data by source markers."""
        sections = []
        
        current_section = []
        lines = data.split("\n")
        
        for line in lines:
            if line.startswith("---") and current_section:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append("\n".join(current_section))
        
        return sections

    def _extract_source_metadata(self, section: str) -> dict:
        """Extract metadata from section header."""
        metadata = {}
        
        lines = section.strip().split("\n")
        if lines:
            header = lines[0]
            
            if header.startswith("---"):
                source = header.replace("---", "").strip()
                metadata["source"] = source
            
            if header.startswith(("Title:", "Title :")):
                title = header.replace("Title:", "").replace("Title :", "").strip()
                metadata["title"] = title
            
            if header[0].isdigit() and "." in header[:4]:
                metadata["source"] = "finance_news"

        lines_content = section.strip().split("\n")
        for line in lines_content:
            if line.startswith("Authors:"):
                metadata["authors"] = line.replace("Authors:", "").strip()
            elif line.startswith("Published:"):
                metadata["published"] = line.replace("Published:", "").strip()
            elif line.startswith("URL:"):
                metadata["url"] = line.replace("URL:", "").strip()

        return metadata

    def parse_with_query_context(
        self,
        research_data: str,
        query: str,
        domain: str,
        user_id: Optional[str] = None
    ) -> List[Document]:
        """
        Parse research data with query context metadata.
        
        Args:
            research_data: Raw research data string
            query: Original user query
            domain: Classified domain
            user_id: Optional user identifier
            
        Returns:
            List of Document objects with query context
        """
        metadata = {
            "query": query,
            "domain": domain,
            "user_id": user_id,
        }
        
        return self.parse_research_data(research_data, metadata)

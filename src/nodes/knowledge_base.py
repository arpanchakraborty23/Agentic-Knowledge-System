from typing import  Optional
from langchain.chat_models import BaseChatModel

from src.constants import GraphState
from src.rag import DataParser, TextChunker, VectorStoreManager
from src.utils import get_logger

logger = get_logger(name="KnowledgeBase")


class KnowledgeBaseManager:
    """Manage knowledge base operations."""

    def __init__(self):
        self.parser = DataParser()
        self.chunker = TextChunker()
        self.vector_store = VectorStoreManager()

    def create_knowledge_base(
        self,
        research_data: str,
        query: str,
        domain: str,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Create knowledge base from research data.
        
        Args:
            research_data: Raw research data string
            query: Original user query
            domain: Classified domain
            user_id: Optional user identifier
            
        Returns:
            Number of chunks stored
        """
        if not research_data:
            logger.warning("No research data to create knowledge base")
            return 0

        documents = self.parser.parse_with_query_context(
            research_data=research_data,
            query=query,
            domain=domain,
            user_id=user_id,
        )

        chunked_docs = self.chunker.chunk_documents(documents)

        self.vector_store.add_documents(
            documents=chunked_docs,
            user_id=user_id,
            query=query,
        )

        logger.info(f"Created knowledge base with {len(chunked_docs)} chunks")
        return len(chunked_docs)

    def retrieve_knowledge(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list:
        """
        Retrieve knowledge from vector store.
        
        Args:
            query: Search query
            k: Number of results
            user_id: Optional user filter
            domain: Optional domain filter
            
        Returns:
            List of retrieved documents
        """
        filter_metadata = {}
        if user_id:
            filter_metadata["user_id"] = user_id
        if domain:
            filter_metadata["domain"] = domain

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata if filter_metadata else None,
        )

        logger.info(f"Retrieved {len(results)} documents from knowledge base")
        return results

    def cleanup_expired(self) -> int:
        """Remove expired documents from knowledge base."""
        deleted = self.vector_store.delete_expired_documents()
        logger.info(f"Cleaned up {deleted} expired documents")
        return deleted

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return self.vector_store.get_stats()


class KnowledgeNode:
    """Node for creating or retrieving knowledge."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.kb_manager = KnowledgeBaseManager()
        logger.info("KnowledgeNode initialized")

    def invoke(self, state: GraphState) -> GraphState:
        """
        Invoke knowledge node.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated graph state
        """
        try:
            operation = state.knowledge_base

            if operation == "add":
                if not state.research_data:
                    logger.warning("Cannot create knowledge base: no research data")
                    return state

                chunks_stored = self.kb_manager.create_knowledge_base(
                    research_data=state.research_data,
                    query=state.query,
                    domain=state.classified_domain or "unknown",
                    user_id=state.user_id,
                )

                logger.info(f"Knowledge base created: {chunks_stored} chunks stored")

            elif operation == "retrieve":
                retrieved_docs = self.kb_manager.retrieve_knowledge(
                    query=state.query,
                    k=5,
                    user_id=state.user_id,
                    domain=state.classified_domain,
                )

                logger.info(f"Retrieved {len(retrieved_docs)} documents from knowledge base")

            self.kb_manager.cleanup_expired()

            return state

        except Exception as e:
            logger.error("Error in knowledge base node: %s", str(e))
            return state

from typing import List, Optional
from datetime import datetime, timedelta

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import get_logger
from src.constants import ProviderConfig

logger = get_logger(name="VectorStore")


class VectorStoreManager:
    """Manage vector storage with TTL support."""

    TTL_DAYS = 10
    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        embedding_model: str = "huggingface",
        persist_directory: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or "./data/qdrant"
        self.embeddings = self._init_embeddings(embedding_model)
        self.client = self._init_client()
        self.vector_store = self._init_vector_store()

    def _init_embeddings(self, model: str):
        """Initialize embeddings model."""
        if model == "openai":
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=ProviderConfig.openai_api_key,
            )
        else:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./models/embeddings",
            )

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client."""
        return QdrantClient(path=self.persist_directory)

    def _init_vector_store(self) -> QdrantVectorStore:
        """Initialize vector store."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384 if isinstance(self.embeddings, HuggingFaceEmbeddings) else 1536,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.COLLECTION_NAME}")

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding=self.embeddings,
        )

    def add_documents(
        self,
        documents: List[Document],
        user_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[str]:
        """
        Add documents to vector store with TTL.
        
        Args:
            documents: List of Document objects
            user_id: User identifier
            query: Original query
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        ttl_timestamp = int((datetime.now() + timedelta(days=self.TTL_DAYS)).timestamp())
        
        for i, doc in enumerate(documents):
            doc.metadata["ttl_timestamp"] = ttl_timestamp
            doc.metadata["created_at"] = datetime.now().isoformat()
            doc.metadata["user_id"] = user_id or "unknown"
            doc.metadata["query"] = query or ""

        ids = self.vector_store.add_documents(documents)
        
        logger.info(
            f"Added {len(documents)} documents with TTL of {self.TTL_DAYS} days"
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search similar documents.
        
        Args:
            query: Search query
            k: Number of results
            filter_metadata: Optional metadata filter
            
        Returns:
            List of matching Document objects
        """
        current_timestamp = int(datetime.now().timestamp())
        
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.ttl_timestamp",
                    match=models.MatchValue(value=current_timestamp),
                    range=models.Range(gte=current_timestamp),
                )
            ]
        )

        if filter_metadata:
            for key, value in filter_metadata.items():
                filter_conditions.must.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value),
                    )
                )

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_conditions,
        )
        
        logger.info(f"Found {len(results)} documents for query")
        return results

    def delete_expired_documents(self) -> int:
        """
        Delete expired documents based on TTL.
        
        Returns:
            Number of deleted documents
        """
        current_timestamp = int(datetime.now().timestamp())
        
        expired_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.ttl_timestamp",
                    range=models.Range(lt=current_timestamp),
                )
            ]
        )

        result = self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=expired_filter),
        )

        deleted_count = result.result.get("deleted_count", 0) if hasattr(result, "result") else 0
        logger.info(f"Deleted {deleted_count} expired documents")
        return deleted_count

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        collection_info = self.client.get_collection(self.COLLECTION_NAME)
        
        return {
            "collection_name": self.COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status,
            "ttl_days": self.TTL_DAYS,
        }

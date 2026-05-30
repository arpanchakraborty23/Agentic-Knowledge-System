from typing import List, Optional
from datetime import datetime, timedelta

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_aws import BedrockEmbeddings

from src.utils import get_logger
from src.constants import ProviderConfig

logger = get_logger(name="VectorStore")


class VectorStoreManager:
    """Manage vector storage with TTL support."""

    TTL_DAYS = 10
    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        embedding_model: str = "aws",
        persist_directory: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or "./data/qdrant"
        self.embeddings = self._init_embeddings(embedding_model)
        self.client = self._init_client()
        self.vector_store = self._init_vector_store()

    def _init_embeddings(self, model: str):
        """Initialize embeddings model."""
        if model == "aws":
            return BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v2:0",
                region_name=ProviderConfig.aws_region,
                credentials_profile_name=None,
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model}")

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model."""
        return 1024

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
                    size=self._get_embedding_dimension(),
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
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
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

        try:
            result = self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=models.FilterSelector(filter=expired_filter),
            )
            deleted_count = 1
        except Exception as e:
            logger.warning(f"Could not delete expired documents: {e}")
            deleted_count = 0

        logger.info(f"Deleted {deleted_count} expired documents")
        return deleted_count

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        collection_info = self.client.get_collection(self.COLLECTION_NAME)
        
        return {
            "collection_name": self.COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
            "ttl_days": self.TTL_DAYS,
        }

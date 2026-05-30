from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from src.constants import RAGToolState
from src.rag.vector_store import VectorStoreManager
from src.utils import get_logger

logger = get_logger(name="RAGTools")


@tool
async def retrieve_from_knowledge_base(
    runtime: ToolRuntime[RAGToolState], query: str, k: int = 5
) -> Command:
    """
    Retrieve relevant documents from the Qdrant vector database knowledge base.
    Use this when the user asks questions that might be answered from stored knowledge.
    Returns documents to be used as context for answer generation.
    """
    try:
        vector_store = VectorStoreManager()
        
        results = vector_store.vector_store.similarity_search_with_score(
            query=query, k=k
        )
        
        if not results:
            logger.info("No documents found in knowledge base for query: %s", query)
            return Command(
                update={
                    "rag_docs": {query: []},
                    "rag_metadata": {query: []},
                    "rag_word_count": {query: 0},
                    "similarity_scores": {query: []},
                    "messages": [
                        ToolMessage(
                            content="No relevant documents found in the knowledge base.",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        
        doc_contents = []
        doc_metadata = []
        scores = []
        
        for doc, score in results:
            doc_contents.append(doc.page_content)
            doc_metadata.append({
                "source": doc.metadata.get("source", "unknown"),
                "created_at": doc.metadata.get("created_at", ""),
            })
            scores.append(round(score, 4))
        
        word_count = sum(len(doc.split()) for doc in doc_contents)
        
        logger.info(
            f"RAG retrieval completed. Found {len(doc_contents)} documents ({word_count} words)."
        )
        
        return Command(
            update={
                "rag_docs": {query: doc_contents},
                "rag_metadata": {query: doc_metadata},
                "rag_word_count": {query: word_count},
                "similarity_scores": {query: scores},
                "messages": [
                    ToolMessage(
                        content=f"Retrieved {len(doc_contents)} documents ({word_count} words) with scores: {scores}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    
    except Exception as e:
        logger.error("Error in retrieve_from_knowledge_base: %s", str(e))
        return f"Knowledge base retrieval failed: {str(e)}"


build_rag_tools = [retrieve_from_knowledge_base]

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def required_env(name: str) -> str:
    value = env(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class MongoConfig:
    mongodb_uri = env("MONGODB_URI")
    mongodb_name = env("MONGODB_DATABASE_NAME")
    mongodb_collection = env("MONGODB_COLLECTION_NAME")


class ProviderConfig:
    groq_api_key = required_env("GROQ_API_KEY")
    tavily_api_key = required_env("TAVLIY_API_KEY")
    news_api_key = required_env("NEWS_API_KEY")
    aws_access_key_id = env("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = env("AWS_SECRET_ACCESS_KEY")
    aws_region = env("AWS_REGION", "us-east-1")




class RAGConfig:
    chunk_size = int(env("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap = int(env("RAG_CHUNK_OVERLAP", "200"))
    ttl_days = int(env("RAG_TTL_DAYS", "10"))
    collection_name = env("RAG_COLLECTION_NAME", "knowledge_base")
    persist_directory = env("RAG_PERSIST_DIRECTORY", "./qdrant_data")
    embedding_model = env("RAG_EMBEDDING_MODEL", "aws")


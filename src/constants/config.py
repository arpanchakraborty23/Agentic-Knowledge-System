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


class StorageConfig:
    research_data_path = Path(env("RESEARCH_DATA_PATH", "Artifacts"))


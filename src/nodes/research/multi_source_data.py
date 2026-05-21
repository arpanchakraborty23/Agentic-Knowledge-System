import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun, ArxivQueryRun
from src.constants import ProviderConfig

from src.constants import GraphState



class TavilyWebLoader:
    """Web loader using Tavily API for search and content extraction"""
    def __init__(self, api_key: str = ProviderConfig.tavily_api_key, timeout: int = 10):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key is required. Set TAVILY_API_KEY env variable.")
        self.timeout = timeout
    
    def search(self, query: str, max_results: int = 5) -> Document:
        """Perform search using Tavily API"""
        try:
            tavily = TavilySearchResults(
                api_key=self.api_key,
                max_results=max_results,
                search_depth="advanced",
                time_range="month",
                include_answer=True
            )
            results = tavily.invoke(query)
            content = str(results) if results else ""
            return Document(page_content=content, metadata={"source": "Tavily", "query": query})
        except Exception as e:
            print(f"Error during Tavily search: {str(e)}")
            return Document(page_content="", metadata={"source": "Tavily", "query": query})
    



class WikipediaLoader:
    """Loader for Wikipedia content using langchain's WikipediaQueryRun"""
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        try:
            self.wiki_query = WikipediaRetriever(verbose=True)
        except ImportError as exc:
            raise ImportError(
                "WikipediaLoader requires the `wikipedia` package. Install it with `pip install wikipedia`."
            ) from exc

    def search(self, query: str) -> Document:
        """Query Wikipedia and return a Document with the content"""
        try:
            result = self.wiki_query.invoke(query)
            data = [item.page_content for item in result]
            content = "\n\n".join(data)[:2000]
            return Document(page_content=content, metadata={"source": "Wikipedia", "query": query})
        except Exception as e:
            print(f"Error during Wikipedia query: {str(e)}")
            return Document(page_content="", metadata={"source": "Wikipedia", "query": query})



class ArxivLoader:
    """Loader for Arxiv content using langchain's ArxivQueryRun"""
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        try:
            self.arxiv_query = ArxivQueryRun(verbose=True)
        except ImportError as exc:
            raise ImportError(
                "ArxivLoader requires the `arxiv` package. Install it with `pip install arxiv`."
            ) from exc

    def search(self, query: str) -> Document:
        """Query Arxiv and return a Document with the content"""
        try:
            result = self.arxiv_query.invoke(query)
            data = [item.page_content for item in result]
            content = "\n\n".join(data)[:2000]
            return Document(page_content=content, metadata={"source": "Arxiv", "query": query})
        except Exception as e:
            print(f"Error during Arxiv query: {str(e)}")
            return Document(page_content="", metadata={"source": "Arxiv", "query": query})
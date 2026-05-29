import requests
import urllib.parse
import asyncio
from pathlib import Path
from langchain_tavily import TavilySearch
from langchain_community.tools import ArxivQueryRun
from langchain.tools import tool

from src.constants import ProviderConfig
from src.utils import get_logger, read_url

logger = get_logger(name="KnowledgeAgent")
DATA_PATH =  Path("Artifacts/research_data.txt")


async def fetch_urls(urls: list[str]) -> list[str]:
    """Fetch and extract readable text content from a list of URLs."""
    try:
        if not urls:
            return []
        data = await asyncio.gather(*[read_url(url) for url in urls])
        return [content[:1500] if content else "" for content in data]
    except Exception as e:
        logger.error("Error in fetch_urls: %s", str(e))
        return []


def save_txt(document):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH,'a', encoding='utf-8') as f:
        for doc in document:
            f.write(str(doc) + "\n")


@tool
async def web_search(query) -> str:
    """Perform a web search using Tavily and store results in state.documents. Use this for general-purpose web research."""
    try:
        query = query
        encoded_query = urllib.parse.quote(query)
        tavily_search = TavilySearch(
            tavily_api_key=ProviderConfig.tavily_api_key,
            num_results=5,
        )
        response = await asyncio.to_thread(tavily_search.invoke, encoded_query)
        urls = [doc.get("url") for doc in response.get("results", [])]
        logger.info("Web search returned %d results", len(urls))

        documents = await fetch_urls(urls)

        save_txt(documents)
        logger.info(f"Web search search completed. Collected {len(documents)} documents.")
        
        return f"Web search completed. Collected {len(documents)} documents."

    except Exception as e:
        logger.error("Error in web_search: %s", str(e))
        return f"Web search failed: {str(e)}"





@tool
async def research_paper(query) -> str:
    """Search for academic papers on Arxiv and store results in state.documents. Use for scientific, technical, or scholarly queries."""
    try:
        query = query
        arxiv = ArxivQueryRun()
        data = await asyncio.to_thread(arxiv.run, query)
        documents = [data] if data else []
        save_txt(documents)
        logger.info(f"Finance news search completed. Collected {len(documents)} documents.")
        
        return f"Arxiv search completed. Collected {len(documents)} documents."
    except Exception as e:
        logger.error("Error in research_paper: %s", str(e))
        return f"Arxiv search failed: {str(e)}"



@tool
async def coding_research(query) -> str:
    """Search for coding answers across StackOverflow, GitHub, MDN, and Microsoft Learn and store results in state.documents."""
    try:
        query = query
        encoded = urllib.parse.quote(query)
        sources = {
            "stackoverflow": f"https://stackoverflow.com/search?q={encoded}",
            "github": f"https://github.com/search?q={encoded}",
            "mdn": f"https://developer.mozilla.org/en-us/search?q={encoded}",
            "microsoft": f"https://learn.microsoft.com/en-us/search?terms={encoded}",
        }
        data = await asyncio.gather(*[read_url(url) for url in sources.values()])

        documents = [
            f"--- {name} ---\n{content[:1500]}"
            for name, content in zip(sources, data)
        ]
        save_txt(documents)
        logger.info(f" Research paper completed. Collected {len(documents)} documents.")

        return f"Coding research completed. Collected {len(documents)} documents."
    except Exception as e:
        logger.error("Error in coding_research: %s", str(e))
        return f"Coding research failed: {str(e)}"


@tool
def get_finance_news(query) -> str:
    """
    Fetch latest finance and market news based on query.
    Supports topics like: stocks, crypto, forex, economy, earnings, IPO, etc.
    Stores results in state.documents.
    """
    query = query
    api_key = ProviderConfig.news_api_key

    if not api_key:
        logger.error("NEWSAPI_KEY environment variable not set")
        return "Finance news search failed: API key not configured."

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{query} finance OR market OR stock OR economy",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return f"No finance news found for: {query}"

        documents = []
        for i, article in enumerate(articles, 1):
            documents.append(
                f"{i}. [{article['source']['name']}] {article['title']}\n"
                f"   Published: {article['publishedAt'][:10]}\n"
                f"   Summary: {article.get('description', 'N/A')}\n"
                f"   URL: {article['url']}"
            )

        save_txt(documents)
        logger.info(f"Finance news search completed. Collected {len(documents)} documents.")

        return f"Finance news search completed. Collected {len(documents)} documents."

    except requests.exceptions.RequestException as e:
        logger.error("Error in get_finance_news: %s", str(e))
        return f"Finance news search failed: {str(e)}"
    

build_research_tools = [web_search,get_finance_news,coding_research,research_paper]
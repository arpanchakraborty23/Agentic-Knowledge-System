import requests
import urllib.parse
import asyncio
from langchain_tavily import TavilySearch
from langchain_community.tools import ArxivQueryRun
from langchain.tools import  tool

from src.constants import ProviderConfig
from src.utils import get_logger, read_url

logger = get_logger(name="Research Tools")



async def fetch_urls(urls_str: str) -> str:
    """Fetch and extract readable text content from one or more URLs. Provide URLs as a comma-separated list."""
    try:
        urls = [u.strip() for u in urls_str.split(",") if u.strip()]
        if not urls:
            return "No valid URLs provided."
        data = await asyncio.gather(*[read_url(url) for url in urls])
        parts = [
            f"--- {url} ---\n{content[:2000]}"
            for url, content in zip(urls, data)
        ]
        return "\n\n".join(parts)
    except Exception as e:
        logger.error("Error in fetch_urls: %s", str(e))





@tool
async def web_search(query: str) ->list[str]:
    """Perform a web search using Tavily and return a list of result URLs. Use this for general-purpose web research."""
    try:
        encoded_query = urllib.parse.quote(query)
        tavily_search = TavilySearch(
            tavily_api_key=ProviderConfig.tavily_api_key,
            num_results=5,
        )
        response = await asyncio.to_thread(tavily_search.invoke, encoded_query)
        urls = [doc.get("url") for doc in response.get("results", [])]
        logger.info("Web search returned %d results", len(urls))

        data = [await fetch_urls(url) for url in urls]

        return data
    

    except Exception as e:
        logger.error("Error in web_search: %s", str(e))





@tool
async def research_paper(query: str) -> str:
    """Search for academic papers on Arxiv. Use for scientific, technical, or scholarly queries."""
    try:
        arxiv = ArxivQueryRun()
        data = await asyncio.to_thread(arxiv.run, query)
        return data
    except Exception as e:
        logger.error("Error in research_paper: %s", str(e))
        return f"Arxiv search failed: {e}"



@tool
async def coding_research(query: str) -> str:
    """Search for coding answers across StackOverflow, GitHub, MDN, and Microsoft Learn."""
    try:
        encoded = urllib.parse.quote(query)
        sources = {
            "stackoverflow": f"https://stackoverflow.com/search?q={encoded}",
            "github": f"https://github.com/search?q={encoded}",
            "mdn": f"https://developer.mozilla.org/en-us/search?q={encoded}",
            "microsoft": f"https://learn.microsoft.com/en-us/search?terms={encoded}",
        }
        data = await asyncio.gather(*[read_url(url) for url in sources.values()])

        return "\n\n".join(
            f"--- {name} ---\n{content[:2000]}"
            for name, content in zip(sources, data)
        )
    except Exception as e:
        logger.error("Error in coding_research: %s", str(e))


@tool
def get_finance_news(query: str) -> str:
    """
    Fetch latest finance and market news based on a query.
    Supports topics like: stocks, crypto, forex, economy, earnings, IPO, etc.
    Stores results in state memory for downstream tools.
    """
    api_key = ProviderConfig.news_api_key  # Get free key at newsapi.org

    if not api_key:
        return "Error: NEWSAPI_KEY environment variable not set."

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

        # Format articles
        results = []
        for i, article in enumerate(articles, 1):
            results.append(
                f"{i}. [{article['source']['name']}] {article['title']}\n"
                f"   Published: {article['publishedAt'][:10]}\n"
                f"   Summary: {article.get('description', 'N/A')}\n"
                f"   URL: {article['url']}"
            )

        formatted = "\n\n".join(results)



        return formatted

    except requests.exceptions.RequestException as e:
        logger.error("Error in coding_research: %s", str(e))
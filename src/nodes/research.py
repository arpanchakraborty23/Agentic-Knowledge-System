from typing import Any
import asyncio
import arxiv
import requests
import urllib.parse

from langchain.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.constants import GraphState, ResearchAgentState
from src.constants.config import ProviderConfig
from src.prompts import RESEARCH_PROMPT
from src.utils import get_logger, llm_chain, read_url

logger = get_logger(name="KnowledgeAgent")
MAX_TOOL_ROUNDS = 3


class ResearchReActAgent:
    """
    This is Research Sub Agent built on a ReAct agent architecture.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def _build_research_context(
        self,
        documents: list[str],
        messages: list[Any] | None = None,
        limit: int = 4,
        max_chars: int = 700,
    ) -> str:
        # Keep the model context compact so tool output does not blow the window.
        snippets = []
        if documents:
            recent_docs = documents[-limit:]
            for index, document in enumerate(recent_docs, 1):
                snippets.append(f"[Doc {index}] {document[:max_chars]}")

        if messages:
            recent_messages = messages[-2:]
            for index, message in enumerate(recent_messages, 1):
                content = getattr(message, "content", "")
                if content:
                    snippets.append(f"[Tool {index}] {str(content)[:200]}")

        return "\n\n".join(snippets)

    def _build_tools(self, documents: list[str]) -> list[BaseTool]:
        # Keep tool schemas narrow so Groq only sees the query field.
        @tool
        async def web_search(query: str) -> str:
            """Search the web and collect a small set of readable page excerpts."""
            try:
                encoded_query = urllib.parse.quote(query)
                tavily_search = TavilySearch(
                    tavily_api_key=ProviderConfig.tavily_api_key,
                    num_results=3,
                )
                response = await asyncio.to_thread(tavily_search.invoke, encoded_query)
                urls = [
                    item.get("url")
                    for item in response.get("results", [])
                    if item.get("url")
                ]
                logger.info("Web search returned %d results", len(urls))

                page_text = await asyncio.gather(*[read_url(url) for url in urls])
                collected = [text[:800] if text else "" for text in page_text]
                documents.extend(collected)

                return f"Web search completed. Collected {len(collected)} documents."
            except Exception as e:
                logger.error("Error in web_search: %s", str(e))
                return f"Web search failed: {str(e)}"

        @tool
        async def research_paper(query: str) -> str:
            """Search Arxiv for relevant papers and capture a short excerpt."""
            try:
                # arxiv 4.x uses Client.results(Search(...)) instead of Search.results().
                def _fetch_arxiv_summaries() -> list[str]:
                    client = arxiv.Client(page_size=3, delay_seconds=0.0, num_retries=2)
                    search = arxiv.Search(query=query[:300], max_results=3)
                    docs: list[str] = []

                    for result in client.results(search):
                        docs.append(
                            f"Published: {result.published.date()}\n"
                            f"Title: {result.title}\n"
                            f"Authors: {', '.join(author.name for author in result.authors)}\n"
                            f"Summary: {result.summary}"
                        )

                    return docs

                documents_found = await asyncio.to_thread(_fetch_arxiv_summaries)
                documents.extend(documents_found)

                return f"Arxiv search completed. Collected {len(documents_found)} documents."
            except Exception as e:
                logger.error("Error in research_paper: %s", str(e))
                return f"Arxiv search failed: {str(e)}"

        @tool
        async def coding_research(query: str) -> str:
            """Search coding documentation and community sites for relevant answers."""
            try:
                encoded = urllib.parse.quote(query)
                sources = {
                    "stackoverflow": f"https://stackoverflow.com/search?q={encoded}",
                    "github": f"https://github.com/search?q={encoded}",
                    "mdn": f"https://developer.mozilla.org/en-US/search?q={encoded}",
                    "microsoft": f"https://learn.microsoft.com/en-us/search?terms={encoded}",
                }
                data = await asyncio.gather(*[read_url(url) for url in sources.values()])

                documents_found = [
                    f"--- {name} ---\n{content[:800]}"
                    for name, content in zip(sources, data)
                ]
                documents.extend(documents_found)

                return f"Coding research completed. Collected {len(documents_found)} documents."
            except Exception as e:
                logger.error("Error in coding_research: %s", str(e))
                return f"Coding research failed: {str(e)}"

        @tool
        def get_finance_news(query: str) -> str:
            """
            Fetch latest finance and market news based on a query.
            Supports topics like stocks, crypto, forex, economy, earnings, and IPOs.
            """
            api_key = ProviderConfig.news_api_key

            if not api_key:
                logger.error("NEWSAPI_KEY environment variable not set")
                return "Finance news search failed: API key not configured."

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{query} finance OR market OR stock OR economy",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 3,
                "apiKey": api_key,
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                articles = data.get("articles", [])
                if not articles:
                    return f"No finance news found for: {query}"

                documents_found = []
                for index, article in enumerate(articles, 1):
                    documents_found.append(
                        f"{index}. [{article['source']['name']}] {article['title']}\n"
                        f"   Published: {article['publishedAt'][:10]}\n"
                        f"   Summary: {article.get('description', 'N/A')}\n"
                        f"   URL: {article['url']}"
                    )

                documents.extend(documents_found)
                return f"Finance news search completed. Collected {len(documents_found)} documents."
            except requests.exceptions.RequestException as e:
                logger.error("Error in get_finance_news: %s", str(e))
                return f"Finance news search failed: {str(e)}"

        return [web_search, research_paper, coding_research, get_finance_news]

    def __llm(self, state: ResearchAgentState, tool_chain, final_chain, documents: list[str]) -> dict[str, Any]:
        query = state.query
        logger.info("No of words in Query: %d", len(query.split()))
        research_context = self._build_research_context(documents, state.messages)
        chain = final_chain if state.tool_rounds >= MAX_TOOL_ROUNDS else tool_chain
        response = chain.invoke(
            input={
                "query": query,
                "domain": state.domain,
                "tool_rounds": state.tool_rounds,
                "max_tool_rounds": MAX_TOOL_ROUNDS,
                "research_context": research_context,
            }
        )

        next_tool_rounds = state.tool_rounds
        if getattr(response, "tool_calls", None):
            next_tool_rounds += 1

        logger.info("Research Agent documents collected so far: %d", len(documents))

        return {
            "messages": [response],
            "tool_rounds": next_tool_rounds,
        }

    def _research_nodes(self, tools: list[BaseTool], tool_chain, final_chain, documents: list[str]):
        try:
            agent_builder = StateGraph(ResearchAgentState)

            agent_builder.add_node("llm", lambda state: self.__llm(state, tool_chain, final_chain, documents))
            agent_builder.add_node("tools", ToolNode(tools))

            agent_builder.set_entry_point("llm")
            agent_builder.add_conditional_edges(
                "llm",
                tools_condition,
                {"tools": "tools", END: END},
            )
            agent_builder.add_edge("tools", "llm")

            return agent_builder.compile()
        except Exception as e:
            logger.error("Error in research agent build: %s", str(e))
            raise


    async def run(self, query: str, domain: str) -> dict[str, Any]:
        """Invoke the research graph and return the final answer with documents."""
        documents: list[str] = []
        tools = self._build_tools(documents)
        tool_chain = llm_chain(template=RESEARCH_PROMPT, llm=self.llm, tools=tools)
        final_chain = llm_chain(template=RESEARCH_PROMPT, llm=self.llm)
        research_agent = self._research_nodes(tools, tool_chain, final_chain, documents)

        final_state = await research_agent.ainvoke(
            input=ResearchAgentState(query=query, domain=domain, tool_rounds=0)
        )

        messages = final_state.get("messages", []) if isinstance(final_state, dict) else final_state.messages

        final_text = ""
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                final_text = last.content or ""

        logger.info("Research complete — documents collected: %d", len(documents))

        return {
            "answer": final_text,
            "documents": documents,
        }


class ResearchNode:
    """ReAct agent with web search, URL fetching, paper search, and coding research tools."""

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.agent = ResearchReActAgent(self.model)

    async def search(self, state: GraphState) -> GraphState:
        """Execute the research ReAct agent and update the graph state."""
        result = await self.agent.run(query=state.query, domain=state.classified_domain)
        return state.model_copy(update={"research_data": result, "knowledge_base": "add"})

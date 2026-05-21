from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from src.constants import GraphState, ResearchOutput
from src.nodes.research.multi_source_data import TavilyWebLoader, WikipediaLoader, ArxivLoader


class ResearchAgent:
    def __init__(self):
        self.tavily = TavilyWebLoader()
        self.wikipedia = WikipediaLoader()
        self.arxiv = ArxivLoader()

    def _search_source(self, loader, query: str, source_name: str) -> Tuple[str, str]:
        document = loader.search(query)
        return source_name, document.page_content or ""

    def run(self, state: GraphState) -> ResearchOutput:
        query = state.topic or (state.messages[-1].content if state.messages else "")

        sources = [
            (self.tavily, "Tavily"),
            (self.wikipedia, "Wikipedia"),
            (self.arxiv, "Arxiv"),
        ]

        retrieved_docs: List[str] = []
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {
                executor.submit(self._search_source, loader, query, name): name
                for loader, name in sources
            }
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    _, content = future.result()
                    if content:
                        retrieved_docs.append(f"Source: {source_name}\n{content}")
                except Exception as exc:
                    retrieved_docs.append(f"Source: {source_name} encountered an error: {exc}")

        return ResearchOutput(
            concepts=[],
            retrieved_docs=retrieved_docs,
            code_examples=[],
            references=[],
            summary=f"Research completed for query: {query}",
        )
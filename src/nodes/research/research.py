from typing import List

from src.constants import GraphState, ResearchOutput
from src.nodes.research.multi_source_data import TavilyWebLoader, WikipediaLoader, ArxivLoader


class ResearchAgent:
    def __init__(self):
        self.tavily = TavilyWebLoader()
        self.wikipedia = WikipediaLoader()
        self.arxiv = ArxivLoader()

    def run(self, state: GraphState) -> ResearchOutput:
        query = state.topic or state.messages[-1].content if state.messages else ""
        
        tavily_doc = self.tavily.search(query)
        wiki_doc = self.wikipedia.search(query)
        arxiv_doc = self.arxiv.search(query)

        all_docs = [tavily_doc.page_content, wiki_doc.page_content, arxiv_doc.page_content]
        retrieved_docs = [d for d in all_docs if d]

        return ResearchOutput(
            concepts=[],
            retrieved_docs=retrieved_docs,
            code_examples=[],
            references=[],
            summary=f"Research completed for query: {query}"
        )
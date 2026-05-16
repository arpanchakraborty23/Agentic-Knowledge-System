
from src.constants import GraphState, ResearchOutput

class ResearchAgent:

    def run(self, query: str) -> ResearchOutput:

        docs = retriever.invoke(query)

        code = github_search(query)

        return ResearchOutput(
            concepts=[
                "Dependency Injection",
                "FastAPI Depends"
            ],
            retrieved_docs=docs,
            code_examples=code,
            references=[
                "FastAPI Docs"
            ],
            summary="FastAPI dependency injection system..."
        )
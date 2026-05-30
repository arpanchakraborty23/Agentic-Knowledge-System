import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from rich import print
from langchain.chat_models import init_chat_model

from src.constants import GraphState
from src.nodes import ResearchNode


async def test_research_node():
    print("[bold green]Testing Research Node...[/bold green]")

    llm = init_chat_model(
        model="groq:openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    node = ResearchNode(llm)

    state = GraphState(
        query="What are the latest advancements in reinforcement learning?",
        classified_domain="general_knowledge"
    )

    result = await node.search(state)

    print("\n[bold blue]Research Data Output:[/bold blue]")

    print(f"\n[bold]knowledge_base:[/bold] {result.knowledge_base}")


if __name__ == "__main__":
    asyncio.run(test_research_node())

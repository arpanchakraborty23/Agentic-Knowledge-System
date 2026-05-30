import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from rich import print
from rich.table import Table
from langchain.chat_models import init_chat_model

from src.constants import GraphState
from src.nodes import ResearchNode, ClassifierNode, KnowledgeNode


async def test_research_node():
    """Test research node only."""
    print("\n[bold green]=== Testing Research Node ===[/bold green]")

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

    print(f"\n[bold]Documents collected:[/bold] {len(result.research_data.split('---')) if result.research_data else 0}")
    print(f"[bold]Knowledge base status:[/bold] {result.knowledge_base}")
    
    if result.research_data:
        print(f"\n[bold]Preview (first 500 chars):[/bold]")
        print(result.research_data[:500] + "...")

    return result


async def test_knowledge_node():
    """Test knowledge node for creating and retrieving knowledge."""
    print("\n[bold green]=== Testing Knowledge Node ===[/bold green]")

    llm = init_chat_model(
        model="groq:openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    research_node = ResearchNode(llm)
    knowledge_node = KnowledgeNode(llm)

    state = GraphState(
        query="Python async await best practices",
        classified_domain="software",
        user_id="test_user_001"
    )

    print("\n[dim]Step 1: Collecting research data...[/dim]")
    research_result = await research_node.search(state)

    if not research_result.research_data:
        print("[red]No research data collected[/red]")
        return

    state = research_result

    print("\n[dim]Step 2: Creating knowledge base...[/dim]")
    knowledge_result = knowledge_node.invoke(state)

    print(f"\n[bold green]Knowledge base created successfully[/bold green]")

    print("\n[dim]Step 3: Getting statistics...[/dim]")
    stats = knowledge_node.kb_manager.get_stats()

    table = Table(title="Knowledge Base Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        table.add_row(str(key), str(value))
    
    print(table)

    return knowledge_result


async def test_retrieve_knowledge():
    """Test retrieving knowledge from vector store."""
    print("\n[bold green]=== Testing Knowledge Retrieval ===[/bold green]")

    llm = init_chat_model(
        model="groq:openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    knowledge_node = KnowledgeNode(llm)

    state = GraphState(
        query="How to use async await in Python?",
        knowledge_base="retrieve",
        user_id="test_user_001"
    )

    print("\n[dim]Retrieving relevant documents...[/dim]")
    result = knowledge_node.invoke(state)

    print(f"[bold green]Retrieval complete[/bold green]")

    return result


async def test_full_pipeline():
    """Test the complete pipeline: Research -> Knowledge."""
    print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    print("[bold magenta]=== Full Pipeline Test ===[/bold magenta]")
    print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")

    llm = init_chat_model(
        model="groq:openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    classifier_node = ClassifierNode(llm)
    research_node = ResearchNode(llm)
    knowledge_node = KnowledgeNode(llm)

    state = GraphState(
        query="What is machine learning?",
        user_id="pipeline_test_user"
    )

    print("[cyan]Step 1: Classifying query...[/cyan]")
    classified = classifier_node.classify(state)
    print(f"   Domain: [bold]{classified.classified_domain}[/bold]")
    print(f"   Note: {classified.classifier_note}")

    print("\n[cyan]Step 2: Researching...[/cyan]")
    research_result = await research_node.search(classified)
    print(f"   Data collected: [bold]{len(research_result.research_data) if research_result.research_data else 0} chars[/bold]")

    if research_result.research_data:
        print("\n[cyan]Step 3: Storing in knowledge base...[/cyan]")
        knowledge_result = knowledge_node.invoke(research_result)
        print("   [bold green]Stored successfully[/bold green]")

        stats = knowledge_node.kb_manager.get_stats()
        print(f"\n   Total documents: [bold]{stats.get('points_count', 0)}[/bold]")

    print("\n[bold green]Pipeline test complete![/bold green]")
    return research_result


async def main():
    """Run all tests."""
    print("\n[bold yellow]═══════════════════════════════════════[/bold yellow]")
    print("[bold yellow]   Knowledge Agent Test Suite[/bold yellow]")
    print("[bold yellow]═══════════════════════════════════════[/bold yellow]")

    tests = {
        "1": ("Research Node", test_research_node),
        "2": ("Knowledge Node (Create)", test_knowledge_node),
        "3": ("Knowledge Node (Retrieve)", test_retrieve_knowledge),
        "4": ("Full Pipeline", test_full_pipeline),
    }

    print("\n[bold]Available tests:[/bold]")
    for key, (name, _) in tests.items():
        print(f"  [cyan]{key}[/cyan]. {name}")
    print(f"  [cyan]all[/cyan]. Run all tests")
    print(f"  [cyan]q[/cyan]. Quit")

    choice = input("\n[bold]Select test to run: [/bold]").strip().lower()

    if choice == "q":
        print("\n[dim]Exiting...[/dim]")
        return

    if choice == "all":
        for key in tests:
            name, test_func = tests[key]
            print(f"\n[dim]Running: {name}[/dim]")
            try:
                await test_func()
            except Exception as e:
                print(f"[red]Error in {name}: {str(e)}[/red]")
    elif choice in tests:
        name, test_func = tests[choice]
        try:
            await test_func()
        except Exception as e:
            print(f"[red]Error: {str(e)}[/red]")
    else:
        print(f"[red]Invalid choice: {choice}[/red]")


if __name__ == "__main__":
    asyncio.run(main())

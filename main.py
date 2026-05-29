import asyncio
from rich import print
from src.tools import ToolMemory, create_research_tools

async def test_research_tools():
    print("[bold green]Testing Research Tools...[/bold green]")

    memory = ToolMemory()
    tools = create_research_tools(memory)
    tool_map = {t.name: t for t in tools}

    print("\n[bold blue]1. Testing Web Search Tool[/bold blue]")
    result = await tool_map["web_search"].ainvoke({"query": "what is LangChain"})
    print(f"URLs: {result[:300]}...")

    print("\n[bold blue]2. Testing Arxiv Research Tool[/bold blue]")
    result = await tool_map["research_paper"].ainvoke({"query": "Attention is all you need"})
    print(f"Paper data: {result[:300]}...")

    print("\n[bold blue]3. Testing Coding Research Tool[/bold blue]")
    result = await tool_map["coding_research"].ainvoke({"query": "asyncio gather python"})
    print(f"Coding data: {result[:300]}...")

    print("\n[bold green]Tool Memory Contents:[/bold green]")
    for tool_name, entries in memory.get_all_results().items():
        print(f"\n  {tool_name}: {len(entries)} call(s)")

if __name__ == "__main__":
    asyncio.run(test_research_tools())
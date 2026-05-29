from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from src.constants import GraphState
from src.tools import ToolMemory, create_research_tools
from src.utils import get_logger

logger = get_logger(name="Research Agent")










class ResearchAgent:
    """ReAct agent with web search, URL fetching, paper search, and coding research tools.

    Every tool call stores its output in a shared ToolMemory so the agent
    can inspect past results across turns.
    """

    def __init__(self, model: BaseChatModel, system_prompt: str | None = None):
        self.model = model
        self.memory = ToolMemory()
        self.tools = create_research_tools(self.memory)

        prompt = system_prompt or (
            "You are a research assistant. Use the available tools to gather information "
            "from the web, academic papers, and coding resources. "
            "Always cite your sources when presenting findings."
        )

        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            state_modifier=prompt,
        )
        logger.info("ResearchAgent initialized with %d tools", len(self.tools))

    async def run(self, state: GraphState) -> GraphState:
        """Execute the research ReAct agent and update the graph state."""
        self.memory.clear()

        response = await self.agent.ainvoke(
            {"messages": [("human", state.query)]}
        )

        messages = response.get("messages", [])
        final_content = messages[-1].content if messages else ""

        research_data = {
            "tool_memory": self.memory.get_all_results(),
            "summary": final_content,
        }

        return state.model_copy(update={"research_data": research_data})

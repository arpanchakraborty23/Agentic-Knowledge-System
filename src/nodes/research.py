from __future__ import annotations 

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent 
from langchain.agents.middleware import ToolCallLimitMiddleware

from src.constants import GraphState, ResearchToolState
from src.tools.research_tools import build_research_tools
from src.prompts import RESEARCH_PROMPT
from src.utils import (
    get_logger,
)

logger = get_logger(name="ResearchNode")



class ResearchReActAgent:
    """
    Research sub-agent using direct tool execution.
    Collects research documents stored as text files for downstream use.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.tools = build_research_tools
        logger.info(f"ResearchReActAgent initialized with {len(self.tools)} tools")

    def _agent(self):
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=RESEARCH_PROMPT,
            state_schema=ResearchToolState,
            middleware=[
                ToolCallLimitMiddleware(
                    run_limit=2,
                    exit_behavior='end'
                )
            ]
        )

    

class ResearchNode:
    """Research node that collects documents and stores them in graph state."""

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.agent = ResearchReActAgent(self.model)._agent()
        logger.info("ResearchNode initialized")

    async def search(self, state: GraphState) -> GraphState:
        """Execute research agent and update graph state."""

        logger.info(
            f"ResearchNode.search - query: {state.query}, "
            f"domain: {state.classified_domain}"
        )
        
        result = await self.agent.ainvoke(
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": f"Query: {state.query}\nDomain: {state.classified_domain}"
                    }
                ]
            }
        )

        research_docs = result.get("research_docs", []) if isinstance(result, dict) else getattr(result, "research_docs", [])
        research_data = "\n\n".join(research_docs) if research_docs else None
        word_count = result.get("research_word_count", 0) if isinstance(result, dict) else getattr(result, "research_word_count", 0)
        logger.info(f"Transferred {len(research_docs)} documents ({word_count} words) from sub-agent to main state")
        
        return state.model_copy(
            update={
                "research_data": research_data,
                "knowledge_base": "add",
            }
        )



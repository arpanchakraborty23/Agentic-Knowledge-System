from __future__ import annotations

import asyncio
from typing import Any
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent 
from langchain.agents.middleware import ToolCallLimitMiddleware

from src.constants import GraphState
from src.tools.research_tools import build_research_tools
from src.prompts import RESEARCH_PROMPT
from src.utils import (
    get_logger,
)

logger = get_logger(name="ResearchNode")
DATA_PATH = Path("Artifacts/research_data.txt")


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
        
        if DATA_PATH.exists():
            research_data = DATA_PATH.read_text(encoding="utf-8")
            logger.info(f"ResearchNode.search complete - collected data from {DATA_PATH}")
        else:
            research_data = ""
            logger.warning(f"ResearchNode.search - no data file found at {DATA_PATH}")

        print(research_data)
        
        return state.model_copy(
            update={
                "research_data": research_data,
                "knowledge_base": "add",
            }
        )


from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.constants import GraphState, ResearchAgentState
from src.prompts import RESEARCH_PROMPT
from src.tools import get_finance_news, coding_research, research_paper, web_search
from src.utils import get_logger,llm_chain

logger = get_logger(name="KnowledgeAgent")


class ResearchReActAgent:
    """
    This is Reseacrh Sub Agent build on ReAct Agent Architechure
    """
    def __init__(self,llm:BaseChatModel) -> None:
        self.tools =[get_finance_news, coding_research, research_paper, web_search]
        self.llm_chain = llm_chain(template=RESEARCH_PROMPT,llm=llm,tools=self.tools)

        self.research_agent = self._research_nodes() 

    def __llm(self, state: ResearchAgentState) -> dict[str, Any]:
        response = self.llm_chain.invoke({"query": state.query, "messages": []})

        return {
            "messages": [response],
        }

    def _research_nodes(self):
        try:
            agent_builder = StateGraph(ResearchAgentState)

            # nodes
            agent_builder.add_node("llm",self.__llm)
            agent_builder.add_node("tools",ToolNode(self.tools))

            # edges
            agent_builder.set_entry_point("llm")
            agent_builder.add_conditional_edges(
                "llm",
                tools_condition,
                {"tools": "tools", END: END}
            )
            agent_builder.add_edge("tools", "llm")

            return agent_builder.compile()
            
        except Exception as e:
            logger.error("Error in coding_research: %s", str(e))

    async def run(self, query: str, domain: str) -> dict[str, Any]:
        """Invoke the research graph and return the final answer with documents."""
        
        final_state = await self.research_agent.ainvoke(
            input=ResearchAgentState(
                query=query, domain=domain
            )
        )

        messages = final_state.get("messages", []) if isinstance(final_state, dict) else final_state.messages
        documents = final_state.get("documents", []) if isinstance(final_state, dict) else final_state.documents

        final_text = ""
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                final_text = last.content or ""

        # Collect documents from ToolMessages in messages
        from langchain.messages import ToolMessage
        collected_docs = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content:
                content = str(msg.content)
                if "completed" in content.lower() or "collected" in content.lower():
                    collected_docs.append(content)

        return {
            "answer": final_text,
            "documents": documents or collected_docs,
        }




class ResearchNode:
    """ReAct agent with web search, URL fetching, paper search, and coding research tools.

    Every tool call stores its output in a shared ToolMemory so the agent
    can inspect past results across turns.
    """

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.agent = ResearchReActAgent(self.model)



    async def search(self, state: GraphState) -> GraphState:
        """Execute the research ReAct agent and update the graph state."""

        result = await self.agent.run(
            query=state.query, domain=state.classified_domain
        )

        return state.model_copy(update={"research_data": result, "knowledge_base": "add"})

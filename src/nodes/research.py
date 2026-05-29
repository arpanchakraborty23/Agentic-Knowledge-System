from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.constants import GraphState, ResearchAgentState
from src.prompts import RESEARCH_PROMPT
from src.tools import get_finance_news, coding_research, research_paper, web_search
from src.utils import get_logger,llm_chain

logger = get_logger(name="Research Agent")


class ResearchReActAgent:
    """
    This is Reseacrh Sub Agent build on ReAct Agent Architechure
    """
    def __init__(self,llm:BaseChatModel) -> None:
        self.tools =[get_finance_news, coding_research, research_paper, web_search]
        self.llm_chain = llm_chain(template=RESEARCH_PROMPT,llm=llm,tools=self.tools) 

        self.research_agent = self._research_nodes() 

    def __llm(self,state:ResearchAgentState)-> dict:

        query = state.query
        response = self.llm_chain.invoke(query)

        logger.info(f"Research Agent Query: {query}, /nRespose {response}")
        
        return {
            "Agent_response" : response,
            "Documents": state.results
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
                "tools",
                tools_condition
            )
            agent_builder.add_edge("tools","llm")
            agent_builder.add_edge("llm",END)

            return agent_builder.compile()
            
        except Exception as e:
            logger.error("Error in coding_research: %s", str(e))

    def run(self,query:str, domain):
        """Invokcation of graph node"""
        
        response = self.research_agent.invoke(
            input=ResearchAgentState(
                query=query,domain=domain
            )
        )

        return response['Documents']




class ResearchAgent:
    """ReAct agent with web search, URL fetching, paper search, and coding research tools.

    Every tool call stores its output in a shared ToolMemory so the agent
    can inspect past results across turns.
    """

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.agent = ResearchReActAgent(self.model)



    async def search(self, state: GraphState) -> GraphState:
        """Execute the research ReAct agent and update the graph state."""

        response = await self.agent.run(
            query=state.query, domain=state.classified_domain
        )


        return state.model_copy(update={"research_data": response})

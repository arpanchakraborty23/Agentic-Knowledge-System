from langchain_core.messages import HumanMessage
from langchain.chat_models import BaseChatModel

from src.constants import GraphState
from src.nodes.profile import ProfileAgent
from src.nodes.router import RouterAgent
from src.nodes.research.research import ResearchAgent
from src.nodes.supervisor import SupervisorNode


class DeepTutorOrchestrator:
    def __init__(self, model: BaseChatModel):
        self.profile_agent = ProfileAgent(model)
        self.router_agent = RouterAgent(model)
        self.research_agent = ResearchAgent()
        self.supervisor_node = SupervisorNode(model)

    def run(self, user_query: str, user_id: str | None = None, board: str | None = None, grade: str | None = None, subject: str | None = None, topic: str | None = None) -> GraphState:
        state = GraphState(
            user_id=user_id,
            messages=[HumanMessage(content=user_query)],
            board=board,
            grade=grade,
            subject=subject,
            topic=topic,
            memory={},
            workflow=["profile", "route", "research", "supervisor"],
        )

        state = self.profile_agent.run(state)
        state = self.router_agent.route(state)

        research_output = self.research_agent.run(state)
        tool_results = dict(state.tool_results or {}, research_output=research_output)
        state = state.model_copy(update={"tool_results": tool_results})

        state = self.supervisor_node.run(state)
        return state

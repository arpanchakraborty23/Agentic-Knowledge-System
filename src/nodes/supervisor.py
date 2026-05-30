from langchain.chat_models import BaseChatModel
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from src.constants import GraphState, SupervisorOutput, RAGToolState
from src.prompts import SUPERVISOR_PROMPT
from src.tools import build_rag_tools, build_research_tools
from src.utils import get_logger

logger = get_logger(name="Supervisor")


class SupervisorReActAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
    
    def _agent(self):
        all_tools = build_rag_tools + build_research_tools
        return create_agent(
            model=self.llm,
            system_prompt=SUPERVISOR_PROMPT,
            context_schema=SupervisorOutput,
            tools=all_tools,
            middleware=[
                ToolCallLimitMiddleware(
                    run_limit=3,
                    exit_behavior='end'
                )
            ]
        )


class SupervisorNode:
    def __init__(self, model: BaseChatModel):
        self._agent = SupervisorReActAgent(model)._agent()

    async def invoke(self, state: GraphState) -> GraphState:
        try:
            tool_state = RAGToolState(
                session_id=state.user_id or "default",
                query=state.query,
                messages=state.messages,
            )
            
            result = await self._agent.ainvoke(tool_state.model_dump())
            
            if isinstance(result, dict):
                output_data = result.get("context", {})
                method = output_data.get("method", "direct")
                answer = output_data.get("answer", "")
                
                rag_docs = result.get("rag_docs", {})
                similarity_scores = result.get("similarity_scores", {})
                
                response_messages = result.get("messages", [])
                
                return GraphState(
                    user_id=state.user_id,
                    messages=state.messages + response_messages,
                    query=state.query,
                    classified_domain=state.classified_domain,
                    classifier_note=state.classifier_note,
                    knowledge_base=state.knowledge_base,
                    research_data=answer,
                )
            
            return state
            
        except Exception as e:
            logger.error("Error in SupervisorNode execution: %s", str(e))
            return state


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
            response_format=SupervisorOutput,
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
            result = await self._agent.ainvoke(
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": state.query
                        }
                    ]
                }
            )
            
            if isinstance(result, dict):
                structured_response = result.get("structured_response")
                
                if structured_response and isinstance(structured_response, SupervisorOutput):
                    method = structured_response.method
                    answer = structured_response.answer
                else:
                    method = "direct"
                    answer = ""
                
                response_messages = result.get("messages", [])
                
                return state.model_copy(
                    update={
                        "messages": state.messages + response_messages,
                        "research_data": answer,
                        "knowledge_base": "retrieve" if method == "rag" else None,
                    }
                )
            
            return state
            
        except Exception as e:
            logger.error("Error in SupervisorNode execution: %s", str(e))
            return state


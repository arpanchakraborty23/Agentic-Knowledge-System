import os
from langchain.chat_models import BaseChatModel

from src.constants import GraphState, RouteQuery
from src.prompts import ROUTE_PROMPT
from src.utils import llm_chain


# RouterAgent is responsible for classifying incoming user queries
# and determining which specialized agent should handle them.
# It uses an LLM with structured output to route queries to categories
# like software, indian_school, job_preparation, general_knowledge, or math.
class RouterAgent:
    def __init__(self, model: BaseChatModel):
        self.model = model

        # Initialize the routing chain with a prompt template,
        # the LLM model, and a Pydantic schema (RouteQuery) for
        # structured output parsing.
        self.chain = llm_chain(
            ROUTE_PROMPT,
            self.model,
            RouteQuery
        )

    def route(self, state: GraphState) -> GraphState:
        # Extract the most recent user message from the conversation history.
        # If no messages exist, use an empty string as fallback.
        query = (
            state.messages[-1].content
            if state.messages
            else ""
        )

        response = self.chain.invoke({
            "query": query,
            "board": state.board or "None",
            "grade": state.grade or "None",
            "subject": state.subject or "None",
            "topic": state.topic or "None",
            "domain": state.memory.get("domain") if isinstance(state.memory, dict) else "None",
        })

        memory = dict(state.memory or {})
        memory["route_note"] = response.note
        memory["route_domain"] = response.route

        return state.model_copy(
            update={
                "current_agent": response.route,
                "memory": memory,
            }
        )
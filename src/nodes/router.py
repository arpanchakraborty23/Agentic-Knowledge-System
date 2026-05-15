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

        # Invoke the LLM chain with the user query.
        # The chain will classify the query into a route category and
        # return a RouteQuery object with the `route` attribute set.
        response = self.chain.invoke({
            "query": query
        })

        # Update the GraphState with the determined routing category
        # so the next node in the workflow knows which agent to use.
        return state.model_copy(
            update={
                "current_agent": response.route
            }
        )
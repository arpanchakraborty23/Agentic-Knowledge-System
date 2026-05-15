from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Literal
from operator import add
from enum import Enum
from langchain.messages import AnyMessage

# GraphState acts as the shared conversation and workflow context
# for the agent system. It holds user metadata, message history,
# task-specific information, tool outputs, and the next step to take.
class GraphState(BaseModel):
    # Unique identifier for the current user or session.
    user_id: Optional[str] = None

    # History of chat messages exchanged with the user or agents.
    # Uses LangChain message objects so the agent can preserve role/type.
    messages: Annotated[List[AnyMessage], add] = Field(default_factory=list)

    # Educational context properties. These can be used to tailor
    # agent behavior, retrieval, or response generation.
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None

    # The identifier of the currently active or selected agent.
    current_agent: Optional[str] = None

    # Documents retrieved from search or knowledge sources during
    # the current session.
    retrieved_docs: list = Field(default_factory=list)

    # A generic memory store for persistent or temporary state.
    memory: dict = Field(default_factory=dict)

    # Store outputs from tools or external systems keyed by tool name.
    tool_results: dict = Field(default_factory=dict)

    # Sequence of workflow actions or steps the system is tracking.
    workflow: Annotated[List[str], add] = Field(default_factory=list)

    # The next action the agent should perform, if any.
    next_action: Optional[str] = None



# RouteType defines the available categories for routing user queries
# to specialized agents based on query content and domain.
class RouteType(str, Enum):
    SOFTWARE = "software"
    INDIAN_SCHOOL = "indian_school"
    JOB_PREPARATION = "job_preparation"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MATH = "math"


# RouteQuery represents the structured output from the routing LLM.
# It contains the determined routing category for a user's query.
class RouteQuery(BaseModel):
    # The category selected for routing the user's query to an appropriate agent.
    route : RouteType = Field(
        ..., description="The category selected for routing the user's query.")
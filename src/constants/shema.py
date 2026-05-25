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
    query: str = Field(..., description="The query to be processed.")

    # The classified domain/category of the query after routing (e.g., software, indian_school, etc.)
    classified_domain : Optional[str] = Field(None, description="The classified domain of the query")
    classifier_note: Optional[str] = Field(None, description="Optional note or explanation from the LLM about the classification decision.")

    research_data : Optional[dict] = Field(None, description="Data retrieved from research nodes, if applicable.")




# RouteType defines the available categories for routing user queries
# to specialized agents based on query content and domain.
class RouteType(str, Enum):
    SOFTWARE = "software"
    INDIAN_SCHOOL = "indian_school"
    JOB_PREPARATION = "job_preparation"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MATH = "math"
    NON_TECH = "non_tech"


# RouteQuery represents the structured output from the routing LLM.
# It contains the determined routing category for a user's query.
class ClassifiQuery(BaseModel):
    # The category selected for routing the user's query to an appropriate agent.
    route : RouteType = Field(
        ..., description="The category selected for routing the user's query.")
    note: Optional[str] = Field(
        None, description="Optional note or explanation from the LLM about the routing decision.")





class ResearchOutput(BaseModel):

    concepts: List[str]

    retrieved_docs: List[str]

    code_examples: List[str]

    references: List[str]

    summary: str
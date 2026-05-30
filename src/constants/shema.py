from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Literal
from operator import add
from enum import Enum
from pathlib import Path
from langchain.messages import AnyMessage
from langchain_core.documents import Document
from langchain.agents import AgentState


class GraphState(BaseModel):
    user_id: Optional[str] = None
    messages: Annotated[List[AnyMessage], add] = Field(default_factory=list, description="Conversation history")
    query: str = Field(..., description="The query to be processed.")
    classified_domain: Optional[str] = Field(None, description="The classified domain of the query")
    classifier_note: Optional[str] = Field(None, description="Optional note or explanation from the LLM about the classification decision.")
    
    knowledge_base: Optional[Literal["retrieve", "add"]] = Field(None, description="Knowledge base need to create if applicable.")
    research_data: Optional[str] = Field(None, description="Collected research documents text")


class RouteType(str, Enum):
    SOFTWARE = "software"
    INDIAN_SCHOOL = "indian_school"
    JOB_PREPARATION = "job_preparation"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MATH = "math"
    NON_TECH = "non_tech"


class ClassifiQuery(BaseModel):
    route: RouteType = Field(..., description="The category selected for routing the user's query.")
    note: Optional[str] = Field(None, description="Optional note or explanation from the LLM about the routing decision.")


class ResearchAgentState(BaseModel):
    session_id: str = Field(..., description="Unique session ID for this research run")
    query: str = Field(..., description="The research query")
    domain: Optional[str] = Field(None, description="Research domain")
    documents: List[Document] = Field(default_factory=list, description="Collected research documents")
    messages: Annotated[List[AnyMessage], add] = Field(default_factory=list, description="Agent messages")
    tool_rounds: int = Field(0, description="Number of tool execution rounds")

class ResearchToolState(AgentState):
    research_docs: Annotated[list[str], add]
    research_word_count: Annotated[int, add]
import os
from langchain.chat_models import BaseChatModel

from src.constants import GraphState
from src.utils import get_logger

logger = get_logger(name="KnowledgeAgent")

class KnowledgeNode:
    def __init__(self,llm: BaseChatModel) -> None:
        self.llm = llm

    def invoke(self,state:GraphState) ->GraphState:
        try:
            # Keep the node harmless until a real knowledge-base action is added.
            return state
        except Exception as e:
            logger.error("Error in knowledge base node: %s", str(e))
            return state

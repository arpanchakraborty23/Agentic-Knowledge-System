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
            pass
        except Exception as e:
            
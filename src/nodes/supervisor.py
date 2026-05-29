from langchain.chat_models import BaseChatModel

from src.constants import GraphState
from src.utils import llm_chain


class SupervisorNode:
    def __init__(self, model: BaseChatModel):
        self.model = model

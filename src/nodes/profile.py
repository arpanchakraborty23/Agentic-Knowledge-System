from langchain.chat_models import BaseChatModel
from src.constants import GraphState, UserProfile
from src.prompts import PROFILE_PROMPT
from src.utils import llm_chain


class ProfileAgent:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.chain = llm_chain(PROFILE_PROMPT, self.model, UserProfile)

    def run(self, state: GraphState) -> GraphState:
        query = state.messages[-1].content if state.messages else ""
        response = self.chain.invoke({"query": query})

        profile_data = response.model_dump() if hasattr(response, "model_dump") else response
        memory = dict(state.memory or {})
        memory["profile"] = profile_data
        if response.domain:
            memory["domain"] = response.domain

        return state.model_copy(
            update={
                "board": response.board,
                "grade": response.grade,
                "subject": response.subject,
                "topic": response.topic,
                "memory": memory,
            }
        )

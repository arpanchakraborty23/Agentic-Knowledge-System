from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.constants import GraphState, ResearchOutput
from src.prompts import DOC_SUMMARY_PROMPT, SUPERVISOR_PROMPT
from src.utils import llm_chain


class SupervisorNode:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.doc_chain = llm_chain(DOC_SUMMARY_PROMPT, self.model)
        self.final_chain = llm_chain(SUPERVISOR_PROMPT, self.model)

    def run(self, state: GraphState) -> GraphState:
        user_query = state.messages[-1].content if state.messages else ""

        research_output: ResearchOutput = state.tool_results.get("research_output")
        if not research_output or not research_output.retrieved_docs:
            return state.model_copy(
                update={
                    "next_action": "final_answer",
                    "messages": state.messages + [AIMessage(content="I could not find enough information to answer that yet.")],
                }
            )

        doc_summaries = []
        for index, doc_text in enumerate(research_output.retrieved_docs, start=1):
            summary = self.doc_chain.invoke({
                "query": user_query,
                "doc_text": doc_text,
            })
            doc_summaries.append(f"Document {index} summary:\n{summary}")

        final_response = self.final_chain.invoke({
            "query": user_query,
            "doc_summaries": "\n\n".join(doc_summaries),
        })

        final_message = AIMessage(content=final_response)
        memory = dict(state.memory or {})
        memory["supervisor_doc_summaries"] = doc_summaries

        return state.model_copy(
            update={
                "messages": state.messages + [final_message],
                "memory": memory,
                "next_action": "final_answer",
            }
        )
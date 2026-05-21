from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.constants import GraphState, ResearchOutput
from src.utils import llm_chain


SUPERVISOR_PROMPT = """You are a supervisor that synthesizes research results into a clear, helpful answer for the user.

# Context
The research agent has gathered information from multiple sources (web search, Wikipedia, and academic papers).

# Task
Based on the user's original query and the retrieved research data, provide a comprehensive, well-structured answer.

# Guidelines
- Summarize the key findings from the research
- Combine information from different sources when relevant
- Present the information in a clear, organized manner
- If there are code examples or references, highlight them appropriately
- Keep the answer focused on what the user asked
"""


class SupervisorNode:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.chain = llm_chain(SUPERVISOR_PROMPT, self.model)

    def run(self, state: GraphState) -> GraphState:
        user_query = (
            state.messages[-1].content
            if state.messages
            else ""
        )

        research_output: ResearchOutput = state.tool_results.get("research_output")
        
        if not research_output:
            return state.model_copy(
                update={
                    "next_action": "final_answer",
                    "messages": state.messages + []
                }
            )

        docs_text = "\n\n".join(research_output.retrieved_docs) if research_output.retrieved_docs else "No data available"
        
        response = self.chain.invoke({
            "query": user_query,
            "research_data": docs_text,
            "concepts": ", ".join(research_output.concepts) if research_output.concepts else "None",
            "code_examples": "\n".join(research_output.code_examples) if research_output.code_examples else "None",
            "references": ", ".join(research_output.references) if research_output.references else "None"
        })

    
        final_message = AIMessage(content=response)

        return state.model_copy(
            update={
                "messages": state.messages + [final_message],
                "next_action": "final_answer"
            }
        )
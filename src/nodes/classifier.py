from langchain.chat_models import BaseChatModel

from src.constants import GraphState, ClassifiQuery
from src.prompts import CLASSIFER_PROMPT
from src.utils import llm_chain, logger



# RouterAgent is responsible for classifying incoming user queries
# and determining which specialized agent should handle them.
# It uses an LLM with structured output to route queries to categories
# like software, indian_school, job_preparation, general_knowledge, or math.
class ClassifierAgent:
    def __init__(self, model: BaseChatModel):
        self.model = model

        # Initialize the routing chain with a prompt template,
        # the LLM model, and a Pydantic schema (ClassifiQuery) for
        # structured output parsing.
        self.chain = llm_chain(
            CLASSIFER_PROMPT,
            self.model,
            ClassifiQuery
        )
        logger.info("ClassifierAgent initialized with model: %s", self.model)

    def classify(self, state: GraphState) -> GraphState:
        try:
            # Extract the most recent user message from the conversation history.
            # If no messages exist, use an empty string as fallback.
            query = state.query or ""


            response = self.chain.invoke({
                "query": query
            })

            logger.info("ClassifierAgent classified query to domain: %s", response.route)


            # Update the GraphState with the determined route and return it.
            state.classified_domain = response.route
            state.classifier_note = response.note
            
            return state.model_copy(update={"classified_domain": response.route, "classifier_note": response.note})
        
        except Exception as e:
            logger.error("Error in ClassifierAgent classification: %s", str(e))
            return state
        
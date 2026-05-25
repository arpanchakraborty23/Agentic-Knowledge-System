import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import START,StateGraph
from langchain.chat_models import init_chat_model

from src.utils import logger
from src.constants import GraphState
from src.nodes import ClassifierAgent

class GraphBuilder:
    def __init__(self):
        self.llm = init_chat_model(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Nodes
        self.clssifier_agent = ClassifierAgent(self.llm)


    def _build_graph(self):
        try:
            # Builder 
            builder = StateGraph(GraphState)

            # Add nodes to the graph
            builder.add_node("classifier", self.clssifier_agent.classify)

            # Define edges between nodes (for now, it's just a single node, so no edges needed)
            builder.add_edge(START, "classifier")


            return builder.compile()
        
        
        except Exception as e:
            logger.error("Error building graph: %s", str(e))

       

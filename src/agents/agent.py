import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

from src.utils import setup_logger
from src.constants import GraphState
from src.nodes import ClassifierNode, ResearchNode

logger = setup_logger(name="KnowledgeAgent",log_file="agent.log")


class GraphBuilder:
    def __init__(self):
        self.llm = init_chat_model(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Nodes
        self.clssifier_agent = ClassifierNode(self.llm)
        self.research_agent = ResearchNode(self.llm)


    def _build_graph(self):
        try:
            # Builder 
            builder = StateGraph(GraphState)

            # Add nodes to the graph
            builder.add_node("classifier", self.clssifier_agent.classify)
            builder.add_node("research_agent",self.research_agent.search)

            # Define edges between nodes (for now, it's just a single node, so no edges needed)
            builder.add_edge(START, "classifier")
            builder.add_edge("classifier","research_agent")


            return builder.compile()
        
        
        except Exception as e:
            logger.error("Error building graph: %s", str(e))

class GraphAgent:
    """"Main Agent Entrypoint"""
    def __init__(self):
        self._graph = GraphBuilder()._build_graph()
        self._graph.get_graph().draw_mermaid_png(output_file_path="data/graph.png")

    def invoke(self,user_id,query):
        try:
            response = self._graph.invoke(
                input=GraphState(
                    user_id=user_id,
                    query=query
                )
            )

            return response
        except Exception as e:
            raise e
       

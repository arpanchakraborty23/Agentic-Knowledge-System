from src.utils import llm_chain
from src.prompts import ROUTE_PROMPT
from src.constants import RouteQuery
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

chain = llm_chain(ROUTE_PROMPT, llm, RouteQuery)

res = chain.invoke({"query": "What is the best software for data analysis?"})

pprint(res)
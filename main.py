# from src.utils import llm_chain
# from src.prompts import ROUTE_PROMPT
# from src.constants import RouteQuery
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from pprint import pprint

# load_dotenv()

# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# chain = llm_chain(ROUTE_PROMPT, llm, RouteQuery)

# res = chain.invoke({"query": "What is the best software for data analysis?"})

# pprint(res)
# """Basic connection example.
# """

# import redis

# r = redis.Redis(
#     host='redis-12980.crce281.ap-south-1-3.ec2.cloud.redislabs.com',
#     port=12980,
#     decode_responses=True,
#     username="default",
#     password="zo9Qk9rTjWYv19ARvEXTtPcDXmf0ARkN",
# )

# success = r.set('foo', 'bar')
# # True

# result = r.get('foo')
# print(result)
# # >>> bar

from langchain_community.retrievers import WikipediaRetriever

wiki = WikipediaRetriever(verbose=True)

data = wiki.invoke("What is the capital of France?")

r = [item.page_content for item in data]
print(r[:2000])


from src.utils import llm_chain
from src.prompts import CLASSIFER_PROMPT
from src.constants import ClassifiQuery
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rich import print

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

chain = llm_chain(CLASSIFER_PROMPT, llm, ClassifiQuery)

questions = [
    "What is the best software for data analysis?",
    "What is the capital of France?",
    "How do I prepare for a software engineering interview?",
    "What are the key features of the Indian education system?",
    "Can you explain the Pythagorean theorem?"
    "What are the latest advancements in AI research?"
]

for question in questions:
    print(f"\n[bold green]Question:[/bold green] {question}")
    res = chain.invoke({"query": question})
    
    print(res)


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



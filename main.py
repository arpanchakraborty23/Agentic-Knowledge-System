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

from langchain.chat_models import ChatOpenAI
from src.services import DeepTutorOrchestrator


def main() -> None:
    model = ChatOpenAI(temperature=0.2)
    orchestrator = DeepTutorOrchestrator(model)

    user_query = "Explain the concept of derivatives in calculus and suggest study resources."
    state = orchestrator.run(user_query, user_id="student-123")

    for message in state.messages:
        print("--- BOT RESPONSE ---")
        print(message.content)


if __name__ == "__main__":
    main()


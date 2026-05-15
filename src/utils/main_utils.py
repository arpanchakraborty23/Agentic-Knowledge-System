from typing import Optional, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

def llm_chain(template: str, llm: BaseChatModel, output_parser_class: Optional[type] = None) -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages([("system", template)])

    if output_parser_class:
        llm_with_parser = llm.with_structured_output(output_parser_class)
        chain = prompt | llm_with_parser
    else:
        chain = prompt | llm
    return chain
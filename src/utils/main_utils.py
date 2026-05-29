import re
import uuid
import unicodedata
from typing import Optional, List, Type
from pathlib import Path
from playwright.async_api import async_playwright
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool

from src.constants import StorageConfig
from src.utils.logger import get_logger

logger = get_logger(name="KnowledgeAgent")


def llm_chain(template: str, llm: BaseChatModel, output_parser_class: Optional[Type[BaseTool]] = None, tools: Optional[List[BaseTool]] = None):
    prompt = ChatPromptTemplate.from_messages([("system", template)])
    model = llm

    if tools:
        model = model.bind_tools(tools)

    if output_parser_class:
        model = model.with_structured_output(output_parser_class)

    return prompt | model


async def read_url(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent="Mozilla/5.0")

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        await page.evaluate("""
        () => {
            document.querySelectorAll("script,style,nav,footer,header,aside").forEach(el => el.remove());
        }
        """)

        text = await page.locator("body").inner_text()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r'\n[a-zA-Z0-9]\n', '\n', text)
        text = text.strip()

        await browser.close()
        return text


async def fetch_site(client, source, url):
    try:
        response = await client.get(url)
        if response.status_code == 200:
            return {
                "source": source,
                "url": url,
                "content": response.text[:5000]
            }
    except Exception:
        pass
    return None

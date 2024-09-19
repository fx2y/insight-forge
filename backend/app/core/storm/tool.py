from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool


@tool
async def search_engine(query: str):
    """Search engine to the internet."""
    results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
    return [{"content": r["body"], "url": r["href"]} for r in results]

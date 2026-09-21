from tools.web_search_tools import (
    search_pubmed,
    search_arxiv,
    research_tavily,
)

from cores.state import SubqueryState
from cores.error_codes import ToolError, StateErrorCode, StateError
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any
from llm import llm
from console import console

from rich.table import Table
import re
import httpx
from bs4 import BeautifulSoup
from readability import Document
import asyncio


def dedupe_hits(hits: List[Dict[str, Any]], limit=25) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for h in hits:
        key = re.sub(r"#.*$", "", (h.get("url") or "").strip())
        if key and key not in seen:
            seen.add(key)
            deduped.append(h)

        if len(deduped) >= limit:
            break

    return deduped


async def fetch_page(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """
    return {
        "url": "",
        "title": "",
        "text": ""
    }
    """
    try:
        r = await client.get(url, follow_redirects=True)
        html = r.text
        doc = Document(html)
        title = doc.short_title()
        cleaned = doc.summary(html_partial=True)
        soup = BeautifulSoup(cleaned, "html.parser")
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

        return {
            "url": str(r.url),
            "title": title,
            "text": text[:120000],
        }

    except Exception as e:
        return {
            "url": url,
            "title": "",
            "text": f"Error fetching page: {e}",
        }


async def node_websearch(state: SubqueryState) -> SubqueryState:
    console.rule("[bold cyan]Web Search")
    tool_names = [tool for tool in state["tools"]]
    if not tool_names:
        return {}
    tool_map = {
        "search_pubmed": search_pubmed,
        "search_arxiv": search_arxiv,
        "research_tavily": research_tavily,
    }
    hits = []
    errors = []
    query = state["subquery"]
    with console.status(f"Searching the web for {query}..."):
        for tool_name in tool_names:
            tool = tool_map[tool_name]
            result = await tool.ainvoke(query)
            if isinstance(result, ToolError):
                if tool_name == "search_pubmed":
                    error_code = StateErrorCode.PUBMED_ERROR
                elif tool_name == "search_arxiv":
                    error_code = StateErrorCode.ARXIV_ERROR
                elif tool_name == "research_tavily":
                    error_code = StateErrorCode.TAVILY_ERROR
                errors.append(
                    StateError(
                        code=error_code,
                        toolcode=result.code,
                        message=str(result.message),
                        node="document_qa",
                        tool=result.tool,
                        retryable=result.retryable,
                    )
                )
            else:
                hits.extend(result)

    hits = dedupe_hits(hits, limit=25)

    tbl = Table(title=f"Top {len(hits)} results (deduped)", show_header=True)
    tbl.add_column("#", width=3)
    tbl.add_column("Title", overflow="fold")
    tbl.add_column("URL", overflow="fold")
    for i, h in enumerate(hits, 1):
        tbl.add_row(str(i), h.get("title") or "", h.get("url") or "")

    console.print(tbl)

    console.rule("[bold cyan]Fetch")

    urls = [h["url"] for h in hits if h.get("url")][:12]

    async with httpx.AsyncClient(
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.36"
            )
        },
    ) as client:

        with console.status(f"Fetching {len(urls)} pages..."):
            pages = await asyncio.gather(*(fetch_page(client, url) for url in urls))

    for p in pages:
        label = p.get("title") or p.get("url")
        if p.get("text", "").startswith("Error"):
            console.print(f"[red]✗[/red] {label}")
        else:
            console.print(f"[green]✓[/green] {label}")

    console.print(f"Fetched {len(pages)} pages")

    return {
        "results": [
            {
                "tool": "websearch",
                "result": {
                    "query": query,
                    "hits": hits,
                    "pages": pages,
                },
            }
        ],
        "errors": errors,
    }

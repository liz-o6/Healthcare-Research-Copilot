from tools.web_search_tools import (
    search_pubmed,
    search_arxiv,
    research_tavily,
    dedupe_hits,
    fetch_page,
)

from cores.state import State, SubqueryState
from cores.error_codes import ToolError, StateErrorCode, StateError
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any
from llm import llm
from console import console
import httpx
from rich.table import Table
import asyncio


async def node_websearch(state: SubqueryState):
    max_retries = 3
    console.rule("[bold cyan]Web Search")

    tool_map = {
        "search_pubmed": search_pubmed,
        "search_arxiv": search_arxiv,
        "research_tavily": research_tavily,
    }

    error_code_map = {
        "search_pubmed": StateErrorCode.PUBMED_ERROR,
        "search_arxiv": StateErrorCode.ARXIV_ERROR,
        "research_tavily": StateErrorCode.TAVILY_ERROR,
    }

    hits = []
    errors = []
    query = state["subquery"]

    # Tools assigned to this subquery
    tool_names = [tool for tool in state["tools"]]

    if not tool_names:
        return {}

    # 1. Search. Each tool retries independently up to 3 times
    with console.status(f"Searching the web for {query}..."):

        for tool_name in tool_names:

            tool = tool_map.get(tool_name)

            if tool is None:
                continue

            last_error = None

            for attempt in range(max_retries):

                result = await tool.ainvoke(query)

                if not isinstance(result, ToolError):
                    hits.extend(result)
                    last_error = None
                    break

                last_error = result

                console.print(
                    f"[yellow]{tool_name} failed "
                    f"(attempt {attempt + 1}/3): "
                    f"{result.message}[/yellow]"
                )

            # All 3 attempts failed
            if last_error is not None:
                errors.append(
                    StateError(
                        code=error_code_map[tool_name],
                        toolcode=last_error.code,
                        message=str(last_error.message),
                        node="web_search",
                        tool=last_error.tool,
                        subquery=query,
                        retryable=last_error.retryable,
                    )
                )

    # 2. Dedupe
    hits = dedupe_hits(hits, limit=25)

    tbl = Table(
        title=f"Top {len(hits)} results (deduped)",
        show_header=True,
    )
    tbl.add_column("#", width=3)
    tbl.add_column("Title", overflow="fold")
    tbl.add_column("URL", overflow="fold")

    for i, h in enumerate(hits, 1):
        tbl.add_row(
            str(i),
            h.get("title") or "",
            h.get("url") or "",
        )

    console.print(tbl)

    # 3. Fetch
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

    # 4. Return
    return {
        "results": [
            {
                "tool": "web_search",
                "result": {
                    "query": query,
                    "hits": hits,
                    "pages": pages,
                },
            }
        ],
        "errors": errors,
    }

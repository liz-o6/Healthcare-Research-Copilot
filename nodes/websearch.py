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


async def node_websearch(state: State | SubqueryState):
    console.rule("[bold cyan]Web Search")

    tool_map = {
        "search_pubmed": search_pubmed,
        "search_arxiv": search_arxiv,
        "research_tavily": research_tavily,
    }

    hits = []
    errors = []

    query = state["subquery"]

    # Determine which tools to run
    retry_targets = state.get("retry_targets", [])

    web_retry_targets = [
        target for target in retry_targets if target["node"] == "websearch"
    ]

    if web_retry_targets:
        # Retry: only run failed web tools
        tool_queries = [
            (target["tool"], target["subquery"]) for target in web_retry_targets
        ]

        console.rule("[bold yellow]Retry Web Search")

    else:
        # First run: run all tools assigned to this subquery
        tool_names = [tool for tool in state["tools"]]

        if not tool_names:
            return {}

        tool_queries = [(tool_name, query) for tool_name in tool_names]

        console.rule("[bold cyan]Web Search")

    # Search
    with console.status(f"Searching the web for {query}..."):
        for tool_name, tool_query in tool_queries:

            tool = tool_map.get(tool_name)

            if tool is None:
                continue

            if web_retry_targets:
                console.print(
                    f"[yellow]↻ Retrying {tool_name}[/yellow]: " f"{tool_query}"
                )

            result = await tool.ainvoke(tool_query)

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
                        node="websearch",
                        tool=result.tool,
                        subquery=tool_query,
                        retryable=result.retryable,
                    )
                )

            else:
                hits.extend(result)

    # Dedupe
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

    # Fetch
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

    # Return
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

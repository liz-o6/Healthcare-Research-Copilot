from cores.state import State
from typing import List, Dict
from pydantic import BaseModel
from llm import llm
from console import console
from rich.panel import Panel
import os, asyncio, sys, time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from nodes.clarification import node_clarify
from nodes.router import node_router
from nodes.planner import node_plan
from nodes.fan_out import route_subqueries
from nodes.local_db import node_localdb
from nodes.websearch import node_websearch
from nodes.document_qa import node_documentqa
from nodes.summary import node_summary
from nodes.aggregate import node_aggregate, route_after_aggregate
from pathlib import Path
import shutil

load_dotenv()


def build_clarification_graph():
    builder = StateGraph(State)
    builder.add_node("clarification", node_clarify)
    builder.add_edge(START, "clarification")
    builder.add_edge("clarification", END)
    clarification_graph = builder.compile()
    return clarification_graph


def build_research_graph():
    builder = StateGraph(State)
    builder.add_node("planner", node_plan)
    builder.add_node("router", node_router)

    builder.add_node("local_db", node_localdb)
    builder.add_node("web_search", node_websearch)
    builder.add_node("document_qa", node_documentqa)

    builder.add_node("aggregate", node_aggregate)
    builder.add_node("summary", node_summary)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "router")

    builder.add_conditional_edges(
        "router",
        route_subqueries,
        {
            "local_db": "local_db",
            "web_search": "web_search",
            "document_qa": "document_qa",
        },
    )

    builder.add_edge("local_db", "aggregate")
    builder.add_edge("web_search", "aggregate")
    builder.add_edge("document_qa", "aggregate")

    builder.add_conditional_edges(
        "aggregate",
        route_after_aggregate,
        {
            "document_qa": "document_qa",
            "local_knowledge": "local_db",
            "web_search": "web_search",
            "summary": "summary",
        },
    )

    builder.add_edge("summary", END)

    research_graph = builder.compile()
    return research_graph


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error: OPENAI_API_KEY is not set. Please set it in your environment variables.[/red]"
        )
        sys.exit(1)
    if not os.getenv("TAVILY_API_KEY"):
        console.print(
            "[red]Error: TAVILY_API_KEY is not set. Please set it in your environment variables.[/red]"
        )
        sys.exit(1)

    console.print(Panel("Deep Research (sreaming) - LangGraph", style="cyan"))
    user_prompt = console.input(
        "[bold cyan]Welcome to the Research Assistant![/bold cyan]\n"
        "I can search the web, answer questions from your documents, and retrieve information from your local knowledge base.\n"
        "To use local document retrieval, please place your files (.txt, .pdf, .docx) in the [bold]documents[/bold] folder before starting.\n\n"
        "[bold]What would you like to research today?[/bold] "
    )

    while not user_prompt:
        user_prompt = console.input(
            "[bold cyan]Welcome to the Research Assistant![/bold cyan]\n"
            "I can search the web, answer questions from your documents, and retrieve information from your local knowledge base.\n"
            "To use local document retrieval, please place your files (.txt, .pdf, .docx) in the [bold]documents[/bold] folder before starting.\n\n"
            "[bold]What would you like to research today?[/bold] "
        )

    DOCUMENT_DIR = Path("documents")
    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
    document_files = [
        f
        for f in DOCUMENT_DIR.glob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if document_files:
        console.print(
            f"[green]✓[/green] Found {len(document_files)} document(s) in 'documents'."
        )
    else:
        console.print(
            "[yellow]![/yellow] No supported documents found in 'documents'. "
            "Local Document QA will be unavailable."
        )

    clarification_graph = build_clarification_graph()
    research_graph = build_research_graph()

    # debug
    # from IPython.display import Image, display

    # png_bytes = research_graph.get_graph().draw_mermaid_png()

    # with open("research_graph.png", "wb") as f:
    #     f.write(png_bytes)

    # print("Graph saved to research_graph.png")

    state: State = await clarification_graph.ainvoke(
        {
            "user_prompt": user_prompt,
            "clarifications": {},
            "needs_clarification": False,
            "subquestions": [],
            "retrieval_plan": [],
            "web_search_plan": {},
            "hits": [],
            "pages": [],
            "documents": [],
            "report_markdown": "",
            "document_uploaded": False,
            "local_documents": [],
            "document_queries": [],
            "local_db_queries": [],
            "local_db_documents": [],
        }
    )

    if state["needs_clarification"]:
        console.print("[yellow]Please answer a few questions: [/yellow]")
        for q in state.get("subquestions", []):
            ans = console.input(f"· {q}")
            state["clarifications"][q] = ans
        state["needs_clarification"] = False

    state = await research_graph.ainvoke(state)

    # clear local db

    if os.path.exists("temp_db"):
        shutil.rmtree("temp_db")


if __name__ == "__main__":
    asyncio.run(main())

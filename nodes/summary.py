import os, re, asyncio, sys, time
from typing import TypedDict, List, Dict, Any

# this is for styling the console output
from console import console
from rich.markdown import Markdown

# core packages required by the agent
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# search agent for LLM
from tavily import TavilyClient
import httpx
from bs4 import BeautifulSoup
from readability import Document

from state import State
from llm import llm


async def node_summary(state: State) -> State:
    sources = []

    console.rule("[bold cyan]Summarize results")

    if state["pages"]:
        for i, p in enumerate(state["pages"], start=1):
            title = p.get("title") or p.get("url")
            sources.append(f"[{i}] {title} - {p.get('url')}")
    sources_text = "\n".join(sources)

    corpus = []
    if state["pages"]:
        for i, p in enumerate(state["pages"], start=1):
            txt = (p.get("text") or "")[:6000]
            corpus.append(
                f"### Source {i}\nURL: {p.get('url')}\nTitle: {p.get('title')}\nTexe:{txt}\n"
            )
    corpus_text = "\n".join(corpus)

    docs = []
    if state["documents"]:
        for i, doc in enumerate(state["documents"], start=1):
            docs.append(f"### Document {i}\n{doc}\n")
    documents_text = "\n".join(docs)

    local_docs = []
    if state["local_db_documents"]:
        for i, doc in enumerate(state["local_db_documents"], start=1):
            local_docs.append(f"### Document {i}\n{doc}\n")
    local_documents_text = "\n".join(local_docs)

    prompt = f"""Write a concise, well-structured research report in Markdown format：
    \"\"\"{state['user_prompt']}\"\"\".

    Use sections:
    - Executive Summary (5-8 bullet points)
    - Key Findings
    - Confilcting View / Risks
    - Data & Numbers
    - Open Questions
    - Next Actions
    - Sources

    Rules:
    - Cite like [1], [2] inlike after claims you derive from sources.
    - Synthesize; do not just copy text.
    - Prefer recent and authoritative sources.
    - If sources conflict, say no.
    - Use information from both the retrieved web sources and the retrieved documents when available.
    - Clearly distinguish between information supported by web sources, local documents, and uploaded documents when necessary.
    - If the uploaded documents, local documents, and web sources disagree, explicitly describe the disagreement instead of choosing one without explanation.

    Web Sources List:
    {sources_text}

    Uploaded Documents:
    {documents_text}

    Web Search Corpus:
    {corpus_text}

    Uploaded Documents:
    {documents_text}

    Local Documents:
    {local_documents_text}
    """
    # STREAM THE FINAL REPORT
    report_chunks: List[str] = []

    async for chunk in llm.astream(prompt):
        piece = chunk.content or ""
        report_chunks.append(piece)

    report = "".join(report_chunks)
    console.print(Markdown(report))
    console.rule("[bold green]Report complete")

    fname = "research_report.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(report)

    console.print(f"Report saved to {fname}")

    return state

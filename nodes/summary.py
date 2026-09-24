import json
from typing import Any, Dict, List

from console import console
from rich.markdown import Markdown

from cores.schemas import Source, SummaryOutput
from cores.state import State
from llm import llm


def build_source_registry(results: List[Dict[str, Any]]) -> List[Source]:
    sources: List[Source] = []
    seen = set()

    def add_source(title: str, content: str, url: str | None = None):
        content = content.strip()
        if not content:
            return

        key = url or content
        if key in seen:
            return

        seen.add(key)
        sources.append(
            Source(
                source_id=f"S{len(sources) + 1}",
                title=title or "Untitled source",
                content=content[:12000],
                url=url,
            )
        )

    for tool_result in results:
        tool_name = tool_result.get("tool")
        payload = tool_result.get("result")

        if not isinstance(payload, dict):
            continue

        if tool_name == "web_search":
            hits = payload.get("hits") or []
            pages = payload.get("pages") or []

            for index, hit in enumerate(hits[:12]):
                if not isinstance(hit, dict):
                    continue

                page = pages[index] if index < len(pages) else {}
                page = page if isinstance(page, dict) else {}
                page_text = page.get("text") or ""

                if page_text.startswith("Error fetching page"):
                    page_text = ""

                add_source(
                    title=page.get("title") or hit.get("title") or "Web source",
                    content=page_text or hit.get("content") or "",
                    url=page.get("url") or hit.get("url"),
                )

        elif tool_name in {"document_qa", "local_knowledge"}:
            documents = payload.get("documents") or []
            source_label = (
                "Uploaded document" if tool_name == "document_qa" else "Local knowledge"
            )

            for index, document in enumerate(documents):
                if isinstance(document, dict):
                    content = document.get("content") or ""
                    metadata = document.get("metadata") or {}
                    title = metadata.get("source") or f"{source_label} {index + 1}"
                else:
                    content = str(document)
                    title = f"{source_label} {index + 1}"

                add_source(title=title, content=content)

    return sources


async def node_summary(state: State) -> State:
    console.rule("[bold cyan]Summarize results")
    results = state.get("results") or []
    source_registry = build_source_registry(results)
    sources_json = json.dumps(
        [source.model_dump() for source in source_registry],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""You are writing the final report for a healthcare research assistant.

    Research question:
    \"\"\"{state['user_prompt']}\"\"\"

    Use sections:
    - Executive Summary (5-8 bullet points)
    - Key Findings
    - Conflicting Views / Risks
    - Data & Numbers
    - Open Questions
    - Next Actions
    - Sources

    Rules:
    - Base the report ONLY on the sources provided below.
    - Cite factual claims inline using the exact source IDs, such as [S1] or [S1][S2].
    - Never invent a source ID or cite a source that does not support the claim.
    - Clearly distinguish web sources, uploaded documents, and local knowledge.
    - Synthesize; do not just copy text.
    - Prefer recent and authoritative sources.
    - If sources conflict, explain the disagreement explicitly.
    - In the Sources section, list only cited source IDs with their titles and URLs when available.
    - If no usable sources are provided, state that there is insufficient evidence and do not invent citations.

    Source registry:
    {sources_json}

    """
    structured_llm = llm.with_structured_output(SummaryOutput)
    summary = await structured_llm.ainvoke(prompt)

    console.print(Markdown(summary.report_markdown))
    console.rule("[bold green]Draft report complete")

    return {
        "source_registry": source_registry,
        "report_markdown": summary.report_markdown,
    }

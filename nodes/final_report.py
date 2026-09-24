import json

from console import console
from cores.schemas import CitationCheck, FinalReportOutput
from cores.state import State
from llm import llm
from rich.markdown import Markdown


REPORT_PATH = "research_report.md"


def save_report(report_markdown: str, path: str = REPORT_PATH) -> None:
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write(report_markdown)


def build_verification_warning(
    report_markdown: str,
    failed_checks: list[CitationCheck],
) -> str:
    lines = [
        report_markdown.rstrip(),
        "",
        "## Citation Verification Warning",
        "The following citations could not be verified:",
    ]

    for check in failed_checks:
        lines.append(f"- [{check.source_id}] {check.claim}: {check.reason}")

    if not failed_checks:
        lines.append("- No verifiable inline citations were found in the report.")

    return "\n".join(lines)


async def node_final_report(state: State) -> State:
    console.rule("[bold cyan]Finalize report")

    draft = state.get("report_markdown") or ""
    checks = state.get("citation_checks") or []
    sources = state.get("source_registry") or []
    failed_checks = [check for check in checks if check.status != "supported"]

    needs_revision = bool(failed_checks) or (bool(sources) and not checks)
    final_report = draft

    if needs_revision:
        relevant_ids = {check.source_id for check in failed_checks}
        relevant_sources = [
            source.model_dump()
            for source in sources
            if not relevant_ids or source.source_id in relevant_ids
        ]

        issues = [check.model_dump() for check in failed_checks]
        if not checks:
            issues.append(
                {
                    "claim": "The report contains no verifiable inline citations.",
                    "source_id": "",
                    "status": "invalid",
                    "reason": "Add citations using the available source registry.",
                }
            )

        prompt = f"""Revise the draft research report using the citation issues below.

        Rules:
        - Return the complete report in Markdown.
        - Preserve claims and citations that are already valid.
        - Correct, qualify, or remove unsupported claims.
        - Remove invalid source IDs.
        - Use only source IDs and content provided below.
        - Never invent evidence or source IDs.
        - Preserve the existing report structure where possible.

        Draft report:
        {draft}

        Citation issues:
        {json.dumps(issues, ensure_ascii=False, indent=2)}

        Relevant sources:
        {json.dumps(relevant_sources, ensure_ascii=False, indent=2)}
        """

        try:
            finalizer = llm.with_structured_output(FinalReportOutput)
            result = await finalizer.ainvoke(prompt)
            final_report = result.final_report_markdown
        except Exception as error:
            console.print(f"[yellow]Could not revise report: {error}[/yellow]")
            final_report = build_verification_warning(draft, failed_checks)

    save_report(final_report)
    console.print(Markdown(final_report))
    console.print(f"Report saved to {REPORT_PATH}")

    return {}

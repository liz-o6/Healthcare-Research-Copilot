import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cores.error_codes import ToolError, ToolErrorCode  # noqa: E402
from cores.schemas import Source  # noqa: E402
from nodes.citation_verification import node_citation_verification  # noqa: E402


DATASET_PATH = Path(__file__).with_name("evaluation_dataset.json")
DEFAULT_RESULTS_PATH = Path(__file__).with_name("evaluation_results.json")
CITATION_PATTERN = re.compile(r"\[S\d+\]")


def load_cases() -> List[Dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)["cases"]


def initial_state(question: str) -> Dict[str, Any]:
    return {
        "user_prompt": question,
        "clarifications": {},
        "needs_clarification": False,
        "subquestions": [],
        "results": [],
        "report_markdown": "",
        "source_registry": [],
        "citation_checks": [],
        "errors": [],
        "retry_counts": {},
        "retry_targets": [],
        "processed_error_count": 0,
    }


def skip_result(case: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "id": case["id"],
        "task_type": case["task_type"],
        "status": "SKIP",
        "reason": reason,
    }


def find_missing_topics(report: str, expected_topics: List[Any]) -> List[List[str]]:
    report_lower = report.lower()
    missing = []

    for topic_group in expected_topics:
        alternatives = (
            topic_group if isinstance(topic_group, list) else [topic_group]
        )
        if not any(topic.lower() in report_lower for topic in alternatives):
            missing.append(alternatives)

    return missing


def has_insufficient_evidence_notice(report: str) -> bool:
    report_lower = report.lower()
    phrases = (
        "insufficient evidence",
        "evidence is insufficient",
        "limited evidence",
        "not enough evidence",
        "no evidence",
        "cannot conclude",
        "does not establish",
        "do not establish",
        "not established",
    )
    return any(phrase in report_lower for phrase in phrases)


class FailingTool:
    def __init__(self, tool_name: str, error_code: ToolErrorCode, message: str):
        self.tool_name = tool_name
        self.error_code = error_code
        self.message = message
        self.call_count = 0

    async def ainvoke(self, query: str) -> ToolError:
        self.call_count += 1
        return ToolError(
            code=self.error_code,
            message=self.message,
            tool=self.tool_name,
            retryable=True,
        )


def injected_error_code(tool_name: str, failure: str) -> ToolErrorCode:
    if tool_name == "search_pubmed":
        if failure == "timeout":
            return ToolErrorCode.PUBMED_TIMEOUT
        return ToolErrorCode.PUBMED_NETWORK_ERROR
    if tool_name == "search_arxiv":
        if failure == "timeout":
            return ToolErrorCode.ARXIV_TIMEOUT_ERROR
        return ToolErrorCode.ARXIV_NETWORK_ERROR
    return ToolErrorCode.TAVILY_API_ERROR


async def run_error_recovery_case(case: Dict[str, Any]) -> Dict[str, Any]:
    import nodes.final_report as final_report
    import nodes.websearch as websearch

    injection = case["fault_injection"]
    tool_names = injection.get("tools") or [injection["tool"]]
    expected_attempts = injection["attempts"]
    failure = injection["failure"]
    fake_tools = {
        tool_name: FailingTool(
            tool_name,
            injected_error_code(tool_name, failure),
            f"Injected {failure}",
        )
        for tool_name in tool_names
    }
    original_tools = {
        tool_name: getattr(websearch, tool_name) for tool_name in tool_names
    }
    saved_reports: List[str] = []
    original_save_report = final_report.save_report

    for tool_name, fake_tool in fake_tools.items():
        setattr(websearch, tool_name, fake_tool)
    final_report.save_report = lambda report, path=None: saved_reports.append(report)

    try:
        retrieval_output = await websearch.node_websearch(
            {
                "subquery": case["input"]["question"],
                "tools": tool_names,
            }
        )
        errors = retrieval_output.get("errors", [])

        await final_report.node_final_report(
            {
                "report_markdown": (
                    "# Research Report\n\n"
                    "Insufficient evidence is available because the selected "
                    "retrieval tools failed."
                ),
                "source_registry": [],
                "citation_checks": [],
            }
        )
    finally:
        for tool_name, original_tool in original_tools.items():
            setattr(websearch, tool_name, original_tool)
        final_report.save_report = original_save_report

    call_counts = {
        tool_name: fake_tool.call_count
        for tool_name, fake_tool in fake_tools.items()
    }
    recorded_tools = {error.tool for error in errors}
    retries_ok = all(count == expected_attempts for count in call_counts.values())
    errors_ok = recorded_tools == set(tool_names) and len(errors) == len(tool_names)
    report_created = bool(saved_reports and saved_reports[0].strip())
    expected = case["expected"]
    insufficient_ok = (
        not expected.get("requires_insufficient_evidence_notice")
        or (
            bool(saved_reports)
            and "insufficient evidence" in saved_reports[0].lower()
        )
    )
    passed = retries_ok and errors_ok and report_created and insufficient_ok

    return {
        "id": case["id"],
        "task_type": case["task_type"],
        "status": "PASS" if passed else "FAIL",
        "workflow_completed": True,
        "call_counts": call_counts,
        "expected_attempts": expected_attempts,
        "recorded_error_tools": sorted(recorded_tools),
        "final_report_created": report_created,
        "insufficient_evidence_notice": insufficient_ok,
        "errors": [error.model_dump(mode="json") for error in errors],
    }


async def run_citation_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_input = case["input"]
    source_registry = [Source(**source) for source in case_input["source_registry"]]

    has_valid_citation = any(
        source.source_id in case_input["report_markdown"] for source in source_registry
    )
    if has_valid_citation and not os.getenv("OPENAI_API_KEY"):
        return skip_result(case, "OPENAI_API_KEY is required for semantic verification")

    output = await node_citation_verification(
        {
            "report_markdown": case_input["report_markdown"],
            "source_registry": source_registry,
        }
    )
    actual_checks = output.get("citation_checks", [])
    actual = {
        (check.source_id, check.status)
        for check in actual_checks
    }
    expected = {
        (check["source_id"], check["status"])
        for check in case["expected"]["citation_checks"]
    }
    passed = actual == expected

    return {
        "id": case["id"],
        "task_type": case["task_type"],
        "status": "PASS" if passed else "FAIL",
        "expected": sorted(expected),
        "actual": sorted(actual),
        "checks": [check.model_dump() for check in actual_checks],
    }


async def run_clarification_case(case: Dict[str, Any]) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return skip_result(case, "OPENAI_API_KEY is required")

    from nodes.clarification import node_clarify

    output = await node_clarify(initial_state(case["input"]["question"]))
    expected = case["expected"]
    questions = output.get("subquestions", [])
    passed = (
        output.get("needs_clarification") == expected["needs_clarification"]
        and len(questions) >= expected["minimum_questions"]
    )
    return {
        "id": case["id"],
        "task_type": case["task_type"],
        "status": "PASS" if passed else "FAIL",
        "needs_clarification": output.get("needs_clarification"),
        "questions": questions,
    }


async def run_end_to_end_case(case: Dict[str, Any]) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return skip_result(case, "OPENAI_API_KEY is required")

    required_tools = set(case["expected"].get("required_tools", []))
    if "research_tavily" in required_tools and not os.getenv("TAVILY_API_KEY"):
        return skip_result(case, "TAVILY_API_KEY is required for this case")

    missing_files = [
        path
        for path in case["input"].get("required_files", [])
        if not (PROJECT_ROOT / path).exists()
    ]
    if missing_files:
        return skip_result(case, f"Missing fixture files: {', '.join(missing_files)}")

    from main import build_research_graph

    graph = build_research_graph()
    state = await graph.ainvoke(initial_state(case["input"]["question"]))
    report_path = PROJECT_ROOT / "research_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    decisions = getattr(state.get("router"), "decisions", [])
    used_tools = {tool for decision in decisions for tool in decision.tools}
    missing_tools = sorted(required_tools - used_tools)

    report_lower = report.lower()
    missing_topics = find_missing_topics(
        report, case["expected"].get("expected_topics", [])
    )
    citations_ok = (
        not case["expected"].get("requires_citations")
        or bool(CITATION_PATTERN.search(report))
    )
    insufficient_ok = (
        not case["expected"].get("requires_insufficient_evidence_notice")
        or has_insufficient_evidence_notice(report)
    )
    passed = bool(report) and not missing_tools and not missing_topics
    passed = passed and citations_ok and insufficient_ok

    return {
        "id": case["id"],
        "task_type": case["task_type"],
        "status": "PASS" if passed else "FAIL",
        "used_tools": sorted(used_tools),
        "missing_tools": missing_tools,
        "missing_topics": missing_topics,
        "citations_present": bool(CITATION_PATTERN.search(report)),
        "errors": [
            error.model_dump() if hasattr(error, "model_dump") else str(error)
            for error in state.get("errors", [])
        ],
    }


async def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if case["task_type"] == "citation_verification":
            return await run_citation_case(case)
        if case["task_type"] == "clarification":
            return await run_clarification_case(case)
        if case["task_type"] == "end_to_end":
            return await run_end_to_end_case(case)
        if case["task_type"] == "error_recovery":
            return await run_error_recovery_case(case)
        return skip_result(case, f"Unknown task type: {case['task_type']}")
    except Exception as error:
        return {
            "id": case["id"],
            "task_type": case["task_type"],
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
        }


def select_cases(
    cases: List[Dict[str, Any]], case_id: str | None, task_type: str
) -> List[Dict[str, Any]]:
    if case_id:
        return [case for case in cases if case["id"] == case_id]
    if task_type == "all":
        return cases
    return [case for case in cases if case["task_type"] == task_type]


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Run the project evaluation dataset")
    parser.add_argument("--case", help="Run one case by id")
    parser.add_argument(
        "--type",
        choices=(
            "citation_verification",
            "clarification",
            "end_to_end",
            "error_recovery",
            "all",
        ),
        default="citation_verification",
        help="Case type to run (default: citation_verification)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="JSON results path",
    )
    args = parser.parse_args()

    cases = select_cases(load_cases(), args.case, args.type)
    if not cases:
        parser.error("No matching evaluation cases")

    results = []
    for case in cases:
        print(f"RUN  {case['id']}")
        result = await run_case(case)
        results.append(result)
        detail = result.get("reason") or result.get("error") or ""
        print(f"{result['status']:<4} {case['id']} {detail}".rstrip())

    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("PASS", "FAIL", "SKIP")
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nPASS {summary['PASS']} | FAIL {summary['FAIL']} | SKIP {summary['SKIP']}"
    )
    print(f"Results: {args.output}")
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))

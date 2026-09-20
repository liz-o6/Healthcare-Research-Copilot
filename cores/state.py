from typing import TypedDict, List, Dict, Any, Annotated, Literal
from .schemas import ResearchPlan, RouterOutput
from .error_codes import ErrorInfo
import operator

ToolName = Literal[
    "local_knowledge",
    "document_qa",
    "websearch",
]


class ToolResult(TypedDict):
    tool: ToolName
    result: object


class State(TypedDict):

    user_prompt: str

    # clarifications
    clarifications: Dict[str, str]
    needs_clarification: bool
    subquestions: List[str]

    # planner
    research_plan: ResearchPlan

    # router
    router: RouterOutput

    results: Annotated[List[ToolResult], operator.add]

    errors: list[ErrorInfo]


class SubqueryState(TypedDict):
    subquery: str
    tools: List[
        Literal[
            "research_tavily",
            "search_pubmed",
            "search_arxiv",
            "document_qa",
            "local_knowledge",
        ]
    ]

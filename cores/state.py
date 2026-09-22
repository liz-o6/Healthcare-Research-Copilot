from typing import TypedDict, List, Dict, Any, Annotated, Literal
from .schemas import ResearchPlan, RouterOutput
from .error_codes import StateError
import operator

ToolNodeName = Literal[
    "local_knowledge",
    "document_qa",
    "websearch",
]

ToolName = Literal[
    "research_tavily",
    "search_pubmed",
    "search_arxiv",
    "document_qa",
    "local_knowledge",
]


class ToolResult(TypedDict):
    tool: ToolNodeName
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

    # errors
    errors: Annotated[List[StateError], operator.add]
    retry_counts: Dict[str, int]
    retry_targets: List[str]
    processed_error_count: int


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

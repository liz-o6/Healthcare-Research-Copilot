from typing import TypedDict, List, Dict, Any, Annotated, Literal
from .schemas import ResearchPlan, RouterOutput, Source, CitationCheck
from .error_codes import StateError
import operator

ToolNodeName = Literal[
    "local_knowledge",
    "document_qa",
    "web_search",
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

    # summary
    report_markdown: str
    source_registry: List[Source]
    citation_checks: List[CitationCheck]

    # errors
    errors: Annotated[List[StateError], operator.add]
    retry_counts: Dict[str, int]
    retry_targets: List[ToolName]
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

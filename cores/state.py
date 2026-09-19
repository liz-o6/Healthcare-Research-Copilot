from typing import TypedDict, List, Dict, Any, Annotated, Literal
from .schemas import ResearchPlan, RouterOutput
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

    # format_result: str
    # # web search
    # hits: List[Dict[str, Any]]  # {url, title, snippet, score}
    # pages: List[Dict[str, Any]]  # {url, title, text}
    # report_markdown: str

    # # document qa
    # document_uploaded: bool = False  # if any documents is uploaded
    # document_queries: List[str]
    # documents: List[str]

    # # local db
    # local_db_queries: List[str]
    # local_db_documents: List[str]


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

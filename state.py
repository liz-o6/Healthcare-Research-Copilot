from typing import TypedDict, List, Dict, Any


class State(TypedDict):

    user_prompt: str

    # clarifications
    clarifications: Dict[str, str]
    needs_clarification: bool
    subquestions: List[str]

    # plan
    retrieval_plan: List[str]  # ["web", "document", "local"]
    web_search_plan: dict[str, list[str]]

    # web search
    hits: List[Dict[str, Any]]  # {url, title, snippet, score}
    pages: List[Dict[str, Any]]  # {url, title, text}
    report_markdown: str

    # document qa
    document_uploaded: bool = False  # if any documents is uploaded
    document_queries: List[str]
    documents: List[str]

    # local db
    local_db_queries: List[str]
    local_db_documents: List[str]

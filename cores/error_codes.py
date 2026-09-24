from enum import Enum
from pydantic import BaseModel
from typing import Optional, Literal

NodeName = Literal[
    "planner",
    "router",
    "local_knowledge",
    "document_qa",
    "web_search",
]

ToolName = Literal[
    "load_retriever",
    "build_vector_db",
    "search_arxiv",
    "search_pubmed",
    "research_tavily",
]


class StateErrorCode(str, Enum):
    # Planner
    PLANNER_LLM_ERROR = "ES1001"
    PLANNER_INVALID_OUTPUT = "ES1002"

    # Router
    ROUTER_LLM_ERROR = "ES2001"
    ROUTER_INVALID_OUTPUT = "ES2002"

    # Search tools
    TAVILY_ERROR = "ES3001"
    PUBMED_ERROR = "ES3002"
    ARXIV_ERROR = "ES3003"

    # Local DB / RAG
    VECTOR_DB_ERROR = "E4001"
    RETRIEVER_ERROR = "ES4001"
    EMBEDDING_ERROR = "ES4002"
    DOCUMENT_LOAD_ERROR = "ES4003"
    LOCAL_DB_LLM_ERROR = "ES4004"

    # Answer / Summary
    DOC_SUMMARY_LLM_ERROR = "ES5001"
    DOC_SUMMARY_INVALID_OUTPUT = "ES5002"


class ToolErrorCode(str, Enum):
    VECTOR_DB_SETUP_ERROR = "ET1001"
    RETRIEVER_LOAD_ERROR = "ET1002"

    # Arxiv
    ARXIV_TIMEOUT_ERROR = "ET2001"
    ARXIV_NETWORK_ERROR = "ET2002"
    ARXIV_PARSE_ERROR = "ET2003"
    ARXIV_RESULT_PARSE_ERROR = "ET2004"

    # PubMed
    PUBMED_INVALID_INPUT = "ET3001"
    PUBMED_TIMEOUT = "ET3002"
    PUBMED_NETWORK_ERROR = "ET3003"
    PUBMED_JSON_PARSE_ERROR = "ET3004"
    PUBMED_XML_PARSE_ERROR = "ET3005"
    PUBMED_RESULT_PARSE_ERROR = "ET3006"

    # Tavily
    TAVILY_INVALID_INPUT = "ET4001"
    TAVILY_API_ERROR = "ET4002"
    TAVILY_INVALID_RESPONSE = "ET4003"
    TAVILY_RESULT_PARSE_ERROR = "ET4004"


class StateError(BaseModel):
    code: StateErrorCode
    message: str
    node: NodeName
    retryable: bool = False
    toolcode: Optional[ToolErrorCode] = None
    tool: Optional[ToolName] = None
    subquery: Optional[str] = None


class ToolError(BaseModel):
    code: ToolErrorCode
    message: str
    tool: Optional[ToolName] = None
    retryable: bool = False

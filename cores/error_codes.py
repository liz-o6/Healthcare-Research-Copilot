from enum import Enum
from pydantic import BaseModel
from typing import Optional


class ErrorCode(str, Enum):
    # Planner
    PLANNER_LLM_ERROR = "E1001"
    PLANNER_INVALID_OUTPUT = "E1002"

    # Router
    ROUTER_LLM_ERROR = "E2001"
    ROUTER_INVALID_OUTPUT = "E2002"

    # Search tools
    TAVILY_ERROR = "E3001"
    PUBMED_ERROR = "E3002"
    ARXIV_ERROR = "E3003"

    # Local DB / RAG
    VECTOR_DB_ERROR = "E4001"
    EMBEDDING_ERROR = "E4002"
    DOCUMENT_LOAD_ERROR = "E4003"

    # Answer / Summary
    SUMMARY_LLM_ERROR = "E5001"
    SUMMARY_INVALID_OUTPUT = "E5002"


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str
    node: str
    tool: Optional[str] = None
    retryable: bool = False

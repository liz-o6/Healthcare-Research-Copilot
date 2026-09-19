from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel
from pydantic.fields import Field


# clarify
class ClarifyResponse(BaseModel):
    needs: bool
    questions: List[str]


class RouteDecision(BaseModel):
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


# plan
class ResearchPlan(BaseModel):
    research_goal: str = Field(
        description="A concise description of the user's overall research goal."
    )

    subqueries: List[str] = Field(
        description="A list of focused and self-contained research subqueries."
    )


# router
class RouterOutput(BaseModel):
    decisions: List[RouteDecision]

from state import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List
from pydantic import BaseModel
from llm import llm
from console import console
from pathlib import Path


class WebSearchPlan(BaseModel):
    search_pubmed: List[str] = []
    search_arxiv: List[str] = []
    research_tavily: List[str] = []


class queryresponse(BaseModel):
    web_search: bool
    document_qa: bool
    local_knowledge: bool
    web_search_plan: WebSearchPlan
    document_queries: List[str]
    local_db_queries: List[str]


async def node_plan(state: State) -> State:
    """
    Plan the retrieval strategy for the current user request.

    This node determines which information sources should be used:
    - Web Search: Retrieve up-to-date information from the Internet
      (e.g., Tavily, PubMed, arXiv).

    - Document QA: Answer questions using documents uploaded during
      the current session. These documents are temporary and only
      available for this conversation.

    - Local Knowledge: Retrieve information from the user's persistent
      local vector database, which has been built in advance and can
      be reused across conversations.

    More than one source may be selected when appropriate. For example,
    the agent may combine Web Search with Local Knowledge to produce
    a more comprehensive answer.

    Returns:
        Updated state containing the selected retrieval strategy.
    """

    # Get clarifications
    clar = (
        "\n".join(f"- {k}:{v}" for k, v in state["clarifications"].items()) or "(none)"
    )

    # Uploaded documents
    documents_dir = Path("documents")
    document_files = [f.name for f in documents_dir.iterdir() if f.is_file()]

    # Local vector database source documents
    local_db_dir = Path("local_documents")
    local_db_files = [f.name for f in local_db_dir.iterdir() if f.is_file()]

    Messages = [
        SystemMessage(
            content=f"""You are responsible for planning the retrieval strategy for a research assistant.

            Based on the user's request and the available resources, decide which retrieval sources should be used.

            Available sources:
            - Web Search: Retrieve up-to-date information from the Internet.
            - Document QA: Search documents uploaded during the current session.
            - Local Knowledge: Search the user's persistent local vector database.

            Current uploaded documents:
            {document_files}

            Current local knowledge documents:
            {local_db_files}

            Rules:
            - Select one or more retrieval sources when appropriate.
            - Use Web Search for recent information, external knowledge, or literature search.
            - First choose the most appropriate search tool:
                - search_pubmed: biomedical and medical literature.
                - search_arxiv: AI, computer science, and quantitative research papers.
                - research_tavily: general web search, news, official websites, and non-academic information.
                    - For PubMed and ArXiv queries:
                        - Keep queries broad enough to retrieve relevant results; avoid combining too many concepts with AND.
                        - Use only 2–4 core concepts per query.
                        - Do not require multiple publication types (e.g., "systematic review" AND "meta-analysis") in the same query.
                        - Do not combine multiple years with AND (e.g., "2022 AND 2023 AND 2024").
                        - For recent literature, prefer a date range such as "2022:2024[dp]" instead of separate years.
                        - Generate several simpler queries rather than one overly restrictive query.
            - If multiple search tools are beneficial, select more than one.
            - For each selected search tool, generate 3–8 focused and specific search queries.
            - Use Document QA only if one or more uploaded documents are likely to contain information relevant to the user's request based on their filenames.
            - If Document QA is selected, use the retriever_tool.
                - Generate 3–8 focused and specific retrieval queries for retriever_tool.
                - The retrieval queries should capture different aspects or phrasings of the user's request to maximize document recall.
                - Store these queries as a list under document_queries.
            - Use Local Knowledge only if one or more documents in the persistent local knowledge base are likely to contain relevant information based on their filenames.
            - If Local Knowledge is selected, use the local_retriever_tool.
                - Generate 3–8 focused and specific retrieval queries for local_retriever_tool.
                - The retrieval queries should capture different aspects or phrasings of the user's request to maximize retrieval recall.
                - Store these queries as a list under local_db_queries.
            - Prefer combining multiple sources when it will improve the final answer.
            - Do not select unnecessary sources or tools.
            - If the uploaded documents or local knowledge are clearly relevant, prefer using them instead of relying solely on Web Search.

            Return the result as structured data with:
            - use_web: true/false
            - web_search_plan: a dictionary where each key is a search tool name and value is a list of search queries. The total values for all tools is 3–8 queries.
            - use_document: true/false
            - use_local: true/false
            - document_queries: a list of 3–8 retrieval queries for retriever_tool (empty if use_document is false)

            Do not answer the user's question. Only generate the retrieval plan."""
        ),
        HumanMessage(content=f"""Original question: {state["user_prompt"]}. 
                    Clarification: {clar}"""),
    ]

    structured_llm = llm.with_structured_output(queryresponse)

    console.rule("[bold cyan]Plan")

    with console.status("Drafting sub-queries..."):
        msg = await structured_llm.ainvoke(Messages)

    retrieval_plan = []
    if msg.web_search:
        retrieval_plan.append("web")

        web_search_plan = {}
        remaining = 8

        plan = msg.web_search_plan

        for tool in ["search_pubmed", "search_arxiv", "research_tavily"]:
            if remaining <= 0:
                break

            queries = getattr(plan, tool)
            if not queries:
                continue
            selected = queries[:remaining]
            if selected:
                web_search_plan[tool] = selected
                remaining -= len(selected)

    if msg.document_qa:
        retrieval_plan.append("document")
    if msg.local_knowledge:
        retrieval_plan.append("local")

    return {
        "retrieval_plan": retrieval_plan,
        "document_queries": msg.document_queries,
        "web_search_plan": web_search_plan,
    }


def route_retrieval(state: State) -> List[str] | str:
    routes = []

    if "web" in state["retrieval_plan"]:
        routes.append("web_search")
    if "document" in state["retrieval_plan"]:
        routes.append("document_qa")
    if "local" in state["retrieval_plan"]:
        routes.append("local_db")

    print(f"plannerroutes: {routes}")
    return routes if routes else "END"

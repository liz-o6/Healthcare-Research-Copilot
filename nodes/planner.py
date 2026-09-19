from cores.state import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List
from llm import llm
from console import console
from pathlib import Path
from cores.state import State
from cores.schemas import ResearchPlan


async def node_plan(state: State) -> State:
    """
    Plan the research strategy for the current user request.

    This node is responsible for understanding the user's research goal and decomposing it into clear, focused, and non-overlapping subqueries.
    It should identify the information needed to answer the question comprehensively, while leaving retrieval tool selection to the downstream Router node.

    The available information sources are:

    * **Web Search:** Retrieve up-to-date information from the Internet, including general web content, PubMed, and arXiv.
    * **Document QA:** Retrieve relevant information from documents uploaded during the current session.
        These documents are temporary and available only within this conversation.
    * **Local Knowledge:** Retrieve relevant information from the user's persistent local vector database, which can be reused across conversations.

    The Planner should produce a structured research plan containing the overall research goal and a list of focused subqueries.
    Each subquery should be self-contained, directly relevant to the user's request, and address a specific aspect of the research task.

    Returns:
    Updated state containing the structured research plan, including the research goal and subqueries.

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
            content=f"""You are the Planner of a healthcare research assistant.

            Your task is to analyze the user's request, understand the research goal, and decompose it into clear, focused subqueries. You must NOT decide which retrieval tool to use. Tool selection will be handled by a separate Router node downstream.

            Available resources:
            - document_qa:
            Use this when the user's question requires information from uploaded documents.

            - local_knowledge:
            Use this when the question requires information from the local knowledge base.

            - web/PubMed/arXiv:
            Use these for external information.

            Current uploaded documents:
            {document_files}

            Current local knowledge documents:
            {local_db_files}

            Planning rules:
            1. Understand the user's original question and clarification.
            2. Decompose the research task into clear, focused, and non-overlapping subqueries.
            3. Each subquery should address one specific aspect of the research question.
            4. Generate multiple subqueries when the question requires different perspectives or sources.
            5. Keep each subquery self-contained and sufficiently specific to be independently researched.
            6. Preserve important medical, scientific, technical, and temporal constraints from the user's request.
            7. Do not generate search queries for specific tools.
            8. Do not select Web Search, PubMed, arXiv, Document QA, or Local Knowledge.
            9. Do not answer the user's question.
            10. Do not include tool names, tool-selection decisions, or retrieval instructions in the output.

            Return ONLY structured data matching the following schema:

            {{
                "research_goal": "A concise description of the user's overall research goal.",
                "subqueries": [
                    "A focused and self-contained research subquery.",
                    "Another focused research subquery."
                ]
            }}

            The subqueries list must contain at least one item.
            Each subquery must be directly relevant to the original question.
            Do not include any additional fields or explanatory text."""
        ),
        HumanMessage(content=f"""Original question: {state["user_prompt"]}

            Clarification: {clar}"""),
    ]

    structured_llm = llm.with_structured_output(ResearchPlan)

    console.rule("[bold cyan]Plan")

    with console.status("Drafting sub-queries..."):
        msg = await structured_llm.ainvoke(Messages)

    return {
        "research_plan": msg,
    }


def route_retrieval(state: State) -> List[str] | str:
    routes = []

    if "web" in state["retrieval_plan"]:
        routes.append("web_search")
    if "document" in state["retrieval_plan"]:
        routes.append("document_qa")
    if "local" in state["retrieval_plan"]:
        routes.append("local_db")

    return routes if routes else "END"

from cores.state import SubqueryState
from console import console
from tools.retriever_toos import reteriever_tool
from tools.retriever_toos import build_vector_db
from tools.retriever_toos import load_documents
from tools.retriever_toos import load_retriever
from pathlib import Path


async def node_documentqa(state: SubqueryState) -> SubqueryState:
    if "document_qa" not in state["tools"]:
        return {}
    query = state["subquery"]
    console.rule("[bold cyan]Document QA")
    docs_path = Path("documents")
    db_path = "temp_db"

    docs = load_documents([docs_path])
    build_vector_db(docs, db_path)
    retriever = load_retriever(db_path)

    documents = []
    with console.status("Retrieving documents..."):
        documents = await reteriever_tool(retriever, query)

    return {
        "results": [
            {"tool": "document_qa", "result": {"query": query, "documents": documents}}
        ]
    }

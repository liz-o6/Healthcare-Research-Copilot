from cores.state import SubqueryState
from console import console
from tools.retriever_toos import retriever_tool
from tools.retriever_toos import build_vector_db
from tools.retriever_toos import load_documents
from tools.retriever_toos import load_retriever
from pathlib import Path
from cores.error_codes import StateError, StateErrorCode, ToolError


async def node_documentqa(state: SubqueryState) -> SubqueryState:
    max_retries = 3
    if "document_qa" not in state["tools"]:
        return {}

    query = state["subquery"]
    console.rule("[bold cyan]Document QA")

    docs_path = Path("documents")
    db_path = "temp_db"

    # 1. Load documents
    docs = load_documents([docs_path])

    # 2. Build vector DB, Retry up to 3 times
    vector_db_error = None

    for attempt in range(max_retries):
        error = build_vector_db(docs, db_path)

        if error is None:
            vector_db_error = None
            break

        vector_db_error = error

        console.print(
            f"[yellow]build_vector_db failed "
            f"(attempt {attempt + 1}/3): {error.message}[/yellow]"
        )

    if vector_db_error is not None:
        state_error = StateError(
            code=StateErrorCode.VECTOR_DB_ERROR,
            toolcode=vector_db_error.code,
            message=str(vector_db_error.message),
            node="document_qa",
            tool=vector_db_error.tool,
            subquery=query,
            retryable=vector_db_error.retryable,
        )

        return {
            "errors": [state_error],
        }

    # 3. Load retriever, Retry up to 3 times
    retriever = None
    retriever_error = None

    for attempt in range(max_retries):
        result = load_retriever(db_path)

        if not isinstance(result, ToolError):
            retriever = result
            retriever_error = None
            break

        retriever_error = result

        console.print(
            f"[yellow]load_retriever failed "
            f"(attempt {attempt + 1}/3): {result.message}[/yellow]"
        )

    if retriever_error is not None:
        state_error = StateError(
            code=StateErrorCode.RETRIEVER_ERROR,
            toolcode=retriever_error.code,
            message=str(retriever_error.message),
            node="document_qa",
            tool=retriever_error.tool,
            subquery=query,
            retryable=retriever_error.retryable,
        )

        return {
            "errors": [state_error],
        }

    # 4. Retrieve documents
    documents = []

    with console.status("Retrieving documents..."):
        try:
            documents = await retriever_tool(retriever, [query])

            # 5. Return results
            return {
                "results": [
                    {
                        "tool": "document_qa",
                        "result": {
                            "query": query,
                            "documents": documents,
                        },
                    }
                ]
            }

        except Exception as e:
            error = StateError(
                code=StateErrorCode.DOC_SUMMARY_LLM_ERROR,
                message=str(e),
                node="document_qa",
                retryable=True,
            )

            return {
                "errors": [error],
            }

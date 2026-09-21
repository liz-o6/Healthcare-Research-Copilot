from cores.state import SubqueryState, State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from pydantic import BaseModel
from llm import llm
from console import console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner
from rich.markdown import Markdown
from tools.retriever_toos import build_vector_db
from tools.retriever_toos import reteriever_tool
from tools.retriever_toos import load_retriever
from tools.retriever_toos import load_documents
import json
import hashlib
from pathlib import Path
from cores.error_codes import StateError, StateErrorCode, ToolError


def file_sha256(file_path: Path) -> str:
    """
    Compute the SHA256 hash of a file.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def get_new_or_modified_files(
    docs_dir: str = "local_documents", index_file: str = "local_documents/index.json"
):
    """
    Scan the local_db folder and compare each file's SHA256 with index.json.

    Returns:
        changed_files: List[Path]
            Files that are new or whose contents have changed.

        index: dict
            Updated index dictionary (remember to save it after
            successfully adding these files into the vector database).
    """

    docs_dir = Path(docs_dir)
    index_file = Path(index_file)

    # Create index.json if it does not exist.
    if not index_file.exists():
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index = {}
        with open(index_file, "w") as f:
            json.dump(index, f, indent=4)
    else:
        with open(index_file, "r") as f:
            index = json.load(f)

    changed_files = []

    # Scan every supported document.
    for file in docs_dir.iterdir():

        if file.suffix not in {".pdf", ".txt", ".docx"}:
            continue

        sha = file_sha256(file)

        # New file OR modified file
        if file.name not in index or index[file.name]["sha256"] != sha:
            changed_files.append(file)

            # Update the in-memory index.
            # Save it only AFTER embedding succeeds.
            index[file.name] = {"sha256": sha}

    return changed_files, index


def save_index(index: dict, index_file: str = "local_documents/index.json"):
    """
    Save the updated index after vector_db.add_documents() succeeds.
    """
    with open(index_file, "w") as f:
        json.dump(index, f, indent=4)


async def node_localdb(state: SubqueryState) -> State:
    if "local_knowledge" not in state["tools"]:
        return {}
    query = state["subquery"]
    console.rule("[bold cyan]Local DB")
    docs_path = "local_documents"
    db_path = "local_db"
    changed_files, index = get_new_or_modified_files(docs_path)

    if changed_files:
        save_index(index)
    docs = load_documents(changed_files)
    if docs:
        error = build_vector_db(docs, db_path)
        if error is not None:
            state_error = StateError(
                code=StateErrorCode.VECTOR_DB_ERROR,
                toolcode=error.code,
                message=str(error.message),
                node="local_knowledge",
                tool=error.tool,
                retryable=error.retryable,
            )
            return {
                "errors": [state_error],
            }

    result = load_retriever(db_path)

    if isinstance(result, ToolError):
        state_error = StateError(
            code=StateErrorCode.RETRIEVER_ERROR,
            toolcode=error.code,
            message=str(error.message),
            node="local_knowledge",
            tool=error.tool,
            retryable=error.retryable,
        )
        return {
            "errors": [state_error],
        }

    retriever = result
    documents = []
    with console.status("Retrieving local_db..."):
        documents = await reteriever_tool(retriever, [query])

    return {
        "results": [
            {
                "tool": "local_knowledge",
                "result": {"query": query, "documents": documents},
            }
        ]
    }

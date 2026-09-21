from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_core.tools import tool
from console import console
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from cores.error_codes import ToolError, ToolErrorCode


def load_documents(paths: list[Path]) -> List[Document]:
    docs = []

    for path in paths:
        if not path.exists():
            console.print(f"[yellow]⚠ Directory '{path}' does not exist.[/yellow]")
            continue

        if not path.is_dir():
            console.print(f"[yellow]⚠ '{path}' is not a directory.[/yellow]")
            continue

        if not any(path.iterdir()):
            console.print(f"[yellow]⚠ No documents found in '{path}'.[/yellow]")
            continue

        for doc in path.iterdir():
            if doc.suffix == ".txt":
                docs.extend(TextLoader(doc).load())
            elif doc.suffix == ".pdf":
                docs.extend(PyPDFLoader(doc).load())
            elif doc.suffix == ".docx":
                docs.extend(Docx2txtLoader(doc).load())
            else:
                console.print(f"[yellow]⚠ Unsupported document type: {doc}[/yellow]")

    return docs


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_db(
    docs: List[Document],
    persist_directory: str,
) -> None:
    """
    Create a new Chroma database or add documents to an existing one.

    Args:
        docs: Documents to be embedded.
        persist_directory: Directory of the persistent Chroma database.
    """

    spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_pages = spliter.split_documents(docs)

    if not all_pages:
        print("No documents found.")
        return
    embeddings = get_embeddings()

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        print(f"Created ChromaDB vector store for path {persist_directory}!")

    try:
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        vector_db.add_documents(all_pages)
        print(f"Added documents to ChromaDB vector store for path {persist_directory}!")

        return None

    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        return ToolError(
            code=ToolErrorCode.VECTOR_DB_SETUP_ERROR,
            message=str(e),
            tool="build_vector_db",
            retryable=False,
        )


def load_retriever(persist_directory):

    if not persist_directory:
        return None

    try:
        embeddings = get_embeddings()
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        print(f"Created ChromaDB vector store for path {persist_directory}!")

    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        return ToolError(
            code=ToolErrorCode.RETRIEVER_LOAD_ERROR,
            message=str(e),
            tool="load_retriever",
            retryable=False,
        )

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # K is the amount of chunks to return
    )

    return retriever


async def reteriever_tool(retriever: BaseRetriever, queries: List[str]) -> List[str]:
    """
    This todol searches and returns the information from the documents under the 'documents' folder.
    """

    if not retriever:
        console.print("[yellow]⚠ Retriever is not initialized.[/yellow]")
        return None

    results = []
    seen = set()
    idx = 1
    for query in queries:
        docs = await retriever.ainvoke(query)

        if not docs:
            continue

        for doc in docs:
            if doc.page_content in seen:
                continue
            seen.add(doc.page_content)
            results.append(f"Document {idx}:\n{doc.page_content}")
            idx += 1

    if not results:
        return None

    return results

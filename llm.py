from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from tools.web_search_tools import search_pubmed, search_arxiv, research_tavily
from tools.retriever_toos import retriever_tool

load_dotenv()

if os.environ["OPENAI_API_KEY"]:
    print("API Key Found")
else:
    raise ValueError("API Key Not Found")

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

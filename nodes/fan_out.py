from langgraph.constants import Send
from cores.state import State
from cores.schemas import RouterOutput
from llm import llm
from langgraph.constants import Send


def route_subqueries(state: State):
    sends = []

    for decision in state["router"].decisions:
        subquery_state = {
            "subquery": decision.subquery,
            "tools": decision.tools,
        }
        if "local_knowledge" in decision.tools:
            subquery_state = {
                "subquery": decision.subquery,
                "tools": ["local_knowledge"],
            }
            sends.append(Send("local_db", subquery_state))

        if "document_qa" in decision.tools:
            subquery_state = {
                "subquery": decision.subquery,
                "tools": ["document_qa"],
            }
            sends.append(Send("document_qa", subquery_state))

        web_tools = [
            tool
            for tool in decision.tools
            if tool
            in [
                "research_tavily",
                "search_pubmed",
                "search_arxiv",
            ]
        ]

        if web_tools:
            subquery_state = {
                "subquery": decision.subquery,
                "tools": web_tools,
            }

            sends.append(Send("web_search", subquery_state))

    return sends

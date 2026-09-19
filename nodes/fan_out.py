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


def format_results(results):
    sections = []

    for item in results:
        tool = item["tool"]
        result = item["result"]

        if tool == "local_knowledge":
            sections.append(
                "## Local Knowledge\n" + "\n".join(f"- {doc}" for doc in result)
            )

        elif tool == "document_qa":
            sections.append(
                "## Document QA\n" + "\n".join(f"- {doc}" for doc in result)
            )

        elif tool == "websearch":
            hits = result.get("hits", [])
            pages = result.get("pages", [])

            web_text = ["## Web Search"]

            web_text.append("\n### Search Results")
            for i, hit in enumerate(hits, 1):
                web_text.append(
                    f"{i}. **{hit.get('title', '')}**\n" f"   URL: {hit.get('url', '')}"
                )

            web_text.append("\n### Retrieved Pages")
            for i, page in enumerate(pages, 1):
                web_text.append(
                    f"#### Source {i}\n"
                    f"Title: {page.get('title', '')}\n"
                    f"URL: {page.get('url', '')}\n"
                    f"Content:\n{page.get('text', '')[:6000]}"
                )

            sections.append("\n".join(web_text))

    return "\n\n".join(sections)

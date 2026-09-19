from cores.state import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List
from pydantic import BaseModel
from llm import llm
from console import console
from rich.panel import Panel
from cores.schemas import ClarifyResponse


async def node_clarify(state: State) -> State:
    """
    Takes the user's research prompt, asks the LLM to decide if any clarifying questions are needed, and extrace 5 short follow-up questions.
    If questions are found, they're stored so the CLI can prompt the user before continuing.

    return {
      "needs": true,
      "questions": [
        "What area of AI?",
        "Academic or industry?"
      ]
    }
    """

    user_prompt = state["user_prompt"]

    messages = [
        SystemMessage(f"""Your are a research assistant. 

    List clarifying questions that would dramatically improve the research plan only if quite needed.
    Return JSON with keys:
    - needs: boolean
    - questions: list of short questions (max 5)
    if nothing is needed, return = false and quesitions = []
    """),
        HumanMessage(user_prompt),
    ]

    console.rule(
        "[bold cyan]Clarify"
    )  # Used to visually separate sections of the terminal output

    structured_llm = llm.with_structured_output(ClarifyResponse)

    with console.status("Thinking about what to clayrify..."):
        strutured_response = await structured_llm.ainvoke(messages)

    needs = strutured_response.needs
    questions = strutured_response.questions

    if needs and questions:
        console.print(
            Panel(
                "\n".join(f"- {q}" for q in questions),
                title="Clarifying Questions",
                style="yellow",
            )
        )
    else:
        console.print("[green]No clarifying questions needed.[/green]")

    return {
        "needs_clarification": needs,
        "subquestions": questions,
    }

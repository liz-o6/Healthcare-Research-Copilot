# Healthcare Research Copilot

A LangGraph research agent that combines web search, uploaded documents, and a
persistent local knowledge base to produce citation-checked Markdown reports.

## Features

- Clarifies ambiguous research questions.
- Plans subqueries and routes them to PubMed, arXiv, Tavily, Document QA, or
  Local Knowledge.
- Runs retrieval branches in parallel and combines their results.
- Retries failed retrieval tools up to three times without stopping the entire
  workflow.
- Builds a source registry and verifies inline citations.
- Revises unsupported citations and saves the final report as
  `research_report.md`.
- Includes an evaluation dataset, automated runner, and fault-injection tests.

## Workflow

```text
User Query
    |
Clarification -> Planner -> Router
                            |
              +-------------+-------------+
              |             |             |
          Web Search    Document QA   Local Knowledge
              +-------------+-------------+
                            |
                         Summary
                            |
                 Citation Verification
                            |
                      Final Report
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Optional input files:

- Place temporary session documents in `documents/`.
- Place persistent knowledge documents in `local_documents/`.
- Supported formats: PDF, DOCX, and TXT.

## Run

```bash
venv/bin/python main.py
```

The completed report is saved to `research_report.md`.

## Evaluation

The dataset is stored in `evals/evaluation_dataset.json` and covers end-to-end
research, clarification, citation verification, and error recovery.

```bash
# Citation verification cases (default)
venv/bin/python evals/run_evals.py

# Fault-injection and retry cases
venv/bin/python evals/run_evals.py --type error_recovery

# All cases; calls external APIs
venv/bin/python evals/run_evals.py --type all

# One case
venv/bin/python evals/run_evals.py --case citation-invalid-id-001
```

Results are written to `evals/evaluation_results.json`.

## Project Structure

```text
cores/                         State, schemas, and error models
nodes/                         LangGraph workflow nodes
tools/                         Search and retrieval tools
documents/                     Temporary Document QA inputs
local_documents/               Persistent knowledge inputs
local_db/                      Persistent Chroma database
evals/evaluation_dataset.json  Evaluation cases
evals/run_evals.py             Evaluation runner
main.py                        Application entry point
research_report.md             Generated report
```

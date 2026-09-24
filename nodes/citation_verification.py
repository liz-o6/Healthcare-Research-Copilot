import json
import re
from typing import List, Tuple

from console import console
from cores.schemas import CitationCheck, CitationVerificationOutput
from cores.state import State
from llm import llm


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
SOURCES_HEADER_PATTERN = re.compile(r"^#{1,6}\s+sources\s*$", re.IGNORECASE)


def extract_cited_claims(report_markdown: str) -> List[Tuple[str, str]]:
    """Return unique (claim, source_id) pairs from the report body."""
    pairs = []
    seen = set()

    for raw_line in report_markdown.splitlines():
        line = raw_line.strip()

        if SOURCES_HEADER_PATTERN.match(line):
            break
        if not line or line.startswith("#"):
            continue

        source_ids = CITATION_PATTERN.findall(line)
        if not source_ids:
            continue

        claim = CITATION_PATTERN.sub("", line)
        claim = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", claim)
        claim = claim.replace("**", "").replace("__", "").replace("`", "").strip()

        for source_id in source_ids:
            pair = (claim, source_id)
            if claim and pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    return pairs


async def node_citation_verification(state: State) -> State:
    console.rule("[bold cyan]Verify citations")

    report = state.get("report_markdown") or ""
    source_registry = state.get("source_registry") or []
    source_map = {source.source_id: source for source in source_registry}
    cited_claims = extract_cited_claims(report)

    checks: List[CitationCheck] = []
    valid_claims = []

    for claim, source_id in cited_claims:
        if source_id not in source_map:
            checks.append(
                CitationCheck(
                    claim=claim,
                    source_id=source_id,
                    status="invalid",
                    reason="The cited source ID does not exist in the source registry.",
                )
            )
        else:
            valid_claims.append({"claim": claim, "source_id": source_id})

    if valid_claims:
        relevant_ids = {item["source_id"] for item in valid_claims}
        relevant_sources = [
            source.model_dump()
            for source_id, source in source_map.items()
            if source_id in relevant_ids
        ]

        prompt = f"""Verify whether each cited source supports its claim.

        Rules:
        - Use only the source content provided below.
        - Return exactly one CitationCheck for every claim-source pair.
        - Preserve each claim and source_id exactly.
        - Use "supported" only when the source directly supports the claim.
        - Use "unsupported" when support is absent, too weak, or contradictory.
        - Keep each reason concise and specific.

        Claim-source pairs:
        {json.dumps(valid_claims, ensure_ascii=False, indent=2)}

        Sources:
        {json.dumps(relevant_sources, ensure_ascii=False, indent=2)}
        """

        try:
            verifier = llm.with_structured_output(CitationVerificationOutput)
            result = await verifier.ainvoke(prompt)
            checks.extend(result.citation_checks)
        except Exception as error:
            checks.extend(
                CitationCheck(
                    claim=item["claim"],
                    source_id=item["source_id"],
                    status="invalid",
                    reason=f"Citation verification failed: {error}",
                )
                for item in valid_claims
            )

    supported = sum(check.status == "supported" for check in checks)
    console.print(f"Verified {len(checks)} citations: {supported} supported")

    return {"citation_checks": checks}

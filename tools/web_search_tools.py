import requests
import xml.etree.ElementTree as ET
from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
from cores.error_codes import ToolError, ToolErrorCode
from bs4 import BeautifulSoup
from readability import Document
import re
import httpx

load_dotenv()
tavily = TavilyClient(api_key=str(os.getenv("TAVILY_API_KEY")))


def dedupe_hits(hits: List[Dict[str, Any]], limit=25) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for h in hits:
        key = re.sub(r"#.*$", "", (h.get("url") or "").strip())
        if key and key not in seen:
            seen.add(key)
            deduped.append(h)

        if len(deduped) >= limit:
            break

    return deduped


async def fetch_page(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """
    return {
        "url": "",
        "title": "",
        "text": ""
    }
    """
    try:
        r = await client.get(url, follow_redirects=True)
        html = r.text
        doc = Document(html)
        title = doc.short_title()
        cleaned = doc.summary(html_partial=True)
        soup = BeautifulSoup(cleaned, "html.parser")
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

        return {
            "url": str(r.url),
            "title": title,
            "text": text[:120000],
        }

    except Exception as e:
        return {
            "url": url,
            "title": "",
            "text": f"Error fetching page: {e}",
        }


@tool
async def search_arxiv(query: str, k: int = 5) -> list:
    """
    Use Arxiv to search query and get k results, input "query" and the number of results "k",
    Arxiv search will return a json response
    json[
        {
            "id": "...",
            "title": "...",
            "authors": [...],
            "published": "...",
            "abstract": "...",
            "pdf_url": "...",
            "url": "...",
        },
        ...
    ]

    Arxiv search will return a list of dict, each dict contains id, title, abstract, and url
    """

    if not isinstance(k, int) or k <= 0:
        k = 5
    # Query the arXiv API
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": k,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as e:
        return ToolError(
            code=ToolErrorCode.ARXIV_TIMEOUT_ERROR,
            message=str(e),
            tool="search_arxiv",
            retryable=True,
        )

    except requests.exceptions.ConnectionError as e:
        return ToolError(
            code=ToolErrorCode.ARXIV_NETWORK_ERROR,
            message=str(e),
            tool="search_arxiv",
            retryable=True,
        )

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None

        retryable = status_code == 429 or (
            status_code is not None and status_code >= 500
        )

        return ToolError(
            code=ToolErrorCode.ARXIV_NETWORK_ERROR,
            message=f"HTTP {status_code}: {e}",
            tool="search_arxiv",
            retryable=retryable,
        )

    # 3. XML parsing
    try:
        root = ET.fromstring(response.text)

    except ET.ParseError as e:
        return ToolError(
            code=ToolErrorCode.ARXIV_PARSE_ERROR,
            message=str(e),
            tool="search_arxiv",
            retryable=False,
        )

    # XML namespace used by the Atom feed
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Convert to a unified list format
    results = []

    try:
        for entry in root.findall("atom:entry", ns):
            authors = [
                author.findtext(
                    "atom:name",
                    default="",
                    namespaces=ns,
                )
                for author in entry.findall("atom:author", ns)
            ]

            pdf_url = ""

            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break

            paper = {
                "id": entry.findtext(
                    "atom:id",
                    default="",
                    namespaces=ns,
                ),
                "title": entry.findtext(
                    "atom:title",
                    default="",
                    namespaces=ns,
                ).strip(),
                "authors": authors,
                "published": entry.findtext(
                    "atom:published",
                    default="",
                    namespaces=ns,
                ),
                "abstract": entry.findtext(
                    "atom:summary",
                    default="",
                    namespaces=ns,
                ).strip(),
                "pdf_url": pdf_url,
                "url": entry.findtext(
                    "atom:id",
                    default="",
                    namespaces=ns,
                ),
            }

            results.append(
                {
                    "source": "arxiv",
                    "id": paper["id"],
                    "title": paper["title"],
                    "content": paper["abstract"],
                    "url": paper["url"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "metadata": {"pdf_url": paper["pdf_url"]},
                }
            )

            if len(results) >= k:
                break

    except Exception as e:
        return ToolError(
            code=ToolErrorCode.ARXIV_RESULT_PARSE_ERROR,
            message=str(e),
            tool="search_arxiv",
            retryable=False,
        )

    if not results:
        print("Arxiv no results found.")

    return results


@tool
async def search_pubmed(query: str, k: int = 5) -> list | ToolError:
    """
    Use PubMed to search query and get k results, input "query" and the number of results "k",
    PubMed search will return a json response
    json[
        {
            "pmid": "41234567",
            "title": "...",
            "authors": ["Alice", "Bob"],
            "journal": "Nature Medicine",
            "pubdate": "2025 Jun",
            "doi": "doi:10.xxxx/xxxx",
            "url": "https://pubmed.ncbi.nlm.nih.gov/41234567/"
        },
        ...
    ]

    PubMed search will return a list of dict, each dict contains pmid, title, abstract, and url
    """

    # 1. Validate input
    if not isinstance(k, int) or k <= 0:
        k = 5

    # 2. Search PubMed and get PMID list
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" "esearch.fcgi"

    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": k,
        "sort": "relevance",
        "retmode": "json",
    }

    try:
        response = requests.get(
            search_url,
            params=search_params,
            timeout=30,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_TIMEOUT_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=True,
        )

    except requests.exceptions.ConnectionError as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_NETWORK_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=True,
        )

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None

        retryable = status_code == 429 or (
            status_code is not None and status_code >= 500
        )

        return ToolError(
            code=ToolErrorCode.PUBMED_NETWORK_ERROR,
            message=f"HTTP {status_code}: {e}",
            tool="search_pubmed",
            retryable=retryable,
        )

    except requests.exceptions.RequestException as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_NETWORK_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=False,
        )

    # 3. Parse JSON response
    try:
        data = response.json()

    except ValueError as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_JSON_PARSE_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=False,
        )

    pmids = data.get("esearchresult", {}).get("idlist", [])

    if not pmids:
        print("PubMed: no results found.")
        return []

    # 4. Fetch paper details
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" "efetch.fcgi"

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    try:
        response = requests.get(
            fetch_url,
            params=fetch_params,
            timeout=30,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_TIMEOUT,
            message=str(e),
            tool="search_pubmed",
            retryable=True,
        )

    except requests.exceptions.ConnectionError as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_NETWORK_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=True,
        )

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None

        retryable = status_code == 429 or (
            status_code is not None and status_code >= 500
        )

        return ToolError(
            code=ToolErrorCode.PUBMED_NETWORK_ERROR,
            message=f"HTTP {status_code}: {e}",
            tool="search_pubmed",
            retryable=retryable,
        )

    except requests.exceptions.RequestException as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_NETWORK_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=False,
        )
    # 5. Parse XML response
    try:
        root = ET.fromstring(response.text)

    except ET.ParseError as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_XML_PARSE_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=False,
        )

    # 6. Convert to unified result format
    results = []

    try:
        for article in root.findall(".//PubmedArticle"):

            if len(results) >= k:
                break

            pmid = article.findtext(
                ".//PMID",
                default="",
            )

            title = article.findtext(
                ".//ArticleTitle",
                default="",
            )

            journal = article.findtext(
                ".//Journal/Title",
                default="",
            )

            pubdate = article.findtext(
                ".//PubDate/Year",
                default="",
            )

            authors = [
                f"{a.findtext('ForeName', '')} " f"{a.findtext('LastName', '')}".strip()
                for a in article.findall(".//Author")
            ]

            # Join multiple AbstractText sections
            abstract = " ".join(
                text.text or "" for text in article.findall(".//AbstractText")
            )

            results.append(
                {
                    "source": "pubmed",
                    "id": pmid,
                    "title": title,
                    "content": abstract,
                    "url": (f"https://pubmed.ncbi.nlm.nih.gov/" f"{pmid}/"),
                    "authors": authors,
                    "published": pubdate,
                    "metadata": {
                        "journal": journal,
                    },
                }
            )

    except Exception as e:
        return ToolError(
            code=ToolErrorCode.PUBMED_RESULT_PARSE_ERROR,
            message=str(e),
            tool="search_pubmed",
            retryable=False,
        )

    # 7. Empty results are not errors
    if not results:
        print("PubMed: no results found.")

    return results


@tool
async def research_tavily(query: str, k: int = 5) -> list | ToolError:
    """
    Use tavily to search query and get k results, input "query" and the number of results "k",
    tavily search will return a json response
    json[
    {
        "query": "",
        "answer": "",
        "results": [
            {
            "url": "",
            "title": "",
            "content": "",
            "score": xxx,
            "published_date": ""
            }
        ]
    }]

    output: a list of dict, each dict contains url, title, snippet, and score
    """
    if not isinstance(k, int) or k <= 0:
        k = 5

    # 2. Call Tavily API
    try:
        res = tavily.search(
            query=query,
            max_results=k,
        )

    except Exception as e:
        error_message = str(e).lower()

        retryable = any(
            keyword in error_message
            for keyword in [
                "timeout",
                "timed out",
                "connection",
                "rate limit",
                "429",
                "500",
                "502",
                "503",
                "504",
            ]
        )

        return ToolError(
            code=ToolErrorCode.TAVILY_API_ERROR,
            message=str(e),
            tool="research_tavily",
            retryable=retryable,
        )

    # 3. Validate API response
    if not isinstance(res, dict):
        return ToolError(
            code=ToolErrorCode.TAVILY_INVALID_RESPONSE,
            message=(
                f"Expected Tavily response to be dict, " f"got {type(res).__name__}"
            ),
            tool="research_tavily",
            retryable=False,
        )

    # 4. Extract results
    raw_results = res.get("results", [])

    if raw_results is None:
        raw_results = []

    if not isinstance(raw_results, list):
        return ToolError(
            code=ToolErrorCode.TAVILY_INVALID_RESPONSE,
            message="'results' field is not a list",
            tool="research_tavily",
            retryable=False,
        )

    # 5. Convert to unified result format
    results = []

    try:
        for r in raw_results[:k]:

            if not isinstance(r, dict):
                continue

            url = r.get("url")

            # A result without a URL is not useful for
            # citation / deduplication.
            if not url:
                continue

            title = r.get("title") or ""
            content = r.get("content") or r.get("snippet") or ""
            score = r.get("score", 0)

            results.append(
                {
                    "source": "tavily",
                    "id": url,
                    "title": title,
                    "content": content,
                    "url": url,
                    "authors": [],
                    "published": None,
                    "metadata": {
                        "score": score,
                    },
                }
            )

    except Exception as e:
        return ToolError(
            code=ToolErrorCode.TAVILY_RESULT_PARSE_ERROR,
            message=str(e),
            tool="research_tavily",
            retryable=False,
        )

    # 6. Empty results are not errors
    if not results:
        print("Tavily: no results found.")

    return results

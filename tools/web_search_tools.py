import requests
import xml.etree.ElementTree as ET
from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

load_dotenv()
tavily = TavilyClient(api_key=str(os.getenv("TAVILY_API_KEY")))


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
    # Query the arXiv API
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": k,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    xml_text = requests.get(url, params=params).text
    root = ET.fromstring(xml_text)

    # XML namespace used by the Atom feed
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Convert to a unified list format
    results = []
    num = 0
    for entry in root.findall("atom:entry", ns):
        if num >= k:
            break
        num += 1
        authors = [
            author.findtext("atom:name", default="", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]

        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
                break

        paper = {
            "id": entry.findtext("atom:id", default="", namespaces=ns),
            "title": entry.findtext("atom:title", default="", namespaces=ns).strip(),
            "authors": authors,
            "published": entry.findtext("atom:published", default="", namespaces=ns),
            "abstract": entry.findtext(
                "atom:summary", default="", namespaces=ns
            ).strip(),
            "pdf_url": pdf_url,
            "url": entry.findtext("atom:id", default="", namespaces=ns),
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

    if not results:
        print("Arxiv no results found.")

    return results


@tool
async def search_pubmed(query: str, k: int = 5) -> list:
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

    # Step 1: Search PubMed and get PMID list
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": k,
        "sort": "relevance",
        "retmode": "json",
    }

    response = requests.get(search_url, params=search_params, timeout=30)

    if not response.ok:
        print(f"PubMed API error: {response.status_code}")
        return []

    data = response.json()

    # print("data:", data)

    pmids = data.get("esearchresult", {}).get("idlist", [])
    # print("pmids:", pmids)

    if not pmids:
        print("Pubmed no results found.")
        return []

    # Step 2: Fetch paper details (including abstract)
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    xml_text = requests.get(fetch_url, params=fetch_params).text

    if not xml_text.strip().startswith("<?xml"):
        print("Invalid XML response:", xml_text[:500])
        return []

    root = ET.fromstring(xml_text)

    # Step 3: Convert to a unified list format
    results = []
    num = 0
    for article in root.findall(".//PubmedArticle"):
        if num >= k:
            break
        num += 1
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="")
        journal = article.findtext(".//Journal/Title", default="")
        pubdate = article.findtext(".//PubDate/Year", default="")

        authors = [
            f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
            for a in article.findall(".//Author")
        ]

        # Join multiple AbstractText sections if present
        abstract = " ".join(
            text.text or "" for text in article.findall(".//AbstractText")
        )

        paper = {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "pubdate": pubdate,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

        results.append(
            {
                "source": "pubmed",
                "id": paper["pmid"],
                "title": paper["title"],
                "content": paper["abstract"],
                "url": paper["url"],
                "authors": paper["authors"],
                "published": paper["pubdate"],
                "metadata": {"journal": paper["journal"]},
            }
        )

    if not results:
        print("Pubmed no results found.")

    return results


@tool
async def research_tavily(query: str, k: int = 5) -> List[Dict[str, Any]]:
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
    res = tavily.search(query=query, max_results=k)
    results = []
    num = 0
    for r in res.get("results", []):
        if num >= k:
            break
        num += 1
        hit = {
            "url": r.get("url"),
            "title": r.get("title"),
            "snippet": r.get("content") or r.get("snippet"),
            "score": r.get("score", 0),
        }

        results.append(
            {
                "source": "tavily",
                "id": hit["url"],
                "title": hit["title"],
                "content": hit["snippet"],
                "url": hit["url"],
                "authors": [],
                "published": None,
                "metadata": {"score": hit["score"]},
            }
        )

    if not results:
        print("tavily no results found.")

    return results

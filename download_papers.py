#!/usr/bin/env python3
"""
Download the N most recent arXiv papers from a given category.

Defaults:
  category   : q-bio.PE   (Populations and Evolution)
  max_results: 10

Output:
  data/papers/<arxiv_id>.pdf   — PDF files  (ready for the ingestion pipeline)
  data/metadata.json           — accumulating metadata store (JSON array)

Usage:
  python download_papers.py
  python download_papers.py --category cs.LG --n 20
"""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARXIV_API_URL = "https://export.arxiv.org/api/query"

BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
PAPERS_DIR   = DATA_DIR / "papers"
METADATA_FILE = DATA_DIR / "metadata.json"

# arXiv Atom feed XML namespaces
_NS = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
}

# Polite delay between consecutive PDF downloads (seconds)
_DOWNLOAD_DELAY = 3


# ---------------------------------------------------------------------------
# arXiv API helpers
# ---------------------------------------------------------------------------

def _fetch_feed(category: str, max_results: int) -> str:
    """Call the arXiv API and return the raw Atom XML string."""
    params = {
        "search_query": f"cat:{category}",
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
        "start":        0,
        "max_results":  max_results,
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_feed(xml_text: str) -> list[dict]:
    """
    Parse an arXiv Atom feed into a list of paper-metadata dicts.

    Each dict has keys:
        arxiv_id, title, authors, year, published, abstract,
        pdf_url, arxiv_url, categories
    """
    root = ET.fromstring(xml_text)
    papers: list[dict] = []

    for entry in root.findall("atom:entry", _NS):

        # --- IDs ----------------------------------------------------------
        raw_id = entry.findtext("atom:id", namespaces=_NS) or ""
        # raw_id looks like:  http://arxiv.org/abs/2501.12345v2
        arxiv_id = raw_id.split("/abs/")[-1]
        base_id  = arxiv_id.rsplit("v", 1)[0]   # strip version suffix

        # --- Title --------------------------------------------------------
        title = (entry.findtext("atom:title", namespaces=_NS) or "").strip()
        title = " ".join(title.split())          # collapse whitespace / newlines

        # --- Authors ------------------------------------------------------
        authors = [
            (a.findtext("atom:name", namespaces=_NS) or "").strip()
            for a in entry.findall("atom:author", _NS)
        ]

        # --- Dates --------------------------------------------------------
        published = entry.findtext("atom:published", namespaces=_NS) or ""
        year = int(published[:4]) if len(published) >= 4 else 0

        # --- Abstract -----------------------------------------------------
        abstract = (entry.findtext("atom:summary", namespaces=_NS) or "").strip()
        abstract = " ".join(abstract.split())

        # --- PDF URL ------------------------------------------------------
        pdf_url: Optional[str] = None
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
        # Ensure HTTPS and .pdf suffix
        pdf_url = pdf_url.replace("http://", "https://")
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"

        # --- Categories ---------------------------------------------------
        categories = [
            t.get("term", "")
            for t in entry.findall("atom:category", _NS)
        ]

        papers.append({
            "arxiv_id":   base_id,
            "title":      title,
            "authors":    authors,
            "year":       year,
            "published":  published,
            "abstract":   abstract,
            "pdf_url":    pdf_url,
            "arxiv_url":  f"https://arxiv.org/abs/{base_id}",
            "categories": categories,
        })

    return papers


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _safe_filename(arxiv_id: str) -> str:
    """Convert an arXiv ID to a safe filename (handle old-style q-bio/XXXXXX)."""
    return arxiv_id.replace("/", "_")


def _download_pdf(paper: dict, out_dir: Path) -> Optional[Path]:
    """
    Download one PDF.  Returns the local path on success, None on failure.
    Already-present files are skipped.
    """
    filename = _safe_filename(paper["arxiv_id"]) + ".pdf"
    out_path = out_dir / filename

    if out_path.exists() and out_path.stat().st_size > 1024:
        print(f"  ↷  Already exists : {filename}")
        return out_path

    print(f"  ↓  {paper['arxiv_id']}  —  {paper['title'][:60]}…")
    try:
        resp = requests.get(paper["pdf_url"], timeout=120, stream=True)
        resp.raise_for_status()

        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=16_384):
                fh.write(chunk)

        size_kb = out_path.stat().st_size // 1024
        print(f"     ✓  {filename}  ({size_kb} KB)")
        return out_path

    except Exception as exc:
        print(f"     ✗  Failed: {exc}")
        if out_path.exists():
            out_path.unlink()          # remove partial file
        return None


# ---------------------------------------------------------------------------
# Metadata store
# ---------------------------------------------------------------------------

def _load_metadata() -> list[dict]:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def _save_metadata(papers: list[dict]) -> None:
    """
    Merge *papers* into the persistent metadata.json.
    Existing entries (matched by arxiv_id) are updated in place.
    """
    existing = _load_metadata()
    index = {p["arxiv_id"]: i for i, p in enumerate(existing)}

    for paper in papers:
        aid = paper["arxiv_id"]
        if aid in index:
            existing[index[aid]] = paper     # update
        else:
            existing.append(paper)           # insert

    with open(METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)

    print(f"\n📋 Metadata → {METADATA_FILE}  ({len(existing)} total entries)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(category: str, n: int) -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch paper metadata from arXiv
    print(f"Querying arXiv  category={category}  n={n} …\n")
    xml_text = _fetch_feed(category, n)
    papers   = _parse_feed(xml_text)

    if not papers:
        print("⚠  No papers returned — check category name or network.")
        return

    print(f"Found {len(papers)} papers. Starting PDF downloads …\n")

    # 2. Download PDFs (with polite delay between requests)
    downloaded = 0
    for i, paper in enumerate(papers):
        path = _download_pdf(paper, PAPERS_DIR)
        if path:
            paper["local_pdf"] = str(path.relative_to(BASE_DIR))
            downloaded += 1
        else:
            paper["local_pdf"] = None

        if i < len(papers) - 1:
            time.sleep(_DOWNLOAD_DELAY)

    # 3. Persist metadata
    _save_metadata(papers)

    # 4. Summary
    print(f"\n✅  {downloaded}/{len(papers)} PDFs downloaded → {PAPERS_DIR}")
    print(f"    Metadata           → {METADATA_FILE}")
    print(f"\nReady to ingest:  python main.py --ingest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="q-bio.PE",
                        help="arXiv category (default: q-bio.PE)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of most-recent papers to fetch (default: 10)")
    args = parser.parse_args()

    main(category=args.category, n=args.n)

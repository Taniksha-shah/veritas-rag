"""
fetch_pubmed_papers.py

Bulk-downloads open-access, full-text biomedical papers, saved as clean .txt
files ready to feed straight into data_generation.py's --input_dir.

Uses Europe PMC's REST API rather than NCBI's E-utilities.
------------------------------------------------------------
Europe PMC mirrors the same underlying PMC corpus but its search results
include an explicit `inEPMC` flag that tells you, before you fetch anything,
whether an article's full text is actually retrievable. That lets us filter
out non-retrievable results up front instead of discovering it one request
at a time (which is what was happening with NCBI's efetch: PubMed's
"open access[filter]" also matches NIH public-access manuscripts that were
never released for full-text reuse, so a lot of "open access" hits still
come back abstract-only).

Two-step process:
1. Query https://www.ebi.ac.uk/europepmc/webservices/rest/search with
   `OPEN_ACCESS:y AND SRC:PMC`, keep only results where inEPMC == "Y".
2. Fetch https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML
   for each one, parse out the title + body paragraphs, save as PMCID.txt.

Usage
-----
# Search-driven: pull up to 150 papers matching a topic
python fetch_pubmed_papers.py \
    --query "adverse drug reaction pharmacovigilance" \
    --max_papers 150 \
    --output_dir ./raw_papers \
    --email you@example.com

# Or fetch specific PMC IDs directly
python fetch_pubmed_papers.py \
    --pmcids PMC10215571,PMC11919300,PMC10300063,PMC12268696 \
    --output_dir ./raw_papers \
    --email you@example.com
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT_URL_TMPL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def _get(url: str, email: str):
    """GET with a descriptive User-Agent, as EBI's usage guidelines ask for."""
    req = urllib.request.Request(url, headers={"User-Agent": f"VeritasStageA/1.0 (mailto:{email})"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def epmc_search_pmc_ids(query: str, max_papers: int, email: str):
    """Search Europe PMC, restricted to open-access PMC articles that actually
    have retrievable full text (inEPMC == 'Y'). Returns a list of PMCIDs."""
    params = {
        "query": f"{query} AND OPEN_ACCESS:y AND SRC:PMC",
        "format": "json",
        "pageSize": min(max_papers, 1000),
    }
    url = f"{EPMC_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    data = json.loads(_get(url, email))
    results = data.get("resultList", {}).get("result", [])
    pmc_ids = [r["pmcid"] for r in results if r.get("pmcid") and r.get("inEPMC") == "Y"]
    return pmc_ids


def fetch_fulltext_xml(pmcid: str, email: str):
    url = EPMC_FULLTEXT_URL_TMPL.format(pmcid=pmcid)
    return _get(url, email)


def extract_body_text(xml_bytes: bytes):
    """Pull the article title and body paragraph text out of the JATS XML."""
    root = ET.fromstring(xml_bytes)

    title_el = root.find(".//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    body = root.find(".//body")
    if body is None:
        return title, ""

    paragraphs = []
    for p in body.iter("p"):
        text = "".join(p.itertext())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)

    return title, "\n\n".join(paragraphs)


def download_and_save(pmcid: str, output_dir: Path, email: str):
    try:
        xml_bytes = fetch_fulltext_xml(pmcid, email)
    except Exception as e:
        print(f"  [{pmcid}] fullTextXML request failed: {e}")
        return False

    try:
        title, body_text = extract_body_text(xml_bytes)
    except ET.ParseError as e:
        print(f"  [{pmcid}] could not parse XML response: {e}")
        return False

    if not body_text:
        print(f"  [{pmcid}] response parsed but had no body paragraphs, skipping.")
        return False

    out_path = output_dir / f"{pmcid}.txt"
    out_path.write_text(f"{title}\n\n{body_text}", encoding="utf-8")
    print(f"  [{pmcid}] saved -> {out_path.name} ({len(body_text.split())} words)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Search term, e.g. 'adverse drug reaction pharmacovigilance'")
    parser.add_argument("--pmcids", type=str, default=None, help="Comma-separated PMC IDs to fetch directly, e.g. PMC10215571,PMC11919300")
    parser.add_argument("--max_papers", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./raw_papers")
    parser.add_argument("--email", type=str, required=True, help="Used in the User-Agent header, per EBI's usage guidelines")
    parser.add_argument("--sleep", type=float, default=0.34, help="Seconds between requests (be polite to the free API)")
    args = parser.parse_args()

    if not args.query and not args.pmcids:
        raise SystemExit("Provide either --query or --pmcids")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pmcids:
        pmc_ids = [p.strip() for p in args.pmcids.split(",") if p.strip()]
    else:
        print(f"Searching Europe PMC (open access, full text available) for: {args.query!r} ...")
        pmc_ids = epmc_search_pmc_ids(args.query, args.max_papers, args.email)
        print(f"Found {len(pmc_ids)} candidate articles with retrievable full text.")

    n_saved = 0
    for pmcid in pmc_ids:
        ok = download_and_save(pmcid, output_dir, args.email)
        if ok:
            n_saved += 1
        time.sleep(args.sleep)

    print(f"\nSaved {n_saved}/{len(pmc_ids)} papers to {output_dir}/")
    print(f"Feed this folder straight into: python data_generation.py --input_dir {output_dir}")


if __name__ == "__main__":
    main()

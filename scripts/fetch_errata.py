#!/usr/bin/env python3
"""Fetch the official Cyberpunk TCG card errata as structured JSON.

Errata matter more here than in a mature game, and they carry an explicit
override: the errata page states that "every card on this page should be
played as if it bears the updated text and characteristics associated with
it. The updated text shown here supersedes all printed text at all levels of
play." So this is not supplementary colour -- it is the authority over the
card database wherever the two disagree.

WHY THIS READS THE CMS API AND NOT THE RENDERED PAGE

`https://cyberpunktcg.com/errata` is server-rendered, so unlike the
Comprehensive Rules it *could* be scraped. It should not be, for a reason
found by comparing the two: the rendered page carried three errata while the
CMS carried FOUR. The Judy Alvarez artist erratum was live in the CMS and
absent from the HTML, on a freshly fetched copy, so a scraper would have
silently missed an erratum that officially supersedes printed text. The
rendered page lags; the CMS does not.

The site's own JS bundle queries content through:

    POST https://api.netdeck.gg/api/cyberpunk/content/query
    {"params": {"content_type": "cyberpunkPage", "fields.slug": "errata"}}

which is a Contentful-shaped response: `items` holds the page, and
`includes.Entry` holds every linked entry, addressed by id. The page links
`sectionEntries` -> a section -> `subsectionEntries` -> markdown blocks, and
the errata themselves are `### <Card Name>` headings inside one block's
`bodyMarkdown`.

Usage:
    python scripts/fetch_errata.py               # -> data/raw/errata.json
    python scripts/fetch_errata.py --print-updated-at   # check for new errata
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

API_BASE = "https://api.netdeck.gg/api"
CONTENT_URL = f"{API_BASE}/cyberpunk/content/query"
USER_AGENT = "Mozilla/5.0 (compatible; cyberpunk-llm-corpus-builder/1.0)"
PAGE_SLUG = "errata"

DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "errata.json"


def fetch_page(slug: str = PAGE_SLUG, url: str = CONTENT_URL, timeout: int = 30) -> dict:
    """One CMS page and every entry it links, as returned by the content API."""
    payload = json.dumps({
        "params": {
            "content_type": "cyberpunkPage",
            "fields.slug": slug,
            # Deep enough for page -> section -> markdown block. Contentful
            # silently returns unresolved Link stubs if this is too shallow,
            # which `validate` below turns into a hard error rather than an
            # errata list that is quietly short.
            "include": 3,
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def validate(doc: dict, slug: str = PAGE_SLUG) -> None:
    """Refuse a response `build_errata.py` cannot rely on.

    The check that earns its keep is the link-resolution one: Contentful
    returns `{"sys": {"type": "Link", ...}}` stubs for anything outside the
    requested include depth, and a parser that walked those would find no
    markdown at all and write an EMPTY errata file -- which looks exactly
    like "this game has no errata yet".
    """
    for key in ("items", "includes"):
        if key not in doc:
            raise SystemExit(f"response is missing top-level {key!r} -- API shape changed?")
    if doc.get("total") != 1 or len(doc["items"]) != 1:
        raise SystemExit(
            f"expected exactly one page with slug {slug!r}, got total={doc.get('total')}")

    page = doc["items"][0]
    if page["fields"].get("slug") != slug:
        raise SystemExit(f"returned page is {page['fields'].get('slug')!r}, not {slug!r}")

    entries = doc["includes"].get("Entry") or []
    if not entries:
        raise SystemExit("no linked entries returned -- the errata body would be empty")

    linked = {e["sys"]["id"] for e in entries}
    for ref in page["fields"].get("sectionEntries") or []:
        if ref["sys"]["id"] not in linked:
            raise SystemExit(
                f"section {ref['sys']['id']} was linked but not included -- raise `include`")

    blocks = [e for e in entries
              if e["sys"]["contentType"]["sys"]["id"] == "cyberpunkPageMarkdownBlock"]
    if not blocks:
        raise SystemExit("no markdown blocks in the response -- nothing to parse")
    if not any((b["fields"].get("bodyMarkdown") or "").strip() for b in blocks):
        raise SystemExit("every markdown block is empty -- refusing to save")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--slug", type=str, default=PAGE_SLUG)
    ap.add_argument("--url", type=str, default=CONTENT_URL)
    ap.add_argument("--print-updated-at", action="store_true",
                    help="fetch and print the page's updatedAt, write nothing")
    args = ap.parse_args()

    doc = fetch_page(args.slug, args.url)
    validate(doc, args.slug)

    page = doc["items"][0]
    updated_at = page["sys"]["updatedAt"]
    if args.print_updated_at:
        print(updated_at)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    headings = sum(
        (e["fields"].get("bodyMarkdown") or "").count("\n### ")
        + (e["fields"].get("bodyMarkdown") or "").startswith("### ")
        for e in doc["includes"]["Entry"]
        if e["sys"]["contentType"]["sys"]["id"] == "cyberpunkPageMarkdownBlock")
    print(f"wrote the {args.slug!r} page ({headings} errata heading(s)) -> {args.out}")
    print(f"updatedAt: {updated_at}")


if __name__ == "__main__":
    main()

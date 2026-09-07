#!/usr/bin/env python3
"""Fetch the official Cyberpunk TCG Comprehensive Rules as structured JSON.

The rules live at https://cyberpunktcg.com/comprehensive-rules, which is a
client-rendered route -- fetching that URL returns an empty SPA shell, so
there is nothing to scrape. The page loads its content from:

    https://api.netdeck.gg/api/cyberpunk/comprehensive-rules

which returns the document as a flat list of nodes with parent links, NOT as
prose. That is a considerably better corpus than the PDF-derived text the One
Piece instance of this template has to work from: the hierarchy, the rule
numbers, and the anchors are all first-class fields, so `scripts/ingest.py`
never has to infer structure from indentation or numbering heuristics.

Each node carries:

    id, parent_id      the tree
    node_type          "section" (a heading) or "rule" (a numbered statement)
    display_number     "1", "1.4", "9.2.3.1" -- blank on unnumbered sections
    title              the rule's first sentence, or the section heading
    body_markdown      the full rule text; empty string on sections
    stable_anchor      "rule-<uuid>", the site's own deep-link target
    sort_order, depth, is_numbered, include_in_toc

The response also carries a document-level `updated_at`, which is the only
version marker this game publishes -- there is no "version 1.2.1" string like
One Piece's rules carry. `games/cyberpunk/prompts.py`'s `RULES_VERSION` is
that date, and refreshing this file is what should prompt changing it.

Usage:
    python scripts/fetch_rules.py            # write data/raw/comprehensive_rules.json
    python scripts/fetch_rules.py --print-updated-at   # check for a new version
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

API_BASE = "https://api.netdeck.gg/api"
RULES_URL = f"{API_BASE}/cyberpunk/comprehensive-rules"
USER_AGENT = "Mozilla/5.0 (compatible; cyberpunk-llm-corpus-builder/1.0)"

DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "comprehensive_rules.json"

REQUIRED_NODE_FIELDS = {
    "id", "parent_id", "node_type", "title", "body_markdown",
    "display_number", "stable_anchor", "sort_order", "depth",
}


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def validate(doc: dict) -> None:
    """Refuse a response whose shape `ingest.py` cannot rely on.

    Checked here rather than in ingest so a bad fetch never lands on disk and
    get mistaken for a real corpus. The parent-link check matters most: an
    orphaned node would silently vanish from the ingested outline, taking its
    whole subtree with it, and the resulting corpus would look complete.
    """
    for key in ("game", "updated_at", "items"):
        if key not in doc:
            raise SystemExit(f"response is missing top-level {key!r} -- API shape changed?")
    items = doc["items"]
    if not items:
        raise SystemExit("response contains zero rule nodes")

    ids = {n["id"] for n in items}
    for n in items:
        missing = REQUIRED_NODE_FIELDS - set(n)
        if missing:
            raise SystemExit(f"node {n.get('id')} is missing fields {sorted(missing)}")
        if n["parent_id"] is not None and n["parent_id"] not in ids:
            raise SystemExit(
                f"node {n['id']} ({n.get('display_number')!r}) points at parent "
                f"{n['parent_id']}, which is not in the response -- the tree is "
                f"incomplete and its whole subtree would be dropped silently."
            )
    if len(ids) != len(items):
        raise SystemExit("response contains duplicate node ids")
    if not any(n["parent_id"] is None for n in items):
        raise SystemExit("no root node -- every node claims a parent")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--url", type=str, default=RULES_URL)
    ap.add_argument("--print-updated-at", action="store_true",
                    help="fetch and print the document's updated_at, write nothing")
    args = ap.parse_args()

    doc = fetch_json(args.url)
    validate(doc)

    if args.print_updated_at:
        print(doc["updated_at"])
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sections = sum(1 for n in doc["items"] if n["node_type"] == "section")
    rules = sum(1 for n in doc["items"] if n["node_type"] == "rule")
    print(f"wrote {len(doc['items'])} nodes ({sections} sections, {rules} rules) -> {args.out}")
    print(f"updated_at: {doc['updated_at']}")


if __name__ == "__main__":
    main()

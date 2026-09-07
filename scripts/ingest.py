#!/usr/bin/env python3
"""Parse the fetched Comprehensive Rules into the flat corpus files.

Input is `data/raw/comprehensive_rules.json` (see `scripts/fetch_rules.py`),
which is already a tree of typed nodes rather than prose. That is the whole
reason this script is short: there is no numbering heuristic, no indentation
parsing, and no PDF text extraction anywhere in it. The One Piece instance of
this template needs all three; this game publishes its rules as data.

Outputs, matching the shapes the rest of the harness expects:

  data/processed/rules.jsonl          one record per node, document order
  data/processed/rule_sections.jsonl  the outline alone (sections/headings)

A `rules.jsonl` record:

    {"id": "9.2.3", "kind": "rule", "section": "9. ATTACK, FIGHT, AND STEAL",
     "title": "...", "text": "9.2.3. ...", "anchor": "rule-<uuid>",
     "parent": "9.2", "depth": 3, "refs": ["9.2.4"]}

`text` is the citable form: the rule number, then the body rendered through
`games.cyberpunk.markup.to_plain_text`. Prefixing the number matters -- it is
what lets a retrieved chunk be cited by a model that only ever saw the chunk
text, rather than needing the id carried alongside out of band.

WHAT "DOCUMENT ORDER" MEANS HERE

The API returns nodes in no particular order and does NOT return a depth-first
traversal. Order is reconstructed by walking the tree from the roots, sorting
each sibling group by `sort_order` then `id` -- the exact comparison the
site's own renderer uses. Getting this wrong would not raise anything; it
would just quietly produce chunks that straddle unrelated rules, so
`scripts/test_cyberpunk.py` pins the resulting order against known
neighbours.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --report-unknown-markup   # beta keyword watch
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk import markup
from harness.core.io import write_jsonl_atomic, guard_shrink

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "comprehensive_rules.json"
DEFAULT_RULES_OUT = REPO_ROOT / "data" / "processed" / "rules.jsonl"
DEFAULT_SECTIONS_OUT = REPO_ROOT / "data" / "processed" / "rule_sections.jsonl"


def node_id(node: dict) -> str:
    """The citable id for a node: its rule number, or its anchor if unnumbered.

    Every `rule` node in this document is numbered; 42 `section` nodes are not
    (they are structural headings like "Official Rules Document"). Those fall
    back to their stable anchor so every record still has a unique key, and
    `is_numbered` records which kind it is rather than leaving a caller to
    guess from the shape of the string.
    """
    return node["display_number"] or node["stable_anchor"]


def walk(nodes: list[dict]) -> list[dict]:
    """Nodes in depth-first document order, roots first.

    Siblings are ordered by `sort_order`, ties broken by `id` -- the same
    comparison the official site's renderer applies.
    """
    children: dict = {}
    for n in nodes:
        children.setdefault(n["parent_id"], []).append(n)
    for group in children.values():
        group.sort(key=lambda n: (n["sort_order"], n["id"]))

    ordered: list[dict] = []

    def visit(parent_id):
        for n in children.get(parent_id, []):
            ordered.append(n)
            visit(n["id"])

    visit(None)
    if len(ordered) != len(nodes):
        # Unreachable given fetch_rules.py's parent-link validation, but a
        # cycle would silently drop a subtree and leave a corpus that looks
        # complete. Cheap to check, impossible to notice downstream.
        raise SystemExit(
            f"walk visited {len(ordered)} of {len(nodes)} nodes -- the tree has a "
            f"cycle or a disconnected component."
        )
    return ordered


def build_records(doc: dict) -> tuple[list[dict], list[dict]]:
    """`(rules, sections)` records from a fetched rules document."""
    nodes = doc["items"]
    ordered = walk(nodes)

    by_id = {n["id"]: n for n in nodes}
    by_anchor = {n["stable_anchor"]: n for n in nodes}

    # The top-level section a node belongs to, e.g. "9. ATTACK, FIGHT, AND
    # STEAL", carried onto every descendant so chunking can refuse to build a
    # chunk that straddles two of them.
    section_of: dict = {}

    def top_section(n: dict) -> str:
        if n["parent_id"] is None:
            number = n["display_number"]
            return f"{number}. {n['title']}" if number else n["title"]
        return section_of[n["parent_id"]]

    rules: list[dict] = []
    sections: list[dict] = []

    for n in ordered:
        section_of[n["id"]] = top_section(n)
        rid = node_id(n)
        parent = by_id.get(n["parent_id"])

        body = n["body_markdown"] or ""
        plain = markup.to_plain_text(body)
        number = n["display_number"]
        text = f"{number}. {plain}" if (number and plain) else (plain or n["title"])

        # Anchor links (`[GIGS](#rule-<uuid>)`, braced or bare) point at another
        # node. `to_plain_text` drops the target, since a DOM fragment is noise
        # in an embedding -- so resolve it to the destination's citable id here
        # and keep it as data. Cross-references to an UNNUMBERED section
        # resolve to that section's anchor rather than a rule number, because
        # an unnumbered section has no number to cite; those ids still join
        # against this file's own `id` column, which uses the same fallback.
        # External links (the gameplay guide, for instance) are not refs and
        # are dropped.
        anchor_refs = []
        for _label, target in markup.links(body):
            if not target.startswith("#"):
                continue
            dest = by_anchor.get(target[1:])
            if dest is not None:
                anchor_refs.append(node_id(dest))

        record = {
            "id": rid,
            "kind": n["node_type"],
            "is_numbered": bool(number),
            "section": section_of[n["id"]],
            "title": n["title"],
            "text": text,
            "anchor": n["stable_anchor"],
            "parent": node_id(parent) if parent else None,
            "depth": n["depth"],
            "refs": markup.rule_references(body) + anchor_refs,
        }
        rules.append(record)

        if n["node_type"] == "section":
            sections.append({
                "id": rid,
                "title": n["title"],
                "kind": "section" if n["parent_id"] is None else "subsection",
                "parent": node_id(parent) if parent else None,
                "depth": n["depth"],
            })

    return rules, sections


def report_unknown_markup(rules: list[dict], doc: dict) -> list[tuple[str, str]]:
    """`(rule_id, token)` for every brace span this repo does not recognise.

    See `games/cyberpunk/markup.py` on why an unknown token is reported rather
    than dropped: the card `keywords` API field is empty for every card, so
    brace markup is the only record of a keyword, and this game is mid-beta.
    """
    out = []
    for node in doc["items"]:
        for token in markup.unknown_tokens(node["body_markdown"] or ""):
            out.append((node_id(node), token))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--rules-output", type=Path, default=DEFAULT_RULES_OUT)
    ap.add_argument("--sections-output", type=Path, default=DEFAULT_SECTIONS_OUT)
    ap.add_argument("--report-unknown-markup", action="store_true",
                    help="list brace spans games/cyberpunk/markup.py does not know")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found -- run scripts/fetch_rules.py first")
    doc = json.loads(args.input.read_text(encoding="utf-8"))

    rules, sections = build_records(doc)

    unknown = report_unknown_markup(rules, doc)
    if unknown:
        print(f"WARNING: {len(unknown)} unrecognised markup span(s) -- "
              f"games/cyberpunk/markup.py may need a new keyword:")
        for rid, token in unknown[:20]:
            print(f"  {rid}: {{{token}}}")
    if args.report_unknown_markup:
        if not unknown:
            print("no unrecognised markup -- the vocabulary covers the corpus")
        return

    guard_shrink(args.rules_output, len(rules), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.rules_output, rules)
    print(f"wrote {len(rules)} rule records -> {args.rules_output}")

    guard_shrink(args.sections_output, len(sections), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.sections_output, sections)
    print(f"wrote {len(sections)} section records -> {args.sections_output}")
    print(f"rules document updated_at: {doc['updated_at']}")


if __name__ == "__main__":
    main()

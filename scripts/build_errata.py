#!/usr/bin/env python3
"""Build the errata records from the fetched errata page.

Input is `data/raw/errata.json` (see `scripts/fetch_errata.py`) plus
`data/processed/card_database.jsonl` for the join; output is
`data/processed/errata.jsonl`, one record per erratum in document order.

The parsing and the join live in `games/cyberpunk/errata.py`; this script is
the CLI around them, and the place the card-database cross-check happens.

WHY THIS REPORTS WHICH ERRATA THE CARD API HAS NOT APPLIED

The card API applies most errata to the card record itself, but not all of
them -- at the launch set, three of four were applied and one (Nocturne
OP55 N1's artist credit) was not. Which ones lag is not knowable from either
source alone, and it changes without notice. So `--check-applied` re-derives
the comparison from live data every time rather than trusting a note in a
docstring, and prints what it finds.

Usage:
    python scripts/build_errata.py
    python scripts/build_errata.py --check-applied   # what has the card API not caught up on?
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk import errata as errata_mod
from harness.core.io import read_jsonl, write_jsonl_atomic, guard_shrink

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "errata.json"
DEFAULT_CARDS = REPO_ROOT / "data" / "processed" / "card_database.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "errata.jsonl"


def build_records(doc: dict, cards: list[dict]) -> list[dict]:
    joined = errata_mod.join_to_cards(errata_mod.parse(doc), cards)
    updated_at = doc["items"][0]["sys"]["updatedAt"]
    out = []
    for i, entry in enumerate(joined, 1):
        out.append({
            # Stable and readable: the card it applies to, plus an ordinal for
            # the case where one card earns a second erratum later. Not the
            # CMS entry id, which is per-block rather than per-erratum.
            "id": f"errata:{entry['card_id']}:{i:02d}",
            "card_id": entry["card_id"],
            "card_name": entry["card_name"],
            "variant": entry["variant"],
            "section": entry["section"],
            "heading": entry["heading"],
            "text": entry["text"],
            "images": entry["images"],
            "page_updated_at": updated_at,
        })
    return out


def check_applied(records: list[dict], cards: list[dict]) -> None:
    """Print, per erratum, whether the card record already reflects it.

    Deliberately a REPORT and not an assertion. Whether the API has caught up
    is a fact about the upstream data, not a property this repo controls, so
    failing a build on it would block work for a reason nobody here can fix.
    What matters is that it is visible and dated.
    """
    import textwrap

    by_id = {c["id"]: c for c in cards}
    print(f"{len(records)} erratum/errata, each against the card record it corrects.\n")
    for rec in records:
        card = by_id.get(rec["card_id"])
        print(f"=== {rec['heading']} -> {rec['card_id']} ===")
        # The erratum's own wording, printed HERE rather than left for the
        # reader to find elsewhere -- the comparison is the whole point of
        # this command, and it cannot be made against text that is not shown.
        errata_text = " ".join(rec["text"].split())
        for line in textwrap.wrap(errata_text, 76, initial_indent="  errata:  ",
                                  subsequent_indent="           "):
            print(line)
        if card is None:
            print("  card:    NOT IN THE DATABASE")
            print()
            continue
        print(f"  card:    artist={card.get('artist')!r}  "
              f"sellable={card.get('is_eddiable')}")
        card_text = " ".join((card.get("text") or "").split())
        for line in textwrap.wrap(card_text, 76, initial_indent="           text: ",
                                  subsequent_indent="                 "):
            print(line)
        print()
    print("The card API applies most errata to the card record, but has been "
          "observed not to apply all of them -- read each pair above rather "
          "than assuming either source is authoritative on its own.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-applied", action="store_true",
                    help="report whether the card database reflects each erratum")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found -- run scripts/fetch_errata.py first")
    doc = json.loads(args.input.read_text(encoding="utf-8"))
    cards = read_jsonl(args.cards, missing_ok=False)

    records = build_records(doc, cards)

    if args.check_applied:
        check_applied(records, cards)
        return

    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate errata ids: {sorted(ids)}")

    guard_shrink(args.output, len(records), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.output, records)
    print(f"wrote {len(records)} errata -> {args.output}")
    print(f"  covering {len({r['card_id'] for r in records})} card(s); "
          f"page updated {records[0]['page_updated_at'] if records else 'n/a'}")


if __name__ == "__main__":
    main()

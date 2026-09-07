#!/usr/bin/env python3
"""Build the structured card database from the fetched card JSON.

Input is one file per card under `data/raw/cards/` plus `_index.json` (see
`scripts/fetch_cards.py`); output is `data/processed/card_database.jsonl`, one
record per card in index order. The normalization itself lives in
`games/cyberpunk/cards.py` -- this script is the CLI around it, and the place
where corpus-level consistency is checked.

WHY THE INDEX IS THE AUTHORITY, NOT THE DIRECTORY LISTING

Building from `data/raw/cards/*.json` alone would silently include a card the
catalog no longer lists -- a stale file left behind when a beta card is pulled
or renamed. It would also silently EXCLUDE a card whose fetch failed, and the
resulting database would look perfectly well-formed. So the index's slug list
decides what belongs in the corpus, a missing file is a hard error, and an
extra file on disk is reported.

Usage:
    python scripts/build_cards.py
    python scripts/build_cards.py --report-unknown-markup   # beta keyword watch
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk import markup
from games.cyberpunk.cards import parse_card
from harness.core.io import write_jsonl_atomic, guard_shrink

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "cards"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "card_database.jsonl"
INDEX_NAME = "_index.json"


def load_raw_cards(raw_dir: Path) -> list[dict]:
    """Every card named by the index, in index order.

    Raises `SystemExit` rather than skipping if a listed card has no file:
    see this module's docstring on why a quietly short catalog is the failure
    worth being loud about.
    """
    index_path = raw_dir / INDEX_NAME
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found -- run scripts/fetch_cards.py first")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    slugs = [c["slug"] for c in index["items"]]
    missing = [s for s in slugs if not (raw_dir / f"{s}.json").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} card(s) in the index have no file on disk "
            f"(e.g. {missing[:3]}) -- re-run scripts/fetch_cards.py")

    on_disk = {p.stem for p in raw_dir.glob("*.json") if not p.stem.startswith("_")}
    extra = sorted(on_disk - set(slugs))
    if extra:
        print(f"WARNING: {len(extra)} card file(s) on disk are not in the index and "
              f"will be excluded: {extra[:5]}")

    return [json.loads((raw_dir / f"{s}.json").read_text(encoding="utf-8")) for s in slugs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-unknown-markup", action="store_true",
                    help="list brace spans games/cyberpunk/markup.py does not know")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw = load_raw_cards(args.raw_dir)
    cards = [parse_card(r) for r in raw]

    ids = [c["id"] for c in cards]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate card ids -- the index contains a repeated slug")

    unknown = [(c["id"], t) for c in cards
               for t in markup.unknown_tokens(c["text_markup"])]
    if unknown:
        print(f"WARNING: {len(unknown)} unrecognised markup span(s) -- a new keyword "
              f"may have shipped, and games/cyberpunk/markup.py needs it:")
        for cid, token in unknown[:20]:
            print(f"  {cid}: {{{token}}}")
    if args.report_unknown_markup:
        if not unknown:
            print(f"no unrecognised markup across {len(cards)} cards -- "
                  f"the vocabulary covers the corpus")
        return

    guard_shrink(args.output, len(cards), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.output, cards)
    print(f"wrote {len(cards)} cards -> {args.output}")

    with_keywords = sum(1 for c in cards if c["keywords"])
    print(f"  {with_keywords} card(s) carry a printed keyword "
          f"(the API's own `keywords` field is empty on all {len(cards)})")


if __name__ == "__main__":
    main()

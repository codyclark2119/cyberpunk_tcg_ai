#!/usr/bin/env python3
"""Cluster the community decklists into archetypes, with its own null.

Reads the fetched decks (`scripts/fetch_decks.py`) and reports, in this order:

  1. the NULL comparison -- how much similarity structure survives replacing
     every deck with a random legal deck of the same colour identity and size
  2. the SENSITIVITY curve -- how the archetype count moves with the
     threshold, because that number is a property of the knob, not of the game
  3. the archetypes themselves at one chosen threshold

The order is deliberate. A list of archetypes is persuasive whether or not it
means anything, so the evidence that it means something is printed before it
rather than after, and the null is not behind a flag.

`games/cyberpunk/archetypes.py` holds the method and the reasoning behind
each choice (weighted Jaccard over copy counts, average linkage, a threshold
rather than a chosen k).

This is a REPORT, not a test: it depends on how many decks happen to be public
today, and archetypes are a description of a moving format rather than a fact
to regress against. `scripts/test_cyberpunk.py` covers the clustering
mechanics on synthetic decks instead, with no network and no corpus.

Usage:
    python scripts/cluster_decks.py
    python scripts/cluster_decks.py --threshold 0.5 --min-size 3
    python scripts/cluster_decks.py --no-null           # skip the null, if rerunning
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk import archetypes
from harness.core.io import read_jsonl

DEFAULT_DECK_DIR = REPO_ROOT / "data" / "raw" / "decks"
DEFAULT_CARDS = REPO_ROOT / "data" / "processed" / "card_database.jsonl"
SENSITIVITY_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.60)


def load_decks(deck_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(deck_dir.glob("*.json")) if not p.name.startswith("_")]


def name_of(slug: str, card_index: dict) -> str:
    card = card_index.get(slug)
    return card["display_name"] if card else slug


def report_null(decks, counts, card_index, matrix, seed):
    rng = random.Random(seed)
    null_counts = archetypes.null_decks(decks, card_index, rng)
    null_matrix = archetypes.similarity_matrix(null_counts)
    print("=== is this structure, or is it just colour and deck size? ===")
    print("    null = every deck replaced by a random legal deck of the SAME")
    print("    colour identity, size and copy-count shape.\n")
    print(f"    {'similarity':>12}  {'real pairs':>10}  {'null pairs':>10}")
    for t in (0.4, 0.5, 0.6, 0.7):
        print(f"    {'>= ' + format(t, '.1f'):>12}  "
              f"{archetypes.pairs_above(matrix, t):>10}  "
              f"{archetypes.pairs_above(null_matrix, t):>10}")
    print()


def report_sensitivity(matrix, min_size):
    print("=== how many archetypes? that depends on the threshold ===\n")
    print(f"    {'threshold':>10}  {'clusters':>9}  {'>= ' + str(min_size) + ' decks':>11}  "
          f"{'largest':>8}  {'singletons':>11}")
    for t in SENSITIVITY_THRESHOLDS:
        clusters = archetypes.cluster(matrix, t)
        print(f"    {t:>10.2f}  {len(clusters):>9}  "
              f"{sum(1 for c in clusters if len(c) >= min_size):>11}  "
              f"{max(len(c) for c in clusters):>8}  "
              f"{sum(1 for c in clusters if len(c) == 1):>11}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck-dir", type=Path, default=DEFAULT_DECK_DIR)
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--threshold", type=float, default=archetypes.DEFAULT_THRESHOLD)
    ap.add_argument("--min-size", type=int, default=3,
                    help="only describe clusters with at least this many decks")
    ap.add_argument("--core-share", type=float, default=archetypes.DEFAULT_CORE_SHARE)
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed for the null model")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="also write the archetypes as jsonl")
    args = ap.parse_args()

    decks = load_decks(args.deck_dir)
    if not decks:
        raise SystemExit(f"no decks under {args.deck_dir} -- run scripts/fetch_decks.py")
    card_index = {c["id"]: c for c in read_jsonl(args.cards, missing_ok=False)}

    counts = [archetypes.deck_counts(d) for d in decks]
    matrix = archetypes.similarity_matrix(counts)
    print(f"{len(decks)} community decks\n")

    dupes = archetypes.duplicate_groups(counts)
    if dupes:
        print("=== decks that are byte-for-byte the same list ===")
        print("    (kept, not removed: a list worth copying is itself a finding)\n")
        for group in dupes:
            names = [decks[i].get("name") for i in group]
            owners = {decks[i]["owner"]["owner_pseudonym"] for i in group}
            print(f"    {len(group)}x  {names}  ({len(owners)} distinct author(s))")
        print()

    if not args.no_null:
        report_null(decks, counts, card_index, matrix, args.seed)
    report_sensitivity(matrix, args.min_size)

    clusters = archetypes.cluster(matrix, args.threshold)
    described = [c for c in clusters if len(c) >= args.min_size]
    print(f"=== archetypes at threshold {args.threshold:.2f} "
          f"({len(described)} with >= {args.min_size} decks, "
          f"{len(clusters) - len(described)} smaller) ===\n")

    records = []
    for n, members in enumerate(described, 1):
        colours = Counter(
            "/".join(sorted(archetypes.colour_identity(decks[i], card_index)))
            for i in members)
        legends = Counter(name_of(s, card_index)
                          for i in members for s in archetypes.legend_slugs(decks[i]))
        core = archetypes.core_cards(members, counts, args.core_share)
        arch_id = f"arch-{n:02d}"
        print(f"  {arch_id}  {len(members)} decks   "
              f"cohesion {archetypes.cohesion(members, matrix):.2f}")
        print(f"      colours:  {', '.join(f'{c} ({k})' for c, k in colours.most_common(4))}")
        print(f"      Legends:  {', '.join(f'{l} ({k})' for l, k in legends.most_common(3))}")
        # " | " and not ", ": one launch-set card is called "Three Mouths, One
        # Desire", so a comma-joined list reads as one card more than it holds.
        print(f"      core ({len(core)} cards in >= {args.core_share:.0%} of members):")
        print(f"        {' | '.join(name_of(s, card_index) for s, _ in core[:8])}")
        print(f"      names:    {[decks[i].get('name') for i in members[:4]]}")
        print()
        records.append({
            "id": arch_id,
            "size": len(members),
            "cohesion": round(archetypes.cohesion(members, matrix), 4),
            "threshold": args.threshold,
            "colours": dict(colours),
            "legends": dict(legends),
            "core_cards": [{"card_id": s, "share": round(sh, 4)} for s, sh in core],
            "deck_ids": [decks[i]["id"] for i in members],
        })

    if args.json_out:
        from harness.core.io import write_jsonl_atomic
        write_jsonl_atomic(args.json_out, records)
        print(f"wrote {len(records)} archetypes -> {args.json_out}")


if __name__ == "__main__":
    main()

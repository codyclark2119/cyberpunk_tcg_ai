#!/usr/bin/env python3
"""Check this repo's deckbuilding rules against the official builder's verdict.

Every fetched community deck carries two things this repo did not compute: the
API's own `is_valid` boolean, and a `stats` block holding `ramBudget` and
`ramUsage` per colour. That makes a few hundred real decks into an INDEPENDENT
LABEL SET for `games/cyberpunk/deckbuilding.py` -- a rules implementation
checked against something other than its author's own reading of the guide.

## The agreement number alone is close to meaningless -- read both halves

Every one of the fetched decks is `is_valid: true`. The official builder
refuses to save an illegal deck, so the public corpus contains no negative
examples at all, and a validator that returned "legal" unconditionally would
also agree with it on every single deck. Quoting that number on its own would
be exactly the one-sided call `harness/core/calibration/controls.py`'s
`separation()` refuses to make.

So this script reports two things that are worth something, and labels the
one that is not:

  - `--negative-control` breaks each real deck in each specific way a rule
    forbids and checks the validator fires. That is the P(fire | broken) half
    the corpus cannot supply, and it is the half that can actually fail.
  - The RAM arithmetic check is not a boolean: it recomputes `ramBudget` and
    `ramUsage` per colour and compares them to the official builder's own
    numbers, which a constant answer cannot fake.

This script is that comparison, and it is deliberately a REPORT rather than a
test. Two reasons:

  - A disagreement is not automatically this repo's bug. The official builder
    may enforce a format restriction this repo does not model, or may have
    saved a deck before a rule changed. What matters is that every
    disagreement is enumerated and looked at, not that the number is zero.
  - It needs the fetched decks, which are a dated snapshot of user content.
    `scripts/test_cyberpunk.py` stays runnable with no network and no
    dependency on how many decks happen to be public today.

The agreement rate this produced when the rules were written is recorded in
`CLAUDE.md`; re-run it after changing anything in `deckbuilding.py`.

Usage:
    python scripts/check_deck_rules.py
    python scripts/check_deck_rules.py --show-disagreements
    python scripts/check_deck_rules.py --negative-control
"""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk import deckbuilding
from harness.core.io import read_jsonl

DEFAULT_DECK_DIR = REPO_ROOT / "data" / "raw" / "decks"
DEFAULT_CARDS = REPO_ROOT / "data" / "processed" / "card_database.jsonl"


def load_decks(deck_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(deck_dir.glob("*.json"))
            if not p.name.startswith("_")]


def _zone(deck: dict, code: str) -> dict | None:
    for zone in deck.get("zones") or []:
        if zone.get("zone_code") == code:
            return zone
    return None


# Each breaker takes a LEGAL deck and returns an illegal copy, or None when
# that deck cannot be broken that particular way. Returning None matters:
# counting an unbreakable deck as a miss would understate the catch rate for a
# reason that has nothing to do with the validator.
def _break_legend_count(deck, card_index):
    out = copy.deepcopy(deck)
    zone = _zone(out, deckbuilding.LEGEND_ZONE)
    if not zone or len(zone["cards"]) < 2:
        return None
    zone["cards"].pop()
    return out


def _break_legend_names(deck, card_index):
    out = copy.deepcopy(deck)
    zone = _zone(out, deckbuilding.LEGEND_ZONE)
    if not zone or len(zone["cards"]) != 3:
        return None
    # Two copies of one Legend plus one other: still three Legends, but two
    # share a name. This is the case a naive count-only check misses.
    zone["cards"] = [dict(zone["cards"][0], quantity=2), zone["cards"][1]]
    return out


def _break_deck_size(deck, card_index):
    out = copy.deepcopy(deck)
    zone = _zone(out, deckbuilding.DECK_ZONE)
    if not zone or not zone["cards"]:
        return None
    while deckbuilding.total_cards(zone["cards"]) >= deckbuilding.DECK_MIN:
        if not zone["cards"]:
            return None
        zone["cards"].pop()
    return out


def _break_max_copies(deck, card_index):
    out = copy.deepcopy(deck)
    zone = _zone(out, deckbuilding.DECK_ZONE)
    if not zone or not zone["cards"]:
        return None
    zone["cards"][0] = dict(zone["cards"][0], quantity=deckbuilding.MAX_COPIES + 1)
    return out


def _break_ram_limit(deck, card_index):
    """Swap in a card whose RAM exceeds the budget in its own colour."""
    out = copy.deepcopy(deck)
    zone = _zone(out, deckbuilding.DECK_ZONE)
    legends = deckbuilding.legend_entries(out)
    if not zone or not zone["cards"]:
        return None
    budget = deckbuilding.ram_budget(legends, card_index)
    for colour, have in budget.items():
        too_big = next((c for c in card_index.values()
                        if c["color"] == colour and (c.get("ram") or 0) > have), None)
        if too_big is not None:
            zone["cards"][0] = dict(zone["cards"][0], card_slug=too_big["id"], quantity=1)
            return out
    return None


BREAKERS = {
    "legend-count": _break_legend_count,
    "legend-unique-names": _break_legend_names,
    "deck-size": _break_deck_size,
    "max-copies": _break_max_copies,
    "ram-limit": _break_ram_limit,
}


def negative_control(decks: list[dict], card_index: dict) -> None:
    """P(fire | broken), per rule -- the half the public corpus cannot supply.

    Reports both the catch rate AND whether the intended rule is the one that
    fired. A breaker that trips a different rule would otherwise look like a
    pass while proving nothing about the rule it was named for.
    """
    print("\n=== negative control: every legal deck, broken one rule at a time ===")
    print("    (the public corpus is 100% valid, so this is the only half that "
          "can fail)\n")
    for rule, breaker in BREAKERS.items():
        attempted = caught = right_rule = 0
        for deck in decks:
            broken = breaker(deck, card_index)
            if broken is None:
                continue
            attempted += 1
            violations = deckbuilding.validate(broken, card_index)
            if violations:
                caught += 1
                if any(v["rule"] == rule for v in violations):
                    right_rule += 1
        if not attempted:
            print(f"  {rule:22s} NOT EXERCISED -- no deck could be broken this way")
            continue
        flag = "" if right_rule == attempted else "   <-- LOOK"
        print(f"  {rule:22s} caught {caught}/{attempted}, "
              f"by the intended rule {right_rule}/{attempted}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck-dir", type=Path, default=DEFAULT_DECK_DIR)
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--show-disagreements", action="store_true")
    ap.add_argument("--negative-control", action="store_true",
                    help="break every legal deck each way a rule forbids, and "
                         "report whether the validator fires")
    args = ap.parse_args()

    decks = load_decks(args.deck_dir)
    if not decks:
        raise SystemExit(f"no decks under {args.deck_dir} -- run scripts/fetch_decks.py")
    card_index = {c["id"]: c for c in read_jsonl(args.cards, missing_ok=False)}

    agree = disagree = 0
    ram_agree = ram_disagree = 0
    we_say_illegal_they_say_legal = []
    we_say_legal_they_say_illegal = []
    rule_counts: Counter = Counter()

    for deck in decks:
        violations = deckbuilding.validate(deck, card_index)
        ours_valid = not violations
        theirs_valid = bool(deck.get("is_valid"))
        for v in violations:
            rule_counts[v["rule"]] += 1

        if ours_valid == theirs_valid:
            agree += 1
        else:
            disagree += 1
            row = (deck["id"], deck.get("name"), violations)
            (we_say_illegal_they_say_legal if theirs_valid
             else we_say_legal_they_say_illegal).append(row)

        # The arithmetic check, independent of the boolean: do our budget and
        # usage numbers reproduce the official builder's?
        stats = deck.get("stats") or {}
        ours = (deckbuilding.ram_budget(deckbuilding.legend_entries(deck), card_index),
                deckbuilding.ram_usage(deckbuilding.deck_entries(deck), card_index))
        theirs = (stats.get("ramBudget"), stats.get("ramUsage"))
        if theirs[0] is None:
            continue
        if ours[0] == theirs[0] and ours[1] == theirs[1]:
            ram_agree += 1
        else:
            ram_disagree += 1
            if args.show_disagreements:
                print(f"  RAM mismatch {deck['id']} ({deck.get('name')!r})")
                print(f"    ours   budget={ours[0]} usage={ours[1]}")
                print(f"    theirs budget={theirs[0]} usage={theirs[1]}")

    total = len(decks)
    print(f"{total} community decks\n")
    labels = Counter(bool(d.get("is_valid")) for d in decks)
    print(f"=== the label distribution, first, because it caps what this proves ===")
    print(f"  is_valid true: {labels[True]}   false: {labels[False]}")
    if not labels[False]:
        print("  NO ILLEGAL DECKS IN THE CORPUS. The official builder refuses to")
        print("  save one, so agreement below is one-sided: a validator that")
        print("  always answered 'legal' would score the same. Run")
        print("  --negative-control for the half that can actually fail.")
    print()
    print("=== validity, ours vs. the official builder's is_valid ===")
    print(f"  agree:    {agree}/{total}")
    print(f"  disagree: {disagree}/{total}")
    if disagree:
        print(f"    we say ILLEGAL, they say legal: {len(we_say_illegal_they_say_legal)}")
        print(f"    we say LEGAL, they say illegal: {len(we_say_legal_they_say_illegal)}")

    print("\n=== RAM arithmetic, ours vs. their stats block ===")
    print(f"  reproduce budget AND usage exactly: {ram_agree}/{ram_agree + ram_disagree}")

    print("\n=== violations we found, by rule ===")
    for rule, n in rule_counts.most_common():
        print(f"  {n:4d}  {rule}")
    if not rule_counts:
        print("  (none)")

    if args.negative_control:
        negative_control(decks, card_index)

    if args.show_disagreements:
        for label, rows in (("we say ILLEGAL, they say legal", we_say_illegal_they_say_legal),
                            ("we say LEGAL, they say illegal", we_say_legal_they_say_illegal)):
            if not rows:
                continue
            print(f"\n=== {label} ===")
            for deck_id, name, violations in rows[:25]:
                print(f"  {deck_id}  {name!r}")
                for v in violations:
                    print(f"      {v['rule']}: {v['detail']}")


if __name__ == "__main__":
    main()

"""Cyberpunk TCG deck construction rules, and a validator for them.

The rules, from the official gameplay guide's "Deck Building & RAM" section:

  - exactly 3 Legend cards, with unique names
  - no fewer than 40 and no more than 50 cards, NOT counting the Legends
  - no more than 3 copies of the same card
  - every card must stay within the RAM limit its Legends provide

## The RAM rule, stated precisely

This is the rule most easily got subtly wrong, so it is worth writing out.
Each Legend supplies RAM in ONE colour. A deck's budget in a colour is the
SUM of its Legends' RAM in that colour. A deck card's demand is its own RAM
value -- and the constraint is on the MAXIMUM demand in each colour, not the
total:

    for every colour c:  max(card.ram for deck cards of colour c) <= budget[c]

So a Yellow budget of 4 admits any number of Yellow cards with RAM <= 4; it
is a ceiling per card, not a pool that spends down. "Cards can only be
included in decks with sufficient RAM. ... The RAM color values on each card
in your deck must be less than or equal to your Legends' total RAM color
values."

This reading is not just the guide's wording -- it was checked against the
official deckbuilder's own arithmetic. Every public deck's API record carries
a `stats` block with `ramBudget` and `ramUsage` per colour; recomputing both
from the deck's cards reproduces them exactly, with `ramUsage` being the max
and `ramBudget` the sum. See `scripts/check_deck_rules.py`, which re-runs that
comparison across every fetched deck rather than trusting this docstring.

## Why the validator returns violations rather than a bool

A deckbuilding assistant has to say WHY a deck is illegal, and a bare False
cannot be turned back into a reason. `validate` returns a list of structured
violations, empty for a legal deck, so a caller can render them, count them
by kind, or compare them against the API's own verdict.
"""

from collections import Counter

LEGEND_COUNT = 3
DECK_MIN = 40
DECK_MAX = 50
MAX_COPIES = 3

LEGEND_ZONE = "legends"
DECK_ZONE = "deck"


def _entries(deck: dict, zone_code: str) -> list[dict]:
    for zone in deck.get("zones") or []:
        if zone.get("zone_code") == zone_code:
            return zone.get("cards") or []
    return []


def legend_entries(deck: dict) -> list[dict]:
    return _entries(deck, LEGEND_ZONE)


def deck_entries(deck: dict) -> list[dict]:
    return _entries(deck, DECK_ZONE)


def _card_of(entry: dict, card_index: dict) -> dict | None:
    return card_index.get(entry.get("card_slug"))


def ram_budget(legends: list[dict], card_index: dict) -> dict:
    """Colour -> total RAM the Legends supply in it (a sum)."""
    budget: Counter = Counter()
    for entry in legends:
        card = _card_of(entry, card_index)
        if card is None:
            continue
        budget[card["color"]] += (card.get("ram") or 0) * entry.get("quantity", 1)
    return dict(budget)


def ram_usage(cards: list[dict], card_index: dict) -> dict:
    """Colour -> the LARGEST single-card RAM demand in it (a max, not a sum).

    Quantity is irrelevant here: three copies of a RAM-2 card demand 2, not 6.
    """
    usage: dict = {}
    for entry in cards:
        card = _card_of(entry, card_index)
        if card is None:
            continue
        colour = card["color"]
        usage[colour] = max(usage.get(colour, 0), card.get("ram") or 0)
    return usage


def total_cards(entries: list[dict]) -> int:
    return sum(e.get("quantity", 0) for e in entries)


def validate(deck: dict, card_index: dict) -> list[dict]:
    """Every deck-construction rule this deck breaks. Empty means legal.

    Each violation is `{"rule": <slug>, "detail": <human sentence>}`. The rule
    slugs are stable, so a caller can count violations by kind without parsing
    prose.
    """
    violations: list[dict] = []
    legends = legend_entries(deck)
    main = deck_entries(deck)

    unknown = sorted({e.get("card_slug") for e in legends + main
                      if e.get("card_slug") not in card_index})
    if unknown:
        # Reported as a violation rather than raised: a deck naming a card the
        # local catalog does not have is a fact about this snapshot, and the
        # remaining rules are still worth checking.
        violations.append({
            "rule": "unknown-card",
            "detail": f"{len(unknown)} card(s) are not in the card database: "
                      f"{unknown[:5]}",
        })

    n_legends = total_cards(legends)
    if n_legends != LEGEND_COUNT:
        violations.append({
            "rule": "legend-count",
            "detail": f"a deck must have exactly {LEGEND_COUNT} Legends, this has {n_legends}",
        })

    # Expanded by QUANTITY, not one name per entry. Two copies of one Legend
    # is a single entry with `quantity: 2`, so counting entries finds no
    # duplicate name and the rule silently never fires -- which is exactly
    # what it did, undetected, while agreeing with the official builder on all
    # 272 public decks. Only the negative control in
    # `scripts/check_deck_rules.py` caught it: the corpus has no illegal decks
    # to disagree about.
    names: list[str] = []
    for entry in legends:
        card = _card_of(entry, card_index)
        if card is not None:
            names.extend([card["name"]] * entry.get("quantity", 1))
    duplicated = sorted({n for n, k in Counter(names).items() if k > 1})
    if duplicated:
        violations.append({
            "rule": "legend-unique-names",
            "detail": f"Legends must have unique names; repeated: {duplicated}",
        })

    n_main = total_cards(main)
    if not (DECK_MIN <= n_main <= DECK_MAX):
        violations.append({
            "rule": "deck-size",
            "detail": f"a deck must hold {DECK_MIN}-{DECK_MAX} cards excluding "
                      f"Legends, this holds {n_main}",
        })

    over = sorted(e["card_slug"] for e in main if e.get("quantity", 0) > MAX_COPIES)
    if over:
        violations.append({
            "rule": "max-copies",
            "detail": f"no more than {MAX_COPIES} copies of a card; over the limit: {over}",
        })

    budget = ram_budget(legends, card_index)
    usage = ram_usage(main, card_index)
    over_ram = sorted(
        (colour, demand, budget.get(colour, 0))
        for colour, demand in usage.items() if demand > budget.get(colour, 0))
    if over_ram:
        detail = "; ".join(
            f"{colour} needs RAM {demand} but the Legends supply {have}"
            for colour, demand, have in over_ram)
        violations.append({"rule": "ram-limit", "detail": detail})

    return violations


def describe(deck: dict, card_index: dict) -> dict:
    """The numbers a deckbuilding answer usually needs, in one call."""
    legends = legend_entries(deck)
    main = deck_entries(deck)
    types: Counter = Counter()
    colours: Counter = Counter()
    for entry in main:
        card = _card_of(entry, card_index)
        if card is None:
            continue
        types[card["card_type"]] += entry.get("quantity", 0)
        colours[card["color"]] += entry.get("quantity", 0)
    return {
        "legend_count": total_cards(legends),
        "deck_count": total_cards(main),
        "type_counts": dict(types),
        "color_counts": dict(colours),
        "ram_budget": ram_budget(legends, card_index),
        "ram_usage": ram_usage(main, card_index),
        "violations": validate(deck, card_index),
    }

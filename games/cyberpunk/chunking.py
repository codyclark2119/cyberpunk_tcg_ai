"""Chunk the Cyberpunk TCG rules corpus and card database for retrieval.

Three outputs: rule chunks, card chunks, and the two concatenated into
`corpus_chunks.jsonl`, which is what the retrieval index is actually built
over.

WHY ONE MERGED INDEX IS ENOUGH HERE, AND WHEN IT STOPS BEING

The One Piece instance of this template cannot use a single index: its card
chunks outnumber its rule chunks 48:1 (2,767 to 58), so card text wins nearly
every slot in a fixed retrieval budget even on a pure rules question, and it
needs a dual-budget router to fix it. This game's launch set is 151 card
chunks against 74 rule chunks -- 2:1 -- so a plain top-k over one index still
returns a mix, and shipping a router now would mean shipping an untested
mechanism to solve a problem this corpus does not have.

That changes as sets ship. The two per-source files are written separately,
and stay written separately, precisely so a router can be added later without
re-shaping the corpus first: once cards outnumber rules by something like 10:1
-- roughly 750 cards, five or six sets -- re-measure retrieval on rules-only
questions before trusting the merged index further.

RULES CHUNKING

Rule nodes arrive from `scripts/ingest.py` already in document order, with a
`section` field naming the top-level section each belongs to. Chunking is a
greedy pack capped at `--max-chars`, flushed whenever the section changes, so
a chunk never straddles two top-level sections and the outline's locality
survives into retrieval.

Heading-only nodes (a `section` node carries an empty body) are packed along
with the rules beneath them rather than dropped: a chunk that opens with
"9.2 Declare Step" is substantially easier to retrieve for a question about
declaring an attack than one that opens mid-rule.

CARD CHUNKING

Cards are already atomic -- one card is one chunk. `card_to_text` reshapes a
parsed card record into the `{"chunk_id", "text"}` shape
`harness.core.rag.index` expects.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.core.io import read_jsonl, write_jsonl_atomic, guard_shrink

DEFAULT_RULES_MAX_CHARS = 1200


def chunk_rules(rules: list[dict], max_chars: int = DEFAULT_RULES_MAX_CHARS) -> list[dict]:
    """Group consecutive same-section rules into <= `max_chars` chunks.

    `rules` must be in document order (`scripts/ingest.py` guarantees it by
    walking the tree, not by trusting the API's response order).
    """
    chunks: list[dict] = []
    current_section = None
    current_ids: list[str] = []
    current_texts: list[str] = []
    current_len = 0

    def flush():
        nonlocal current_ids, current_texts, current_len
        if current_ids:
            chunk_id = (f"rule:{current_ids[0]}-{current_ids[-1]}"
                        if len(current_ids) > 1 else f"rule:{current_ids[0]}")
            chunks.append({
                "chunk_id": chunk_id,
                "text": "\n".join(current_texts),
                "rule_ids": list(current_ids),
                "section": current_section,
                "source": "rules",
            })
        current_ids = []
        current_texts = []
        current_len = 0

    for rule in rules:
        text = rule["text"]
        section = rule["section"]
        if section != current_section or current_len + len(text) + 1 > max_chars:
            flush()
            current_section = section
        current_ids.append(rule["id"])
        current_texts.append(text)
        current_len += len(text) + 1

    flush()
    return chunks


def card_to_text(card: dict) -> str:
    """Render one parsed card record as retrievable prose.

    Field order mirrors how a player reads the physical card: name and
    subtitle, then the type line, then the stat line, then tags, then rules
    text. Stats that a card genuinely does not have are omitted rather than
    written as "null" -- 34 Programs have no power at all, and 19 Legends have
    neither cost nor power because they are never played from hand as Units.

    Keywords are stated explicitly even though they also appear inside the
    rules text, because the rules text spells them as `[Blocker]` inline while
    a question is far more likely to ask "which Units have Blocker". Only
    PRINTED keywords go on that line; a card that merely refers to a keyword
    is not listed as having it (see `games/cyberpunk/markup.py`).

    Printings are deliberately NOT rendered. A card in six printings would
    otherwise carry six near-identical set/rarity lines into its embedding,
    diluting the text that actually distinguishes it; the printing list stays
    in `card_database.jsonl` for lookups that need it.
    """
    header = card["display_name"]
    lines = [f"{header} ({card['card_type']}, {card['color']})"]

    stat_bits = []
    if card.get("cost") is not None:
        stat_bits.append(f"Cost {card['cost']}")
    if card.get("power") is not None:
        stat_bits.append(f"Power {card['power']}")
    if card.get("ram") is not None:
        stat_bits.append(f"RAM {card['ram']}")
    if card.get("is_eddiable") is not None:
        stat_bits.append("Sellable for Eddies" if card["is_eddiable"]
                         else "Not sellable for Eddies")
    if stat_bits:
        lines.append(" | ".join(stat_bits))

    if card.get("classifications"):
        lines.append(f"Tags: {', '.join(card['classifications'])}")
    if card.get("keywords"):
        lines.append(f"Keywords: {', '.join(card['keywords'])}")
    if card.get("timing_triggers"):
        lines.append(f"Triggers: {', '.join(card['timing_triggers'])}")
    if card.get("text"):
        lines.append(f"Rules text: {card['text']}")

    return "\n".join(lines)


def chunk_cards(cards: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": f"card:{card['id']}",
            "text": card_to_text(card),
            "card_id": card["id"],
            "card_name": card["display_name"],
            "source": "cards",
        }
        for card in cards
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules-input", type=Path,
                    default=REPO_ROOT / "data/processed/rules.jsonl")
    ap.add_argument("--cards-input", type=Path,
                    default=REPO_ROOT / "data/processed/card_database.jsonl")
    ap.add_argument("--rules-output", type=Path,
                    default=REPO_ROOT / "data/processed/rule_chunks.jsonl")
    ap.add_argument("--cards-output", type=Path,
                    default=REPO_ROOT / "data/processed/card_chunks.jsonl")
    ap.add_argument("--corpus-output", type=Path,
                    default=REPO_ROOT / "data/processed/corpus_chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_RULES_MAX_CHARS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rules = read_jsonl(args.rules_input, missing_ok=False)
    rule_chunks = chunk_rules(rules, max_chars=args.max_chars)
    guard_shrink(args.rules_output, len(rule_chunks), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.rules_output, rule_chunks)
    print(f"wrote {len(rule_chunks)} rule chunks -> {args.rules_output}")

    cards = read_jsonl(args.cards_input, missing_ok=False)
    card_chunks = chunk_cards(cards)
    guard_shrink(args.cards_output, len(card_chunks), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.cards_output, card_chunks)
    print(f"wrote {len(card_chunks)} card chunks -> {args.cards_output}")

    # Rules first, so the merged file's order is stable and reviewable rather
    # than depending on which source happened to be read first.
    corpus = rule_chunks + card_chunks
    chunk_ids = [c["chunk_id"] for c in corpus]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise SystemExit(
            "a chunk_id is shared between the rule and card chunk sets -- the "
            "merged index would silently drop one of them")
    guard_shrink(args.corpus_output, len(corpus), min_ratio=0.9, force=args.force)
    write_jsonl_atomic(args.corpus_output, corpus)
    ratio = len(card_chunks) / len(rule_chunks) if rule_chunks else float("inf")
    print(f"wrote {len(corpus)} merged chunks -> {args.corpus_output}")
    print(f"  cards:rules = {ratio:.1f}:1 "
          f"({'one merged index is fine' if ratio < 10 else 'TIME FOR A DUAL-BUDGET ROUTER'})")


if __name__ == "__main__":
    main()

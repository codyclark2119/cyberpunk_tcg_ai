"""Normalize one raw Cyberpunk TCG card API record into a corpus record.

Kept separate from `chunking.py` because these are two different jobs: this
module decides what a card IS in this corpus (which fields, under which
names, with which derived values), while chunking decides how a card reads
once it is one. The card record is also what `data/processed/card_database.jsonl`
holds, which is the file a card-lookup path reads directly without any
chunking involved.

WHAT IS DERIVED HERE, AND WHY

Two fields the API declares but never populates, on any of the 151 cards in
the launch set:

  - `keywords` is `[]` on every card. The keywords are really there, in the
    `{Blocker}` brace markup inside `rules_text`; nothing but this repo reads
    them out. See `games/cyberpunk/markup.py`.
  - `flavor_text` is `null` on every card. Nothing to recover -- recorded
    here only so that a future set which does populate it is not mistaken for
    a parser bug.

`printings` is carried through because it is the only record of which
physical products a card appears in, and this game is in beta with Beta and
Retail versions of the same card at different rarities and collector numbers.
It is dropped from the retrievable text (see `chunking.card_to_text`) but
kept in the database record.
"""

from games.cyberpunk import markup

# The card's identity in this corpus. The slug is what the official site puts
# in its own URLs, it is unique across all 151 cards, and it is readable --
# unlike the API's uuid `id`, which is a printing-dependent surrogate. The
# `name` field is NOT usable as an identity: sixteen names are shared by two
# or three different cards (three separate cards are named "V"), which is
# also why prompts tell the model to always give the subtitle.
ID_FIELD = "slug"

_PRINTING_FIELDS = ("collector_number", "rarity", "finish", "artist")


def _printing(raw: dict) -> dict:
    out = {k: raw.get(k) for k in _PRINTING_FIELDS}
    out["set"] = (raw.get("set") or {}).get("name")
    out["set_code"] = (raw.get("set") or {}).get("code")
    out["image_url"] = raw.get("source_image_url")
    return out


def parse_card(raw: dict) -> dict:
    """One raw API record -> one corpus record.

    Raises `KeyError` on a record missing a field this corpus treats as
    required, rather than defaulting it: a card with no `card_type` or no
    `slug` means the API shape moved, and silently writing a record with
    `None` in those places would corrupt the database in a way nothing
    downstream could detect.
    """
    for required in (ID_FIELD, "name", "card_type", "color"):
        if raw.get(required) in (None, ""):
            raise KeyError(f"card record is missing required field {required!r}: "
                           f"{raw.get('slug') or raw.get('id')!r}")

    rules_markup = raw.get("rules_text") or ""
    set_info = raw.get("set") or {}

    return {
        "id": raw[ID_FIELD],
        "external_id": raw.get("external_id"),
        "name": raw["name"],
        "subname": raw.get("subname"),
        "display_name": raw.get("display_name") or raw["name"],
        "card_type": raw["card_type"],
        "color": raw["color"],
        "cost": raw.get("cost"),
        "power": raw.get("power"),
        "ram": raw.get("ram"),
        "is_eddiable": raw.get("is_eddiable"),
        "classifications": list(raw.get("classifications") or []),
        # Derived from brace markup -- the API's own `keywords` field is empty
        # on every card. See markup.printed_keywords on why "printed" and
        # "referenced" are two fields rather than one.
        "keywords": markup.printed_keywords(rules_markup),
        "keywords_referenced": markup.referenced_keywords(rules_markup),
        "timing_triggers": markup.printed_timing_triggers(rules_markup),
        # Both forms are kept: the rendered text is what gets embedded and
        # shown to a model, the markup is the source of truth that the derived
        # fields above are recomputed from if the rules for deriving change.
        "text": markup.to_plain_text(rules_markup),
        "text_markup": rules_markup,
        "flavor_text": raw.get("flavor_text"),
        "rarity": raw.get("rarity"),
        "set": set_info.get("name"),
        "set_code": set_info.get("code"),
        "print_number": raw.get("print_number"),
        "artist": raw.get("artist"),
        "legality": raw.get("legality"),
        "image_url": raw.get("source_image_url"),
        "printings": [_printing(p) for p in (raw.get("printings") or [])],
    }

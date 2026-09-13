"""Read-only candidate records for admission into the authoritative TS engine.

This module deliberately does *not* construct engine CardRevision snapshots.
It exports source facts and suggested identity mappings from the API corpus so
that the application repo can review/admit them.  The authoritative engine is
responsible for revision numbers, executable mechanics, execution scope and
review status.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable

SCHEMA_VERSION = 1
FORBIDDEN_AUTHORITATIVE_FIELDS = {
    "revision",
    "mechanics",
    "execution",
    "reviewed",
    "status",
}

_CARD_TYPES = {
    "Legend": "LEGEND",
    "Unit": "UNIT",
    "Gear": "GEAR",
    "Program": "PROGRAM",
}

_COLORS = {
    "Red": "RED",
    "Blue": "BLUE",
    "Green": "GREEN",
    "Yellow": "YELLOW",
    "Neutral": "NEUTRAL",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapped(mapping: dict[str, str], value: str | None, field: str) -> str:
    if value not in mapping:
        raise ValueError(f"unsupported {field} for engine candidate: {value!r}")
    return mapping[value]


def _printing_candidate(printing: dict) -> dict:
    return {
        "setCode": printing.get("set_code"),
        "setName": printing.get("set"),
        "collectorNumber": printing.get("collector_number"),
        "rarity": printing.get("rarity"),
        "finish": printing.get("finish"),
        "artist": printing.get("artist"),
        "imageUrl": printing.get("image_url"),
    }


def build_card_candidate(card: dict, errata: Iterable[dict] = ()) -> dict:
    """Convert one normalized API-corpus card into a non-authoritative candidate."""
    subtitle = card.get("subname") or ""
    candidate = {
        "schemaVersion": SCHEMA_VERSION,
        "sourceCardSlug": card["id"],
        "sourceRecordHash": sha256_json(card),
        "identityCandidate": {
            # The site slug is stable/readable and unique in the captured corpus.
            "cardId": card["id"],
            # Deckbuilding uniqueness follows printed base name, not the slug.
            "deckbuildingIdentity": card["name"],
            "subtitle": subtitle,
            "displayName": card.get("display_name") or card["name"],
        },
        "catalog": {
            "type": _mapped(_CARD_TYPES, card.get("card_type"), "card_type"),
            "colors": [_mapped(_COLORS, card.get("color"), "color")],
            "cost": card.get("cost"),
            "power": card.get("power"),
            "ram": card.get("ram"),
            "sellable": card.get("is_eddiable"),
            "classifications": list(card.get("classifications") or []),
            "rarity": card.get("rarity"),
            "setCode": card.get("set_code"),
            "setName": card.get("set"),
            "collectorNumber": card.get("print_number"),
            "artist": card.get("artist"),
            "legality": card.get("legality"),
            "imageUrl": card.get("image_url"),
        },
        "rulesSource": {
            "markup": card.get("text_markup") or "",
            "rendered": card.get("text") or "",
            # These are parser-derived hints, not executable mechanics.
            "keywordHints": list(card.get("keywords") or []),
            "referencedKeywordHints": list(card.get("keywords_referenced") or []),
            "timingTriggerHints": list(card.get("timing_triggers") or []),
        },
        "printings": [_printing_candidate(p) for p in (card.get("printings") or [])],
        "errata": [
            {
                "id": e["id"],
                "heading": e["heading"],
                "variant": e.get("variant"),
                "text": e["text"],
                "pageUpdatedAt": e["page_updated_at"],
            }
            for e in errata
        ],
    }
    assert_no_authoritative_fields(candidate)
    return candidate


def build_candidates(cards: Iterable[dict], errata: Iterable[dict]) -> list[dict]:
    by_card: dict[str, list[dict]] = defaultdict(list)
    for item in errata:
        by_card[item["card_id"]].append(item)
    return [build_card_candidate(card, by_card.get(card["id"], ())) for card in cards]


def assert_no_authoritative_fields(candidate: dict) -> None:
    """Fail closed if this bridge starts pretending to be the rules authority."""
    stack = [candidate]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            bad = FORBIDDEN_AUTHORITATIVE_FIELDS.intersection(value)
            if bad:
                raise ValueError(f"engine candidate contains authoritative field(s): {sorted(bad)}")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)

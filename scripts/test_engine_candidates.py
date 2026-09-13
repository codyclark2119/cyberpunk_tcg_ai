#!/usr/bin/env python3
"""Offline tests for the API -> engine candidate bridge."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk.engine_candidates import (  # noqa: E402
    FORBIDDEN_AUTHORITATIVE_FIELDS,
    assert_no_authoritative_fields,
    build_card_candidate,
    build_candidates,
    canonical_json,
)
from harness.core.io import read_jsonl  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from export_engine_candidates import export  # noqa: E402

CARDS = REPO_ROOT / "data" / "processed" / "card_database.jsonl"
ERRATA = REPO_ROOT / "data" / "processed" / "errata.jsonl"
INDEX = REPO_ROOT / "data" / "raw" / "cards" / "_index.json"


def test_candidate_is_source_only() -> None:
    card = {
        "id": "v-corporate-exile",
        "name": "V",
        "subname": "Corporate Exile",
        "display_name": "V: Corporate Exile",
        "card_type": "Legend",
        "color": "Yellow",
        "cost": None,
        "power": None,
        "ram": 2,
        "is_eddiable": False,
        "classifications": ["Merc"],
        "keywords": ["Go Solo"],
        "keywords_referenced": [],
        "timing_triggers": [],
        "text": "Go Solo 4.",
        "text_markup": "{Go Solo} 4.",
        "rarity": "Rare",
        "set": "The Heist",
        "set_code": "theheist",
        "print_number": "001",
        "artist": "Artist",
        "legality": "legal",
        "image_url": "https://example.test/v.webp",
        "printings": [],
    }
    candidate = build_card_candidate(card)
    assert candidate["identityCandidate"]["cardId"] == "v-corporate-exile"
    assert candidate["identityCandidate"]["deckbuildingIdentity"] == "V"
    assert candidate["catalog"]["type"] == "LEGEND"
    assert candidate["rulesSource"]["keywordHints"] == ["Go Solo"]
    assert_no_authoritative_fields(candidate)
    serialized = canonical_json(candidate)
    for field in FORBIDDEN_AUTHORITATIVE_FIELDS:
        assert f'"{field}"' not in serialized


def test_unknown_catalog_enum_fails_closed() -> None:
    card = {
        "id": "new-shape",
        "name": "New Shape",
        "card_type": "Location",
        "color": "Red",
    }
    try:
        build_card_candidate(card)
    except ValueError as exc:
        assert "unsupported card_type" in str(exc)
    else:
        raise AssertionError("unknown card type should fail closed")


def test_full_corpus_bridge_is_complete_and_deterministic() -> None:
    cards = read_jsonl(CARDS, missing_ok=False)
    errata = read_jsonl(ERRATA, missing_ok=False)
    candidates = build_candidates(cards, errata)
    assert len(candidates) == len(cards) == 151
    assert len({c["sourceCardSlug"] for c in candidates}) == 151
    assert sum(len(c["errata"]) for c in candidates) == len(errata) == 4
    assert all(c["sourceRecordHash"] for c in candidates)
    for candidate in candidates:
        assert_no_authoritative_fields(candidate)

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ca, ma = export(CARDS, ERRATA, INDEX, Path(a))
        cb, mb = export(CARDS, ERRATA, INDEX, Path(b))
        assert ca.read_bytes() == cb.read_bytes()
        assert ma.read_bytes() == mb.read_bytes()
        manifest = json.loads(ma.read_text(encoding="utf-8"))
        assert manifest["authority"] == "SOURCE_CANDIDATES_ONLY"
        assert manifest["recordCount"] == 151
        assert manifest["errataCount"] == 4


def main() -> None:
    tests = [
        test_candidate_is_source_only,
        test_unknown_catalog_enum_fails_closed,
        test_full_corpus_bridge_is_complete_and_deterministic,
    ]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()

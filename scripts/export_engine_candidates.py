#!/usr/bin/env python3
"""Export normalized API corpus records as non-authoritative engine candidates.

This is a bridge, not an admission step.  It preserves source facts, hashes,
printings and errata while deliberately omitting authoritative engine-only
fields such as revision, executable mechanics, execution scope and reviewed
status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk.engine_candidates import (  # noqa: E402
    SCHEMA_VERSION,
    build_candidates,
    canonical_json,
    sha256_bytes,
)
from harness.core.io import read_jsonl  # noqa: E402

DEFAULT_CARDS = REPO_ROOT / "data" / "processed" / "card_database.jsonl"
DEFAULT_ERRATA = REPO_ROOT / "data" / "processed" / "errata.jsonl"
DEFAULT_CARD_INDEX = REPO_ROOT / "data" / "raw" / "cards" / "_index.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "engine-candidates"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def export(cards_path: Path, errata_path: Path, index_path: Path, out_dir: Path) -> tuple[Path, Path]:
    cards = read_jsonl(cards_path, missing_ok=False)
    errata = read_jsonl(errata_path, missing_ok=False)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    index_slugs = [item["slug"] for item in index["items"]]
    card_ids = [card["id"] for card in cards]
    if card_ids != index_slugs:
        raise SystemExit("processed card database order/content does not match raw card index")

    candidates = build_candidates(cards, errata)
    catalog_path = out_dir / "card-catalog.v1.jsonl"
    manifest_path = out_dir / "manifest.v1.json"

    catalog_text = "".join(canonical_json(candidate) + "\n" for candidate in candidates)
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "CYBERPUNK_ENGINE_CANDIDATES_V1",
        "authority": "SOURCE_CANDIDATES_ONLY",
        "recordCount": len(candidates),
        "errataCount": len(errata),
        "source": {
            "cardDatabasePath": str(cards_path.relative_to(REPO_ROOT)),
            "cardDatabaseSha256": sha256_bytes(cards_path.read_bytes()),
            "errataPath": str(errata_path.relative_to(REPO_ROOT)),
            "errataSha256": sha256_bytes(errata_path.read_bytes()),
            "cardIndexPath": str(index_path.relative_to(REPO_ROOT)),
            "cardIndexSha256": sha256_bytes(index_path.read_bytes()),
        },
        "catalogSha256": sha256_bytes(catalog_text.encode("utf-8")),
        "forbiddenAuthority": ["revision", "mechanics", "execution", "reviewed", "status"],
    }

    _write_text_atomic(catalog_path, catalog_text)
    _write_text_atomic(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return catalog_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--errata", type=Path, default=DEFAULT_ERRATA)
    parser.add_argument("--index", type=Path, default=DEFAULT_CARD_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            catalog, manifest = export(args.cards, args.errata, args.index, temp_dir)
            expected_catalog = args.out_dir / "card-catalog.v1.jsonl"
            expected_manifest = args.out_dir / "manifest.v1.json"
            if not expected_catalog.exists() or not expected_manifest.exists():
                raise SystemExit("candidate artifacts missing; run without --check first")
            if catalog.read_bytes() != expected_catalog.read_bytes():
                raise SystemExit(f"{expected_catalog} is stale")
            if manifest.read_bytes() != expected_manifest.read_bytes():
                raise SystemExit(f"{expected_manifest} is stale")
        print("engine candidate artifacts are current")
        return

    catalog, manifest = export(args.cards, args.errata, args.index, args.out_dir)
    print(f"wrote engine candidates -> {catalog}")
    print(f"wrote candidate manifest -> {manifest}")


if __name__ == "__main__":
    main()

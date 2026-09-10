#!/usr/bin/env python3
"""Fetch the public community decklists.

`https://cyberpunktcg.com/decks?tab=community` is a community decklist host --
distinct from the `/decks` "My Decks" tab, which is a signed-in user's own
private decks. The community tab reads an unauthenticated endpoint:

    GET https://api.netdeck.gg/api/cyberpunk/decks/public?limit=&offset=&sort=recent
    GET https://api.netdeck.gg/api/cyberpunk/decks/<uuid>

The index is summary-only (colours, type counts, featured cards); the full
zone lists come from the per-deck endpoint, so every deck is fetched
individually. `limit` is capped server-side at 100 however large a value is
asked for, so the index is paged.

WHY THIS IS WORTH HAVING

Two things, beyond example decks:

  - Every deck carries the API's own `is_valid` verdict and a `stats` block
    with `ramBudget`/`ramUsage` per colour. That is an INDEPENDENT LABEL to
    check a deckbuilding validator against, on a few hundred real decks,
    rather than a validator checked only against its author's own reading of
    the rules.
  - It is a snapshot of what people actually build in the opening weeks of a
    format, which is not recoverable later.

## Owner handles are pseudonymized, deliberately

Each deck carries `owner.display_name` and `owner.user_id` -- real handles
and account ids of third parties who published a decklist, not of this repo's
user. This workspace already has a standing rule for exactly this, set when
handles appeared in captured combat logs: anonymize them, because the goal is
a general model of play rather than a reconstruction of any particular
person's.

That reason applies here in full, so `pseudonymize_owner` replaces both
fields with a salted hash. It is a STABLE pseudonym rather than a blank:
"these two decks share an author" is a real signal, for deduplication and for
noticing one person uploading twenty variants of one list, and dropping the
field entirely would destroy that while protecting nobody further. The salt
is generated per snapshot and stored beside the decks, so the mapping is
consistent within a snapshot and cannot be reversed to a handle from outside
it.

`credit`, the free-text field the platform offers for attribution, is kept as
written: its entire purpose is to be published, and its author chose it.

## Embedded card objects are reduced to slugs

Each deck entry embeds a full copy of every card in it. That is ~23KB per
deck, almost all of it duplicating `data/raw/cards/`, and keeping it would
put card rules text in 273 places instead of one. This repo has just made
errata the authority over printed card text; 272 stale copies of that text is
precisely the shape of the problem errata exist to fix. So a card reference
is stored as its slug, and `scripts/build_decks.py` joins against the card
database.

Usage:
    python scripts/fetch_decks.py                 # index + every public deck
    python scripts/fetch_decks.py --index-only
    python scripts/fetch_decks.py --max-decks 20  # a sample, for a quick check
"""

import argparse
import hashlib
import json
import secrets
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# One definition of the signed-URL normalization, shared with the card fetch.
# Two copies that had to agree would be a bug waiting to happen.
from fetch_cards import strip_signed_urls  # noqa: E402

API_BASE = "https://api.netdeck.gg/api"
GAME_CODE = "cyberpunk"
USER_AGENT = "Mozilla/5.0 (compatible; cyberpunk-llm-corpus-builder/1.0)"
PAGE_SIZE = 100  # the server's own cap; asking for more returns 100

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "decks"
INDEX_NAME = "_index.json"
SALT_NAME = "_salt.json"


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pseudonymize_owner(owner: dict, salt: str) -> dict:
    """Replace a handle and account id with a stable, salted pseudonym."""
    if not owner:
        return owner
    raw = owner.get("user_id") or owner.get("display_name") or ""
    digest = hashlib.sha256(f"{salt}:{raw}".encode("utf-8")).hexdigest()[:12]
    return {"owner_pseudonym": f"author-{digest}"}


def reduce_cards(node, seen_slugs: set):
    """Replace every embedded card object with its slug, recursively.

    Records each slug it drops into `seen_slugs`, so the caller can check the
    decks reference no card the fetched catalog is missing -- a deck naming a
    card this repo has never fetched means the catalog is stale.
    """
    if isinstance(node, dict):
        if "card" in node and isinstance(node["card"], dict) and "slug" in node["card"]:
            out = {k: reduce_cards(v, seen_slugs) for k, v in node.items() if k != "card"}
            seen_slugs.add(node["card"]["slug"])
            out["card_slug"] = node["card"]["slug"]
            return out
        return {k: reduce_cards(v, seen_slugs) for k, v in node.items()}
    if isinstance(node, list):
        return [reduce_cards(v, seen_slugs) for v in node]
    return node


def clean_deck(deck: dict, salt: str, seen_slugs: set) -> dict:
    out = strip_signed_urls(reduce_cards(deck, seen_slugs))
    out["owner"] = pseudonymize_owner(out.get("owner"), salt)
    return out


def fetch_index(api_base: str = API_BASE, delay: float = 0.0) -> dict:
    """Every public deck summary, paged through until `total` is reached."""
    url = f"{api_base}/{GAME_CODE}/decks/public"
    first = fetch_json(f"{url}?limit={PAGE_SIZE}&offset=0&sort=recent")
    total = first["total"]
    items = list(first["items"])
    offset = PAGE_SIZE
    while offset < total and len(first["items"]):
        if delay:
            time.sleep(delay)
        page = fetch_json(f"{url}?limit={PAGE_SIZE}&offset={offset}&sort=recent")
        if not page["items"]:
            break
        items.extend(page["items"])
        offset += PAGE_SIZE

    ids = [d["id"] for d in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("index contains duplicate deck ids -- paging shifted mid-pull")
    if len(items) != total:
        # A warning rather than a hard error, unlike the card catalog: decks
        # are user content and one can genuinely be deleted or made private
        # between two pages of the same pull. The count is reported so a large
        # shortfall is visible instead of silent.
        print(f"WARNING: index reported total={total} but {len(items)} came back "
              f"-- a deck may have been unpublished mid-pull")
    return {"total": total, "items": items}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--api-base", type=str, default=API_BASE)
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--max-decks", type=int, default=None,
                    help="stop after this many decks (for a quick check)")
    ap.add_argument("--delay", type=float, default=0.25)
    ap.add_argument("--force", action="store_true",
                    help="re-fetch decks whose json is already on disk")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The salt is generated once and reused, so re-running does not reshuffle
    # every pseudonym and produce a diff in which nothing actually changed.
    salt_path = args.out_dir / SALT_NAME
    if salt_path.exists():
        salt = json.loads(salt_path.read_text(encoding="utf-8"))["salt"]
    else:
        salt = secrets.token_hex(16)
        salt_path.write_text(json.dumps({
            "salt": salt,
            "note": "Salts the owner pseudonyms in this snapshot. Rotating it "
                    "renames every author; keep it with the data.",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"generated a new pseudonym salt -> {salt_path}")

    print(f"fetching the public deck index from {args.api_base} ...")
    seen_slugs: set = set()
    index = fetch_index(args.api_base, delay=args.delay)
    index["items"] = [clean_deck(d, salt, seen_slugs) for d in index["items"]]
    index_path = args.out_dir / INDEX_NAME
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"wrote {len(index['items'])} deck summaries -> {index_path}")

    if args.index_only:
        return

    deck_ids = [d["id"] for d in index["items"]]
    if args.max_decks:
        deck_ids = deck_ids[:args.max_decks]

    fetched = 0
    for i, deck_id in enumerate(deck_ids, 1):
        path = args.out_dir / f"{deck_id}.json"
        if path.exists() and not args.force:
            continue
        deck = clean_deck(
            fetch_json(f"{args.api_base}/{GAME_CODE}/decks/{deck_id}"), salt, seen_slugs)
        path.write_text(json.dumps(deck, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        fetched += 1
        if i % 25 == 0 or i == len(deck_ids):
            print(f"[{i}/{len(deck_ids)}] {fetched} fetched")
        if args.delay:
            time.sleep(args.delay)

    print(f"done -- {fetched} deck(s) fetched, "
          f"{len(list(args.out_dir.glob('*.json'))) - 2} on disk")

    catalog = REPO_ROOT / "data" / "raw" / "cards" / "_index.json"
    if catalog.exists():
        known = {c["slug"] for c in json.loads(catalog.read_text())["items"]}
        unknown = sorted(seen_slugs - known)
        if unknown:
            print(f"WARNING: {len(unknown)} card(s) referenced by a deck are not in "
                  f"the fetched catalog (e.g. {unknown[:5]}) -- run fetch_cards.py")
        else:
            print(f"every card referenced by a deck is in the catalog "
                  f"({len(seen_slugs)} distinct)")


if __name__ == "__main__":
    main()

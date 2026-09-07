#!/usr/bin/env python3
"""Fetch the official Cyberpunk TCG card database as structured JSON.

Unlike the One Piece instance of this template -- which has to scrape
server-rendered HTML because the official card list has no data endpoint --
cyberpunktcg.com is a thin client over a public JSON API, so there is no
HTML parsing anywhere in this repo's card path. The site's own JS bundle
names the endpoints:

    https://api.netdeck.gg/api/cards/cyberpunk?limit=&offset=   # paged index
    https://api.netdeck.gg/api/cards/cyberpunk/<slug>            # one card
    https://api.netdeck.gg/api/cards/cyberpunk/filters           # vocabularies

`robots.txt` is `Allow: /`, and these are the same unauthenticated requests
the public site makes to render its own card pages.

WHY EACH CARD IS FETCHED INDIVIDUALLY WHEN A PAGED INDEX EXISTS

The index returns every field EXCEPT `printings`, which comes back as `[]`
for all 151 cards -- only the per-card endpoint populates it. Printings are
not cosmetic here: this game is in beta, and the same card exists in Beta and
Retail sets at different rarities and collector numbers, so the printing list
is the only record of which physical products a card appears in. The index is
still fetched first, as the authoritative slug set and total.

WHY SIGNED IMAGE URLS ARE STRIPPED BEFORE THE RESPONSE IS SAVED

Every `image_url` in the response is a CloudFront URL signed with an
`Expires` timestamp, so two fetches of an UNCHANGED card differ in several
hundred bytes of signature. Left alone, `git diff` on a re-fetch would show
all 151 files changed and tell you nothing about which cards actually
changed -- which is precisely the question worth asking during a beta with
live errata. So `strip_signed_urls` drops the query string from any URL
carrying CloudFront signing parameters, and the saved snapshot is stable
across fetches. This is a deliberate, documented normalization of the raw
data, not a silent one; it is lossless because the API also returns
`source_image_url`, which is the same URL unsigned, and the check that they
agree is a test (`test_cyberpunk.py`). Re-sign by re-fetching; the images
themselves are not part of this corpus.

Usage:
    python scripts/fetch_cards.py                 # full pull, index + 151 cards
    python scripts/fetch_cards.py --index-only    # refresh the index alone
    python scripts/fetch_cards.py --slug v-streetkid   # re-fetch one card
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

API_BASE = "https://api.netdeck.gg/api"
GAME_CODE = "cyberpunk"
USER_AGENT = "Mozilla/5.0 (compatible; cyberpunk-llm-corpus-builder/1.0)"
PAGE_SIZE = 100

# CloudFront's signing parameters. A URL carrying any of these has a query
# string that changes on every fetch and means nothing to this corpus.
SIGNED_URL_PARAMS = {"Expires", "Signature", "Key-Pair-Id", "Policy"}

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "cards"
INDEX_NAME = "_index.json"
FILTERS_NAME = "_filters.json"


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_signed_url(value: str) -> bool:
    """True for a URL whose query carries CloudFront signing parameters."""
    if not isinstance(value, str) or "://" not in value or "?" not in value:
        return False
    query = urllib.parse.urlsplit(value).query
    keys = {k for k, _ in urllib.parse.parse_qsl(query, keep_blank_values=True)}
    return bool(keys & SIGNED_URL_PARAMS)


def strip_signed_urls(node):
    """Recursively drop the query string from every signed URL in `node`.

    Returns a new structure; the input is not mutated. Anything that is not a
    signed URL -- including ordinary URLs that merely have a query string --
    is passed through untouched.
    """
    if isinstance(node, dict):
        return {k: strip_signed_urls(v) for k, v in node.items()}
    if isinstance(node, list):
        return [strip_signed_urls(v) for v in node]
    if is_signed_url(node):
        return urllib.parse.urlsplit(node)._replace(query="", fragment="").geturl()
    return node


def fetch_index(api_base: str = API_BASE, page_size: int = PAGE_SIZE,
                delay: float = 0.0) -> dict:
    """Every card summary, paged through until `total` is reached.

    Returns `{"total": N, "items": [...]}`. Refuses a short read rather than
    silently returning a partial catalog: a truncated index would quietly
    shrink the corpus on the next ingest, and that is the one failure this
    repo's `guard_shrink` exists to make loud.
    """
    first = fetch_json(f"{api_base}/cards/{GAME_CODE}?limit={page_size}&offset=0")
    total = first["total"]
    items = list(first["items"])
    offset = page_size
    while offset < total:
        if delay:
            time.sleep(delay)
        page = fetch_json(f"{api_base}/cards/{GAME_CODE}?limit={page_size}&offset={offset}")
        items.extend(page["items"])
        offset += page_size
    if len(items) != total:
        raise SystemExit(
            f"index is short: API reported total={total} but {len(items)} items "
            f"came back. Refusing to save a partial catalog."
        )
    slugs = [c["slug"] for c in items]
    if len(set(slugs)) != len(slugs):
        raise SystemExit("index contains duplicate slugs -- paging may have shifted mid-pull")
    return {"total": total, "items": items}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--api-base", type=str, default=API_BASE)
    ap.add_argument("--slug", type=str, default=None,
                    help="re-fetch only this card instead of the whole catalog")
    ap.add_argument("--index-only", action="store_true",
                    help="refresh the index and filter vocabularies, no per-card fetches")
    ap.add_argument("--delay", type=float, default=0.25,
                    help="seconds between requests (default 0.25)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch cards whose json is already on disk")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.slug:
        card = strip_signed_urls(
            fetch_json(f"{args.api_base}/cards/{GAME_CODE}/{urllib.parse.quote(args.slug)}"))
        path = args.out_dir / f"{args.slug}.json"
        path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return

    print(f"fetching card index from {args.api_base} ...")
    index = strip_signed_urls(fetch_index(args.api_base, delay=args.delay))
    index_path = args.out_dir / INDEX_NAME
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {index['total']} card summaries -> {index_path}")

    filters = strip_signed_urls(fetch_json(f"{args.api_base}/cards/{GAME_CODE}/filters"))
    filters_path = args.out_dir / FILTERS_NAME
    filters_path.write_text(json.dumps(filters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote filter vocabularies -> {filters_path}")

    if args.index_only:
        return

    slugs = [c["slug"] for c in index["items"]]
    fetched = 0
    for i, slug in enumerate(slugs, 1):
        path = args.out_dir / f"{slug}.json"
        if path.exists() and not args.force:
            continue
        card = strip_signed_urls(
            fetch_json(f"{args.api_base}/cards/{GAME_CODE}/{urllib.parse.quote(slug)}"))
        path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fetched += 1
        print(f"[{i}/{len(slugs)}] {slug}")
        if args.delay:
            time.sleep(args.delay)

    on_disk = len(list(args.out_dir.glob("*.json"))) - 2  # minus _index / _filters
    print(f"done -- {fetched} card(s) fetched, {on_disk} on disk, {len(slugs)} in the index")
    if on_disk != len(slugs):
        raise SystemExit(
            f"MISMATCH: {on_disk} card files on disk but {len(slugs)} cards in the index. "
            f"A card may have been removed upstream (stale file) or a fetch failed."
        )


if __name__ == "__main__":
    main()

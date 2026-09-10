"""Parse the Cyberpunk TCG errata CMS page into per-card errata records.

Errata are the one part of this corpus that OVERRIDES another part. The
official page is explicit: "every card on this page should be played as if it
bears the updated text and characteristics associated with it. The updated
text shown here supersedes all printed text at all levels of play." So an
erratum that fails to reach the corpus, or reaches it attached to the wrong
card, is worse than a missing card -- it makes the rest of the data wrong
without looking wrong.

That is why both failure modes here are hard errors rather than warnings:

  - a heading that matches no card raises, instead of being dropped
  - a normalized name that matches more than one card raises, instead of
    picking one

WHY THE CARD DATABASE IS NOT SIMPLY TRUSTED TO BE ERRATA'D ALREADY

It mostly is, and that is the trap. Of the four launch-set errata, three are
already reflected in the card API -- Johnny Silverhand is correctly
`is_eddiable: false`, Kiroshi Optics carries its corrected reminder text, and
Judy Alvarez's artist credit is fixed. The fourth is not: Nocturne OP55 N1
still carries its pre-errata artist. So "the API already applies errata" is
true often enough to be tempting and false often enough to be wrong, and the
errata list has to be carried as its own joined source.

## Document shape

The CMS response is Contentful-shaped: one `cyberpunkPage`, with every linked
entry in `includes.Entry` addressed by id. The page links sections, sections
link markdown blocks, and the errata live as `### <Card Name>` headings
inside a block's `bodyMarkdown`, each followed by prose and a `![...](...)`
image of the corrected card.
"""

import re
import unicodedata

CONTENT_TYPE_SECTION = "cyberpunkPageSection"
CONTENT_TYPE_MARKDOWN = "cyberpunkPageMarkdownBlock"

# `### Card Name` opens one erratum. Errata are always h3 in this document;
# matching any heading level would also swallow the page's own structure if
# the CMS ever nests one.
HEADING_RE = re.compile(r"^###\s+(?P<heading>.+?)\s*$", re.M)

# A trailing parenthetical on a heading names WHICH PRINTING the erratum
# applies to -- "Johnny Silverhand: Never Stop Fighting (Beta Iconic Rare)".
# It is stripped for the card join and kept as `variant`, because the card
# itself is one record with several printings and the erratum is about one
# of them.
VARIANT_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<variant>[^()]*)\)\s*$")

MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]*)\)")


def absolute_url(src: str) -> str:
    """Give a protocol-relative CMS asset URL a scheme.

    Contentful serves images as `//images.ctfassets.net/...`, which resolves
    fine inside a browser page and not at all anywhere else -- a stored
    `//host/path` is read as a path by most tooling, so it would be a broken
    reference the moment anything outside a browser touched it.
    """
    return f"https:{src}" if src.startswith("//") else src


def normalize_name(name: str) -> str:
    """A card name reduced to a join key: no case, no accents, no punctuation.

    Needed because the errata headings are hand-written and do not always
    match the database's spelling exactly -- the launch set's "Nocturne
    OP55N1" is "Nocturne OP55 N1" in the card database, a single space apart.
    Verified safe rather than assumed: across all 151 cards this collapses to
    151 distinct keys, so it cannot merge two different cards. `build_errata`
    re-checks that on the live data instead of trusting this comment.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", stripped.casefold())


def _entries_by_id(doc: dict) -> dict:
    return {e["sys"]["id"]: e for e in (doc.get("includes", {}).get("Entry") or [])}


def _content_type(entry: dict) -> str:
    return entry["sys"]["contentType"]["sys"]["id"]


def markdown_blocks(doc: dict) -> list[tuple[str, str]]:
    """`(section_title, body_markdown)` for each block, in document order.

    Walks page -> `sectionEntries` -> `subsectionEntries`, ordering by each
    entry's own `order` field. A block reached through no section is skipped:
    the errata page's structure is the section headings ("Welcome to Night
    City Card Errata"), and a stray unlinked block is CMS scaffolding, not an
    erratum.
    """
    page = doc["items"][0]
    by_id = _entries_by_id(doc)
    out: list[tuple[str, str]] = []

    def refs(field):
        return [r["sys"]["id"] for r in (field or [])]

    sections = [by_id[i] for i in refs(page["fields"].get("sectionEntries")) if i in by_id]
    sections.sort(key=lambda e: (e["fields"].get("order", 0), e["sys"]["id"]))

    for section in sections:
        if _content_type(section) != CONTENT_TYPE_SECTION:
            continue
        title = section["fields"].get("title", "")
        blocks = [by_id[i] for i in refs(section["fields"].get("subsectionEntries"))
                  if i in by_id]
        blocks.sort(key=lambda e: (e["fields"].get("order", 0), e["sys"]["id"]))
        for block in blocks:
            if _content_type(block) != CONTENT_TYPE_MARKDOWN:
                continue
            body = block["fields"].get("bodyMarkdown") or ""
            if body.strip():
                out.append((title, body))
    return out


def split_entries(section_title: str, body: str) -> list[dict]:
    """One markdown block -> one record per `### Card Name` heading.

    Text before the first heading is intentionally discarded: it is a block
    preamble, not an erratum, and attaching it to the first card would put
    words in that card's record it does not own.
    """
    matches = list(HEADING_RE.finditer(body))
    records = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[match.end():end]

        images = [absolute_url(m.group("src")) for m in MD_IMAGE_RE.finditer(chunk)]
        text = MD_IMAGE_RE.sub("", chunk).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)

        heading = match.group("heading").strip()
        variant_match = VARIANT_RE.match(heading)
        if variant_match:
            card_name = variant_match.group("name").strip()
            variant = variant_match.group("variant").strip()
        else:
            card_name = heading
            variant = None

        records.append({
            "section": section_title,
            "heading": heading,
            "card_name": card_name,
            "variant": variant,
            "text": text,
            "images": images,
        })
    return records


def parse(doc: dict) -> list[dict]:
    """Every erratum on the page, in document order, before the card join."""
    out = []
    for section_title, body in markdown_blocks(doc):
        out.extend(split_entries(section_title, body))
    return out


def join_to_cards(entries: list[dict], cards: list[dict]) -> list[dict]:
    """Attach a `card_id` to each erratum, raising rather than guessing.

    Matching is exact on `display_name` first, then on `normalize_name`. An
    unmatched heading raises: see this module's docstring on why a dropped
    erratum is the failure that matters most here.
    """
    exact = {c["display_name"]: c["id"] for c in cards}

    normalized: dict = {}
    collisions: dict = {}
    for card in cards:
        key = normalize_name(card["display_name"])
        if key in normalized:
            collisions.setdefault(key, [normalized[key]]).append(card["id"])
        normalized[key] = card["id"]
    if collisions:
        raise ValueError(
            f"normalize_name collapses different cards together: {collisions}. "
            f"Joining an erratum by normalized name could attach it to the wrong "
            f"card, so this refuses rather than picking one.")

    out = []
    unmatched = []
    for entry in entries:
        name = entry["card_name"]
        card_id = exact.get(name) or normalized.get(normalize_name(name))
        if card_id is None:
            unmatched.append(entry["heading"])
        out.append({**entry, "card_id": card_id})

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} erratum heading(s) match no card: {unmatched}. "
            f"Errata supersede printed card text, so an unjoinable one is a hard "
            f"error -- check for a renamed card or a new set not yet fetched.")
    return out


def to_text(entry: dict) -> str:
    """One erratum rendered as retrievable prose.

    Leads with the word "Errata" and the override, because a retrieved chunk
    has to carry its own authority: a model that sees only the prose has no
    way to know this text outranks the card record it also retrieved.
    """
    who = entry["heading"]
    lines = [f"Errata: {who}"]
    if entry.get("section"):
        lines.append(entry["section"])
    lines.append(entry["text"])
    lines.append("This errata supersedes the card's printed text at all levels of play.")
    return "\n".join(lines)

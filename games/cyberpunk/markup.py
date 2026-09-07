"""Cyberpunk TCG's `{...}` text markup, shared by the card and rules corpora.

Both card `rules_text` and Comprehensive Rules `body_markdown` wrap certain
spans in braces, and the same brace syntax means three different things
depending on what is inside it:

    {Play} {Call} {Attack} {Defeated}     a TIMING TRIGGER  ("when" an effect happens)
    {Adrenaline} {Go Solo} {Quick} {Blocker}   a KEYWORD     (a standard effect)
    {Spend}                                a SYMBOL          (the spend cost symbol)
    {7.6}  {7.5-7.9}                       a RULE REFERENCE  (rules document only)
    {[GIGS](#rule-<uuid>)}                 a LINK            (rules document only)

On the printed card these render as coloured highlights -- convex for timing
triggers, concave for keywords -- which is why the game's own gameplay guide
treats them as two separate categories rather than one list of "keywords".
This module keeps that distinction because it is the distinction a rules
question turns on: "when does it happen" versus "what happens".

WHY THE VOCABULARY IS ASSERTED, NOT ASSUMED

The API exposes a `keywords` field on every card. It is `[]` on all 151 of
them -- populated nowhere, for any card. So the ONLY machine-readable record
of which keywords a card has is this brace markup, and deriving it here is
the difference between a corpus that can answer "which cards have Blocker"
and one that cannot.

That makes an unrecognised token dangerous in a specific way: this game is in
open beta, new sets are landing, and a keyword this module has never seen
would otherwise be silently classified as "unknown" and quietly disappear
from every derived field. So `classify` never drops anything -- it returns
`"unknown"` and `scripts/ingest.py` reports those loudly, while
`scripts/test_cyberpunk.py` asserts the vocabulary below still covers the
whole committed corpus. A new keyword is meant to FAIL the suite, not pass
through it.
"""

import re

# From the official gameplay guide's "Timing Triggers & Keywords" section.
TIMING_TRIGGERS = ("Play", "Call", "Attack", "Defeated")
KEYWORDS = ("Adrenaline", "Go Solo", "Quick", "Blocker")
SYMBOLS = ("Spend",)

# A brace span, captured without its delimiters. Non-greedy and forbidding a
# nested `}` so a run of markup in one sentence yields one token per span.
MARKUP_RE = re.compile(r"\{([^{}]*)\}")

# An ordinary markdown link. The rules document writes cross-references BOTH
# ways -- 12 wrapped in braces and 91 bare, in the corpus as fetched -- so
# anything that only understood the braced form would leave nine tenths of
# the document's links as raw `#rule-<uuid>` fragments embedded in the
# retrievable text. The site's own renderer unwraps `{[x](y)}` to `[x](y)`
# first and then renders markdown normally, which is what `to_plain_text`
# below does too.
#
# The label alternation is not decoration: rule 11.11.1.4 links to the CALL
# keyword's definition and writes the label as `[\[CALL\]]`, escaping the
# brackets that are part of the keyword's own name. A plain `[^\]]*` label
# stops at the first `]` -- the escaped one -- so the link matches nothing at
# all and survives into the corpus as a raw `#rule-<uuid>` fragment. That is
# one record out of 713, found only by grepping the ingested output for
# `](`, which is why `scripts/test_cyberpunk.py` asserts on the whole file
# rather than on a hand-picked example.
# The leading `!` makes it a markdown IMAGE rather than a link. Four rules
# embed a figure this way -- the card diagram, the rarity chart, the RAM
# colour key, and the game-area layout. Matching the `!` here is what stops a
# link-only pattern from eating the `[alt](url)` and leaving an orphaned "!"
# glued to the front of the alt text, which is how the corpus first came out.
# The alt text is kept (it names the figure, and the surrounding rules refer
# to figures by name); only the bang and the image URL are dropped.
_MD_LABEL = r"(?:\\.|[^\]\\])*"
MD_LINK_RE = re.compile(
    rf"(?P<bang>!?)\[(?P<label>{_MD_LABEL})\]\((?P<target>[^)]*)\)")
BRACED_LINK_RE = re.compile(rf"\{{(\[{_MD_LABEL}\]\([^)]*\))\}}")

# The braced form, anchored, for classifying one already-extracted token.
LINK_RE = re.compile(rf"^\[(?P<label>{_MD_LABEL})\]\((?P<target>[^)]*)\)$")

# A markdown backslash escape, for turning a link label back into the text it
# stands for: `\[CALL\]` -> `[CALL]`.
MD_ESCAPE_RE = re.compile(r"\\(.)")


def unescape_markdown(text: str) -> str:
    return MD_ESCAPE_RE.sub(r"\1", text)

# A rule number: dot-separated, two or more segments, so a bare section number
# ("7") is never mistaken for a specific rule. An optional `-<number>` tail
# covers the document's own range references, e.g. "7.5-7.9".
RULE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)+(?:-\d+(?:\.\d+)*)?$")


def classify(token: str) -> str:
    """What one brace span means: the category name, or `"unknown"`.

    `token` is the text BETWEEN the braces. Never raises and never returns
    `None`: an unrecognised span is reported as `"unknown"` so a caller can
    surface it, rather than being dropped where nobody would notice.
    """
    if token in TIMING_TRIGGERS:
        return "timing_trigger"
    if token in KEYWORDS:
        return "keyword"
    if token in SYMBOLS:
        return "symbol"
    if LINK_RE.match(token):
        return "link"
    if RULE_NUMBER_RE.match(token):
        return "rule_reference"
    return "unknown"


def tokens(text: str) -> list[tuple[str, str]]:
    """Every brace span in `text` as `(token, category)`, in document order.

    Duplicates are kept -- a card that states a keyword twice really does
    state it twice, and callers that want a set can build one.
    """
    if not text:
        return []
    return [(t, classify(t)) for t in MARKUP_RE.findall(text)]


def _ordered_unique(values):
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# A brace span at the very start of a line, allowing several in a row --
# `{Quick} 1 €$, {Spend} ...` opens one card's ability with three of them.
# Anchoring is `.match()`'s job below, not this pattern's: a leading `^` here
# would be exactly redundant with it, and a mutation run that "proves" the
# anchoring works by deleting a no-op proves nothing at all.
_LEADING_MARKUP_RE = re.compile(r"\s*\{([^{}]*)\}\s*")


def _leading_tokens(text: str) -> list[tuple[str, str]]:
    """Brace spans that OPEN a line, in order, with their categories.

    A card prints its own abilities at the start of an ability line; when it
    talks about a keyword in the middle of a sentence it is talking ABOUT that
    keyword, not claiming it. See `printed_keywords` for why that distinction
    is worth drawing and how it was checked.
    """
    out = []
    for line in (text or "").split("\n"):
        rest = line
        while True:
            match = _LEADING_MARKUP_RE.match(rest)
            if not match:
                break
            token = match.group(1)
            kind = classify(token)
            if kind not in ("keyword", "timing_trigger", "symbol"):
                break
            out.append((token, kind))
            rest = rest[match.end():]
    return out


def printed_keywords(text: str) -> list[str]:
    """The KEYWORDS this card has as its own printed ability.

    NOT every keyword the text mentions. Four cards in the launch set talk
    about a keyword they do not have -- Riot Shield says rivals must pay more
    to use Go Solo, Valentino: Guerrera can attack ready Units that have
    Blocker, and two cards GRANT Adrenaline to something else. A field that
    conflated those with actually having the keyword would answer "which
    cards have Blocker" wrongly, which is exactly the kind of question this
    corpus exists to answer.

    The rule is positional: a keyword that opens an ability line is printed on
    the card; one that appears mid-sentence is being referred to. That is a
    heuristic, so it was checked against a field derived independently of the
    rules text entirely -- only a Legend with GO SOLO has a printed power, and
    across all 27 Legends the set this function calls Go Solo and the set with
    a non-null `power` agree exactly, with no card in either difference.
    `scripts/test_cyberpunk.py` pins that invariant against the committed
    corpus, so a future set that breaks the rule fails the suite rather than
    quietly producing a wrong field.
    """
    return _ordered_unique(t for t, kind in _leading_tokens(text) if kind == "keyword")


def printed_timing_triggers(text: str) -> list[str]:
    """The TIMING TRIGGERS this card's own abilities fire on."""
    return _ordered_unique(
        t for t, kind in _leading_tokens(text) if kind == "timing_trigger")


def referenced_keywords(text: str) -> list[str]:
    """Keywords the text mentions WITHOUT the card having them.

    Kept as its own field rather than discarded: "which cards care about
    Blocker" is a real question about this game, and it has a different answer
    from "which cards have Blocker".
    """
    printed = set(printed_keywords(text))
    return _ordered_unique(
        t for t, kind in tokens(text) if kind == "keyword" and t not in printed)


def mentioned_keywords(text: str) -> list[str]:
    """Every keyword the text names, printed or merely referenced."""
    return _ordered_unique(t for t, kind in tokens(text) if kind == "keyword")


def unknown_tokens(text: str) -> list[str]:
    return _ordered_unique(t for t, kind in tokens(text) if kind == "unknown")


def rule_references(text: str) -> list[str]:
    """Rule numbers a rule's body cross-references via `{7.6}` markup.

    Only the brace form counts. A rule number written as bare prose is not a
    reference the document itself marked up, and treating it as one would
    invent links the source never made.
    """
    return _ordered_unique(t for t, kind in tokens(text) if kind == "rule_reference")


def links(text: str) -> list[tuple[str, str]]:
    """Every markdown link in `text` as `(label, target)`, document order.

    Covers both the braced and the bare form -- see `MD_LINK_RE` on why both
    exist. Targets are returned exactly as written, so an in-document anchor
    (`#rule-<uuid>`) and an external URL are told apart by the caller rather
    than by a guess made here. Images (`![alt](url)`) are NOT links and are
    excluded: a figure is not a cross-reference to another rule.
    """
    if not text:
        return []
    return [(unescape_markdown(m.group("label")), m.group("target"))
            for m in MD_LINK_RE.finditer(text) if not m.group("bang")]


def to_plain_text(text: str) -> str:
    """Render markup as readable prose for embedding and prompting.

    Three passes, in the same order the official site's own renderer uses:

      1. unwrap a braced link, `{[x](y)}` -> `[x](y)`, so both link forms are
         handled by one rule instead of two that could drift apart
      2. unwrap the remaining brace spans:
           - a keyword or timing trigger keeps its name in square brackets,
             `{Blocker}` -> `[Blocker]`, which is exactly how the rules
             document already writes keywords in its own prose (`[QUICK]`),
             so cards and rules end up in one convention instead of two
           - a rule reference keeps its number, `{7.6}` -> `7.6`, which is
             what `prompts.extract_citations` then finds
           - an unknown span is left completely alone, braces included, so it
             stays visible rather than being silently laundered into prose
      3. reduce every markdown link to its label, dropping the target:
         `[GIGS](#rule-<uuid>)` -> `GIGS`. A `#rule-<uuid>` fragment is
         meaningless outside the site's own DOM and would be pure noise in an
         embedding; `scripts/ingest.py` keeps the resolved destinations in a
         `refs` field instead, so the cross-reference survives as data.

    Markdown emphasis (`*italic*`, `**bold**`) and headings (`### [QUICK]`)
    are left as-is: they are ordinary markdown, they survive embedding
    harmlessly, and stripping them would mean hand-rolling a markdown parser
    for no gain.
    """
    if not text:
        return ""

    def replace_brace(match: re.Match) -> str:
        token = match.group(1)
        kind = classify(token)
        if kind in ("keyword", "timing_trigger", "symbol"):
            return f"[{token}]"
        if kind == "rule_reference":
            return token
        return match.group(0)

    out = BRACED_LINK_RE.sub(r"\1", text)
    out = MARKUP_RE.sub(replace_brace, out)
    return MD_LINK_RE.sub(
        lambda m: unescape_markdown(m.group("label")).strip(), out)

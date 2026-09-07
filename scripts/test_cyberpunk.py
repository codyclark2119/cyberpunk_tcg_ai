#!/usr/bin/env python3
"""Tests for the Cyberpunk TCG game module and its ingestion pipeline.

No pytest, no GPU, no network: plain asserts in runnable functions, matching
every other repo cloned from this template. Run with:

    python scripts/test_cyberpunk.py

Two kinds of test live here on purpose:

  - UNIT tests over synthetic strings, which pin the parsing rules
  - CORPUS tests over the committed `data/` files, which pin facts about the
    real launch set

The corpus tests are the ones that matter most while this game is in beta.
The card API returns an EMPTY `keywords` list for every card, so this repo
derives keywords from brace markup, and a new keyword shipping in a new set
would otherwise be silently classified "unknown" and dropped from every
derived field. `test_no_unknown_markup_in_corpus` turns that into a failing
test instead. Likewise `test_go_solo_matches_printed_power` checks the
keyword derivation against a field derived independently of the rules text --
if those two ever disagree, the positional heuristic in
`markup.printed_keywords` has stopped being true and needs revisiting, not
patching around.

Every test here has been mutation-verified: break the behaviour it claims to
protect, confirm this suite fails, put it back.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.base.game_interface import GameConfig
from games.cyberpunk import markup, prompts, chunking
from games.cyberpunk.cards import parse_card
from games.cyberpunk.config import GAME
from harness.core.io import read_jsonl

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import ingest  # noqa: E402 -- scripts/ is not a package; path is fixed just above

DATA = REPO_ROOT / "data"
PROCESSED = DATA / "processed"

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAILURES.append((name, str(exc)))
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 -- a crash is a failure, report it as one
        FAILURES.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {name}")


# --------------------------------------------------------------------------
# markup: brace vocabulary
# --------------------------------------------------------------------------

def test_classify_covers_every_category():
    assert markup.classify("Blocker") == "keyword"
    assert markup.classify("Play") == "timing_trigger"
    assert markup.classify("Spend") == "symbol"
    assert markup.classify("7.6") == "rule_reference"
    assert markup.classify("7.5-7.9") == "rule_reference"
    assert markup.classify("[GIGS](#rule-abc)") == "link"
    assert markup.classify("Overclock") == "unknown"


def test_classify_rejects_a_bare_section_number():
    # A single segment is a section, not a citable rule. Accepting it would
    # make every "10" in the corpus look like a cross-reference.
    assert markup.classify("10") == "unknown"
    assert markup.classify("10.2") == "rule_reference"


def test_unknown_token_is_reported_not_dropped():
    text = "{Blocker} then {Overclock} happens"
    assert markup.unknown_tokens(text) == ["Overclock"]
    # ...and survives rendering untouched, so it stays visible downstream.
    assert "{Overclock}" in markup.to_plain_text(text)


# --------------------------------------------------------------------------
# markup: printed vs referenced keywords
# --------------------------------------------------------------------------

def test_printed_keyword_opens_a_line():
    text = "{Blocker} (You may spend this Unit to redirect an attack.)"
    assert markup.printed_keywords(text) == ["Blocker"]
    assert markup.referenced_keywords(text) == []


def test_referenced_keyword_is_not_printed():
    text = "Rivals must pay +2 to use {Go Solo}."
    assert markup.printed_keywords(text) == []
    assert markup.referenced_keywords(text) == ["Go Solo"]


def test_one_card_can_both_print_and_reference():
    # Riot Shield, verbatim shape: has Blocker, only talks about Go Solo.
    text = ("(Equip to a friendly Unit or face-up Legend.)\n"
            "{Blocker} (You may spend this Unit to redirect an attack.)\n"
            "Rivals must pay +2 to use {Go Solo}.")
    assert markup.printed_keywords(text) == ["Blocker"]
    assert markup.referenced_keywords(text) == ["Go Solo"]
    assert markup.mentioned_keywords(text) == ["Blocker", "Go Solo"]


def test_several_markup_spans_can_open_one_line():
    # Goro Takemura opens an ability with a keyword, a cost, then a symbol.
    text = "{Quick} 1 EUR, {Spend} Give a friendly Unit {Blocker} this turn."
    assert markup.printed_keywords(text) == ["Quick"]
    # Blocker here is granted to something else, mid-sentence -- not printed.
    assert markup.referenced_keywords(text) == ["Blocker"]


def test_timing_triggers_are_separate_from_keywords():
    text = "{Play} Draw 1."
    assert markup.printed_timing_triggers(text) == ["Play"]
    assert markup.printed_keywords(text) == []


# --------------------------------------------------------------------------
# markup: link and image rendering
# --------------------------------------------------------------------------

def test_bare_and_braced_links_render_the_same():
    braced = "See {[GIGS](#rule-abc)}."
    bare = "See [GIGS](#rule-abc)."
    assert markup.to_plain_text(braced) == "See GIGS."
    assert markup.to_plain_text(bare) == "See GIGS."


def test_escaped_brackets_in_a_link_label():
    # Rule 11.11.1.4 writes the CALL keyword's own brackets escaped. A label
    # pattern that stops at the first `]` misses this link entirely and leaks
    # the raw anchor into the corpus.
    text = r"See [\[CALL\]](#rule-abc)."
    assert markup.to_plain_text(text) == "See [CALL]."


def test_image_loses_its_bang_and_keeps_its_alt_text():
    text = "![Figure 4 - Game Area Layout](https://example.test/fig4.jpg)"
    assert markup.to_plain_text(text) == "Figure 4 - Game Area Layout"


def test_an_image_is_not_a_link():
    text = "![Figure 1](https://example.test/f.png) and [GIGS](#rule-abc)"
    assert markup.links(text) == [("GIGS", "#rule-abc")]


def test_rule_reference_survives_as_a_citable_number():
    text = "complete instructions {7.5-7.9} in listed order"
    assert markup.rule_references(text) == ["7.5-7.9"]
    assert "7.5-7.9" in markup.to_plain_text(text)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def test_extract_citations_finds_dotted_rule_ids():
    assert prompts.extract_citations("see 9.2.3 and 1.4") == {"9.2.3", "1.4"}


def test_extract_citations_ignores_a_bare_section_number():
    assert prompts.extract_citations("section 9 covers attacks") == set()


def test_extract_citations_drops_a_trailing_period():
    assert prompts.extract_citations("as stated in 9.2.3.") == {"9.2.3"}


def test_judge_prompt_carries_the_literal_label_phrase():
    # harness/core/eval/judge.py rewrites this exact phrase for arm counts
    # other than four, with a plain string replace. Paraphrasing it silently
    # no-ops the substitution.
    assert "labeled A, B, C, D" in prompts.JUDGE_SYSTEM_PROMPT


def test_judge_prompt_names_json_and_shows_the_shape():
    # judge_batch_rubric parses with raw.find("{") and returns {} -- scoring
    # nothing, which reads as "not yet run" -- if the judge never emits a
    # brace. A prompt that omits the format is how that happens.
    assert "JSON" in prompts.JUDGE_SYSTEM_PROMPT
    assert '"points_hit"' in prompts.JUDGE_SYSTEM_PROMPT
    assert '"errors_made"' in prompts.JUDGE_SYSTEM_PROMPT
    assert "{" in prompts.JUDGE_SYSTEM_PROMPT


def test_build_messages_differs_with_and_without_context():
    bare = prompts.build_messages("Can a ready Unit be attacked?")
    grounded = prompts.build_messages("Can a ready Unit be attacked?", context="9.2. ...")
    assert [m["role"] for m in bare] == ["system", "user"]
    assert bare[0]["content"] != grounded[0]["content"], \
        "grounded prompt must use the grounded system prompt"
    assert "9.2. ..." in grounded[1]["content"]
    assert "9.2. ..." not in bare[1]["content"]


# --------------------------------------------------------------------------
# GameConfig wiring
# --------------------------------------------------------------------------

def test_game_config_is_wired():
    assert isinstance(GAME, GameConfig)
    assert GAME.name == "cyberpunk"
    assert GAME.retrieval is not None
    assert GAME.build_messages is prompts.build_messages
    assert GAME.extract_citations is prompts.extract_citations


def test_game_config_paths_exist():
    assert GAME.retrieval.chunks_path.exists(), \
        f"{GAME.retrieval.chunks_path} missing -- run scripts/chunk_corpus.py"
    for key in ("rules_path", "card_database_path", "rules_chunks_path",
                "cards_chunks_path", "rule_sections_path"):
        assert GAME.extra[key].exists(), f"{key} -> {GAME.extra[key]} missing"


# --------------------------------------------------------------------------
# card parsing
# --------------------------------------------------------------------------

def _raw_card(**overrides):
    raw = {
        "slug": "test-card", "external_id": "cb-test-card",
        "name": "Test", "subname": "Card", "display_name": "Test: Card",
        "card_type": "Unit", "color": "Red",
        "cost": 3, "power": 4, "ram": 2, "is_eddiable": True,
        "classifications": ["Merc"], "keywords": [],
        "rules_text": "{Blocker} (You may spend this Unit.)",
        "flavor_text": None, "rarity": "Common",
        "set": {"name": "Welcome to Night City", "code": "wnc"},
        "print_number": "001", "artist": "Someone", "legality": "legal",
        "source_image_url": "https://example.test/a.webp",
        "printings": [],
    }
    raw.update(overrides)
    return raw


def test_parse_card_derives_keywords_the_api_leaves_empty():
    card = parse_card(_raw_card())
    assert card["keywords"] == ["Blocker"]


def test_parse_card_refuses_a_record_missing_an_identity_field():
    for field in ("slug", "name", "card_type", "color"):
        try:
            parse_card(_raw_card(**{field: None}))
        except KeyError:
            continue
        raise AssertionError(f"parse_card accepted a card with {field}=None")


def test_parse_card_keeps_both_rendered_and_raw_text():
    card = parse_card(_raw_card())
    assert "{Blocker}" in card["text_markup"]
    assert "[Blocker]" in card["text"]


def test_card_to_text_omits_stats_a_card_does_not_have():
    card = parse_card(_raw_card(cost=None, power=None, card_type="Legend"))
    text = chunking.card_to_text(card)
    assert "Cost" not in text and "Power" not in text
    assert "RAM 2" in text


def test_card_to_text_lists_printed_keywords_only():
    card = parse_card(_raw_card(
        rules_text="{Blocker} (reminder)\nRivals must pay +2 to use {Go Solo}."))
    text = chunking.card_to_text(card)
    keyword_line = [l for l in text.split("\n") if l.startswith("Keywords:")]
    assert keyword_line == ["Keywords: Blocker"], keyword_line


# --------------------------------------------------------------------------
# rules chunking
# --------------------------------------------------------------------------

def _rule(rid, section, text):
    return {"id": rid, "section": section, "text": text}


def test_chunk_never_straddles_two_sections():
    rules = [_rule("1.1", "1. A", "short"), _rule("2.1", "2. B", "short")]
    chunks = chunking.chunk_rules(rules, max_chars=1000)
    assert len(chunks) == 2
    assert {c["section"] for c in chunks} == {"1. A", "2. B"}


def test_chunk_respects_max_chars():
    rules = [_rule(f"1.{i}", "1. A", "x" * 40) for i in range(10)]
    chunks = chunking.chunk_rules(rules, max_chars=100)
    assert len(chunks) > 1
    for c in chunks:
        if len(c["rule_ids"]) > 1:
            assert len(c["text"]) <= 100, f"{c['chunk_id']} is {len(c['text'])} chars"


def test_chunk_id_names_its_span():
    rules = [_rule("1.1", "1. A", "a"), _rule("1.2", "1. A", "b")]
    chunks = chunking.chunk_rules(rules, max_chars=1000)
    assert chunks[0]["chunk_id"] == "rule:1.1-1.2"
    assert chunks[0]["rule_ids"] == ["1.1", "1.2"]


# --------------------------------------------------------------------------
# rules ingestion
# --------------------------------------------------------------------------

def _node(nid, parent, sort_order, number, title, body="", node_type="rule", depth=0):
    return {"id": nid, "parent_id": parent, "node_type": node_type,
            "title": title, "body_markdown": body, "display_number": number,
            "stable_anchor": f"rule-{nid}", "stable_key": nid,
            "sort_order": sort_order, "depth": depth,
            "is_numbered": bool(number), "include_in_toc": True}


def test_walk_orders_siblings_by_sort_order_not_response_order():
    """The API returns nodes in no useful order; `walk` imposes document order.

    Covered at the code level as well as against the committed corpus,
    because a corpus test alone would keep passing for anyone who broke the
    ordering and did not re-run ingestion.

    The ids here sort OPPOSITE to `sort_order` on purpose. The first version
    of this test named them "a", "a1", "a2", "b", which sort into the correct
    order by id alone -- so it passed just as happily against a `walk` that
    ignored `sort_order` entirely, and a mutation proved it.
    """
    nodes = [
        _node("aaa", None, 2000, "2", "SECOND", node_type="section"),
        _node("zzz", None, 1000, "1", "FIRST", node_type="section"),
        _node("mmm", "zzz", 2000, "1.2", "second child"),
        _node("nnn", "zzz", 1000, "1.1", "first child"),
    ]
    assert [n["id"] for n in ingest.walk(nodes)] == ["zzz", "nnn", "mmm", "aaa"]


def test_walk_refuses_to_drop_a_disconnected_node():
    nodes = [_node("a", None, 1000, "1", "ROOT", node_type="section"),
             _node("orphan", "nonexistent", 1000, "9.9", "lost")]
    try:
        ingest.walk(nodes)
    except SystemExit:
        return
    raise AssertionError("walk silently dropped a node with a dangling parent")


def test_build_records_carries_the_top_level_section_down():
    doc = {"items": [
        _node("s", None, 1000, "9", "ATTACK", node_type="section"),
        _node("r", "s", 1000, "9.1", "A rule.", body="A rule."),
    ]}
    rules, sections = ingest.build_records(doc)
    assert [r["section"] for r in rules] == ["9. ATTACK", "9. ATTACK"]
    assert [s["id"] for s in sections] == ["9"]


def test_build_records_numbers_the_citable_text():
    doc = {"items": [
        _node("s", None, 1000, "9", "ATTACK", node_type="section"),
        _node("r", "s", 1000, "9.1", "A rule.", body="A rule."),
    ]}
    rules, _ = ingest.build_records(doc)
    assert rules[1]["text"] == "9.1. A rule.", rules[1]["text"]


def test_build_records_falls_back_to_an_anchor_for_an_unnumbered_section():
    doc = {"items": [
        _node("s", None, 1000, "9", "ATTACK", node_type="section"),
        _node("u", "s", 1000, "", "Ending an Attack", node_type="section"),
    ]}
    rules, _ = ingest.build_records(doc)
    assert rules[1]["id"] == "rule-u"
    assert rules[1]["is_numbered"] is False


# --------------------------------------------------------------------------
# corpus: facts about the committed launch set
# --------------------------------------------------------------------------

def _cards():
    return read_jsonl(PROCESSED / "card_database.jsonl", missing_ok=False)


def _rules():
    return read_jsonl(PROCESSED / "rules.jsonl", missing_ok=False)


def test_corpus_card_count_matches_the_index():
    index = json.loads((DATA / "raw" / "cards" / "_index.json").read_text())
    assert len(_cards()) == index["total"], \
        "card_database.jsonl and the fetched index disagree -- re-run build_cards.py"


def test_no_unknown_markup_in_corpus():
    """A new keyword must fail this suite, not vanish into an unread field."""
    offenders = []
    for card in _cards():
        for token in markup.unknown_tokens(card["text_markup"]):
            offenders.append((card["id"], token))
    assert not offenders, (
        f"{len(offenders)} unrecognised brace span(s), e.g. {offenders[:5]}. "
        f"A new keyword probably shipped: add it to games/cyberpunk/markup.py.")


def test_go_solo_matches_printed_power():
    """The keyword heuristic, checked against a field it cannot see.

    Only a Legend with GO SOLO can be played as a Unit, so only those have a
    printed power. `power` comes straight from the API and owes nothing to the
    brace markup, which makes it a genuine independent check on
    `markup.printed_keywords`' positional rule.
    """
    legends = [c for c in _cards() if c["card_type"] == "Legend"]
    assert legends, "no Legends in the corpus -- did build_cards.py run?"
    with_keyword = {c["id"] for c in legends if "Go Solo" in c["keywords"]}
    with_power = {c["id"] for c in legends if c["power"] is not None}
    assert with_keyword == with_power, (
        f"Go Solo and printed power disagree on {sorted(with_keyword ^ with_power)}. "
        f"The positional keyword rule in markup.printed_keywords may no longer hold.")


def test_corpus_rule_text_has_no_unrendered_markup():
    leaks = [r["id"] for r in _rules()
             if "](" in r["text"] or "#rule-" in r["text"] or "{" in r["text"]]
    assert not leaks, f"{len(leaks)} rule(s) leak raw markup into their text: {leaks[:5]}"


def test_every_ref_joins_a_rule_id():
    rules = _rules()
    ids = {r["id"] for r in rules}
    # Ranges ("7.5-7.9") name a span rather than one rule and have no id of
    # their own; everything else must resolve.
    dangling = [(r["id"], ref) for r in rules for ref in r["refs"]
                if ref not in ids and "-" not in ref]
    assert not dangling, f"cross-references pointing nowhere: {dangling[:5]}"


def test_rules_are_in_document_order():
    ids = [r["id"] for r in _rules() if r["is_numbered"]]
    first_top_level = [i for i in ids if "." not in i]
    assert first_top_level[:4] == ["1", "2", "3", "4"], \
        f"top-level sections are out of order: {first_top_level[:6]}"
    # A specific neighbour pair, so a reordering that happens to keep the
    # sections ascending still fails.
    assert ids.index("7.4") < ids.index("7.9.3.2")


def test_merged_corpus_is_the_two_chunk_sets():
    rules = read_jsonl(PROCESSED / "rule_chunks.jsonl", missing_ok=False)
    cards = read_jsonl(PROCESSED / "card_chunks.jsonl", missing_ok=False)
    corpus = read_jsonl(PROCESSED / "corpus_chunks.jsonl", missing_ok=False)
    assert len(corpus) == len(rules) + len(cards)
    assert len({c["chunk_id"] for c in corpus}) == len(corpus), \
        "duplicate chunk_id in the merged corpus -- one chunk would be lost"


def test_card_names_are_ambiguous_so_ids_are_slugs():
    """Sixteen names are shared. This is why prompts insist on the subtitle."""
    cards = _cards()
    names = [c["name"] for c in cards]
    assert len(set(names)) < len(names), "no shared card names -- has the set changed?"
    assert len({c["id"] for c in cards}) == len(cards), "slugs are not unique"


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    print(f"running {len(tests)} tests")
    for name, fn in tests:
        check(name, fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED of {len(tests)}")
        for name, msg in FAILURES:
            print(f"  {name}: {msg}")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()

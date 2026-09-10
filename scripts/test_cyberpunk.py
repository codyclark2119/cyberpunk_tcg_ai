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
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.base.game_interface import GameConfig
from games.cyberpunk import markup, prompts, chunking
from games.cyberpunk import errata as errata_mod
from games.cyberpunk import deckbuilding
from games.cyberpunk import archetypes
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
# errata
# --------------------------------------------------------------------------

# The `## Current Errata` line is load-bearing: errata are always h3 in this
# document, and a pattern loose enough to accept any heading level would turn
# a structural sub-heading into an erratum for a card that does not exist --
# which `join_to_cards` would then raise on, failing the whole build. Without
# a non-h3 heading in this fixture the anchoring is untested, and a mutation
# proved exactly that.
_ERRATA_MD = (
    "## Current Errata\n\n"
    "### Kiroshi Optics\n\n"
    "The reminder text has been updated.\n\n"
    "![Kiroshi Errata](//images.ctfassets.net/x/y/z/Kiroshi.png)\n\n"
    "### Johnny Silverhand: Never Stop Fighting (Beta Iconic Rare)\n\n"
    "This card cannot be sold for an Eddie.\n"
)

_ERRATA_CARDS = [
    {"id": "kiroshi-optics", "display_name": "Kiroshi Optics"},
    {"id": "johnny-silverhand-never-stop-fighting",
     "display_name": "Johnny Silverhand: Never Stop Fighting"},
    {"id": "nocturne-op55-n1", "display_name": "Nocturne OP55 N1"},
]


def test_split_entries_one_record_per_heading():
    entries = errata_mod.split_entries("WNC Errata", _ERRATA_MD)
    assert [e["card_name"] for e in entries] == [
        "Kiroshi Optics", "Johnny Silverhand: Never Stop Fighting"], \
        "only h3 headings are errata; a structural heading is not a card"


def test_split_entries_ignores_a_non_h3_heading():
    entries = errata_mod.split_entries("WNC Errata", _ERRATA_MD)
    assert "Current Errata" not in [e["card_name"] for e in entries]
    # ...and the h2 must not be swallowed into the first erratum's body either.
    assert "Current Errata" not in entries[0]["text"]


def test_split_entries_extracts_the_printing_variant():
    entries = errata_mod.split_entries("WNC Errata", _ERRATA_MD)
    assert entries[0]["variant"] is None
    assert entries[1]["variant"] == "Beta Iconic Rare"


def test_split_entries_lifts_images_out_of_the_text():
    entries = errata_mod.split_entries("WNC Errata", _ERRATA_MD)
    assert "![" not in entries[0]["text"], entries[0]["text"]
    assert entries[0]["images"] == ["https://images.ctfassets.net/x/y/z/Kiroshi.png"], \
        "a protocol-relative CMS URL must be given a scheme"


def test_split_entries_drops_a_block_preamble():
    entries = errata_mod.split_entries("S", "Intro prose.\n\n### Kiroshi Optics\n\nBody.")
    assert len(entries) == 1
    assert "Intro prose" not in entries[0]["text"]


def test_join_matches_a_name_the_database_spells_differently():
    # "Nocturne OP55N1" on the errata page; "Nocturne OP55 N1" in the database.
    entries = [{"heading": "Nocturne OP55N1", "card_name": "Nocturne OP55N1",
                "variant": None, "section": "S", "text": "t", "images": []}]
    joined = errata_mod.join_to_cards(entries, _ERRATA_CARDS)
    assert joined[0]["card_id"] == "nocturne-op55-n1"


def test_join_refuses_an_unmatched_heading():
    entries = [{"heading": "Not A Card", "card_name": "Not A Card", "variant": None,
                "section": "S", "text": "t", "images": []}]
    try:
        errata_mod.join_to_cards(entries, _ERRATA_CARDS)
    except ValueError:
        return
    raise AssertionError("join_to_cards silently dropped an unmatched erratum")


def test_join_refuses_colliding_normalized_names():
    cards = [{"id": "a", "display_name": "Nocturne OP55 N1"},
             {"id": "b", "display_name": "Nocturne OP55N1"}]
    try:
        errata_mod.join_to_cards([], cards)
    except ValueError:
        return
    raise AssertionError("join_to_cards accepted two cards with the same normalized name")


def test_card_text_carries_its_erratum_last():
    card = parse_card(_raw_card(slug="kiroshi-optics", display_name="Kiroshi Optics"))
    entry = {"card_id": "kiroshi-optics", "heading": "Kiroshi Optics",
             "text": "The reminder text has been updated."}
    text = chunking.card_to_text(card, [entry])
    assert "ERRATA" in text
    assert "supersedes" in text
    assert text.rstrip().endswith("This supersedes the printed text above."), \
        "the override line must be last, so a tail-preserving truncation keeps it"


def test_card_without_errata_says_nothing_about_them():
    card = parse_card(_raw_card())
    assert "ERRATA" not in chunking.card_to_text(card, [])
    assert "ERRATA" not in chunking.card_to_text(card, None)


def test_chunk_cards_flags_which_cards_have_errata():
    cards = [parse_card(_raw_card(slug="a", display_name="A")),
             parse_card(_raw_card(slug="b", display_name="B"))]
    entry = {"card_id": "a", "heading": "A", "text": "corrected"}
    chunks = {c["card_id"]: c for c in chunking.chunk_cards(cards, [entry])}
    assert chunks["a"]["has_errata"] is True
    assert chunks["b"]["has_errata"] is False


def test_merge_chunks_keeps_every_source():
    rules = [{"chunk_id": "rule:1"}]
    cards = [{"chunk_id": "card:a"}]
    errata = [{"chunk_id": "errata:a:01"}]
    merged = chunking.merge_chunks(rules, cards, errata)
    assert [c["chunk_id"] for c in merged] == ["rule:1", "card:a", "errata:a:01"], \
        "a source dropped here is a source that can never be retrieved"


def test_merge_chunks_refuses_a_shared_chunk_id():
    try:
        chunking.merge_chunks([{"chunk_id": "x"}], [{"chunk_id": "x"}])
    except ValueError:
        return
    raise AssertionError("merge_chunks accepted a chunk_id shared between sources")


def test_chunk_corpus_cli_wires_every_source_into_the_merged_index():
    """End-to-end over `chunking.main()`, against the real committed inputs.

    `merge_chunks` being correct does not prove `main` passes it every source:
    dropping `errata_chunks` from that one call removes an entire source from
    retrieval, and every other test here would still pass, because the corpus
    tests read committed files rather than re-running the CLI.

    Outputs go to a temp directory, and the real `data/processed/` files are
    read before and compared after. That guard is not paranoia -- a test in a
    sibling repo in this workspace wrote into real gold data twice, because
    the path it thought it had redirected was a default argument bound at
    definition time, so the write went to the real file while the assertions
    talked about the temp one and passed.
    """
    import tempfile
    real = {p.name: p.read_bytes() for p in PROCESSED.glob("*.jsonl")}
    argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        sys.argv = [
            "chunk_corpus.py",
            "--rules-output", str(out / "rule_chunks.jsonl"),
            "--cards-output", str(out / "card_chunks.jsonl"),
            "--errata-output", str(out / "errata_chunks.jsonl"),
            "--corpus-output", str(out / "corpus_chunks.jsonl"),
        ]
        try:
            chunking.main()
        finally:
            sys.argv = argv

        corpus = read_jsonl(out / "corpus_chunks.jsonl", missing_ok=False)
        assert {c["source"] for c in corpus} == {"rules", "cards", "errata"}, \
            f"a source never reached the merged index: {sorted({c['source'] for c in corpus})}"

    after = {p.name: p.read_bytes() for p in PROCESSED.glob("*.jsonl")}
    assert after == real, \
        f"the CLI wrote into real data/processed/: {sorted(set(after) ^ set(real)) or 'contents changed'}"


def test_errata_chunk_states_its_own_authority():
    entry = {"id": "errata:x:01", "card_id": "x", "card_name": "X",
             "heading": "X", "section": "S", "text": "body"}
    chunks = chunking.chunk_errata([entry])
    assert chunks[0]["chunk_id"] == "errata:x:01"
    assert "supersedes" in chunks[0]["text"], \
        "a retrieved erratum must carry its own override, not rely on context"


def test_grounded_prompt_tells_the_model_errata_win():
    grounded = prompts.build_messages("q", context="c")[0]["content"]
    assert "ERRATA" in grounded and "OVERRIDES" in grounded


# --------------------------------------------------------------------------
# deckbuilding
# --------------------------------------------------------------------------

_DB_CARDS = {
    "legend-y1": {"id": "legend-y1", "name": "Ay", "display_name": "Ay: One",
                  "card_type": "Legend", "color": "Yellow", "ram": 2},
    "legend-y2": {"id": "legend-y2", "name": "Bee", "display_name": "Bee: Two",
                  "card_type": "Legend", "color": "Yellow", "ram": 2},
    "legend-r1": {"id": "legend-r1", "name": "Cee", "display_name": "Cee: Three",
                  "card_type": "Legend", "color": "Red", "ram": 2},
    "unit-y2": {"id": "unit-y2", "name": "Yunit", "display_name": "Yunit",
                "card_type": "Unit", "color": "Yellow", "ram": 2},
    "unit-y4": {"id": "unit-y4", "name": "Ybig", "display_name": "Ybig",
                "card_type": "Unit", "color": "Yellow", "ram": 4},
    "unit-y5": {"id": "unit-y5", "name": "Yhuge", "display_name": "Yhuge",
                "card_type": "Unit", "color": "Yellow", "ram": 5},
    "unit-r2": {"id": "unit-r2", "name": "Runit", "display_name": "Runit",
                "card_type": "Unit", "color": "Red", "ram": 2},
}

# Enough distinct filler to build a legal deck without exceeding three copies
# of anything -- the first version of this fixture used 39 copies of one card,
# which is itself illegal, so the "a legal deck has no violations" test failed
# against a deck that was never legal.
for _i in range(20):
    _DB_CARDS[f"filler-{_i}"] = {
        "id": f"filler-{_i}", "name": f"Filler {_i}", "display_name": f"Filler {_i}",
        "card_type": "Unit", "color": "Red", "ram": 2,
    }


def _filler(total, per=3, slug="filler"):
    """`total` cards spread over as many distinct cards as `per` allows."""
    out = []
    remaining = total
    i = 0
    while remaining > 0:
        take = min(per, remaining)
        out.append({"card_slug": f"{slug}-{i}", "quantity": take})
        remaining -= take
        i += 1
    return out


def _deck(legends=None, main=None):
    """A legal 3-Legend, 42-card deck unless overridden."""
    if legends is None:
        legends = [{"card_slug": "legend-y1", "quantity": 1},
                   {"card_slug": "legend-y2", "quantity": 1},
                   {"card_slug": "legend-r1", "quantity": 1}]
    if main is None:
        main = _filler(42)
    return {"zones": [{"zone_code": "legends", "cards": legends},
                      {"zone_code": "deck", "cards": main}]}


def _rules_fired(deck):
    return {v["rule"] for v in deckbuilding.validate(deck, _DB_CARDS)}


def test_a_legal_deck_has_no_violations():
    assert deckbuilding.validate(_deck(), _DB_CARDS) == []


def test_ram_budget_is_a_sum_per_colour():
    budget = deckbuilding.ram_budget(deckbuilding.legend_entries(_deck()), _DB_CARDS)
    assert budget == {"Yellow": 4, "Red": 2}


def test_ram_usage_is_a_max_not_a_sum():
    """Three copies of a RAM-2 card demand 2, not 6."""
    main = [{"card_slug": "unit-y2", "quantity": 3}]
    usage = deckbuilding.ram_usage(main, _DB_CARDS)
    assert usage == {"Yellow": 2}, usage


def test_ram_limit_is_a_ceiling_per_card():
    # Yellow budget is 4. A RAM-4 card is legal; a RAM-5 card is not, however
    # few copies it appears in.
    ok = _deck(main=[{"card_slug": "unit-y4", "quantity": 3}] + _filler(39))
    assert "ram-limit" not in _rules_fired(ok), _rules_fired(ok)
    bad = _deck(main=[{"card_slug": "unit-y5", "quantity": 1}] + _filler(41))
    assert "ram-limit" in _rules_fired(bad)


def test_wrong_legend_count_is_caught():
    assert "legend-count" in _rules_fired(
        _deck(legends=[{"card_slug": "legend-y1", "quantity": 1},
                       {"card_slug": "legend-y2", "quantity": 1}]))


def test_duplicate_legend_names_are_caught_even_as_one_entry():
    """Two copies of a Legend is ONE entry with quantity 2.

    Counting entries instead of copies makes this rule silently never fire --
    which is what it did, agreeing with the official builder on all 272 public
    decks, because none of them is illegal. Found only by a negative control.
    """
    deck = _deck(legends=[{"card_slug": "legend-y1", "quantity": 2},
                          {"card_slug": "legend-r1", "quantity": 1}])
    fired = _rules_fired(deck)
    assert "legend-unique-names" in fired, fired
    assert "legend-count" not in fired, "three Legends is the right count; the name is the problem"


def test_deck_size_bounds():
    assert "deck-size" in _rules_fired(_deck(main=_filler(39)))
    assert "deck-size" in _rules_fired(_deck(main=_filler(51)))
    assert "deck-size" not in _rules_fired(_deck(main=_filler(40))), "40 is legal"
    assert "deck-size" not in _rules_fired(_deck(main=_filler(50))), "50 is legal"


def test_legends_do_not_count_toward_deck_size():
    # Exactly 50 in the deck zone is legal. Counting the 3 Legends too would
    # make it 53 and illegal, so this boundary is what separates the two
    # readings of the rule.
    deck = _deck(main=_filler(50))
    assert "deck-size" not in _rules_fired(deck), \
        "50 deck cards + 3 Legends is legal; Legends are excluded from the count"


def test_more_than_three_copies_is_caught():
    deck = _deck(main=[{"card_slug": "unit-r2", "quantity": 4}] + _filler(38))
    assert "max-copies" in _rules_fired(deck)


def test_exactly_three_copies_is_legal():
    deck = _deck(main=_filler(42, per=3))
    assert "max-copies" not in _rules_fired(deck), _rules_fired(deck)


def test_a_card_not_in_the_database_is_reported_not_crashed():
    deck = _deck(main=[{"card_slug": "does-not-exist", "quantity": 3}] + _filler(39))
    fired = _rules_fired(deck)
    assert "unknown-card" in fired


def test_violations_carry_a_stable_rule_slug_and_a_reason():
    deck = _deck(legends=[{"card_slug": "legend-y1", "quantity": 1}])
    violations = deckbuilding.validate(deck, _DB_CARDS)
    assert violations and all(v["rule"] and v["detail"] for v in violations)


# --------------------------------------------------------------------------
# archetype clustering
# --------------------------------------------------------------------------

def _arch_deck(main, legends=None):
    if legends is None:
        legends = [{"card_slug": "legend-y1", "quantity": 1},
                   {"card_slug": "legend-r1", "quantity": 1}]
    return {"zones": [{"zone_code": "legends", "cards": legends},
                      {"zone_code": "deck", "cards": main}]}


def test_deck_counts_excludes_legends():
    deck = _arch_deck([{"card_slug": "unit-y2", "quantity": 3}])
    counts = archetypes.deck_counts(deck)
    assert counts == Counter({"unit-y2": 3})
    assert "legend-y1" not in counts, \
        "Legends are identity, not content; including them adds a similarity floor"


def test_similarity_counts_copies_not_just_presence():
    """Three copies and one copy are different decisions."""
    three = Counter({"a": 3})
    one = Counter({"a": 1})
    assert archetypes.similarity(three, three) == 1.0
    assert archetypes.similarity(three, one) == 1 / 3, archetypes.similarity(three, one)


def test_similarity_of_disjoint_decks_is_zero():
    assert archetypes.similarity(Counter({"a": 3}), Counter({"b": 3})) == 0.0
    assert archetypes.similarity(Counter(), Counter()) == 0.0


def test_cluster_uses_average_linkage_not_single():
    """A-B and B-C similar, A-C not: single linkage chains all three, average does not.

    This is the whole reason for average linkage. On the real corpus single
    linkage collapses most decks into one blob via a chain of pairwise
    resemblances, none of which says the ends belong together.
    """
    m = [[0.0, 0.6, 0.0],
         [0.6, 0.0, 0.6],
         [0.0, 0.6, 0.0]]
    clusters = archetypes.cluster(m, threshold=0.5)
    # A+B merge at 0.6; {A,B} vs C averages (0.0 + 0.6)/2 = 0.3, below 0.5.
    assert len(clusters) == 2, clusters
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_cluster_merge_averages_over_the_whole_cluster():
    """After merging A and B, C is compared against BOTH of them, not the best.

    A-B 0.9, B-C 0.6, A-C 0.4. Merging A and B first, the average against C is
    (0.4 + 0.6) / 2 = 0.5, which clears a 0.45 threshold, so all three join. A
    max-based merge would carry only the 0.6, average it to 0.3, and stop --
    that is single linkage wearing an average's name, and the difference is
    invisible on a matrix where the two happen to coincide.
    """
    m = [[0.0, 0.9, 0.4],
         [0.9, 0.0, 0.6],
         [0.4, 0.6, 0.0]]
    assert archetypes.cluster(m, threshold=0.45) == [[0, 1, 2]]


def test_cluster_threshold_is_inclusive_at_the_boundary():
    """Exactly `threshold` merges; a hair under does not.

    Weighted Jaccard is a ratio of integer copy counts, so exact ties with a
    round threshold are common, not hypothetical: 22 real deck pairs sit at
    exactly 0.40 and 35 at exactly 0.50. Flipping this one comparison changes
    the archetype count on real data.
    """
    assert archetypes.cluster([[0.0, 0.5], [0.5, 0.0]], threshold=0.5) == [[0, 1]]
    assert archetypes.cluster([[0.0, 0.49], [0.49, 0.0]], threshold=0.5) == [[0], [1]]


def test_cluster_threshold_above_every_similarity_leaves_singletons():
    m = [[0.0, 0.4], [0.4, 0.0]]
    assert archetypes.cluster(m, threshold=0.5) == [[0], [1]]


def test_cluster_merges_below_threshold_pairs():
    m = [[0.0, 0.9], [0.9, 0.0]]
    assert archetypes.cluster(m, threshold=0.5) == [[0, 1]]


def test_cluster_output_is_deterministic_and_sorted():
    """Biggest cluster first -- and the fixture must be able to show it.

    Deck 0 stays a singleton while 1 and 2 merge, so the natural insertion
    order is [[0], [1, 2]] and the required order is [[1, 2], [0]]. An earlier
    fixture had the merge happen first, where both orders coincide, so
    deleting the sort entirely changed nothing and the test passed anyway.
    """
    m = [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.9],
         [0.0, 0.9, 0.0]]
    first = archetypes.cluster(m, 0.5)
    assert first == archetypes.cluster(m, 0.5), "two runs must agree"
    assert first == [[1, 2], [0]], first
    assert all(c == sorted(c) for c in first)


def test_cluster_of_nothing_is_nothing():
    assert archetypes.cluster([], 0.5) == []


def test_core_cards_needs_the_share_of_members():
    counts = [Counter({"a": 1, "b": 1}), Counter({"a": 1, "c": 1}),
              Counter({"a": 1, "d": 1})]
    core = dict(archetypes.core_cards([0, 1, 2], counts, share=0.8))
    assert set(core) == {"a"}, core
    loose = dict(archetypes.core_cards([0, 1, 2], counts, share=0.3))
    assert set(loose) == {"a", "b", "c", "d"}, loose


def test_duplicate_groups_needs_matching_counts_not_just_cards():
    counts = [Counter({"a": 3, "b": 1}), Counter({"a": 3, "b": 1}),
              Counter({"a": 1, "b": 3})]
    groups = archetypes.duplicate_groups(counts)
    assert groups == [[0, 1]], groups


def test_cohesion_is_the_mean_within_a_cluster():
    m = [[0.0, 0.4, 0.6], [0.4, 0.0, 0.8], [0.6, 0.8, 0.0]]
    assert abs(archetypes.cohesion([0, 1, 2], m) - 0.6) < 1e-9
    assert archetypes.cohesion([0], m) == 1.0


def test_null_decks_preserve_colour_size_and_shape():
    """The null must randomize WHICH cards and nothing else.

    A null that also changed colour identity or deck size would be beaten by
    any clustering at all, and would license claims the data does not support.
    """
    import random as _random
    deck = _arch_deck([{"card_slug": "unit-y2", "quantity": 3},
                       {"card_slug": "unit-y4", "quantity": 2}],
                      legends=[{"card_slug": "legend-y1", "quantity": 1}])
    null = archetypes.null_decks([deck], _DB_CARDS, _random.Random(0))[0]
    real = archetypes.deck_counts(deck)
    assert sum(null.values()) == sum(real.values()), "deck size must be preserved"
    assert sorted(null.values(), reverse=True) == sorted(real.values(), reverse=True), \
        "copy-count shape must be preserved"
    colours = {_DB_CARDS[s]["color"] for s in null}
    assert colours <= {"Yellow"}, f"null drew outside the colour identity: {colours}"


def test_null_decks_actually_randomize():
    import random as _random
    deck = _arch_deck([{"card_slug": f"filler-{i}", "quantity": 2} for i in range(10)],
                      legends=[{"card_slug": "legend-r1", "quantity": 1}])
    a = archetypes.null_decks([deck], _DB_CARDS, _random.Random(1))[0]
    b = archetypes.null_decks([deck], _DB_CARDS, _random.Random(2))[0]
    assert a != b, "two seeds produced the same null -- it is not sampling"


def test_pairs_above_counts_each_pair_once():
    m = [[0.0, 0.9, 0.9], [0.9, 0.0, 0.1], [0.9, 0.1, 0.0]]
    assert archetypes.pairs_above(m, 0.5) == 2


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


def test_merged_corpus_is_every_chunk_set():
    rules = read_jsonl(PROCESSED / "rule_chunks.jsonl", missing_ok=False)
    cards = read_jsonl(PROCESSED / "card_chunks.jsonl", missing_ok=False)
    errata = read_jsonl(PROCESSED / "errata_chunks.jsonl", missing_ok=False)
    corpus = read_jsonl(PROCESSED / "corpus_chunks.jsonl", missing_ok=False)
    assert len(corpus) == len(rules) + len(cards) + len(errata), (
        f"merged corpus has {len(corpus)} chunks but the per-source files hold "
        f"{len(rules)}+{len(cards)}+{len(errata)} -- a source is being dropped")
    assert len({c["chunk_id"] for c in corpus}) == len(corpus), \
        "duplicate chunk_id in the merged corpus -- one chunk would be lost"


def test_card_names_are_ambiguous_so_ids_are_slugs():
    """Sixteen names are shared. This is why prompts insist on the subtitle."""
    cards = _cards()
    names = [c["name"] for c in cards]
    assert len(set(names)) < len(names), "no shared card names -- has the set changed?"
    assert len({c["id"] for c in cards}) == len(cards), "slugs are not unique"


def _errata():
    return read_jsonl(PROCESSED / "errata.jsonl", missing_ok=False)


def test_corpus_every_erratum_joins_a_real_card():
    card_ids = {c["id"] for c in _cards()}
    orphans = [e["id"] for e in _errata() if e["card_id"] not in card_ids]
    assert not orphans, f"errata pointing at no card: {orphans}"


def test_corpus_errata_are_in_the_merged_index():
    corpus = read_jsonl(PROCESSED / "corpus_chunks.jsonl", missing_ok=False)
    sources = {c["source"] for c in corpus}
    assert sources == {"rules", "cards", "errata"}, sources
    n_errata = sum(1 for c in corpus if c["source"] == "errata")
    assert n_errata == len(_errata()), \
        "an erratum exists but is not retrievable -- it would never be surfaced"


def test_corpus_errata_cards_are_annotated():
    """Retrieval can return a card without its erratum; the card must say so."""
    errata_cards = {e["card_id"] for e in _errata()}
    chunks = {c["card_id"]: c for c in
              read_jsonl(PROCESSED / "card_chunks.jsonl", missing_ok=False)}
    for cid in errata_cards:
        assert chunks[cid]["has_errata"] is True, cid
        assert "ERRATA" in chunks[cid]["text"], cid
    unflagged = [c for c in chunks.values()
                 if c["has_errata"] and c["card_id"] not in errata_cards]
    assert not unflagged, f"cards flagged with errata that have none: {unflagged[:3]}"


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

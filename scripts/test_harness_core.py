#!/usr/bin/env python3
"""Plain-assert tests for harness/core -- the game-agnostic engine itself.

No GPU/mlx dependency required. This is the "gold standard" template's own
regression coverage: a fix made here and never tested is a fix a future game
clone can silently lose the moment someone "simplifies" the code without
knowing why it looks the way it does.

Most of these are ported from `magic-llm/scripts/test_eval.py`, which had
real coverage of code this repo is the SOURCE of and this repo did not test
at all. Only the game-agnostic cases came across; the MTG-specific
assertions and that repo's own document-section citations stayed behind.

They deliberately concentrate on the modules nothing in this repo calls
(`stats.py`, `templating.py`, `calibration/controls.py`,
`eval/harness.py`), because those are the ones where a change here is
otherwise unexercised until a game repo pulls it -- into a repo with real
published numbers. See CLAUDE.md.

Usage:
    python scripts/test_harness_core.py
"""

import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import random

from harness.core.calibration.adjudicate import build_queue
from harness.core.calibration.controls import (
    build_error_assertions, half_answer, separation,
)
from harness.core.eval.harness import stratified_sample
from harness.core.eval.judge import (
    apply_judge_template, judge_batch_rubric, judge_prompt_for, rubric_correctness,
    verify_quoted_claims,
)
from harness.core.io import guard_shrink, write_jsonl_atomic
from harness.core.stats import pearson_r
from harness.core.templating import templatize, untemplatize


class _StubTokenizer:
    """`apply_chat_template` is all `judge_batch_rubric` needs from a tokenizer."""

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)


def _judge_returning(payload: str):
    """A fake `lm_generate` that answers with `payload`, recording the prompt."""
    seen = {}

    def gen(model, tokenizer, prompt, max_tokens, verbose=False):
        seen["prompt"] = prompt
        return payload

    return gen, seen


# A deliberately game-neutral answer: generic combat vocabulary only, no card
# names, no rule ids, nothing tied to a specific game's rules document.
_ANSWER = ("The attacking card assigns only lethal damage to the blocker, and "
           "the remainder carries through to the defending player.")
_KEY_POINTS = ["Excess damage carries past the blocker",
               "The defending player takes the remainder"]
_COMMON_ERRORS = ["Believes blocking prevents all of the damage"]


def test_build_queue_draws_disagreements_first():
    """The disputed-first ordering must actually reorder the queue.

    `disputed` held (rid, arm) tuples while the sort tested the "rid::arm"
    string, so membership was never true and the documented ordering was a
    silent no-op -- the queue was a plain shuffle. Nothing downstream could
    notice: the queue was still the right length and still well-formed.
    This exact bug shipped here (base-training-repo's own copy) even after
    it was found and fixed in one_piece_llm's independent copy, because
    nothing pushed the fix back and nothing here tested for it. Ported
    from one_piece_llm/scripts/test_one_piece.py.
    """
    # Several disputed records, not one: with a single disputed record the
    # shuffle alone puts it first often enough that the assertion passes even
    # when the ordering is a no-op (it did, on seed 42).
    ids = [f"q{i:02d}" for i in range(40)]
    disputed_ids = {"q07", "q19", "q33"}
    rubric_index = {i: {"id": i, "common_errors": ["e1"]} for i in ids}
    # Judge A fires on every record; judge B agrees except on the disputed ones.
    run_a = [{"id": i, "arms": {"base": {"answer": "a", "errors_made": ["e1"]}}}
             for i in ids]
    run_b = [{"id": i,
              "arms": {"base": {"answer": "a",
                                "errors_made": [] if i in disputed_ids else ["e1"]}}}
             for i in ids]

    with tempfile.TemporaryDirectory() as td:
        pa, pb = Path(td) / "a.jsonl", Path(td) / "b.jsonl"
        write_jsonl_atomic(pa, run_a)
        write_jsonl_atomic(pb, run_b)
        queue = build_queue([pa, pb], rubric_index, n=10)

    assert len(queue) == 10
    drawn = [q["record_id"] for q in queue]
    assert set(drawn[:3]) == disputed_ids, (
        f"all three disputed records must be drawn first, got {drawn}")


# --- stats.py ---------------------------------------------------------------

def test_pearson_r_endpoints_and_undefined_cases():
    """Inter-judge agreement, including when it is not defined.

    The NaN cases are the point. A judge that gives every answer the same
    score has zero variance, so r is undefined -- returning 0.0 there would
    read as "the two judges are uncorrelated", a far weaker and quite
    different claim than "one of them said nothing".
    """
    assert pearson_r([(1, 1), (2, 2), (3, 3), (4, 4)]) == 1.0
    assert pearson_r([(1, 4), (2, 3), (3, 2), (4, 1)]) == -1.0

    assert math.isnan(pearson_r([(3, 1), (3, 2), (3, 3)])), "constant series -> NaN"
    assert math.isnan(pearson_r([(1, 1), (2, 2)])), "under three pairs -> NaN"
    assert math.isnan(pearson_r([])), "no pairs -> NaN"

    r = pearson_r([(5, 5), (4, 3), (3, 4), (2, 2), (1, 1)])
    assert 0.8 < r < 1.0, f"a realistic pair correlates but is not 1.0, got {r}"


# --- eval/judge.py ----------------------------------------------------------

def test_rubric_correctness_scale():
    """The 1-5 score every rubric-judged number in this project comes from."""
    def score(ph, em, n_p, n_e):
        return rubric_correctness(ph, em, n_p, n_e)["correctness"]

    assert score([1, 2, 3, 4], [], 4, 3) == 5.0, "all points -> 5"
    assert score([], [], 4, 3) == 1.0, "no points -> 1"
    assert score([1, 2], [], 4, 3) == 3.0, "half the points -> 3"
    assert score([1], [], 4, 3) == 2.0, "one of four -> 2"


def test_rubric_correctness_ignores_claims_a_judge_could_not_have_meant():
    """A judge that miscounts must not be able to move the score.

    Not hypotheticals: real judge output has claimed point 7 of 4, and has
    listed the same point twice in one response. Every one of these is
    dropped rather than clamped, so a miscount costs nothing rather than
    silently crediting or penalising an answer.
    """
    def score(ph):
        return rubric_correctness(ph, [], 4, 3)["correctness"]

    assert score([1, 2, 7]) == 3.0, "index above n_points is dropped"
    assert score([2, 2, 2, 2, 2]) == 2.0, "duplicates cannot exceed the scale"
    assert score([0, 1]) == 2.0, "index 0 is dropped (points are 1-based)"
    assert score([-1, 1]) == 2.0, "negative index is dropped"

    # `bool` is a subclass of `int`, so a judge answering `"points_hit": true`
    # would be read as claiming point 1 -- a fabricated claim, awarded
    # silently, from a response that named no point at all.
    assert score(True) == 1.0, "a bare `true` claims nothing, not point 1"
    assert score([True, False]) == 1.0, "booleans are never rubric indices"
    assert score([True, 2]) == 2.0, "a real index alongside a bool still counts once"

    # Wrong-typed claims are dropped rather than coerced: "1" is a judge
    # writing prose, not a claim this scale can honour.
    assert score(["1", None, 2]) == 2.0, "only real int indices count"


def test_rubric_correctness_errors_are_reported_but_do_not_move_the_score():
    """Errors are extracted and reported, and deliberately do not score.

    The halving term was removed after positive controls measured the judge
    inventing errors against known-correct reference answers at a
    nontrivial rate -- a property of the judge, not of the answer. Dropping
    the FIELD would have been a different and much worse change than
    dropping its effect on the score, since blunder rate is defined on it.
    """
    assert rubric_correctness([1, 2, 3, 4], [1], 4, 3)["correctness"] == 5.0
    assert rubric_correctness([], [1], 4, 3)["correctness"] == 1.0
    assert (rubric_correctness([1, 2], [1, 2, 3], 4, 3)["correctness"]
            == rubric_correctness([1, 2], [], 4, 3)["correctness"])

    r = rubric_correctness([1, 2, 3, 4], [1, 3], 4, 3)
    assert r["errors_made"] == [1, 3], "errors still reported"
    assert r["scoring"] == "points_only", "the scoring version is recorded"


def test_rubric_correctness_legacy_halving_still_reproduces_old_numbers():
    """The superseded rule stays reachable, and is labelled as itself.

    Numbers published under the old convention can only be re-derived while
    this path exists. Removing it would make the earlier record
    unreproducible, which is a break rather than a revision.
    """
    def old(ph, em):
        return rubric_correctness(ph, em, 4, 3, halve_on_error=True)["correctness"]

    assert old([1, 2, 3, 4], []) == 5.0, "no error is unchanged"
    assert old([1, 2, 3, 4], [1]) == 3.0, "all points + an error -> halved"
    assert old([1, 2, 3, 4], [1, 2]) == 3.0, "two errors halve once, not twice"
    assert rubric_correctness([1], [], 4, 3, halve_on_error=True)["scoring"] != "points_only", (
        "the legacy path must record its own scoring version, not the current one")


def test_judge_prompt_for_needs_the_literal_phrase_and_silently_no_ops_without_it():
    """The GameConfig contract that nothing enforces.

    `judge_prompt_for` rewrites the label list by plain string replacement
    on the literal phrase "labeled A, B, C, D". A game module that
    paraphrases it gets no substitution and no error -- the judge is then
    told about four candidates when it was given two. This test exists to
    make that failure visible here rather than in a game repo's run.
    """
    base = "Grade the candidates labeled A, B, C, D against the rubric."

    two = judge_prompt_for(base, ["A", "B"])
    assert "labeled A and B" in two, two
    assert "A, B, C, D" not in two, "the four-label phrase must be gone"

    one = judge_prompt_for(base, ["A"])
    assert "labeled A " in one, one
    assert "A, B, C, D" not in one

    assert judge_prompt_for(base, ["A", "B", "C", "D"]) == base, (
        "four labels is the phrase already written, so it returns unchanged")

    paraphrased = "Grade the candidates labelled A through D against the rubric."
    assert judge_prompt_for(paraphrased, ["A", "B"]) == paraphrased, (
        "a paraphrase silently no-ops -- this is the documented trap, and if "
        "this assertion ever fails the contract in games/base has changed")


# --- calibration/controls.py ------------------------------------------------

def test_separation_refuses_a_one_sided_call():
    """The non-negotiable, enforced in code rather than by convention.

    A judge that never fires scores perfectly on a clean-only control; one
    that fires at everything scores perfectly on a planted-only control.
    Neither rate means anything alone, so there is deliberately no way to
    compute one.
    """
    clean = [{"fired": False}, {"fired": False}]
    planted = [{"fired": True}, {"fired": True}]

    for bad_clean, bad_planted in (([], planted), (clean, []), ([], [])):
        try:
            separation(bad_clean, bad_planted)
        except ValueError:
            pass
        else:
            raise AssertionError("separation must refuse a one-sided call")

    s = separation(clean, planted)
    assert s["p_fire_clean"] == 0.0
    assert s["p_fire_error"] == 1.0
    assert s["separation"] == 1.0
    assert s["n_clean"] == 2 and s["n_planted"] == 2


def test_separation_adds_optional_stats_only_when_every_row_supports_them():
    """Richer stats appear only if the caller's rows actually carry them.

    A caller that tracks a plain bool `fired` must not get a fabricated
    `hit_planted: 0%`, which would read as "the judge never caught the
    planted error" rather than "this caller never claimed to know".
    """
    plain_clean = [{"fired": False}]
    plain_planted = [{"fired": True}]
    s = separation(plain_clean, plain_planted)
    assert "mean_fired_clean" not in s
    assert "hit_planted" not in s

    rich = separation(
        [{"fired": False, "n_fired": 0}],
        [{"fired": True, "n_fired": 2, "hit": True}],
    )
    assert rich["mean_fired_clean"] == 0.0
    assert rich["mean_fired_error"] == 2.0
    assert rich["hit_planted"] == 1.0

    # One planted row missing `hit` is enough to withhold the key entirely.
    partial = separation(
        [{"fired": False, "n_fired": 0}],
        [{"fired": True, "n_fired": 1, "hit": True}, {"fired": True, "n_fired": 1}],
    )
    assert "hit_planted" not in partial, "a partial signal must not be averaged"


def test_build_error_assertions_emits_one_case_per_record_and_rotates():
    """One case per RECORD, not one per common_error entry.

    Easy to misread as "one case per listed error" and then report a
    planted-case count that does not match the corpus -- which has already
    caused a real miscount when reconciling two calibration runs. A record
    with several errors still contributes exactly one case; which error it
    plants rotates by the record's index, so the result describes the error
    list rather than the habits of whichever entry happens to be first.
    """
    records = [
        {"id": "r0", "common_errors": ["a0", "b0"]},
        {"id": "r1", "common_errors": ["a1", "b1"]},
        {"id": "r2", "common_errors": ["only2"]},
    ]
    cases, skipped_entries, skipped_records = build_error_assertions(records, " TAIL")

    assert len(cases) == 3, "one case per record, not per error entry"
    assert [c["id"] for c in cases] == ["r0", "r1", "r2"]
    assert cases[0]["n"] == 1 and cases[0]["claim"] == "a0", "record 0 -> error 0"
    assert cases[1]["n"] == 2 and cases[1]["claim"] == "b1", "record 1 -> error 1"
    assert cases[2]["n"] == 1, "a single-error record always plants that one"
    assert cases[0]["answer"] == "a0 TAIL", "the tail is appended verbatim"
    assert skipped_entries == 0 and skipped_records == 0

    # A record with nothing plantable is counted, not silently dropped: the
    # surviving records are the ones authored in claim form, so a rate over
    # them is a rate over a writing style unless the skip is visible.
    filtered, n_entries, n_records = build_error_assertions(
        records + [{"id": "r3", "common_errors": ["skipme"]}],
        " TAIL",
        is_usable_error=lambda e: e != "skipme",
    )
    assert len(filtered) == 3
    assert n_entries == 1 and n_records == 1


def test_half_answer_is_always_a_proper_prefix():
    """`partial` is a length-neutrality control, so it must really be half.

    Returning the whole answer would make the partial arm identical to the
    oracle arm and quietly turn the control into a no-op that always passes.
    """
    text = "One sentence. Two sentence. Three sentence. Four sentence."
    got = half_answer(text)
    assert got and got != text, "never the whole answer"
    assert text.startswith(got.rstrip()), "must be a genuine prefix"
    assert "Four sentence." not in got, "the final sentence is never included"

    # Nothing to split on: returned unchanged rather than emptied, since an
    # empty candidate would score 1.0 and read as a real measurement.
    single = "Only one sentence here."
    assert half_answer(single) == single
    assert half_answer("") == ""


# --- templating.py ----------------------------------------------------------

def test_templatize_replaces_the_longest_name_first():
    """A name that contains another name must not be half-replaced.

    Replacing the short name first leaves "[[entity1]] Lee" behind, which
    then never matches the long slot -- the entity is silently split in two.
    """
    slots = {"entity1": "Ann", "entity2": "Ann Lee"}
    out = templatize("Ann Lee met Ann at noon.", slots)
    assert out == "[[entity2]] met [[entity1]] at noon.", out


def test_templatize_matches_whole_words_only():
    slots = {"entity1": "Ann"}
    assert templatize("Anna went out.", slots) == "Anna went out."
    assert templatize("", slots) == ""
    assert templatize("Ann", {}) == "Ann", "no slots is a no-op, not a crash"


def test_untemplatize_leaves_an_unknown_slot_visible():
    """A typo must stay visible rather than silently deleting a claim."""
    assert untemplatize("[[entity1]] wins.", {"entity1": "Ann"}) == "Ann wins."
    assert untemplatize("[[entity9]] wins.", {"entity1": "Ann"}) == "[[entity9]] wins."
    assert untemplatize("", {"entity1": "Ann"}) == ""


def test_templatize_round_trips():
    slots = {"entity1": "Ann", "entity2": "Ann Lee"}
    original = "Ann Lee met Ann at noon."
    assert untemplatize(templatize(original, slots), slots) == original


# --- eval/harness.py --------------------------------------------------------

def test_stratified_sample_is_a_cross_section_not_a_prefix():
    """A flat stride under-samples whichever category is rarest.

    The rare category is the whole reason to stratify, so a sampler that
    drops it produces an eval set that cannot measure the thing it was
    built to measure.
    """
    rows = ([{"id": f"a{i}", "category": "common"} for i in range(50)]
            + [{"id": f"b{i}", "category": "medium"} for i in range(10)]
            + [{"id": "c0", "category": "rare"}])

    picked = stratified_sample(rows, limit=6)
    assert len(picked) == 6
    cats = {r["category"] for r in picked}
    assert cats == {"common", "medium", "rare"}, (
        f"every category must appear, got {cats}")

    # A small category exhausts and its budget spills to the rest, rather
    # than the sample coming up short.
    everything = stratified_sample(rows, limit=61)
    assert len(everything) == 61
    assert len({r["id"] for r in everything}) == 61, "no row is picked twice"

    # Asking for more than exists returns what exists, not a padded list.
    assert len(stratified_sample(rows, limit=999)) == 61


def test_stratified_sample_handles_rows_with_no_category():
    rows = [{"id": f"x{i}"} for i in range(5)]
    picked = stratified_sample(rows, limit=3)
    assert len(picked) == 3


# --- io.py ------------------------------------------------------------------

def test_guard_shrink_refuses_a_silent_corpus_collapse():
    """A rebuild that writes a subset over a full corpus is silent otherwise.

    Nothing downstream necessarily notices: retrieval and citation-checking
    just quietly get worse against a corpus that is still well-formed.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "corpus.jsonl"
        write_jsonl_atomic(path, [{"i": i} for i in range(100)])

        try:
            guard_shrink(path, 10)
        except SystemExit as e:
            assert "refusing to shrink" in str(e)
            assert "--force" in str(e), "the message must name the way out"
        else:
            raise AssertionError("a 100 -> 10 rebuild must be refused")

        guard_shrink(path, 60), "60% of 100 clears the default 50% floor"
        guard_shrink(path, 10, force=True), "--force is the deliberate override"
        guard_shrink(Path(td) / "absent.jsonl", 1), "a first write is never a shrink"

        # A stricter ratio is for pinned corpora that should never shrink.
        try:
            guard_shrink(path, 99, min_ratio=1.0)
        except SystemExit:
            pass
        else:
            raise AssertionError("min_ratio=1.0 must refuse even a 1-record loss")


def test_verify_quoted_claims_keeps_bare_integers():
    """A prompt variant that never asked for a quote must not be zeroed.

    Bare-integer claims are the older response shape. Reading them through
    the quote-checking path and dropping them would silently score every
    answer as claiming nothing.
    """
    assert verify_quoted_claims([1, 3], _ANSWER, 4) == ([1, 3], 0)
    # Out of range is dropped, but is NOT counted as a quote failure --
    # the two numbers mean different things and must not be conflated.
    assert verify_quoted_claims([1, 9], _ANSWER, 4) == ([1], 0)


def test_verify_quoted_claims_drops_a_claim_it_cannot_quote():
    """The receipt check: a claim the judge cannot quote is not a claim."""
    good = [{"n": 1, "quote": "assigns only lethal damage to the blocker"}]
    assert verify_quoted_claims(good, _ANSWER, 4) == ([1], 0)

    bad = [{"n": 2, "quote": "a sentence that is nowhere in the answer"}]
    assert verify_quoted_claims(bad, _ANSWER, 4) == ([], 1)

    assert verify_quoted_claims(good + bad, _ANSWER, 4) == ([1], 1), (
        "a mixed response keeps the honest claim and drops only the other")

    assert verify_quoted_claims(good, "", 4) == ([], 1), (
        "an empty answer cannot support any long quote")


def test_verify_quoted_claims_normalizes_case_and_whitespace():
    """A quote must survive reformatting, or the check punishes formatting.

    BOTH sides need normalizing, and they need separate assertions. The
    first version of this test only reflowed the QUOTE, against an answer
    that was already lowercase and single-spaced in the quoted span -- so
    it passed with the answer side left raw, and a mutation removing that
    normalization went uncaught.
    """
    reflowed_quote = [{"n": 1, "quote": "ASSIGNS ONLY LETHAL\n  DAMAGE to the blocker"}]
    assert verify_quoted_claims(reflowed_quote, _ANSWER, 4) == ([1], 0)

    # ...and now the other side: an ordinary quote against an answer that
    # arrives wrapped and inconsistently cased, as a generated answer does.
    wrapped_answer = ("The attacking card ASSIGNS ONLY lethal\n   damage to the "
                      "blocker, and the rest carries through.")
    plain_quote = [{"n": 1, "quote": "assigns only lethal damage to the blocker"}]
    assert verify_quoted_claims(plain_quote, wrapped_answer, 4) == ([1], 0)


def test_verify_quoted_claims_ignores_quotes_too_short_to_be_evidence():
    """Below the length threshold a quote matches too easily to mean anything.

    Asserted in BOTH directions -- a short quote that is absent from the
    answer must also be kept. If only the present case were checked, a
    threshold of zero would pass this test while doing nothing.
    """
    assert verify_quoted_claims([{"n": 1, "quote": "damage"}], _ANSWER, 4) == ([1], 0)
    assert verify_quoted_claims([{"n": 1, "quote": "zzzz"}], _ANSWER, 4) == ([1], 0)


def test_verify_quoted_claims_degrades_rather_than_collapsing():
    """Malformed claims are dropped individually, never taken as a whole score."""
    assert verify_quoted_claims([{"n": 1}], _ANSWER, 4) == ([1], 0), "no quote key -> kept"
    assert verify_quoted_claims([{"n": 1, "quote": "   "}], _ANSWER, 4) == ([1], 0)
    assert verify_quoted_claims([{"n": "1", "quote": "x" * 20}], _ANSWER, 4) == ([], 0)
    assert verify_quoted_claims(["point 1"], _ANSWER, 4) == ([], 0), "bare string ignored"
    assert verify_quoted_claims(None, _ANSWER, 4) == ([], 0)
    assert verify_quoted_claims(2, _ANSWER, 4) == ([2], 0), "a bare scalar is wrapped"

    # The out-of-range index is rejected BEFORE the quote is checked, so it
    # never counts as a quote failure.
    assert verify_quoted_claims(
        [{"n": 9, "quote": "not in the answer at all"}], _ANSWER, 4) == ([], 0)

    # bool is a subclass of int -- `true` must not be read as claim 1.
    assert verify_quoted_claims(True, _ANSWER, 4) == ([], 0)
    assert verify_quoted_claims([{"n": True, "quote": "x" * 20}], _ANSWER, 4) == ([], 0)


class _NoSystemRoleTokenizer:
    """A chat template that raises on a system message, as gemma-2-27b-it does."""

    def __init__(self):
        self.seen = None

    def apply_chat_template(self, messages, add_generation_prompt=True):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.seen = messages
        return "\n".join(m["content"] for m in messages)


def test_apply_judge_template_folds_system_when_the_template_refuses_it():
    """A whole judge model family is otherwise unusable over a formatting rule.

    gemma-2-27b-it raises rather than ignoring a system message, and every
    judge prompt built here opens with one -- so it died before grading
    anything.
    """
    tok = _NoSystemRoleTokenizer()
    out = apply_judge_template(tok, [{"role": "system", "content": "SYS"},
                                     {"role": "user", "content": "USER"}])
    assert [m["role"] for m in tok.seen] == ["user"], "the system turn is folded away"
    assert tok.seen[0]["content"] == "SYS\n\nUSER", tok.seen[0]["content"]
    assert "SYS" in out and "USER" in out, "no content is lost in the fold"


def test_apply_judge_template_is_a_no_op_for_a_normal_template():
    """Models that accept a system role must render byte-identically.

    If this ever changed, every judge run under a normal template would move,
    which would silently reprice published numbers.
    """
    tok = _StubTokenizer()
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert (apply_judge_template(tok, messages)
            == tok.apply_chat_template(messages, add_generation_prompt=True))


def test_apply_judge_template_reraises_an_unrelated_template_error():
    """Only a system-role complaint is retried; a real bug stays itself.

    Folding on every exception would turn an unrelated template failure into a
    confusing second failure with a different message.
    """
    # This tokenizer fails ONLY while a system turn is present, with an error
    # that has nothing to do with the system role. A tokenizer that failed
    # unconditionally could not tell the two behaviours apart: the retry would
    # raise the same error again and the test would pass either way. Here, a
    # wrongly-broadened fallback folds, retries, SUCCEEDS, and returns a string
    # instead of raising -- which is the failure this asserts against.
    class _FailsOnSystemForAnotherReason:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            if any(m["role"] == "system" for m in messages):
                raise ValueError("tokenizer is out of memory")
            return "folded"

    try:
        got = apply_judge_template(_FailsOnSystemForAnotherReason(),
                                   [{"role": "system", "content": "S"},
                                    {"role": "user", "content": "U"}])
    except ValueError as e:
        assert "out of memory" in str(e), "the original error must survive"
    else:
        raise AssertionError(
            f"an unrelated error must not be swallowed, got {got!r}")


def test_apply_judge_template_handles_a_system_turn_with_no_user_after_it():
    """A trailing system message must still reach the model, not vanish."""
    tok = _NoSystemRoleTokenizer()
    apply_judge_template(tok, [{"role": "system", "content": "ONLY SYS"}])
    assert [m["role"] for m in tok.seen] == ["user"]
    assert tok.seen[0]["content"] == "ONLY SYS"


def test_judge_batch_rubric_grades_under_a_template_that_refuses_system():
    """The end-to-end reason the fold exists: such a judge must still score."""
    payload = json.dumps({"A": {"points_hit": [1, 2], "errors_made": [],
                                "citation": 5, "note": ""}})
    gen, _ = _judge_returning(payload)
    out = judge_batch_rubric(gen, None, _NoSystemRoleTokenizer(), "sys", "q?",
                             _KEY_POINTS, _COMMON_ERRORS, {"only": "x"}, 256,
                             random.Random(0))
    assert out["only"]["correctness"] == 5.0, "a gemma-family judge grades normally"


def test_judge_batch_rubric_maps_shuffled_labels_back_to_the_right_arm():
    """The single most dangerous thing this module does.

    Candidates are hidden behind randomized A/B/C/D labels and mapped back
    afterwards. An off-by-one attributes every arm's score to a different
    arm -- a result that looks entirely normal and is exactly wrong. Six
    seeds, because one shuffle can be the identity by luck.
    """
    cands = {"base": "A", "base_rag": "B", "finetuned": "C", "finetuned_rag": "D"}
    for seed in range(6):
        arms = list(cands)
        random.Random(seed).shuffle(arms)
        label_of = {arm: chr(ord("A") + i) for i, arm in enumerate(arms)}
        # Only base_rag is awarded points, whichever label it landed on.
        payload = json.dumps({
            lbl: {"points_hit": [1, 2] if arm == "base_rag" else [],
                  "errors_made": [], "citation": 5, "note": ""}
            for arm, lbl in label_of.items()})
        gen, _ = _judge_returning(payload)
        out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                                 _KEY_POINTS, _COMMON_ERRORS, dict(cands), 256,
                                 random.Random(seed))
        scored = [a for a in out if out[a]["correctness"] > 1.0]
        assert scored == ["base_rag"], f"seed {seed}: scored {scored}, expected base_rag"


def test_judge_batch_rubric_reads_an_unwrapped_single_candidate():
    """Asked to grade one answer "labeled A", a judge often omits the wrapper.

    It emits the entry directly instead of {"A": {...}}, which a plain
    `scored.get("A")` reads as "the judge said nothing about A" -- a
    coverage loss that looks like a property of the prompt and is an
    artifact of the harness.
    """
    entry = json.dumps({"points_hit": [1, 2], "errors_made": [], "citation": 5, "note": ""})
    gen, _ = _judge_returning(entry)
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                             _KEY_POINTS, _COMMON_ERRORS, {"only": "x"}, 256,
                             random.Random(0))
    assert list(out) == ["only"]
    assert out["only"]["points_hit"] == [1, 2]

    # With several arms an unwrapped object cannot be attributed to any of
    # them, so it must still fail rather than be handed to an arbitrary arm.
    gen, _ = _judge_returning(entry)
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                             _KEY_POINTS, _COMMON_ERRORS,
                             {"a": "1", "b": "2"}, 256, random.Random(0))
    assert out == {}, "an unwrapped multi-arm response must be rejected"

    # A properly wrapped single candidate must not be double-wrapped.
    gen, _ = _judge_returning(json.dumps(
        {"A": {"points_hit": [1], "errors_made": [], "citation": 5, "note": ""}}))
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                             _KEY_POINTS, _COMMON_ERRORS, {"only": "x"}, 256,
                             random.Random(0))
    assert out["only"]["points_hit"] == [1]


def test_judge_batch_rubric_survives_valid_json_of_the_wrong_shape():
    """`"points_hit": 1` where a list was expected parses cleanly, then kills a run.

    Scoring it is right; crashing loses the whole run at whatever question
    the judge happened to fumble, and discarding it would score the answer
    as having hit nothing -- which reads as measured rather than broken.
    """
    gen, _ = _judge_returning('{"A": {"points_hit": 1, "errors_made": null, "citation": 5}}')
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                             _KEY_POINTS, _COMMON_ERRORS, {"only": "x"}, 256,
                             random.Random(0))
    assert out["only"]["correctness"] == 3.0, "1 of 2 points, scored not crashed"


def test_judge_batch_rubric_returns_nothing_rather_than_a_partial_score():
    """Unusable judge output must be empty, and an omitted arm must be absent.

    An arm reported at 1.0 because the judge never mentioned it reads as
    "this arm answered badly" -- a fabricated measurement, and the failure
    mode that once made an entire run score as unjudged without erroring.
    """
    cands = {"base": "A", "base_rag": "B", "finetuned": "C", "finetuned_rag": "D"}
    for label, payload in (("no JSON at all", "I cannot judge this."),
                           ("malformed JSON", '{"A": {"points_hit": [1,}')):
        gen, _ = _judge_returning(payload)
        out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                                 _KEY_POINTS, _COMMON_ERRORS, dict(cands), 256,
                                 random.Random(0))
        assert out == {}, f"{label} must yield nothing, not a partial score"

    gen, _ = _judge_returning(
        '{"A": {"points_hit": [1, 2], "errors_made": [], "citation": 5}}')
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "sys", "q?",
                             _KEY_POINTS, _COMMON_ERRORS, dict(cands), 256,
                             random.Random(0))
    assert len(out) == 1, "arms the judge omitted are absent, never zero-scored"


# --- webui/monitor.py --------------------------------------------------------
#
# A console that can bind beyond loopback, so most of what matters is a
# REFUSAL. In THIS repo the module is unexercised by any script -- the template
# ships no `scripts/monitor.py` -- which is exactly why it needs tests here: a
# game repo that pulls it gets whatever state it is in, and there is no local
# caller to notice a regression first.


def _monitor_actions():
    return [{
        "id": "echo", "group": "T", "label": "Echo", "cmd": "scripts/nonexistent.py",
        "args": [{"name": "--mode", "type": "choice", "default": "a",
                  "choices": ["a", "b"], "label": "m"},
                 {"name": "--secret", "type": "fixed", "value": ["hidden-value"]}],
    }]


def test_monitor_imports_without_fastapi_installed():
    """Importing the module must not require the web stack.

    `fastapi` is imported inside `build_app`, not at module scope, so a repo
    that has not installed the optional web dependencies can still import
    `harness.core` as a whole. Moving that import to the top would break
    `import harness.core.webui.monitor` for every such repo, and the failure
    would look like a broken harness rather than a missing extra.
    """
    import ast
    src = (REPO_ROOT / "harness" / "core" / "webui" / "monitor.py").read_text()
    tree = ast.parse(src)
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "fastapi" not in top_level, "fastapi must be imported lazily"
    assert "uvicorn" not in top_level, "uvicorn must be imported lazily"


def test_monitor_never_publishes_the_command_it_runs():
    """The client names an id; it must not be handed the script path.

    Publishing `cmd` invites building a request around the path rather than
    around the allowlist, which is the only thing standing between this server
    and remote code execution for anyone who can reach the port.
    """
    from harness.core.webui.monitor import _public_action
    pub = _public_action(_monitor_actions()[0])
    assert "cmd" not in pub
    # ...and a `fixed` argument is a server constant, not a client field
    assert [a.get("name") for a in pub["args"]] == ["--mode"]
    assert "hidden-value" not in json.dumps(pub)


def test_monitor_refuses_duplicate_action_ids():
    """Which one wins would depend on list order, so it would surface as
    'that ran the wrong script' rather than as an error."""
    from harness.core.webui.monitor import build_app
    dupes = _monitor_actions() * 2
    try:
        build_app(dupes, REPO_ROOT)
    except ValueError as e:
        assert "echo" in str(e), e
        return
    raise AssertionError("duplicate action ids were accepted")


def test_monitor_runner_rejects_a_choice_outside_its_list():
    """Otherwise `choice` is `text` with a nicer widget."""
    from harness.core.webui.runner import Runner
    action = _monitor_actions()[0]
    r = Runner({action["id"]: action}, REPO_ROOT)
    try:
        r.build_cmd(action, {"--mode": "; rm -rf /"})
    except ValueError as e:
        assert "not an allowed value" in str(e), e
        return
    raise AssertionError("an out-of-list choice was accepted")


def test_monitor_runner_passes_free_text_as_one_argv_entry():
    """Never through a shell, so metacharacters stay inert."""
    from harness.core.webui.runner import Runner
    action = {"id": "t", "cmd": "s.py", "label": "t",
              "args": [{"name": "--q", "type": "text", "default": ""}]}
    cmd = Runner({"t": action}, REPO_ROOT).build_cmd(
        action, {"--q": "a; rm -rf / && echo pwned"})
    assert cmd[-1] == "a; rm -rf / && echo pwned", cmd
    assert len(cmd) == 4, cmd


def test_monitor_runner_refuses_an_unknown_action():
    from harness.core.webui.runner import Runner
    r = Runner({}, REPO_ROOT)
    out = r.start("anything", {})
    assert out["ok"] is False and "unknown action" in out["error"], out


# --- gameplay/review.py + gameplay/store.py ----------------------------------
#
# The draft -> review -> promote gate, and the position store under it. Both are
# unexercised in this template -- no game, no gold set -- so the tests ARE the
# exercise. What they guard is a refusal: a gold position is gold because a
# person reviewed it, and this is the only code that can make that structural.

_CATS = ("counter-step", "don-allocation")
_DIFFS = ("basic", "intermediate", "advanced")


def _valid_pos(pid="pos-counter-step-0001"):
    return {"id": pid, "category": "counter-step", "difficulty": "intermediate",
            "key_points": ["a real claim"], "common_errors": ["a false claim"],
            "players": {"you": {}, "opp": {}}}


def _fresh_draft():
    from harness.core.gameplay.review import DRAFT_MARK
    d = dict(_valid_pos(), reviewed=False)
    d["key_points"] = [f"{DRAFT_MARK} write the real claim here"]
    d["common_errors"] = [f"{DRAFT_MARK} write the false claim here"]
    d["_board"] = "scaffolding a reviewer reads"
    d["_source"] = {"observed_next_action": "PASS"}
    return d


def test_review_needs_BOTH_signals_not_either():
    """Each signal is the obvious shortcut past the other.

    Flipping `reviewed` without rewriting leaves machine text in gold;
    rewriting without flipping means nobody claimed to have read the board.
    Requiring both is what makes "reviewed" mean the work happened.
    """
    from harness.core.gameplay.review import unreviewed_problems
    fresh = _fresh_draft()
    assert len(unreviewed_problems(fresh, _CATS)) >= 3

    flipped = dict(fresh, reviewed=True)
    probs = unreviewed_problems(flipped, _CATS)
    assert probs and all("reviewed" not in p for p in probs), probs

    rewritten = dict(fresh, key_points=["real"], common_errors=["false"])
    probs = unreviewed_problems(rewritten, _CATS)
    assert len(probs) == 1 and "reviewed" in probs[0], probs

    signed = dict(rewritten, reviewed=True)
    assert unreviewed_problems(signed, _CATS) == []


def test_promotion_strips_scaffolding_including_what_a_player_DID():
    """`observed_next_action` records what a player did, not what was correct.

    Carrying it into the gold set would sit an unreviewed answer beside a
    reviewed rubric with nothing marking which is which.
    """
    from harness.core.gameplay.review import to_position
    pos = to_position(dict(_fresh_draft(), reviewed=True))
    assert "reviewed" not in pos
    assert not [k for k in pos if k.startswith("_")], sorted(pos)
    assert "observed_next_action" not in json.dumps(pos)
    assert pos["id"] and pos["players"], "the position itself survived"


def test_promote_writes_nothing_for_a_refused_draft(tmp=None):
    """A refusal has to be visible AND inert."""
    import tempfile
    from harness.core.gameplay.review import promote
    appended = []
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "draft.json"
        path.write_text(json.dumps(_fresh_draft()), encoding="utf-8")
        promoted, refused = promote([path], validate=lambda p: [],
                                    append=appended.append, categories=_CATS,
                                    log=lambda *a: None)
    assert (promoted, refused) == (0, 1)
    assert appended == [], "a refused draft was appended anyway"


def test_promote_appends_a_signed_draft_and_reports_it():
    """The control: a gate that never opens is a wall."""
    import tempfile
    from harness.core.gameplay.review import promote
    appended = []
    signed = dict(_fresh_draft(), reviewed=True,
                  key_points=["real"], common_errors=["false"])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "draft.json"
        path.write_text(json.dumps(signed), encoding="utf-8")
        promoted, refused = promote([path], validate=lambda p: [],
                                    append=appended.append, categories=_CATS,
                                    log=lambda *a: None)
    assert (promoted, refused) == (1, 0)
    assert appended and appended[0]["id"] == signed["id"]
    assert "_board" not in appended[0], "scaffolding reached the append"


def test_write_draft_never_overwrites_a_half_written_rubric():
    """Regenerating is one command; recovering the prose is not."""
    import tempfile
    from harness.core.gameplay.review import write_draft
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.json"
        assert write_draft({"a": 1}, path) is True
        assert write_draft({"a": 2}, path) is False
        assert json.loads(path.read_text())["a"] == 1
        assert write_draft({"a": 2}, path, overwrite=True) is True
        assert json.loads(path.read_text())["a"] == 2


def test_store_common_problems_survives_a_malformed_position():
    """A validator that dies on the input it exists to reject reports nothing.

    Not hypothetical: a game's own card-id walk indexed `pos["players"][side]`
    directly and killed validation with a KeyError from inside the very check
    that had just recorded "players.you is missing".
    """
    from harness.core.gameplay.store import common_problems
    for bad in ({}, {"players": None}, {"players": []}, {"players": {"you": {}}}):
        probs = common_problems(bad, _CATS, _DIFFS)      # must not raise
        assert any("players." in p for p in probs) or "players must be an object" in probs


def test_store_common_problems_reports_every_problem_not_just_the_first():
    """A reviewer fixing one error per run stops running the check."""
    from harness.core.gameplay.store import common_problems
    probs = common_problems({}, _CATS, _DIFFS)
    assert len(probs) >= 6, probs
    assert common_problems(_valid_pos(), _CATS, _DIFFS) == []


def test_store_append_refuses_invalid_and_duplicate_ids():
    import tempfile
    from harness.core.gameplay.store import append_position, load_positions
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gold.jsonl"
        assert load_positions(path) == [], "a missing gold set reads as empty"
        try:
            append_position(_valid_pos(), path, validate=lambda p: ["nope"])
        except ValueError as e:
            assert "nope" in str(e)
        else:
            raise AssertionError("an invalid position was appended")
        assert not path.exists(), "a refused append wrote to the gold set"

        append_position(_valid_pos(), path, validate=lambda p: [])
        assert len(load_positions(path)) == 1
        try:
            append_position(_valid_pos(), path, validate=lambda p: [])
        except ValueError as e:
            assert "already exists" in str(e)
        else:
            raise AssertionError("a duplicate id was appended")
        assert len(load_positions(path)) == 1


def test_store_next_position_id_fills_gaps_and_scopes_by_category():
    from harness.core.gameplay.store import next_position_id
    assert next_position_id([], "counter-step") == "pos-counter-step-0001"
    taken = [{"id": "pos-counter-step-0001"}, {"id": "pos-counter-step-0002"}]
    assert next_position_id(taken, "counter-step") == "pos-counter-step-0003"
    assert next_position_id(taken, "don-allocation") == "pos-don-allocation-0001"
    assert next_position_id([{"id": "pos-counter-step-0002"}],
                            "counter-step") == "pos-counter-step-0001"


def test_monitor_console_javascript_actually_parses():
    """A string in one language inside a file of another gets no checking.

    magic-llm shipped exactly this: two Python-style `#` comments inside a
    raw-string HTML template were shipped verbatim to the browser, where `#`
    is a syntax error -- fatal to the whole <script>, so every view rendered
    blank. Nothing caught it: the module imported, the API returned correct
    JSON, --help exited 0.

    Skipped where `osascript` is unavailable: a missing interpreter is not a
    broken page.
    """
    import re
    import shutil
    import subprocess

    from harness.core.webui.monitor import CONSOLE_HTML

    blocks = re.findall(r"<script>(.*?)</script>", CONSOLE_HTML, re.S)
    assert blocks, "the console lost its script block"
    for block in blocks:
        for lineno, line in enumerate(block.splitlines(), 1):
            assert not line.strip().startswith("#"), (
                f"line {lineno} starts with '#' -- a Python comment inside a raw "
                f"string is JS, and a syntax error blanks the page")
    if not shutil.which("osascript"):
        return
    for i, js in enumerate(blocks):
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", "function _f(){" + js + "}"],
            capture_output=True, text=True)
        assert "SyntaxError" not in proc.stderr, (
            f"console script block {i} does not parse:\n{proc.stderr[:400]}")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()

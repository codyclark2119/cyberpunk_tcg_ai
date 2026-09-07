"""Generic positive-control mechanics for judge calibration.

Generic judge-calibration mechanics. These build
known-quality candidates (a correct reference, a half-complete answer, a
wrong-but-fluent answer, a refusal, and a planted-error assertion) from
existing rubric records, with no model generation or authoring involved --
and compute the one number a judge report should lead with: separation
between firing on a genuine error and firing on a clean answer.

Game-specific concerns intentionally excluded: how to render a "question"
(may be a rendered board, a rules question, ...) and what counts as a
"behaviour-shaped" (unusable) common_error entry. Callers supply
`question_text`/`reference_answer`/`is_usable_error` from their own game
module.
"""

import random
import re

REFUSAL_DEFAULT = "The rules provided do not cover this, so I cannot answer the question."

# Sentence boundaries that survive citation-style abbreviations ("601.2h",
# "3."): the split requires whitespace then a capital letter or an opening
# bracket, so a naive split on every period does not truncate mid-citation.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def half_answer(text: str) -> str:
    """Roughly the first half of an answer, cut at a sentence boundary.

    Picks the prefix whose length is CLOSEST to half, and never returns every
    sentence -- at least one sentence and at most n-1, which guarantees the
    result is a genuine proper prefix of the input. This makes `partial` a
    length-neutrality check: a half-length answer stating half the points
    should score about half, not less and not (nearly) full credit.
    """
    text = (text or "").strip()
    parts = _SENT_RE.split(text)
    if len(parts) < 2:
        return text
    target = len(text) / 2
    best_i, best_gap, acc = 0, None, 0
    for i, part in enumerate(parts[:-1]):     # never the whole answer
        acc += len(part) + 1
        gap = abs(acc - target)
        if best_gap is None or gap < best_gap:
            best_i, best_gap = i, gap
    return " ".join(parts[:best_i + 1]).strip()


def build_candidates(questions: list[dict], rng: random.Random,
                      refusal: str = REFUSAL_DEFAULT,
                      category_key: str = "category",
                      reference_key: str = "reference") -> list[dict]:
    """Four known-quality candidates per question/record.

        oracle    the reference answer, verbatim
        partial   the reference cut to roughly half, at a sentence boundary
        wrong     a DIFFERENT record's reference answer, same category
        refusal   a fixed refusal string

    `wrong` is another real answer (not gibberish) on purpose: a judge that
    separates the reference from noise but not from a fluent answer to a
    different question is being fooled by fluency alone.
    """
    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(q.get(category_key) or "?", []).append(q)

    out = []
    for q in questions:
        pool = [o for o in by_cat.get(q.get(category_key) or "?", []) if o is not q]
        if not pool:
            pool = [o for o in questions if o is not q]
        if not pool:
            continue
        other = rng.choice(pool)
        out.append({
            "q": q,
            "candidates": {
                "oracle": q[reference_key],
                "partial": half_answer(q[reference_key]),
                "wrong": other[reference_key],
                "refusal": refusal,
            },
        })
    return out


def build_error_assertions(records: list[dict], assertion_tail: str,
                            is_usable_error=None) -> tuple[list[dict], int, int]:
    """One candidate per record that asserts a listed `common_error` verbatim.

    Rotates which error is planted (record i takes error i % len(usable)) so
    the result describes the error list rather than the habits of whichever
    entry happens to be first.

    `is_usable_error(text) -> bool` lets a game module skip entries not
    written as an assertable claim (e.g. a description of behaviour rather
    than a sentence a wrong answer could contain verbatim); defaults to
    accepting everything. Returns (cases, n_entries_skipped,
    n_records_with_nothing_plantable) -- the second count is the one that
    biases a result, since the records that survive are exactly the ones
    authored in claim form, and a rate over them is a rate over a writing
    style rather than over the corpus.
    """
    is_usable_error = is_usable_error or (lambda _: True)
    cases, skipped_entries, skipped_records = [], 0, 0
    for i, rec in enumerate(records):
        errs = rec.get("common_errors") or []
        usable = [(n, e) for n, e in enumerate(errs, 1) if is_usable_error(e)]
        skipped_entries += len(errs) - len(usable)
        if not usable:
            skipped_records += 1
            continue
        n, claim = usable[i % len(usable)]
        cases.append({
            "rec": rec,
            "id": rec.get("id") or rec.get("gold_id"),
            "n": n,
            "claim": claim,
            "answer": claim + assertion_tail,
        })
    return cases, skipped_entries, skipped_records


def separation(clean: list[dict], planted: list[dict]) -> dict:
    """The one number a judge report is allowed to lead with.

    `P(fire | error) - P(fire | clean)`. Raises on a one-sided call rather
    than returning a partial result: a judge that never fires scores
    perfectly on the clean half alone, and one that always fires scores
    perfectly on the planted half alone -- neither rate means anything by
    itself, so there is deliberately no way to compute only one side.

    `r["fired"]` only needs to be truthy-or-not -- a plain bool is enough for
    the three core keys below. Two richer, OPTIONAL stats are included only
    when the input rows actually support them, rather than crashing or
    silently coercing a caller's simpler shape into something it isn't:

      - `mean_fired_*` needs `r["n_fired"]` (a count, not just whether
        anything fired) on every row of that half.
      - `hit_planted` needs `r["hit"]` on every planted row -- whether the
        judge caught the SPECIFIC error planted, not just whether it fired
        at all. A caller that never plants a specific, checkable error index
        (and so never has an honest way to say `hit`) correctly gets no
        `hit_planted` key rather than a misleading 0%.
    """
    if not clean or not planted:
        raise ValueError(
            "separation needs BOTH halves. A judge that never fires scores perfectly "
            "on the clean half; one that always fires scores perfectly on the planted "
            "half. Neither rate means anything alone.")
    p_clean = sum(1 for r in clean if r["fired"]) / len(clean)
    p_error = sum(1 for r in planted if r["fired"]) / len(planted)
    result = {
        "p_fire_clean": p_clean,
        "p_fire_error": p_error,
        "separation": p_error - p_clean,
        "n_clean": len(clean),
        "n_planted": len(planted),
    }
    if all("n_fired" in r for r in clean) and all("n_fired" in r for r in planted):
        result["mean_fired_clean"] = sum(r["n_fired"] for r in clean) / len(clean)
        result["mean_fired_error"] = sum(r["n_fired"] for r in planted) / len(planted)
    if all("hit" in r for r in planted):
        result["hit_planted"] = sum(1 for r in planted if r["hit"]) / len(planted)
    return result

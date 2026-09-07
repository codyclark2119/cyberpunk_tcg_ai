"""Generic rubric-judge scoring: turning extracted judge claims into a score.

This is the game-agnostic half of a rubric-judge's scoring machinery. It
knows nothing about any specific game, its cards, or its rule IDs -- it
operates on:

  - a rubric: `key_points` (facts a correct answer must state) and
    `common_errors` (false claims a correct answer must avoid), both supplied
    by the game/gold-data layer;
  - a set of anonymized candidate answers to grade against that rubric;
  - a judge LLM's structured JSON response naming which numbered points/errors
    each candidate hit.

The system PROMPT TEXT is deliberately NOT hardcoded here. Prompt wording is
part of what a game module supplies (it needs to name the game, its rules
document, and any citation-ID convention), so callers pass in `judge_system_prompt`
built from their own game-specific prompt module. Only the label-matching,
JSON-shape tolerance, quote-verification, and score arithmetic live here --
this is what catches silent scoring bugs
(`points_hit: 3` where a list was expected; a single-candidate response
returned unwrapped; a bool read as a fabricated claimed index), so keeping ONE
definition matters more than in most modules.
"""

import json
import random


def _normalize_for_quote(s: str) -> str:
    """Collapse whitespace and case so a quote survives reformatting."""
    return " ".join((s or "").split()).lower()


def _as_claim_list(v) -> list:
    """A judge's `points_hit`/`errors_made` field, coerced to a list.

    Valid JSON of the wrong SHAPE (`"points_hit": 3` where `[3]` was expected)
    parses cleanly and then raises `TypeError` in a naive set comprehension,
    silently killing a run at whatever question the judge happened to fumble.
    A bare scalar is wrapped rather than discarded, since discarding would
    score that answer as though it hit nothing -- worse than the crash it
    replaces, because it reads as measured rather than broken.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _claim_index(i, n_max: int) -> bool:
    """A usable 1-based rubric index.

    `bool` is a subclass of `int`, so a judge answering `"points_hit": true`
    would otherwise be read as claiming point 1 -- a fabricated claim, awarded
    silently, from a response that named no point at all.
    """
    return isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n_max


def verify_quoted_claims(claims, answer: str, n_max: int) -> tuple[list[int], int]:
    """Keep only claims whose quote really appears in the answer.

    Returns (kept_numbers, n_dropped). Also accepts bare-integer claims (no
    quote requested/required), so a judge prompt variant that does not ask for
    quotes degrades gracefully rather than dropping every claim.
    """
    kept: list[int] = []
    dropped = 0
    hay = _normalize_for_quote(answer)
    for c in _as_claim_list(claims):
        if isinstance(c, int) and not isinstance(c, bool):
            if 1 <= c <= n_max:
                kept.append(c)
            continue
        if not isinstance(c, dict):
            continue
        n = c.get("n")
        if not _claim_index(n, n_max):
            continue
        quote = c.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            kept.append(n)
            continue
        # Short quotes match too easily to be evidence of anything.
        needle = _normalize_for_quote(quote)
        if len(needle) >= 12 and needle not in hay:
            dropped += 1
            continue
        kept.append(n)
    return kept, dropped


# Correctness scoring version, recorded in every row it produces.
#
# "points_only" (current): correctness = 1 + 4 * (points_hit / n_points).
# "halved_v3" (legacy): the same, halved whenever `errors_made` was non-empty.
#
# Named rather than implied, because the two produce different numbers from
# the SAME judge output, and a run file that does not say which one it used
# cannot be compared to anything.
SCORING = "points_only"


def rubric_correctness(points_hit, errors_made, n_points: int, n_errors: int,
                        halve_on_error: bool = False) -> dict:
    """Turn rubric extraction into a 1-5 correctness score, in Python.

    `halve_on_error=True` reproduces a legacy scoring convention where a
    non-empty `errors_made` halved the point-based score. That term was found to
    subtract real credit from correct answers because
    the judge itself invented errors against known-correct reference answers
    at a nontrivial rate (a property of the *judge*, not of the scoring rule)
    -- kept here only so a stored run can be re-derived either way, not as a
    recommendation to use it. `errors_made` is still extracted and returned
    regardless, and remains what any blunder-rate metric should be defined on.
    """
    hit = {i for i in _as_claim_list(points_hit) if _claim_index(i, n_points)}
    err = {i for i in _as_claim_list(errors_made) if _claim_index(i, n_errors)}
    fraction = len(hit) / n_points if n_points else 0.0
    if err and halve_on_error:
        fraction *= 0.5
    return {
        "correctness": round(1 + 4 * fraction, 2),
        "points_hit": sorted(hit),
        "points_total": n_points,
        "errors_made": sorted(err),
        "scoring": "halved_v3" if halve_on_error else SCORING,
    }


# Everything a rubric grading carries besides the score itself. Both a
# fresh-run writer and a rescore-from-stored-output writer need to carry
# exactly the same set of diagnostic fields onto the record they produce, or
# the two code paths silently diverge on which fields survive.
RUBRIC_DIAGNOSTICS = ("scoring", "quote_drops", "all_errors_fired", "error_contradiction")


def carry_diagnostics(dest: dict, entry: dict) -> dict:
    """Copy the non-score fields of a rubric grading onto a stored arm record."""
    for k in RUBRIC_DIAGNOSTICS:
        if entry.get(k) is not None:
            dest[k] = entry[k]
    return dest


# The judge system prompt says "labeled A, B, C, D" in prose while the labels
# are generated dynamically from however many candidate arms are present. That
# agrees at four arms and silently invites the judge to drop the last label at
# any other count. Rewritten only when the count differs from 4, so runs using
# exactly four arms reproduce byte-for-byte.
_FOUR_LABEL_PHRASE = "labeled A, B, C, D"


def apply_judge_template(judge_tokenizer, messages: list[dict]) -> str:
    """Render a JUDGE prompt, folding `system` into the first user turn when the
    model's chat template refuses a system role.

    Some chat templates raise on a system message rather than ignoring it --
    `gemma-2-27b-it` fails with `TemplateError: System role not supported` and
    dies before grading anything. Every judge prompt built here opens with a
    system message, so without this an entire model family is unusable as a
    judge over a formatting convention.

    Only the fold is attempted, and only when the error actually mentions the
    system role: anything else re-raises, so a real template bug still surfaces
    as itself rather than being retried into a confusing second failure.

    **Judge paths only, deliberately.** The prompt for a model UNDER TEST is
    built by the caller's own game module, and an adapter is only valid for the
    prompt format it was trained on -- silently reshaping that would invalidate
    the adapter and present as a capability result. Judges are stateless
    graders with no adapter, so the same reshaping is free here. It does change
    the prompt for such a model, which is a reason to compare its numbers only
    against other runs of itself.
    """
    try:
        return judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as exc:                      # jinja2.TemplateError, and kin
        if "system" not in str(exc).lower():
            raise
        folded, carried = [], ""
        for m in messages:
            if m["role"] == "system":
                carried += m["content"].rstrip() + "\n\n"
            elif m["role"] == "user" and carried:
                folded.append({"role": "user", "content": carried + m["content"]})
                carried = ""
            else:
                folded.append(m)
        if carried:                               # system with no user turn after it
            folded.append({"role": "user", "content": carried.rstrip()})
        return judge_tokenizer.apply_chat_template(folded, add_generation_prompt=True)


def judge_prompt_for(base: str, labels: list[str]) -> str:
    """A judge system prompt, with its label list matching the real one.

    `base` must contain the literal phrase "labeled A, B, C, D" wherever the
    prompt names the candidate labels -- game-specific judge prompt modules
    should reuse that phrase for this substitution to have any effect.
    """
    if len(labels) == 4:
        return base
    if len(labels) == 1:
        named = f"labeled {labels[0]}"
    else:
        named = "labeled " + ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return base.replace(_FOUR_LABEL_PHRASE, named)


def judge_batch_rubric(
    lm_generate, judge_model, judge_tokenizer,
    judge_system_prompt: str,
    question: str, key_points: list[str], common_errors: list[str],
    candidates: dict[str, str], max_tokens: int, rng: random.Random,
    quote_required: str = "none",
) -> dict:
    """Score against an enumerated rubric behind randomized A/B/C/D labels.

    `judge_system_prompt` is supplied by the caller's game-specific prompt
    module; it must instruct the judge to return per-label
    `{"points_hit": [...], "errors_made": [...], "citation": <1-5>, "note": "..."}`
    JSON (optionally with per-claim quotes, see `quote_required`).

    `quote_required`:
      - "none"   -- points_hit/errors_made are bare integers (no receipts).
      - "errors" -- errors_made entries must carry a verbatim quote, checked
                    against the candidate's own text; unverifiable claims are
                    dropped. points_hit is left as bare integers.
      - "all"    -- both points_hit and errors_made require quotes.
    """
    arms = list(candidates)
    rng.shuffle(arms)
    label_to_arm = dict(zip((chr(ord("A") + i) for i in range(len(arms))), arms))

    points_block = "\n".join(f"{i}. {p}" for i, p in enumerate(key_points, 1))
    user = f"QUESTION:\n{question}\n\nKEY POINTS:\n{points_block}\n"
    if common_errors:
        errors_block = "\n".join(f"{i}. {e}" for i, e in enumerate(common_errors, 1))
        user += f"\nCOMMON ERRORS:\n{errors_block}\n"
    user += "\n" + "\n\n".join(f"CANDIDATE {label}:\n{candidates[arm]}" for label, arm in label_to_arm.items())

    messages = [
        {"role": "system", "content": judge_prompt_for(judge_system_prompt, list(label_to_arm))},
        {"role": "user", "content": user},
    ]
    prompt = apply_judge_template(judge_tokenizer, messages)
    raw = lm_generate(judge_model, judge_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        scored = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}

    # A single candidate is often returned UNWRAPPED: asked to grade one
    # answer "labeled A", the judge emits
    #   {"points_hit": [...], "errors_made": [...], "citation": 3}
    # rather than {"A": {...}} -- which is reasonable, and which a naive
    # `scored.get("A")` silently reads as "the judge said nothing about A".
    # Only rewritten when there is exactly one label and the object looks like
    # an entry rather than a label map; with several arms an unwrapped object
    # cannot be attributed and must still fail.
    if (len(label_to_arm) == 1 and not any(k in scored for k in label_to_arm)
            and any(k in scored for k in ("points_hit", "errors_made", "citation"))):
        scored = {next(iter(label_to_arm)): scored}

    out = {}
    for label, arm in label_to_arm.items():
        entry = scored.get(label)
        if not isinstance(entry, dict):
            continue
        raw_points = entry.get("points_hit") or []
        raw_errors = entry.get("errors_made") or []
        drops = 0
        if quote_required == "errors":
            raw_errors, drops = verify_quoted_claims(raw_errors, candidates[arm], len(common_errors))
        elif quote_required == "all":
            answer_text = candidates[arm]
            raw_points, d1 = verify_quoted_claims(raw_points, answer_text, len(key_points))
            raw_errors, d2 = verify_quoted_claims(raw_errors, answer_text, len(common_errors))
            drops = d1 + d2
        computed = rubric_correctness(raw_points, raw_errors, len(key_points), len(common_errors))
        computed["quote_drops"] = drops
        # The "every error at once" signature: a rubric's common_errors are
        # alternative wrong answers, so committing all of them simultaneously
        # is usually the judge using the error list as a "this answer is bad"
        # flag rather than reading errors off the text. Reported, never
        # corrected -- an answer really can be wrong on every axis, so
        # silently dropping errors would change a metric on a heuristic.
        n_err = len(common_errors)
        fired_all = n_err >= 3 and len(computed["errors_made"]) == n_err
        computed["all_errors_fired"] = fired_all
        computed["error_contradiction"] = bool(
            fired_all and key_points and len(computed["points_hit"]) >= len(key_points) / 2)
        citation = entry.get("citation")
        out[arm] = {
            **computed,
            "citation": citation if isinstance(citation, (int, float)) else 3,
            "note": entry.get("note", ""),
            "scored_by": "rubric",
        }
    return out


def judge_batch_anonymized(
    lm_generate, judge_model, judge_tokenizer,
    judge_system_prompt: str,
    question: str, reference: str,
    candidates: dict[str, str], max_tokens: int, rng: random.Random,
) -> dict:
    """Score candidates behind randomized A/B/C/D labels against a reference
    answer, using a holistic (non-rubric) judge prompt.

    Use `judge_batch_rubric` instead whenever a `key_points` rubric exists --
    this is the fallback for questions that carry only a reference answer.
    """
    arms = list(candidates)
    rng.shuffle(arms)
    label_to_arm = dict(zip((chr(ord("A") + i) for i in range(len(arms))), arms))

    user = (f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
            + "\n\n".join(f"CANDIDATE {label}:\n{candidates[arm]}" for label, arm in label_to_arm.items()))

    messages = [
        {"role": "system", "content": judge_prompt_for(judge_system_prompt, list(label_to_arm))},
        {"role": "user", "content": user},
    ]
    prompt = apply_judge_template(judge_tokenizer, messages)
    raw = lm_generate(judge_model, judge_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        scored = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}

    out = {}
    for label, arm in label_to_arm.items():
        entry = scored.get(label)
        if not isinstance(entry, dict):
            continue
        out[arm] = {
            "correctness": entry.get("correctness"),
            "citation": entry.get("citation"),
            "note": entry.get("note", ""),
            "scored_by": "anonymized",
        }
    return out

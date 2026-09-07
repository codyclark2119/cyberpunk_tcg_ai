"""Cyberpunk TCG prompt templates: the SFT/inference message shape and the
rubric judge's system prompt.

`build_messages` is the SINGLE definition of this game's prompt shape, used
identically by SFT dataset assembly (`harness.core.sft.build`) and by
eval-time generation (`harness.core.eval.harness`) -- see
`games/base/game_interface.py`'s `GameConfig.build_messages` contract.
Changing this shape after training an adapter invalidates it; an adapter is
only valid for the prompt format it saw. Use `harness.core.prompt_fingerprint`
to stamp an adapter to its training-time prompt shape once this stabilizes.
"""

import re

# This game publishes no rules version string -- unlike One Piece, whose
# Comprehensive Rules carry a "Last updated" line. The document-level
# `updated_at` from the rules API is the only version marker there is, so
# that is what this tracks. `scripts/fetch_rules.py --print-updated-at`
# reports the live value; when it moves past this date, re-fetch, re-ingest,
# and change this string in the same commit.
RULES_VERSION = "2026-09-01"

# Rule ids are dot-separated multi-level numbers, e.g. "9.2.3.1", "1.4".
# Two or more segments are required so a bare top-level section number ("9")
# is never mistaken for a specific rule. A rule cited at the end of a sentence
# ("...see 9.2.3.") yields "9.2.3" without the trailing period, because each
# `\.` here must be followed by a digit.
RULE_ID_RE = re.compile(r"\b\d+(?:\.\d+)+\b")


def extract_citations(text: str) -> set:
    """Rule ids cited in an answer, for citation-validity scoring."""
    return set(RULE_ID_RE.findall(text))


SYSTEM_PROMPT = (
    "You are a rules assistant for Cyberpunk TCG (the Cyberpunk Trading Card "
    "Game, produced by Weird Co. under licence from CD PROJEKT RED). Answer "
    "questions about game rules, card interactions, and legal plays precisely "
    "and concisely, grounded in the official Comprehensive Rules (last updated "
    f"{RULES_VERSION}). When you cite a rule, use its exact numeric id (e.g. "
    "\"9.2.3\"). When you reference a specific card, give its full name "
    "including the subtitle (e.g. \"V: Streetkid\"), since several characters "
    "appear on more than one card and only the subtitle tells them apart."
)

GROUNDED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + "\n\nYou will be given retrieved rules text and/or card "
    "text relevant to the question below. Use it to ground your answer, and "
    "prefer it over your own recollection if the two ever conflict -- the "
    "retrieved text is pinned to the current rules version and card database; "
    "your training data may be stale. This game launched in 2026 and is still "
    "in beta, so it is very likely that you have never seen it before and that "
    "anything you seem to recall about it is confabulated. Say plainly when "
    "the retrieved text does not answer the question."
)


def build_messages(question: str, context: str | None = None) -> list[dict]:
    """Chat messages for one question, EXCLUDING the final assistant turn.

    Used as-is for bare question -> answer training examples
    (`harness.core.sft.build.build_messages`) and, with `context` supplied,
    for retrieval-grounded examples/inference
    (`harness.core.sft.build.build_grounded_messages`,
    `harness.core.eval.harness`). Keeping ONE function for both paths is what
    prevents training and inference prompts from silently drifting apart.
    """
    if context:
        system = GROUNDED_SYSTEM_PROMPT
        user = f"Rules/card text:\n{context}\n\nQuestion: {question}"
    else:
        system = SYSTEM_PROMPT
        user = question
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# The literal phrase "labeled A, B, C, D" is required here -- see
# `harness.core.eval.judge.judge_prompt_for`, which does a plain string
# replace on that phrase to rewrite this prompt for candidate counts other
# than 4. Paraphrase it and the substitution silently no-ops.
#
# The JSON shape below is not decoration either. `judge_batch_rubric` parses
# the response with raw.find("{") / raw.rfind("}") and gives up entirely --
# returning {}, scoring nothing -- if the response contains no braces at all.
# A prompt that describes the grading task perfectly but never names the
# output format gets fluent markdown back and every arm silently scores as
# *unjudged*, which reads exactly like "not yet run" rather than "broken".
# That happened for real in the One Piece instance of this template. Keep the
# word JSON and the shape explicit.
JUDGE_SYSTEM_PROMPT = (
    "You are grading answers to a rules question about Cyberpunk TCG (the "
    "Cyberpunk Trading Card Game), against a rubric of key points a correct "
    "answer must state and common errors it must avoid.\n\n"
    "You will be given the question, the rubric, and several candidate "
    "answers labeled A, B, C, D. For each candidate, identify which numbered "
    "key points it asserts and which numbered common errors it asserts. Only "
    "count a point as hit if the candidate's own words assert it -- do not "
    "give credit for points a correct answer could imply but this candidate "
    "did not actually state.\n\n"
    "This is a new game, released in 2026, so grade strictly against the "
    "rubric and not against your own knowledge of it -- you almost certainly "
    "have none, and several of its terms mean something specific here that "
    "they do not mean in other card games. In particular: a card is \"spent\" "
    "when turned sideways and \"ready\" when upright; only READY Units may "
    "attack and only SPENT Units may be attacked, which is the reverse of the "
    "convention in most other games. \"Eddies\" are the resource, \"Gigs\" are "
    "dice and the win condition, and \"RAM\" is a deckbuilding limit rather "
    "than anything spent during play.\n\n"
    "When a candidate cites a rule id (e.g. \"9.2.3\"), do not penalize an "
    "unfamiliar-looking id -- rule ids in this game are not sequential or "
    "intuitive, and a correct citation can look arbitrary.\n\n"
    "Respond with ONLY a JSON object, no other text before or after it. One "
    "key per candidate label. Each value has this exact shape:\n\n"
    "{\n"
    '  "A": {\n'
    '    "points_hit": [1, 3],\n'
    '    "errors_made": [2],\n'
    '    "citation": 4,\n'
    '    "note": "one short sentence on why"\n'
    "  },\n"
    '  "B": { ... }\n'
    "}\n\n"
    "\"points_hit\" and \"errors_made\" are lists of the NUMBERED key points / "
    "common errors this candidate's answer asserts -- bare integers, an empty "
    "list if it asserts none. \"citation\" is your own 1-5 rating of how well "
    "the candidate's own citations (if any) support its answer; use 3 if the "
    "candidate cited nothing. \"note\" is one short sentence, not a restated "
    "rubric."
)

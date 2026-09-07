"""The draft -> review -> promote gate. Game-agnostic, and it is a REFUSAL.

A gold position is gold because a person reviewed it. That sentence is in every
repo's CLAUDE.md, and this module is the only part of the pipeline that can
make it structurally true rather than a convention people mean to follow.

## Two steps with a human edit in between, on purpose

    draft     a file with every derived fact filled in and the rubric left as
              clearly-marked DRAFT text; `reviewed` is false
    (edit)    a person rewrites the rubric and sets `reviewed: true`
    promote   validates, REFUSES anything still unsigned, then appends

A single interactive prompt would be fewer keystrokes and worse. A rubric is
prose that wants a real editor; the draft file is diffable in git, so a review
shows up as a reviewable change; and an accept-by-keypress flow makes
"reviewed" mean *a key was pressed*.

## Why two independent signals, not one

`reviewed: true` alone is trivially flipped without doing the work. Generated
text alone is removable by deleting lines. Requiring both means the cheap
shortcut for each is blocked by the other -- and the two tests that matter are
exactly those: flipping the flag without rewriting, and rewriting without
flipping.

## What is game-specific and stays out

What to draft, how to render a board, and what makes a position valid. Those
arrive as callables and data. This module knows only that some strings are
marked, some keys are scaffolding, and nothing unsigned may be written.
"""

import json
from pathlib import Path

# The marker that keeps a draft out of gold. Every generated rubric line
# carries it and `promote` refuses while any survive. Deliberately ugly and
# deliberately not a Unicode character: it has to survive being pasted between
# an editor, a terminal and a browser without normalising into something the
# check no longer matches.
DRAFT_MARK = "DRAFT:"

# Keys that exist to help a reviewer and must never reach the gold set.
# `_`-prefixed by convention; `reviewed` is the signature itself, not content.
SCAFFOLD_PREFIX = "_"
REVIEW_FLAG = "reviewed"


def is_draft_text(value: str) -> bool:
    return DRAFT_MARK in (value or "")


def unreviewed_problems(draft: dict, categories=None,
                        rubric_fields=("key_points", "common_errors")) -> list[str]:
    """Why this draft is not ready to be gold. Empty means a person signed it.

    Both signals are required. Neither implies the other, and each is the
    obvious shortcut past the other.
    """
    problems: list[str] = []
    if not draft.get(REVIEW_FLAG):
        problems.append(
            f'"{REVIEW_FLAG}" is false -- set it to true only after you have '
            "actually read the board and written the rubric")
    for field in rubric_fields:
        for i, line in enumerate(draft.get(field) or []):
            if is_draft_text(line):
                problems.append(f"{field}[{i}] is still generated text: {line[:60]}...")
    if categories is not None and draft.get("category") not in categories:
        problems.append(f"category {draft.get('category')!r} is not one of "
                        + ", ".join(sorted(categories)))
    return problems


def to_position(draft: dict) -> dict:
    """Strip the reviewing scaffolding; what is left is the position.

    The scaffolding is not merely noise. It carries `observed_next_action` --
    what a real player DID, which is not a claim that it was correct. Letting
    that reach the gold set would put an unreviewed answer next to a reviewed
    rubric, where the next reader has no way to tell them apart.
    """
    return {k: v for k, v in draft.items()
            if not k.startswith(SCAFFOLD_PREFIX) and k != REVIEW_FLAG}


def write_draft(draft: dict, path: Path, overwrite: bool = False) -> bool:
    """Write one draft. Returns False if it already existed and was kept.

    Never overwrites by default: the file may hold a half-written rubric that
    took real thought, and regenerating it is a one-line command while
    recovering the prose is not.
    """
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return True


def promote(paths, validate, append, categories=None, log=print) -> tuple[int, int]:
    """Validate and promote reviewed drafts. Returns (promoted, refused).

    `validate(pos) -> list[str]` and `append(pos) -> None` are the game's, so
    this never imports a game or touches a path itself. A refusal is reported
    per problem and writes nothing -- and the whole batch is not atomic on
    purpose: one bad draft should not block the good ones a reviewer already
    finished.
    """
    promoted = refused = 0
    for path in paths:
        path = Path(path)
        draft = json.loads(path.read_text(encoding="utf-8"))
        problems = unreviewed_problems(draft, categories)
        pos = to_position(draft)
        problems += list(validate(pos))
        if problems:
            refused += 1
            log(f"REFUSED {path.name}")
            for p in problems:
                log(f"    - {p}")
            continue
        append(pos)
        promoted += 1
        log(f"promoted {pos.get('id')}  <- {path.name}")
    return promoted, refused

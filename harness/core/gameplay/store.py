"""The gold position store: read, id, validate-skeleton, append.

Game-agnostic. A "position" here is only a dict with an `id`, a `category`, a
`difficulty`, a rubric (`key_points` / `common_errors`) and two `players` --
everything past that is the game's business and is checked by a callable the
game supplies.

## Why these four functions and not more

They were not chosen by guessing what a third game might want. Both existing
games had already written all four independently, with the same semantics and
slightly different spellings:

    magic-llm  scripts/gameplay/positions.py      1569 lines
    one_piece  games/one_piece/gameplay/positions.py  487 lines

and the shared part is small: loading, id allocation, the presence-and-
membership half of validation, and an atomic append that refuses. Everything
that made those files big -- MTG's mana/timing/phase/targeting checks, One
Piece's card-line grammar and DON!! rendering -- is game-specific and stays
where it is.

`common_problems` is deliberately a *skeleton*, not a validator. It returns the
problems it can see and knows nothing about cards, actions or rules; a game
composes it with its own checks. Making it complete for one game would make it
wrong for the next, which is the trap this template's own CLAUDE.md warns about
-- a change here is unexercised until a game repo pulls it into a repo with
real published numbers.

## `append_position` refuses rather than warns

`data/gold/positions.jsonl` backs published measurements, and the loudest
convention in every repo here is that gold means a person reviewed it. So the
write path validates and raises; there is no "append anyway" flag, because the
moment one exists it becomes the path of least resistance at 2am.
"""

from pathlib import Path

from harness.core.io import read_jsonl, write_jsonl_atomic

PLAYERS = ("you", "opp")

# The fields every game's position needs before anything can score it. A
# position missing `key_points` cannot be graded at all, and one missing
# `common_errors` can never register a blunder -- so both are structural, not
# stylistic.
REQUIRED_FIELDS = ("id", "category", "difficulty", "key_points", "common_errors")


def load_positions(path: Path) -> list[dict]:
    """Every position in the file, or `[]` when it does not exist yet.

    Missing-is-empty rather than an error because the gold set legitimately
    does not exist until the first position is promoted, and every caller --
    the eval, the id allocator, the appender -- has to work on that day too.
    """
    return read_jsonl(path, missing_ok=True)


def next_position_id(existing: list[dict], category: str, prefix: str = "pos") -> str:
    """`<prefix>-<category-slug>-NNNN`, skipping ids already taken.

    Note what this does NOT do: ids authored in some other shape do not reserve
    a slot, because the scan only recognises this one. Two hand-typed ids in a
    category would both leave `...-0001` free.
    """
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", (category or "position").lower()).strip("-")
    taken = {p.get("id") for p in existing}
    n = 1
    while f"{prefix}-{slug}-{n:04d}" in taken:
        n += 1
    return f"{prefix}-{slug}-{n:04d}"


def common_problems(pos: dict, categories, difficulties,
                    players=PLAYERS, required=REQUIRED_FIELDS) -> list[str]:
    """The game-neutral half of validation. Every problem, not just the first.

    Returning all of them matters more than it looks: a reviewer fixing a
    position one error at a time re-runs the check once per mistake, and the
    slow feedback is what makes people stop running it.

    Tolerates a malformed `pos` rather than raising on one. A validator that
    dies on the input it exists to reject reports nothing at all -- which is
    not hypothetical: `position_card_ids` in one_piece indexed
    `pos["players"][side]` directly and killed `validate_position` with a
    KeyError from inside the very check that had just recorded "players.you is
    missing".
    """
    problems: list[str] = []

    for field in required:
        if not pos.get(field) and pos.get(field) != 0:
            problems.append(f"missing required field: {field}")

    if categories is not None and pos.get("category") not in categories:
        problems.append("category must be one of: " + ", ".join(sorted(categories)))
    if difficulties is not None and pos.get("difficulty") not in difficulties:
        problems.append("difficulty must be one of: " + ", ".join(sorted(difficulties)))

    sides = pos.get("players") or {}
    if not isinstance(sides, dict):
        problems.append("players must be an object")
        sides = {}
    for side in players:
        if side not in sides:
            problems.append(f"players.{side} is missing")

    for field in ("active_player", "priority"):
        val = pos.get(field)
        if val and val not in players:
            problems.append(f"{field} must be one of {tuple(players)}")

    return problems


def append_position(pos: dict, path: Path, validate=None) -> None:
    """Append one position, rewriting the file atomically. Refuses on problems.

    `validate(pos) -> list[str]` is the game's full validator. It is a
    parameter rather than an import because `harness/` may not reach into
    `games/`, and optional only so a game that has not written one yet still
    gets the duplicate-id guard -- not so validation can be skipped by
    preference.

    Raises ValueError listing EVERY problem, so a reviewer fixes the position
    in one pass.
    """
    problems = list(validate(pos)) if validate is not None else []
    if problems:
        raise ValueError(
            f"refusing to append {pos.get('id')!r} -- not a valid position:\n  "
            + "\n  ".join(problems))
    existing = load_positions(path)
    if any(p.get("id") == pos.get("id") for p in existing):
        raise ValueError(f"id already exists: {pos.get('id')}")
    existing.append(pos)
    write_jsonl_atomic(path, existing)

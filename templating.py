"""Generic entity-slot templating.

Used to replace card names with `[[card1]]`-style placeholders so a single
rubric could score any instantiation of a ruling that names cards. The
mechanism itself has nothing to do with cards: it is
"replace named entities with numbered placeholders and back", which is
useful for any game whose rubrics reference specific named things (units,
locations, characters, ...).

The placeholder pattern intentionally avoids curly braces, since many games'
card/rules text already uses `{...}` for costs or reminder text, which would
collide with `str.format`-style templating.
"""

import re

SLOT_RE = re.compile(r"\[\[(entity\d+)\]\]")


def templatize(text: str, slots: dict[str, str]) -> str:
    """Named entities -> [[entityN]]. Longest name first, so a name that
    contains another name cannot be half-replaced."""
    if not text or not slots:
        return text
    for slot, name in sorted(slots.items(), key=lambda kv: len(kv[1]), reverse=True):
        text = re.sub(r"\b" + re.escape(name) + r"\b", f"[[{slot}]]", text)
    return text


def untemplatize(text: str, slots: dict[str, str]) -> str:
    """[[entityN]] -> entity names. An unknown slot is left as-is rather than
    blanked, so a typo stays visible instead of silently deleting a claim."""
    if not text:
        return text
    return SLOT_RE.sub(lambda m: (slots or {}).get(m.group(1), m.group(0)), text)

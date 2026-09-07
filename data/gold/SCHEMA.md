# Gold set schema (generic, game-agnostic template)

This is the **one** gold-data schema this template ships. It is deliberately
generic across games -- a citation-based rules game (Magic, One Piece TCG)
and a game with no numbered rulebook can both use it -- rather than each game
inventing its own fields. Add a field only if a game genuinely needs one, and
document it here as an *additive* extension so other games' tooling can keep
ignoring it safely.

One JSON object per line in `data/gold/gold_questions.jsonl`.

## Why a rubric instead of one reference answer

Scoring a model's answer against a single prose reference answer has two
common failure modes: the judge rewards length, and two
independent judges agree only weakly, because "how similar is this to my one
phrasing?" has no objective answer.

`key_points` and `common_errors` replace that with something checkable: *did
the response assert these facts, and did it avoid these mistakes?* That is
length-neutral by construction (a one-line answer hitting every key point
scores full marks) and far more reproducible across judges.

## Writing good key_points

1. **Fold the verdict into a substantive claim.** Never a bare `"Yes"`,
   `"No"`, or a single noun as point 1 -- write out what is actually true, so
   an answer cannot score on its opening word alone.
2. **One atomic, independently checkable assertion per point.** A point that
   bundles multiple claims with "and"/"so" cannot be answered yes/no, and
   that is exactly where two judges will diverge.
3. **Cut scaffolding.** Prefer 2-4 sharp points over 5-8 soft ones. A point
   true of every instance of the mechanic (rather than this specific
   situation) discriminates nothing.
4. **`common_errors` should encode the real trap**, not the negation of a key
   point. Negating a key point adds nothing the key point did not already
   catch.
5. **Write a `common_error` as the false CLAIM, not as the player's
   behaviour.** The judge is asked the same extraction question about both
   lists -- *which of these did the candidate assert?* -- and that question
   is only answerable when the entry is itself a sentence a wrong answer
   could contain verbatim.

   | Don't (behaviour) | Do (claim) |
   | --- | --- |
   | Thinks the effect blocks all the damage | The effect blocks all the damage |
   | Plays card X, spending a resource to save a small amount of life | Playing card X here is worth the life cost |
   | Holds a removal spell for a better target | Holding the removal spell for a better target is correct here |

   The test: **could a wrong answer contain this sentence verbatim?** If not,
   it is a description of a mistake rather than the mistake itself.

## Fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | stable slug, e.g. `combat-lethal-damage-01`. Never reuse. |
| `question` | yes | the question as a **player** would ask it -- natural, first-person, messy is fine |
| `paraphrases` | no | other natural phrasings of the *same* question; used to test robustness to wording |
| `answer` | yes | the ruling in the judge's own words, 1-4 sentences |
| `key_points` | yes | list of assertions a correct answer **must** contain. Scoring unit. |
| `common_errors` | no | wrong beliefs a response must **not** assert |
| `citations` | yes if the game has a numbered rulebook | citation ids supporting the ruling (e.g. rule/section numbers), validated against the pinned rules corpus. Generic name so a citation-less game can simply omit it. |
| `entities` | no | named game entities referenced (cards, units, characters, ...), validated against that game's card/entity database. |
| `category` | yes | one of this game's defined categories (see the validator) |
| `difficulty` | yes | `basic` \| `intermediate` \| `advanced` |
| `format_context` | no | e.g. a specific match format, if rulings differ by format |
| `source` | yes | `judge:<name-or-initials>`, `<community-source>:<url>`, `official-ruling`, `rules-derived` |
| `verified_by` | no | second reviewer who confirmed it -- mark disputed rulings clearly |
| `rules_version` | yes | the pinned rules-corpus version this was checked against (this game's analog of a comprehensive-rules effective date) |
| `notes` | no | edge cases, why players get it wrong, related interactions |

## Isolation from other games

This schema, and the gold data built on it, live entirely under
`data/<this game>/gold/`. A sibling game instantiated from this same template
gets its **own** clone of this file and its **own** `data/<that game>/gold/`
directory -- gold sets are never shared or merged across game instances, to
keep cross-game contamination impossible by construction (see the top-level
README's "Cloning this template for a new game" section).

## Example

```json
{
  "id": "example-lethal-damage-01",
  "question": "my creature has 5 toughness and got hit by something that only deals 1 damage but has a special lethal-damage keyword. does it die?",
  "paraphrases": [
    "does a special lethal-damage effect kill something even if the raw damage number is way less than its toughness?"
  ],
  "answer": "Yes -- the keyword marks any nonzero amount of damage from that source as lethal, regardless of the printed damage number.",
  "key_points": [
    "Any nonzero damage from a source with the lethal-damage keyword is treated as lethal",
    "The printed damage amount does not need to meet or exceed toughness"
  ],
  "common_errors": [
    "The damage amount must still meet or exceed toughness for the creature to die"
  ],
  "citations": ["000.0a"],
  "entities": [],
  "category": "combat-damage",
  "difficulty": "basic",
  "source": "judge:CC",
  "rules_version": "2026-01-01",
  "notes": "Common confusion: players conflate the keyword with damage prevention rather than a lethality override."
}
```

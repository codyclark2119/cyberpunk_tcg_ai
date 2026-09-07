# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A local LLM harness for **Cyberpunk TCG** -- the Cyberpunk Trading Card Game
produced by Weird Co. under licence from CD PROJEKT RED. Rules Q&A over the
official Comprehensive Rules and card database, via retrieval-augmented
generation, LoRA fine-tuning, and a rubric-based judge, running on Apple
Silicon via MLX.

The repo is a clone of a game-agnostic template with one game filled in. Two
layers, and the boundary between them is the main architectural rule:

| Layer | What belongs there |
| --- | --- |
| `harness/core/` | game-agnostic engine: jsonl I/O, retrieval index, judge math, SFT assembly, calibration |
| `games/cyberpunk/` | everything Cyberpunk-specific: markup, card normalization, chunking, prompts |
| `scripts/` | thin CLIs wiring the two together |

**Nothing under `harness/` may import from `games/`.** The dependency is
strictly one-directional; the game layer is reached only through callables and
strings passed as ordinary arguments. Keep it that way -- it is what makes the
`harness/core` subtree link back to `base-training-repo` possible at all.

`README.md` is the how-to-run document and the file tree. This file is the
working guidance.

## The one fact that should change how you work here

**This game launched in 2026 and is in open beta.** That is not colour; it is
the single most important operational fact about this repo, and it has
consequences everywhere:

- **The corpus is a moving target.** The Comprehensive Rules carry an
  `updated_at` (currently `2026-09-01`), card errata are published, and new
  sets are landing. `data/` is a *snapshot*, and any measurement is a
  measurement against that snapshot.
- **You have no prior knowledge of this game, and neither does any base
  model.** Anything a model appears to recall about Cyberpunk TCG is
  confabulated. Several of its terms mean something specific here that they do
  not mean elsewhere -- most dangerously, only **ready** (upright) Units may
  attack and only **spent** (sideways) Units may be attacked, which is the
  reverse of the convention in most other card games. Both the system prompt
  and the judge prompt say so explicitly, on purpose.
- **Being early is the point.** Resources for this game are thin, which is
  the reason this repo exists. That makes capturing data promptly worth more
  than polishing the pipeline around it, and it makes a *dated, reproducible*
  snapshot worth more than a fresh one.

## Setup

```bash
source mlx_env/bin/activate     # every command below assumes this
```

`requirements.txt` is curated, not a `pip freeze`.

No pytest. Tests are runnable scripts with plain asserts, and none need a GPU
or the network:

```bash
python scripts/test_cyberpunk.py      # 42 tests -- the game module and ingestion
python scripts/test_harness_core.py   # 48 tests -- harness/core itself
```

That is the whole suite. Every test in `test_cyberpunk.py` has been
mutation-verified -- 18 mutations, 18 caught -- and doing it found two tests
that passed for the wrong reason (see *Traps* below).

## The data pipeline

```bash
python scripts/fetch_rules.py     # -> data/raw/comprehensive_rules.json
python scripts/fetch_cards.py     # -> data/raw/cards/{_index,_filters,<slug>}.json
python scripts/ingest.py          # -> data/processed/rules.jsonl, rule_sections.jsonl
python scripts/build_cards.py     # -> data/processed/card_database.jsonl
python scripts/chunk_corpus.py    # -> rule_chunks, card_chunks, corpus_chunks
```

Every step is idempotent and re-runnable from `data/raw/` with no network.

### This game publishes its data as JSON, not HTML

`cyberpunktcg.com` is a thin client over a public JSON API at
`https://api.netdeck.gg/api`, and `robots.txt` is `Allow: /`. **There is no
HTML parsing anywhere in this repo**, which is the single biggest difference
from `one_piece_llm`, whose card path is a 199-line HTML scraper because its
official card list has no data endpoint.

Endpoints, discovered by reading the site's own JS bundle rather than guessing:

| What | Endpoint |
| --- | --- |
| Comprehensive Rules | `/cyberpunk/comprehensive-rules` |
| Card index (paged) | `/cards/cyberpunk?limit=&offset=` |
| One card | `/cards/cyberpunk/<slug>` |
| Filter vocabularies | `/cards/cyberpunk/filters` |

The rules come back as a **tree of typed nodes**, not prose -- `parent_id`,
`display_number`, `body_markdown`, `stable_anchor`. So `ingest.py` has no
numbering heuristic, no indentation parsing, and no PDF extraction. Do not add
one.

### Things the API declares but does not populate

Checked across all 151 cards in the launch set, and the reason two derived
fields exist at all:

- **`keywords` is `[]` on every card.** The keywords are really there, inside
  the `{Blocker}` brace markup in `rules_text`. `games/cyberpunk/markup.py`
  reads them out. Nothing else does.
- **`flavor_text` is `null` on every card.** Nothing to recover. Carried
  through so a future set that populates it is not mistaken for a parser bug.
- **`printings` is `[]` on the index endpoint** but populated on the per-card
  endpoint. That is why `fetch_cards.py` makes 151 individual requests rather
  than two paged ones: with Beta and Retail versions of the same card at
  different rarities and collector numbers, the printing list is the only
  record of which products a card appears in.

### Printed versus referenced keywords

`markup.printed_keywords` returns the keywords a card **has**, not every
keyword its text mentions -- and the difference is real. Riot Shield says
rivals must pay more to use Go Solo; Valentino: Guerrera can attack ready
Units that have Blocker; two cards *grant* Adrenaline to something else. A
field that conflated those answers "which cards have Blocker" wrongly.

The rule is positional: a keyword that opens an ability line is printed on the
card, one that appears mid-sentence is being referred to. **That is a
heuristic**, so it is checked against a field derived independently of the
rules text: only a Legend with GO SOLO has a printed power, and across all 27
Legends the two sets agree exactly. `test_go_solo_matches_printed_power` pins
that. If it ever fails, the heuristic has stopped being true -- revisit it,
do not patch around it.

### An unknown keyword must fail the suite

Because the API's `keywords` field is empty, an unrecognised brace span would
otherwise be silently classified `"unknown"` and vanish from every derived
field with nothing to notice. So `classify` never drops anything, both build
scripts report unknown spans loudly, and `test_no_unknown_markup_in_corpus`
fails on one. **A new set shipping a new keyword is meant to break the build.**
When it does, add it to the vocabulary in `games/cyberpunk/markup.py` --
`TIMING_TRIGGERS`, `KEYWORDS` or `SYMBOLS`, matching the official gameplay
guide's own two-category distinction.

## Traps this repo has already fallen into

- **A link regex that only understood the braced form.** The rules document
  writes cross-references two ways -- 12 as `{[x](y)}` and 91 as plain
  `[x](y)`. A first pass handled only the braced form, so nine tenths of the
  document's links survived into the corpus as raw `#rule-<uuid>` fragments.
  Found by grepping the *ingested output* for `](`, not by reading the code.
- **One record in 713 with escaped brackets.** Rule 11.11.1.4 writes its link
  label as `[\[CALL\]]`. A `[^\]]*` label stops at the escaped `]`, so the
  link matched nothing at all. One record out of 713 -- which is why
  `test_corpus_rule_text_has_no_unrendered_markup` asserts over the whole
  file rather than a hand-picked example.
- **A markdown image eaten as a link.** Four rules embed a figure as
  `![alt](url)`. A link-only pattern consumed the `[alt](url)` and left an
  orphaned `!` glued to the front of the alt text.
- **A test that passed for the wrong reason.** The first version of
  `test_walk_orders_siblings_by_sort_order_not_response_order` used ids "a",
  "a1", "a2", "b" -- which sort into the correct order *by id alone*, so it
  passed just as happily against a `walk` that ignored `sort_order` entirely.
  A mutation proved it. The fixture now names them so that id order and
  `sort_order` disagree.
- **A mutation that is not the bug proves nothing.** `_LEADING_MARKUP_RE`
  originally carried a leading `^`, which is exactly redundant with the
  `.match()` that applies it. Deleting the `^` changed no behaviour, so the
  mutation run reported "NOT CAUGHT" and looked like a coverage gap. The real
  mutation is `.match()` -> `.search()`; the `^` is gone and a comment says
  why.

## Conventions

Read `README.md`'s "Conventions this template assumes" section, inherited from
`base-training-repo` -- never train on the eval set, never silently change a
measurement, report negative results as negative, calibrate a judge on both
sides, verify by running rather than by reading. This file does not repeat
them.

Code conventions:

- Scripts are CLIs: `argparse` with `description=__doc__`, and a module
  docstring explaining *why*, with a usage block.
- Data is `.jsonl`, one record per line. Read with `harness.core.io.read_jsonl`;
  write with `write_jsonl_atomic`.
- Destructive rebuilds refuse to shrink a corpus without `--force`, via
  `harness.core.io.guard_shrink` -- never a re-implementation.
- **`REPO_ROOT` anchors every canonical path** so scripts run from any working
  directory.
- **A card's identity is its slug**, never its name. Sixteen names are shared
  by two or three different cards -- three separate cards are named "V" -- and
  that is also why the prompts insist on the subtitle.

## Non-negotiables

**Never commit.** The user commits; prepare a message and say it is ready.

**Never train on the eval set.**

**Never silently change a measurement.** Once `data/gold/` and `eval/runs/`
hold real data they back published numbers. Check `git diff --quiet` on them
before touching, and say so if they change.

**Gold data means a person reviewed it.** A model-generated candidate is a
draft until a human promotes it.

**Never re-fetch and re-ingest as a side effect of some other task.** The
corpus is dated and the game is in beta; a refresh is its own change, with its
own commit, and it moves `RULES_VERSION` in `games/cyberpunk/prompts.py` in
the same commit.

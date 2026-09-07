# cyberpunk-llm -- a rules/gameplay AI harness for Cyberpunk TCG

A local LLM harness for **Cyberpunk TCG**, the Cyberpunk Trading Card Game
produced by Weird Co. under licence from CD PROJEKT RED: rules Q&A grounded in
the official Comprehensive Rules and card database, via retrieval-augmented
generation, LoRA fine-tuning, and a rubric-based judge eval harness, running
locally on Apple Silicon via MLX.

Cloned from `base-training-repo`, the game-agnostic template, with
`harness/core/` linked back to it via `git subtree`. See that repo's README
for the template contract and `CLAUDE.md` here for the working conventions.

## Status

The corpus is captured, ingested and tested. There is no gold set, no trained
adapter and no eval run yet -- those are the next steps, in that order.

| | |
| --- | --- |
| Comprehensive Rules | 713 nodes (660 rules, 53 sections), `updated_at` 2026-09-01 |
| Cards | 151, across 13 sets, 2-6 printings each |
| Retrieval corpus | 225 chunks (74 rules + 151 cards) |
| Tests | 90 (42 game module + 48 harness core), all mutation-verified |
| Gold set | **none yet** |
| Eval runs | **none yet** |

This game launched in 2026 and is in open beta. Everything in `data/` is a
dated snapshot, not a stable reference.

## Setup

```bash
python -m venv mlx_env && source mlx_env/bin/activate
pip install -r requirements.txt
```

Or, from the workspace root, `scripts/setup.sh cyberpunk_llm`.

## Running the pipeline

Every step is idempotent, and everything after the two `fetch_` steps runs
offline from `data/raw/`.

```bash
python scripts/fetch_rules.py     # Comprehensive Rules  -> data/raw/
python scripts/fetch_cards.py     # 151 cards + index    -> data/raw/cards/
python scripts/ingest.py          # rules  -> rules.jsonl, rule_sections.jsonl
python scripts/build_cards.py     # cards  -> card_database.jsonl
python scripts/chunk_corpus.py    # chunks -> rule_chunks, card_chunks, corpus_chunks
python scripts/rag.py build       # embedding index over corpus_chunks.jsonl
```

Two commands worth knowing for a beta game:

```bash
python scripts/fetch_rules.py --print-updated-at    # has the rules document moved?
python scripts/build_cards.py --report-unknown-markup   # has a new keyword shipped?
```

## Tests

No pytest. Runnable scripts with plain asserts, none needing a GPU or the
network:

```bash
python scripts/test_cyberpunk.py      # 42 tests -- game module + ingestion
python scripts/test_harness_core.py   # 48 tests -- harness/core itself
```

## Where the data comes from

`cyberpunktcg.com` is a thin client over a **public JSON API**
(`https://api.netdeck.gg/api`), and `robots.txt` is `Allow: /`. These are the
same unauthenticated requests the public site makes to render its own pages.
There is no HTML parsing in this repo.

The Comprehensive Rules arrive as a tree of typed nodes rather than prose --
`parent_id`, `display_number`, `body_markdown`, `stable_anchor` -- so
ingestion reads structure straight off the source instead of inferring it.

Two normalizations are applied to the raw snapshot, both deliberate and both
tested:

- **CloudFront image signatures are stripped.** Every `image_url` is signed
  with an `Expires` timestamp, so two fetches of an unchanged card differ by
  several hundred bytes. Left alone, `git diff` on a re-fetch would report all
  151 files changed and tell you nothing about which cards actually changed --
  exactly the question worth asking during a beta with live errata. Lossless:
  the API also returns `source_image_url`, the same URL unsigned, and the two
  agree after stripping.
- **Brace markup is rendered to prose** for the retrievable text, with the raw
  form kept alongside it. See `games/cyberpunk/markup.py`.

## What's here

```text
games/cyberpunk/
  markup.py       the {...} text markup: keywords, triggers, links, rule refs
  cards.py        one raw card API record -> one corpus record
  chunking.py     rules chunking policy + how a card reads once retrieved
  prompts.py      the SFT/inference message shape and the judge's system prompt
  config.py       the GameConfig that wires all of the above into harness/core

scripts/
  fetch_rules.py    fetch the Comprehensive Rules
  fetch_cards.py    fetch the card catalog
  ingest.py         rules JSON -> rules.jsonl + rule_sections.jsonl
  build_cards.py    card JSON  -> card_database.jsonl
  chunk_corpus.py   -> rule_chunks / card_chunks / corpus_chunks
  rag.py            build and query the embedding index
  rescore_stored.py, stamp_adapter.py    (inherited template CLIs)
  test_cyberpunk.py, test_harness_core.py

data/
  raw/         the dated snapshot: comprehensive_rules.json, cards/*.json
  processed/   everything derived from it
  gold/        SCHEMA.md -- the gold-data schema; no gold data yet

harness/core/  the game-agnostic engine, subtree-linked to base-training-repo
```

## Conventions this instance keeps

Inherited from the template, and they are the rules that actually matter:

- **Never train on the eval set.** Exclude eval records from training on more
  than one key -- an id that changes when a record is promoted will silently
  pass an equality check that then *asserts* it caught everything.
- **Never silently change a measurement.** Once `data/gold/` and `eval/runs/`
  hold real data they back published numbers.
- **Report negative results as negative.** "Retrieval did not beat a
  full-context prompt" is a result worth writing down plainly, and a number
  from one judge is a statement about that judge.
- **Calibrate a judge on both sides.** `separation()` computes
  `P(fire | error) - P(fire | clean)` and refuses a one-sided call by design.
  Never quote half of it.
- **Verify by running, not by reading.** A `--help` that exits 0 proves almost
  nothing. Both real bugs found while building this repo's ingestion were
  found by grepping the *output*, not by reviewing the code that produced it.

One convention specific to this game, because it is in beta:

- **A corpus refresh is its own change.** Re-fetching and re-ingesting moves
  every derived file at once; do it deliberately, in its own commit, and move
  `RULES_VERSION` in `games/cyberpunk/prompts.py` in the same commit.

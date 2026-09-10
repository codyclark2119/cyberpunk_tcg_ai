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
| Errata | 4, `updated_at` 2026-08-29 |
| Community decks | 272 public lists, 174 distinct authors (pseudonymized) |
| Retrieval corpus | 229 chunks (74 rules + 151 cards + 4 errata) |
| Tests | 137 (89 game module + 48 harness core), all mutation-verified |
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
python scripts/fetch_errata.py    # card errata          -> data/raw/errata.json
python scripts/ingest.py          # rules  -> rules.jsonl, rule_sections.jsonl
python scripts/build_cards.py     # cards  -> card_database.jsonl
python scripts/build_errata.py    # errata -> errata.jsonl  (joins to cards)
python scripts/chunk_corpus.py    # chunks -> rule/card/errata chunks + corpus_chunks
python scripts/rag.py build       # embedding index over corpus_chunks.jsonl
```

`build_errata.py` must run after `build_cards.py`: it joins each erratum to a
card and refuses to write if any erratum matches none.

Commands worth knowing for a beta game:

```bash
python scripts/fetch_rules.py --print-updated-at        # has the rules document moved?
python scripts/fetch_errata.py --print-updated-at       # have new errata landed?
python scripts/build_cards.py --report-unknown-markup   # has a new keyword shipped?
python scripts/build_errata.py --check-applied          # errata vs. the card record, side by side
```

## Deckbuilding

`games/cyberpunk/deckbuilding.py` implements the four construction rules
(exactly 3 uniquely-named Legends, 40-50 cards excluding them, at most 3
copies of a card, and the per-colour RAM ceiling) and returns structured
violations rather than a bare boolean, so an assistant can say *why* a deck is
illegal.

The community decklist host at `/decks?tab=community` is fetched by
`scripts/fetch_decks.py` -- 272 public lists, each carrying the official
builder's own `is_valid` verdict and a `stats` block with `ramBudget` and
`ramUsage`. That is an independent check on the rules implementation:

```bash
python scripts/fetch_decks.py                            # -> data/raw/decks/
python scripts/check_deck_rules.py --negative-control    # both halves, see below
```

**Read both halves of that check.** Every public deck is `is_valid: true` --
the official builder refuses to save an illegal one -- so agreeing with it on
all 272 is something a validator that always answered "legal" would also
achieve. The half that can fail is `--negative-control`, which breaks each
real deck in each way a rule forbids and checks the validator fires. It is
what caught the one real bug in these rules: duplicate Legend *names* were
counted per entry rather than per copy, so two copies of one Legend never
tripped the rule, and the 272/272 agreement said nothing about it.

Deck owner handles are **pseudonymized** at fetch time, following this
workspace's standing rule for captured third-party handles. The pseudonym is
stable per author, so "these decks share a builder" survives while the handle
does not.

## Archetypes

`scripts/cluster_decks.py` groups the 272 decks by what they actually contain
-- weighted Jaccard over copy counts, average-linkage agglomerative, with a
similarity threshold rather than a chosen number of clusters.

```bash
python scripts/cluster_decks.py                       # null, sensitivity, archetypes
python scripts/cluster_decks.py --threshold 0.5 --json-out /tmp/arch.jsonl
```

It prints its own evidence before its own conclusions, in that order:

- **A null model.** Decks sharing a colour identity overlap by construction,
  so a cluster of same-colour decks is not a finding. The null replaces every
  deck with a random legal deck of the *same* colour identity, size and
  copy-count shape. At similarity >= 0.5 the real corpus has **339 pairs and
  the null has 1**; above 0.6 the null has none. That gap is what licenses
  calling these archetypes rather than colour groupings.
- **A sensitivity curve.** The archetype count is a property of the
  threshold, not of the game -- 20 clusters of three or more decks at 0.40,
  16 at 0.50 -- so the whole curve is printed rather than one number.

The clusters are corroborated by something the clustering never saw: deck
names. "Burn 'em Down Tri V3" and "V3.5" land together; so do
"control+alt+placide", "Controlinho" and "Braindancer".

Archetype output is **not committed**. It is a description of a moving format
derived from a dated snapshot, and it regenerates in about two seconds from
the decks that are committed.

## Tests

No pytest. Runnable scripts with plain asserts, none needing a GPU or the
network:

```bash
python scripts/test_cyberpunk.py      # 89 tests -- game module + ingestion
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

Errata come from the site's CMS API rather than its rendered `/errata` page,
for a reason worth stating: at capture time the rendered page carried **three**
errata and the CMS carried **four**. The missing one was live and absent from
freshly fetched HTML, so a scraper would have silently dropped an erratum that
officially supersedes printed card text.

Errata are not redundant with the card database either. Three of the four are
already reflected in the card records; the fourth is not, so neither source is
authoritative alone. `scripts/build_errata.py --check-applied` prints each
erratum beside the card record it corrects.

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
  errata.py       the errata page -> per-card errata, joined by name
  deckbuilding.py the deck construction rules, and a validator for them
  archetypes.py   deck similarity, clustering, and the null model for it
  chunking.py     chunking policy + how a card reads once retrieved
  prompts.py      the SFT/inference message shape and the judge's system prompt
  config.py       the GameConfig that wires all of the above into harness/core

scripts/
  fetch_rules.py    fetch the Comprehensive Rules
  fetch_cards.py    fetch the card catalog
  fetch_errata.py   fetch the errata page from the CMS API
  fetch_decks.py    fetch the public community decklists (handles pseudonymized)
  ingest.py         rules JSON -> rules.jsonl + rule_sections.jsonl
  build_cards.py    card JSON  -> card_database.jsonl
  build_errata.py   errata JSON -> errata.jsonl, joined to cards
  chunk_corpus.py   -> rule / card / errata chunks + the merged corpus
  check_deck_rules.py  the deck rules vs. the official builder, both halves
  cluster_decks.py  archetypes, with the null and sensitivity curve first
  rag.py            build and query the embedding index
  rescore_stored.py, stamp_adapter.py    (inherited template CLIs)
  test_cyberpunk.py, test_harness_core.py

data/
  raw/         the dated snapshot: comprehensive_rules.json, errata.json,
               cards/*.json, decks/*.json
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

## Offline TypeScript engine contracts

The Cyberpunk-specific [engine adapter](docs/engine-interop.md) connects to the
sibling application's authoritative engine through JSONL. It adds no model or
infrastructure requirement and does not replace this project's RAG/SFT/judge
pipeline. See that document for the actionId interface, shared golden tests,
known differential validator gaps and explicitly unsupported gameplay rules.

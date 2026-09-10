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
python scripts/test_cyberpunk.py      # 89 tests -- the game module and ingestion
python scripts/test_harness_core.py   # 48 tests -- harness/core itself
```

That is the whole suite. Every test in `test_cyberpunk.py` has been
mutation-verified -- 54 mutations, 54 caught, plus two deliberately not
pinned -- and doing it found seven tests that passed for the wrong reason
(see *Traps* below).

## The data pipeline

```bash
python scripts/fetch_rules.py     # -> data/raw/comprehensive_rules.json
python scripts/fetch_cards.py     # -> data/raw/cards/{_index,_filters,<slug>}.json
python scripts/fetch_errata.py    # -> data/raw/errata.json
python scripts/ingest.py          # -> data/processed/rules.jsonl, rule_sections.jsonl
python scripts/build_cards.py     # -> data/processed/card_database.jsonl
python scripts/build_errata.py    # -> data/processed/errata.jsonl
python scripts/chunk_corpus.py    # -> rule/card/errata chunks + corpus_chunks
```

Every step is idempotent and re-runnable from `data/raw/` with no network.
`build_errata.py` depends on `build_cards.py` having run: it joins each
erratum to a card by name.

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
| CMS pages (errata) | `POST /cyberpunk/content/query` |
| Public decks (paged) | `/cyberpunk/decks/public?limit=&offset=&sort=` |
| One deck | `/cyberpunk/decks/<uuid>` |

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

### Errata outrank everything else in the corpus

This is the one place where two parts of the corpus can disagree and one has
to win. The official errata page is explicit: the updated text "supersedes all
printed text at all levels of play." Three consequences, all of them load-bearing:

- **Errata come from the CMS API, not the rendered `/errata` page.** That page
  is server-rendered and *could* be scraped. It should not be: at capture time
  it carried three errata while the CMS carried four, and the missing one was
  live. A scraper would have silently dropped an erratum that supersedes
  printed card text. `games/cyberpunk/errata.py` reads the CMS.
- **The card database is not already errata'd.** It mostly is, which is the
  trap -- three of four launch errata are reflected in the card record and the
  fourth (Nocturne OP55 N1's artist) is not. Neither source is authoritative
  alone. `scripts/build_errata.py --check-applied` prints each pair.
- **A card's own chunk carries its erratum, as its last line.** Retrieval can
  return a card without the erratum correcting it, and a model reading only
  the card has no way to know its text was superseded. The line is last so a
  tail-preserving truncation keeps it. The grounded system prompt separately
  tells the model that ERRATA-marked text overrides printed card text.

An erratum whose heading matches no card is a **hard error**, not a warning.
Matching is exact on `display_name` first, then on a normalized form (no case,
accents or punctuation) because the headings are hand-written -- the launch set
writes "Nocturne OP55N1" for a card the database calls "Nocturne OP55 N1". A
normalized key that collapses two different cards is also a hard error, since
joining an erratum to the wrong card is worse than not joining it at all.

### Deckbuilding, and why its 272/272 agreement means almost nothing

`games/cyberpunk/deckbuilding.py` holds the four construction rules. The
community decklist host supplies 272 real decks, each with the official
builder's own `is_valid` verdict -- which looks like a strong label set and is
not, because **every one of them is valid**. The builder refuses to save an
illegal deck, so the corpus has no negative examples, and a validator that
returned "legal" unconditionally scores 272/272 too. Quoting that number alone
is the one-sided call `calibration/controls.py`'s `separation()` exists to
refuse.

What the corpus *does* support is two real checks, both in
`scripts/check_deck_rules.py`:

- **RAM arithmetic**, which is not a boolean: recomputing `ramBudget` and
  `ramUsage` per colour reproduces the official builder's own numbers on all
  272 decks. A constant answer cannot fake that. It is also how the RAM rule's
  exact semantics were established rather than guessed --
  `budget[colour]` is a SUM of Legend RAM, `usage[colour]` is the MAX of any
  single deck card's RAM, and the constraint is a per-card ceiling rather than
  a pool that spends down.
- **`--negative-control`**, which breaks each real deck in each way a rule
  forbids and checks the validator fires. This is the P(fire | broken) half the
  corpus cannot supply, and it is the half that found the only real bug in
  these rules (see *Traps*). Run it after any change to `deckbuilding.py`.

Decks are **not** in the retrieval corpus, deliberately: 272 decks against 229
rule/card/errata chunks would win nearly every slot on every question. See
`config.DECKS_DIR`.

Deck owner handles are **pseudonymized at fetch time**, per this workspace's
standing rule for third-party handles in captured data. The pseudonym is
salted and stable per author, so deduplication and "one person uploaded twenty
variants" stay visible while the handle does not. The salt lives beside the
snapshot; rotating it renames every author.

### Archetypes, and the null that licenses the word

`games/cyberpunk/archetypes.py` clusters the community decks by content.
Clustering will produce groups from noise as readily as from structure, so the
module ships the check alongside the method and `scripts/cluster_decks.py`
prints the check FIRST:

- **The null model is the load-bearing part.** Decks of one colour can only
  draw from that colour's cards, so same-colour decks overlap by
  construction and a cluster of them is not a finding. `null_decks` replaces
  each deck with a random legal deck of the same colour identity, size and
  copy-count shape. Real corpus at similarity >= 0.5: **339 pairs. Null: 1.**
  Above 0.6 the null has none. Re-run it rather than citing that number.
- **The archetype count is a knob, not a fact.** 20 clusters of three or more
  decks at threshold 0.40, 16 at 0.50. The CLI prints the whole sensitivity
  curve so nobody quotes a single number as if the format had one.
- **Average linkage, not single.** Single linkage chains -- A~B and B~C merge
  A with C however unalike -- which on this corpus collapses most decks into
  one blob.
- Similarity is a **weighted** Jaccard over copy counts. Three copies and one
  copy are different decisions; a set-based measure calls those decks
  identical.

Archetype output is deliberately **not committed**: it describes a moving
format from a dated snapshot and regenerates in seconds.

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
- **A corpus test cannot catch a code bug.** Three errata mutations went
  uncaught at first -- dropping errata from the merged index, dropping the
  `has_errata` flag, and loosening the h3 heading match -- because the tests
  covering them read the COMMITTED `data/processed/` files, which a source-only
  mutation does not change. A test over data pins the data; pinning the code
  needs a test that calls the code. The fix was a unit test per behaviour, an
  extracted `merge_chunks` so the merge is callable without writing files, and
  one end-to-end test that runs `chunking.main()` against a temp directory.
- **A fixture that never exercises the rule.** The errata heading pattern
  matches `###` only, but the fixture contained nothing but `###` headings, so
  relaxing it to any heading level changed no test outcome. The fixture now
  opens with an `## Current Errata` heading that must NOT become an erratum.
- **A rule that agreed with the official builder on 272 decks and was still
  wrong.** `validate` built its list of Legend names one per ENTRY, but two
  copies of a Legend is a single entry with `quantity: 2` -- so the
  unique-names rule could never fire, and no public deck could reveal it,
  because every public deck is legal. Only the negative control caught it. The
  lesson generalises past this bug: a corpus with one label value measures
  nothing about the other.
- **A test fixture that was itself illegal.** The first deckbuilding fixture
  built its "legal" 42-card deck as 39 copies of one card, which breaks the
  three-copy rule, so `test_a_legal_deck_has_no_violations` failed against a
  deck that had never been legal. Fixtures asserting legality have to satisfy
  every rule, not just the one under test.
- **Three clustering fixtures too weak to distinguish their own mutation.**
  The chaining test used a 3x3 matrix on which a sum-based and a max-based
  merge coincide; the determinism test used one where insertion order already
  equalled sorted order, so deleting the sort changed nothing; the boundary
  was untested entirely. All three passed against the mutation and all three
  needed a fixture where the two behaviours actually differ.
- **A plausible claim, asserted in a docstring, that was false.** `cluster`'s
  docstring said weighted Jaccard "almost never lands on a round number", so
  the inclusive-threshold boundary supposedly never fired on real data. It is
  a ratio of integer copy counts: 22 real pairs sit at exactly 0.40 and 35 at
  0.50, and the boundary moves the cluster count at 0.50 from 198 to 200.
  Checked only because a cluster count shifted after the change.
- **Two mutations deliberately left uncaught, because neither is a bug.**
  `_LEADING_MARKUP_RE`'s leading `^` was redundant with the `.match()` that
  applies it, and `cluster`'s tie-break direction between exactly-equal pairs
  is arbitrary and stable either way. Writing tests to pin those would pin
  arbitrary choices and read as coverage. Both are noted in comments where
  they live.

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

**Never let an erratum fail to join silently.** Errata override printed card
text, so a dropped or mis-attached one makes the rest of the corpus wrong
without looking wrong. Both are hard errors; keep them that way.

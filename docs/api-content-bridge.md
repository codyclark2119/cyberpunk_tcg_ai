# API Content Bridge V1

`cyberpunk_tcg_ai` captures and normalizes the public Cyberpunk TCG API.  The
TypeScript application remains the authoritative gameplay engine.  This bridge
moves source facts between those systems without making the source corpus an
executable rules authority.

## Direction

```text
public API snapshots
  -> data/raw/
  -> data/processed/card_database.jsonl + errata.jsonl
  -> scripts/export_engine_candidates.py
  -> data/engine-candidates/card-catalog.v1.jsonl
  -> engine-side review/admission
  -> immutable CardRevision snapshots
```

The export is intentionally one-way and read-only.  It does not write the
application repository or assign authoritative engine revisions.

## Candidate record

Each JSONL row is `schemaVersion: 1` and contains:

- `sourceCardSlug` — source-corpus identity from the official-site slug.
- `sourceRecordHash` — SHA-256 of the canonical normalized corpus record.
- `identityCandidate` — proposed `cardId`, printed-name deckbuilding identity,
  subtitle and display name.
- `catalog` — source card type/color/stats/sellability/classifications/product
  metadata.
- `rulesSource` — raw brace markup, rendered prose, and parser-derived keyword/
  timing hints.
- `printings` — source printing facts without authoritative engine printing IDs.
- `errata` — joined source errata and the upstream page timestamp.

The manifest separately pins the exact processed card database, errata file,
raw card index and serialized catalog with SHA-256 hashes.

## Authority boundary

The bridge must never emit these authoritative engine fields:

- `revision`
- `mechanics`
- `execution`
- `reviewed`
- `status`

Those fields require application-side review/admission.  In particular,
keyword/timing fields in this export are **hints derived from source markup**;
they are not executable `Ability`, `Effect`, `Trigger`, `Condition`, or
`ContinuousModifier` records.

The bridge also does not assign `CardPrintingId`.  Printing source facts are
preserved so the admission step can create stable engine printing identities.

## Fail-closed behavior

Unknown source card types or colors fail export rather than being mapped to a
nearby engine enum.  The source database must also match the raw `_index.json`
exactly in identity and order; stale/extraneous raw files cannot silently enter
the bridge.

## Commands

Generate the candidate artifacts:

```bash
python scripts/export_engine_candidates.py
```

Verify checked-in artifacts are reproducible:

```bash
python scripts/export_engine_candidates.py --check
```

Run the bridge tests:

```bash
python scripts/test_engine_candidates.py
```

## Next consumer

The next application-side milestone should consume this contract with an
**admission validator**, not direct runtime loading.  That validator should:

1. validate the bridge schema/version and manifest hashes;
2. compare candidate identity/printings/source text with existing immutable
   revisions;
3. reject unknown or unsupported mechanics unless explicitly reviewed;
4. require an explicit revision/admission decision;
5. produce the normal engine `CardRevisionSnapshot` and content manifest only
   after successful review.

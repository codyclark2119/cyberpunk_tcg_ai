# TypeScript engine interop

The authoritative Cyberpunk TCG gameplay engine lives in the sibling
`cyberpunk_tcg_llm` repository. This repository does **not** reimplement game
rules in Python.

## Boundary

The Node worker owns:

- state validation
- observations and privacy
- legal-action enumeration
- Descriptor V2 projection
- actionId generation and resolution
- state transitions and events

Python owns only transport and model/policy selection.

The model-facing request flow is:

```text
GameState + actor
  -> Node worker op=modelInput
  -> ModelInputV2 {
       schemaVersion: 2,
       observation,
       legalActions: [{ actionId, descriptor }]
     }
  -> Python/model chooses one offered actionId
  -> Node worker op=applyAction
  -> authoritative transition
```

The model never receives raw `GameAction`, RNG state, deck order, private
opponent information, or trusted replay state through this adapter.

## Adapter

`games/cyberpunk/engine_adapter.py` launches the sibling app's
`scripts/engine-worker.ts` as a long-lived JSONL subprocess.

`EngineWorker.model_input(...)` requests the Node-owned public ModelInputV2.
`EngineWorker.apply_action(...)` submits only an `actionId`.
`choose_action_id(...)` validates that a policy/model selected one of the
offered IDs.

The adapter deliberately does not parse descriptor labels or reproduce
Descriptor V2 schema logic. Descriptor fields are consumed as data from Node.

## Compatibility pin

The current engine-side contract was validated against:

```text
engine version: 0.4.0-descriptor-v2-1
```

The actual worker artifact hash is content-derived by the engine repo and must
match the content bundle supplied to each request. A stale fixture/bundle is
expected to fail with `ENGINE_ARTIFACT_MISMATCH` rather than silently run.

## Smoke test

From this repository:

```bash
python scripts/test_engine_adapter.py \
  --app-root /path/to/cyberpunk-tcg-online \
  --node /path/to/node
```

The test uses the sibling app's committed, regenerated
`demo-matrix-empty-draw-replay.v1.json` position and verifies:

- Node returns `schemaVersion: 2`
- observation equals the trusted stored observation
- ModelInputV2 actionId set/order equals trusted legal actions
- model-facing payload contains no raw `actorId`, `choiceId`, `optionIndices`,
  `cardInstanceId`, or `sourceInstanceId`
- an offered actionId resolves to an authoritative transition
- an unknown actionId is rejected by Node
- Python rejects a chooser result that was not offered

This is an interop test, not a gameplay-rules test. Rules remain covered in the
engine repository.

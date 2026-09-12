#!/usr/bin/env python3
"""Offline smoke/integration tests for the TypeScript engine adapter.

Usage:
    python scripts/test_engine_adapter.py --app-root ../cyberpunk-tcg-online

The test consumes the sibling app's regenerated Demo matrix fixture. It does
not duplicate any game rules in Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from games.cyberpunk.engine_adapter import EngineProtocolError, EngineWorker, choose_action_id


def load_probe(app_root: Path):
    path = app_root / "tests" / "fixtures" / "demo-matrix-empty-draw-replay.v1.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing regenerated Demo matrix fixture: {path}")
    replay = json.loads(path.read_text(encoding="utf-8"))
    positions = replay.get("positions") or []
    if not positions:
        raise AssertionError("Demo replay contains no strategic positions")
    position = positions[0]
    state = position["state"]
    actor = state["match"]["playerOrder"][position["actingSeat"]]
    return replay["content"], position, actor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", required=True, type=Path)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    app_root = args.app_root.resolve()
    content, position, actor = load_probe(app_root)

    with EngineWorker(app_root, node=args.node) as worker:
        public = worker.model_input(content=content, state=position["state"], actor_id=actor)
        assert public["schemaVersion"] == 2
        assert public["observation"] == position["observation"]
        expected_ids = [a["actionId"] for a in position["legalActions"]]
        actual_ids = [a["actionId"] for a in public["legalActions"]]
        assert actual_ids == expected_ids
        assert len(actual_ids) >= 2

        serialized = json.dumps(public, sort_keys=True)
        for forbidden in ("actorId", "choiceId", "optionIndices", "cardInstanceId", "sourceInstanceId"):
            assert f'"{forbidden}"' not in serialized, forbidden

        chosen = choose_action_id(public, lambda _observation, actions: actions[0]["actionId"])
        assert chosen == actual_ids[0]
        transition = worker.apply_action(content=content, state=position["state"], actor_id=actor, action_id=chosen)
        assert transition["kind"] == "transition"
        assert isinstance(transition.get("state"), dict)
        assert isinstance(transition.get("events"), list)
        assert isinstance(transition.get("stateHash"), str)

        try:
            worker.apply_action(content=content, state=position["state"], actor_id=actor, action_id="0" * 64)
        except EngineProtocolError as exc:
            assert "UNKNOWN_ACTION_ID" in str(exc)
        else:
            raise AssertionError("unknown actionId was accepted")

        try:
            choose_action_id(public, lambda _observation, _actions: "0" * 64)
        except ValueError as exc:
            assert "not offered" in str(exc)
        else:
            raise AssertionError("unoffered actionId was accepted by Python boundary")

    print(f"PASS modelInputV2 adapter: {len(actual_ids)} legal actions, actionId-only transition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

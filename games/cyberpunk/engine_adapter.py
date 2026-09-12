"""Thin JSONL client for the authoritative TypeScript gameplay engine.

This module contains transport only. It never implements Cyberpunk TCG rules,
legal-action generation, descriptor projection, or state transitions. The Node
worker owns all of those semantics; Python receives ModelInputV2 and returns an
actionId selection.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class EngineProtocolError(RuntimeError):
    """Raised when the Node worker rejects a request or violates the JSONL contract."""


class EngineWorker:
    """Long-lived JSONL subprocess wrapper around ``scripts/engine-worker.ts``."""

    def __init__(self, app_root: str | Path, node: str = "node") -> None:
        self.app_root = Path(app_root).resolve()
        worker = self.app_root / "scripts" / "engine-worker.ts"
        if not worker.is_file():
            raise FileNotFoundError(f"engine worker not found: {worker}")
        self._next_request = 1
        self._process = subprocess.Popen(
            [node, "--import", "tsx", str(worker)],
            cwd=self.app_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def __enter__(self) -> "EngineWorker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    def _request(self, op: str, *, content: Mapping[str, Any], state: Mapping[str, Any], actor_id: str, action_id: str | None = None) -> Mapping[str, Any]:
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise EngineProtocolError(f"engine worker exited early: {stderr.strip()}")
        request_id = f"py-{self._next_request}"
        self._next_request += 1
        request: dict[str, Any] = {
            "schemaVersion": 1,
            "requestId": request_id,
            "op": op,
            "content": content,
            "state": state,
            "actorId": actor_id,
        }
        if action_id is not None:
            request["actionId"] = action_id
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise EngineProtocolError(f"engine worker produced no response: {stderr.strip()}")
        response = json.loads(line)
        if response.get("requestId") != request_id:
            raise EngineProtocolError(f"request/response mismatch: expected {request_id!r}, got {response.get('requestId')!r}")
        if response.get("schemaVersion") != 1:
            raise EngineProtocolError(f"unsupported wire response version: {response.get('schemaVersion')!r}")
        if not response.get("ok"):
            errors = response.get("errors") or []
            detail = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in errors)
            raise EngineProtocolError(detail or "engine request failed")
        value = response.get("value")
        if not isinstance(value, dict):
            raise EngineProtocolError("successful engine response is missing object value")
        return value

    def model_input(self, *, content: Mapping[str, Any], state: Mapping[str, Any], actor_id: str) -> Mapping[str, Any]:
        """Return the Node-owned public ModelInputV2 payload for ``actor_id``."""
        value = self._request("modelInput", content=content, state=state, actor_id=actor_id)
        if value.get("kind") != "modelInput":
            raise EngineProtocolError(f"expected modelInput response, got {value.get('kind')!r}")
        model_input = value.get("modelInput")
        if not isinstance(model_input, dict) or model_input.get("schemaVersion") != 2:
            raise EngineProtocolError("worker returned invalid ModelInputV2 payload")
        return model_input

    def apply_action(self, *, content: Mapping[str, Any], state: Mapping[str, Any], actor_id: str, action_id: str) -> Mapping[str, Any]:
        """Submit only an authoritative actionId and return the trusted transition."""
        value = self._request("applyAction", content=content, state=state, actor_id=actor_id, action_id=action_id)
        if value.get("kind") != "transition":
            raise EngineProtocolError(f"expected transition response, got {value.get('kind')!r}")
        return value


def choose_action_id(model_input: Mapping[str, Any], chooser: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], str]) -> str:
    """Run a policy/model callback and validate that it selected an offered actionId.

    The callback receives only ``observation`` and public Descriptor V2 actions.
    It never receives GameState, RNG, raw GameAction payloads, or hidden IDs.
    """
    if model_input.get("schemaVersion") != 2:
        raise ValueError("expected ModelInputV2")
    observation = model_input.get("observation")
    actions = model_input.get("legalActions")
    if not isinstance(observation, dict) or not isinstance(actions, list):
        raise ValueError("invalid ModelInputV2 shape")
    public_actions = [a for a in actions if isinstance(a, dict)]
    action_id = chooser(observation, public_actions)
    offered = {a.get("actionId") for a in public_actions}
    if action_id not in offered:
        raise ValueError("chooser returned an actionId that was not offered")
    return action_id

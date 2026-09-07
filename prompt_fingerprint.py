"""Prompt-fingerprinting so a trained adapter can be checked against the
prompts it was trained under.

An adapter is only valid for the exact prompt shape it saw during training.
Editing a prompt after training silently invalidates every adapter already on
disk, and the failure looks like a capability regression rather than a
mismatch bug -- the model appears to have gotten worse. This module makes that
invariant checkable: a stamp file written beside the adapter weights, compared
against the current prompt set at eval time.

This is deliberately generic: it hashes whatever named prompt strings and
message shapes the caller supplies. Each game module is responsible for
building its own `named_prompts` dict and list of representative message
shapes (e.g. via its own `build_rag_messages`-equivalent) and passing them in.
"""

import hashlib
import json
from pathlib import Path

PROMPT_STAMP_FILE = "prompt_fingerprint.json"


def fingerprint(named_prompts: dict[str, str], message_shapes: list[list[dict]]) -> dict:
    """Hash a set of named prompt strings AND assembled message shapes.

    Hashing only the prompt strings would miss a prompt-shape change that
    leaves every string untouched (e.g. reordering how context is inserted
    into the user message), so both are hashed together.
    """
    shapes = [
        "|".join(m["role"] + ":" + m["content"] for m in shape)
        for shape in message_shapes
    ]
    parts = {**named_prompts, "message_shapes": "\n----\n".join(shapes)}
    digests = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in parts.items()}
    combined = hashlib.sha256(
        "\n".join(f"{k}={digests[k]}" for k in sorted(digests)).encode()
    ).hexdigest()
    return {"prompt_fingerprint": combined, "parts": digests}


def read_stamp(adapter_dir: Path) -> dict | None:
    p = adapter_dir / PROMPT_STAMP_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_stamp(adapter_dir: Path, stamp: dict) -> Path:
    out = adapter_dir / PROMPT_STAMP_FILE
    out.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return out


def check(adapter_dir: Path, current_fingerprint: dict) -> tuple[bool, str]:
    """(ok, message). Missing stamp is not a failure -- older adapters predate this."""
    stamp = read_stamp(adapter_dir)
    if stamp is None:
        return True, (f"no {PROMPT_STAMP_FILE} in {adapter_dir} -- cannot verify the prompt "
                      f"this adapter was trained under. Stamp it after training.")
    if stamp.get("prompt_fingerprint") == current_fingerprint["prompt_fingerprint"]:
        return True, f"prompt fingerprint matches ({current_fingerprint['prompt_fingerprint'][:12]})"

    changed = [k for k, v in current_fingerprint["parts"].items()
               if stamp.get("parts", {}).get(k) != v]
    return False, (
        f"PROMPT MISMATCH: {adapter_dir} was trained under a different prompt.\n"
        f"  stamped : {stamp.get('prompt_fingerprint', '?')[:12]}\n"
        f"  current : {current_fingerprint['prompt_fingerprint'][:12]}\n"
        f"  changed : {', '.join(changed) or 'unknown'}\n"
        "An adapter is only valid for the format it saw. Either revert the prompt "
        "change or retrain -- evaluating across a prompt mismatch measures the "
        "mismatch, not the model."
    )

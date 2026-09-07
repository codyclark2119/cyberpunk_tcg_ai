"""Record which prompt(s) an adapter was trained under, and check it at eval time.

Usage:
    python scripts/stamp_adapter.py models/my-adapter --dataset data/datasets/verified
    python scripts/stamp_adapter.py models/my-adapter --verify

Generic version: rather than requiring a hardcoded list of "known" system
prompts (which would tie this tool to one game's prompt module), this stamps
directly from whatever system prompts and message shapes are actually present
in the training dataset. `--dataset` is how a stamp is earned: the training
data contains the exact prompts the adapter saw, so the stamp is verified
against the file rather than asserted by whoever ran the command.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.core.io import read_jsonl  # noqa: E402
from harness.core.prompt_fingerprint import (  # noqa: E402
    check,
    fingerprint,
    write_stamp,
)


def dataset_fingerprint(dataset_dir: Path) -> dict:
    """Fingerprint built directly from a dataset's own train/valid messages.

    Every distinct system prompt found becomes a named part (`system_prompt_0`,
    `system_prompt_1`, ...), and every distinct message shape (roles in order,
    ignoring the assistant content) is hashed alongside them.
    """
    system_prompts: dict[str, int] = {}
    shapes: list[list[dict]] = []
    seen_shapes: set[str] = set()
    for name in ("train.jsonl", "valid.jsonl"):
        p = dataset_dir / name
        if not p.exists():
            continue
        for rec in read_jsonl(p):
            messages = rec.get("messages") or []
            for m in messages:
                if m.get("role") == "system":
                    system_prompts[m["content"]] = system_prompts.get(m["content"], 0) + 1
            shape_key = "|".join(m.get("role", "") for m in messages)
            if shape_key not in seen_shapes:
                seen_shapes.add(shape_key)
                shapes.append(messages)

    if not system_prompts:
        raise SystemExit(f"{dataset_dir} holds no system prompts to stamp from")

    named = {f"system_prompt_{i}": s for i, s in enumerate(sorted(system_prompts))}
    fp = fingerprint(named, shapes)
    fp["dataset_prompt_counts"] = {k: system_prompts[s] for k, s in named.items()}
    return fp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir", type=Path)
    ap.add_argument("--dataset", type=Path, default=None,
                     help="training set to fingerprint and verify the stamp against")
    ap.add_argument("--verify", action="store_true", help="check only, write nothing")
    args = ap.parse_args()

    if not args.adapter_dir.is_dir():
        raise SystemExit(f"no such adapter directory: {args.adapter_dir}")

    if args.verify:
        if not args.dataset:
            raise SystemExit("--verify requires --dataset to compute the current fingerprint against")
        fp = dataset_fingerprint(args.dataset)
        ok, msg = check(args.adapter_dir, fp)
        print(msg)
        raise SystemExit(0 if ok else 1)

    if not args.dataset:
        raise SystemExit("stamping requires --dataset (there is no game-specific prompt "
                          "module to fall back to in this generic harness)")

    fp = dataset_fingerprint(args.dataset)
    fp["verified_against"] = str(args.dataset)
    print(f"verified against {args.dataset}: "
          + ", ".join(f"{k}x{n}" for k, n in fp["dataset_prompt_counts"].items()))

    out = write_stamp(args.adapter_dir, fp)
    print(f"prompt fingerprint {fp['prompt_fingerprint'][:12]} -> {out}")


if __name__ == "__main__":
    main()

"""Game-agnostic jsonl I/O helpers.

These helpers
have no dependency on any particular game's data schema -- they operate on
plain `dict` records and file paths supplied by the caller.

Import as part of the `harness.core` package:

    from harness.core.io import read_jsonl, write_jsonl_atomic
"""

import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path: Path, missing_ok: bool = True) -> list[dict]:
    """Read a .jsonl file, skipping blank lines.

    Blank-line tolerance matters: appending to a jsonl file by hand or with a
    text editor commonly leaves a trailing blank line, and a naive
    `json.loads` per line raises `JSONDecodeError` on it.

    Use `iter_jsonl` instead when the file may be large enough that holding
    every row in memory at once is undesirable.
    """
    return list(iter_jsonl(path, missing_ok=missing_ok))


def iter_jsonl(path: Path, missing_ok: bool = True):
    """Streaming `read_jsonl`, for files too large to hold in memory."""
    if missing_ok and not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Write via a temp file in the same directory, then replace.

    A half-written data file is worse than no file at all -- a truncated
    final line fails to parse and any hand-authored work not yet flushed
    elsewhere is lost. Writing to a temp file and atomically replacing avoids
    ever leaving a partial file on disk after a crash mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    """Content digest of a file, read in chunks so large files are cheap."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def guard_shrink(path: Path, new_count: int, force: bool = False,
                  min_ratio: float = 0.5, what: str = "records",
                  hint: str = "") -> None:
    """Refuse to overwrite a corpus with a much smaller one. Raises SystemExit.

    A destructive rebuild that accidentally writes a small subset over a full
    corpus is a silent regression -- nothing downstream necessarily notices
    until retrieval or citation-checking quietly gets worse. Any script that
    rebuilds a corpus file in place should call this before writing.

    `min_ratio` is the fraction of the existing corpus the new one must
    reach. Pass a stricter ratio (e.g. 1.0) for content that is
    pinned/versioned and should never shrink at all except on a deliberate
    version bump; a looser ratio (e.g. 0.5) suits derived corpora where the
    known failure mode is an order-of-magnitude subset.
    """
    if force or not path.exists():
        return
    existing = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if new_count >= existing * min_ratio:
        return
    detail = (f"refusing to shrink {path} from {existing} to {new_count} {what}.\n"
              f"  (the new corpus must be at least {min_ratio:.0%} of the existing one)")
    if hint:
        detail += f"\n  {hint}"
    raise SystemExit(detail + "\n  Pass --force if the shrink is intended.")

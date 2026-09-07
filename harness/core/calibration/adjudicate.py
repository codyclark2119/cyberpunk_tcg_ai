"""Human-ground-truth adjudication of judge error-detection: is the judge
RIGHT, not just internally consistent?

Game-agnostic adjudication core. Operates on any
rubric records carrying `common_errors` and any stored judge run carrying
`errors_made` per answer -- it does not know what a "record" represents
(a rules question, a board position, ...), only that it has an id, a
`common_errors` list, and answers scored against it.

The caller supplies:
  - `rubric_index`: dict mapping every id a record might be found under to
    the record dict (a record is included only if it has `common_errors`).
  - the run file(s) to sample from / score against.
"""

import json
import random
from pathlib import Path

from harness.core.io import read_jsonl


def build_queue(runs: list[Path], rubric_index: dict, n: int, seed: int = 42) -> list[dict]:
    """Sample (record, arm, answer) triples for blind human review.

    When two runs are given, pairs they disagree on (one fired an error,
    the other did not) are drawn FIRST -- sampling on disagreement is a
    property of the pair, not a claim about either judge's correctness, and
    it concentrates human review time where a tiebreaker matters most. The
    task itself must never reveal that a disagreement exists (see
    `task_for`) -- a verdict collected while looking at judge output is not
    independent of it.
    """
    loaded = [{r.get("id") or r.get("gold_id"): r for r in read_jsonl(p)} for p in runs]
    base = loaded[0]

    cands, disputed = [], set()
    for rid, row in base.items():
        rec = rubric_index.get(rid)
        if not rec:
            continue
        for arm, d in (row.get("arms") or {}).items():
            if d.get("errors_made") is None or not (d.get("answer") or "").strip():
                continue
            key = f"{rid}::{arm}"
            cands.append({
                "key": key,
                "record_id": rid,
                "arm": arm,
                "answer": d["answer"],
                "n_errors": len(rec["common_errors"]),
            })
            for other in loaded[1:]:
                od = ((other.get(rid) or {}).get("arms") or {}).get(arm) or {}
                if od.get("errors_made") is None:
                    continue
                if bool(od["errors_made"]) != bool(d["errors_made"]):
                    disputed.add(key)

    rng = random.Random(seed)
    rng.shuffle(cands)
    cands.sort(key=lambda c: (c["key"] not in disputed,))
    return cands[:n]


def task_for(item: dict, rubric_index: dict, question_text) -> dict | None:
    """The blind review task: question/board text, the answer, numbered errors.

    `question_text(record) -> str` is supplied by the caller since rendering
    a "question" (which may be a rendered board position for a gameplay
    record) is game-specific.
    """
    rec = rubric_index.get(item["record_id"])
    if not rec:
        return None
    return {
        "key": item["key"],
        "record_id": item["record_id"],
        "arm": item["arm"],
        "question": question_text(rec),
        "answer": item["answer"],
        "common_errors": rec["common_errors"],
        "key_points": rec.get("key_points") or [],
        "category": rec.get("category"),
    }


def export_tasks(queue: list[dict], rubric_index: dict, question_text, out: Path) -> int:
    """Write a self-contained task file for a deployable review form.

    No judge output (not the errors it fired, its score, or the run it came
    from) is ever included, and no rubric ids beyond the item being reviewed
    -- a verdict collected while looking at judge output is not independent
    of it, and there is nothing on the far side of a rubric id for the
    deployed form to fetch from.
    """
    tasks = []
    for item in queue:
        t = task_for(item, rubric_index, question_text)
        if not t:
            continue
        leaked = set(t) & {"errors_made", "judge_model", "fired", "blundered", "run"}
        assert not leaked, f"judge output would leak to the public form: {leaked}"
        tasks.append(t)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kind": "adjudication", "tasks": tasks},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return len(tasks)


def append_verdict(verdict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(verdict, ensure_ascii=False) + "\n")


def score_run(run: Path, verdicts: list[dict]) -> dict:
    """Score one judge run against human verdicts.

    Per-error precision/recall, plus blunder-call accuracy (did the judge get
    "this answer blundered at all" right) -- not derivable from precision/
    recall alone, since a judge can fire the wrong error on an answer that did
    blunder and still get the binary call right.
    """
    by_key = {}
    for row in read_jsonl(run):
        rid = row.get("id") or row.get("gold_id")
        for arm, d in (row.get("arms") or {}).items():
            if d.get("errors_made") is not None:
                by_key[f"{rid}::{arm}"] = [e["n"] if isinstance(e, dict) else e
                                           for e in d["errors_made"]]

    tp = fp = fn = 0
    b_right = b_total = 0
    fp_only = fn_only = 0
    for v in verdicts:
        if v["key"] not in by_key:
            continue
        human = set(v["errors_present"])
        judge = set(by_key[v["key"]])
        tp += len(human & judge)
        fp += len(judge - human)
        fn += len(human - judge)
        b_total += 1
        if bool(human) == bool(judge):
            b_right += 1
        elif judge and not human:
            fp_only += 1
        else:
            fn_only += 1

    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "run": run.name, "n": b_total,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan"),
        "blunder_accuracy": b_right / b_total if b_total else float("nan"),
        "false_blunder": fp_only, "missed_blunder": fn_only,
        "tp": tp, "fp": fp, "fn": fn,
    }

"""Generic eval-harness scaffolding: sampling, generation loop, judge
comparison and rescoring, independent of any one game's corpus or schema.

Game-specific concerns deliberately excluded from this module (each game
supplies its own): retrieval/context-building, rule-ID citation checking,
rubric provenance/authorship metadata, and the system prompts used for
generation. Callers inject those via plain function arguments.
"""

import json
import random
from pathlib import Path

from harness.core.io import read_jsonl
from harness.core.stats import pearson_r


def stratified_sample(rows: list[dict], limit: int, key: str = "category") -> list[dict]:
    """Take `limit` rows spread as evenly as possible across `key`.

    A flat stride under-samples whatever category is rarest in the source
    corpus. This strides within each category instead, round-robining across
    categories so the pick is a cross-section rather than a prefix of each
    group. Small categories exhaust and drop out; their budget spills to the
    rest.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(key) or "uncategorized", []).append(r)

    ordered = {
        name: [members[int(i * len(members) / min(len(members), limit))]
               for i in range(min(len(members), limit))]
        for name, members in sorted(groups.items())
    }

    picked: list[dict] = []
    round_idx = 0
    while len(picked) < limit and any(round_idx < len(v) for v in ordered.values()):
        for name in sorted(ordered):
            if round_idx < len(ordered[name]) and len(picked) < limit:
                picked.append(ordered[name][round_idx])
        round_idx += 1
    return picked


def load_gold_questions(path: Path, limit: int | None = None, stratify: bool = True,
                         category_key: str = "category") -> list[dict]:
    """Load rubric-bearing eval rows (any gold set carrying `key_points`).

    Never truncated to a naive prefix by default: gold files are commonly
    ordered by id, which correlates with topic, so a bare prefix is not a
    cross-section.
    """
    rows = read_jsonl(path, missing_ok=False)
    if limit and len(rows) > limit:
        if stratify:
            rows = stratified_sample(rows, limit, key=category_key)
        else:
            stride = len(rows) / limit
            rows = [rows[int(i * stride)] for i in range(limit)]
    return rows


def generate_all_answers(
    questions: list[dict], build_prompt, arm_specs: list[tuple[str, str | None]],
    base_model_id: str, max_tokens: int,
    context_by_arm: dict[str, list[str | None]] | None = None,
    prompt_stamp_check=None,
) -> dict[str, list[str]]:
    """Run every question through several (adapter, context) arms.

    `arm_specs` is a list of (arm_name, adapter_path) pairs; `adapter_path=None`
    means the base model with no adapter. `build_prompt(tokenizer, question,
    context)` builds the chat prompt for one question -- supplied by the game
    module, since prompt construction embeds that game's system prompts.

    `context_by_arm` optionally supplies pre-retrieved context strings per arm
    name (same length/order as `questions`); arms not present in it run with
    no context. `prompt_stamp_check(adapter_path) -> (ok, msg)` is called
    before generating with a non-None adapter, if supplied, so a prompt-format
    mismatch is caught before spending the compute to generate hundreds of
    answers under it.
    """
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    answers: dict[str, list[str]] = {}
    context_by_arm = context_by_arm or {}

    checked_adapters = set()
    for arm_name, adapter_path in arm_specs:
        if adapter_path and adapter_path not in checked_adapters and prompt_stamp_check:
            ok, msg = prompt_stamp_check(Path(adapter_path))
            if not ok:
                raise SystemExit(msg)
            print(f"  {msg}")
            checked_adapters.add(adapter_path)

    for arm_name, adapter_path in arm_specs:
        print(f"loading {base_model_id} for arm={arm_name} adapter_path={adapter_path} ...")
        model, tokenizer = load_lm(base_model_id, adapter_path=adapter_path)
        ctxs = context_by_arm.get(arm_name)
        out = []
        for i, q in enumerate(questions, 1):
            ctx = ctxs[i - 1] if ctxs is not None else None
            prompt = build_prompt(tokenizer, q["question"], ctx)
            out.append(lm_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False))
            if i % 25 == 0 or i == len(questions):
                print(f"  {arm_name}: {i}/{len(questions)}")
        answers[arm_name] = out

    return answers


def compare_judges(path_a: Path, path_b: Path, report_out: Path,
                    id_key: str = "gold_id",
                    group_of=None) -> None:
    """Inter-judge agreement on identical stored answers.

    `group_of(record_id) -> str` optionally buckets records (e.g. by rubric
    author/source) for a per-group breakdown; defaults to a single "ALL"
    group when omitted. Both files must score the SAME stored answers
    (produced via a `--rescore-from`-style rerun) -- comparing two
    independent generations conflates judge variance with generation
    variance and settles neither.
    """
    def load(p: Path) -> dict:
        return {r[id_key]: r for r in read_jsonl(p, missing_ok=False) if r.get(id_key)}

    a, b = load(path_a), load(path_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(f"{path_a} and {path_b} share no {id_key} -- are both from a rescore of the same run?")
    arms = [x for x in a[shared[0]]["arms"] if x in b[shared[0]]["arms"]]
    group_of = group_of or (lambda _id: "ALL")

    groups: dict[str, list] = {}
    for qid in shared:
        label = group_of(qid)
        for arm in arms:
            xa, xb = a[qid]["arms"][arm], b[qid]["arms"][arm]
            ca, cb = xa.get("correctness"), xb.get("correctness")
            if ca is None or cb is None:
                continue
            row = (ca, cb, xa.get("points_hit"), xb.get("points_hit"), qid, arm)
            groups.setdefault(label, []).append(row)
            groups.setdefault("ALL", []).append(row)

    def stats(rows: list) -> dict:
        pairs = [(r[0], r[1]) for r in rows]
        n = len(pairs)
        if not n:
            return {}
        exact = sum(1 for x, y in pairs if x == y) / n
        mean_gap = sum(abs(x - y) for x, y in pairs) / n
        ph = [(r[2], r[3]) for r in rows if r[2] is not None and r[3] is not None]
        same_ph = (sum(1 for x, y in ph if sorted(x) == sorted(y)) / len(ph)) if ph else float("nan")
        return {"n": n, "r": pearson_r(pairs), "exact": exact,
                "gap": mean_gap, "same_points": same_ph,
                "n_questions": len({r[4] for r in rows})}

    order = ["ALL"] + sorted(g for g in groups if g != "ALL")
    lines = [f"# Inter-judge agreement: `{path_a.name}` vs `{path_b.name}`", "",
             f"{len(shared)} questions x {len(arms)} arms, identical stored answers.", "",
             "| Group | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for g in order:
        s = stats(groups[g])
        name = f"**{g}**" if g == "ALL" else g
        lines.append(f"| {name} | {s['n_questions']} | {s['n']} | {s['r']:+.2f} | "
                     f"{s['exact']:.0%} | {s['gap']:.2f} | {s['same_points']:.0%} |")

    lines += ["", "## Questions the judges disagree on most", "",
              "| id | Arm | Judge A | Judge B | Gap |",
              "| --- | --- | --- | --- | --- |"]
    worst = sorted(groups.get("ALL", []), key=lambda r: -abs(r[0] - r[1]))[:12]
    for ca, cb, _, _, qid, arm in worst:
        if ca == cb:
            break
        lines.append(f"| `{qid}` | {arm} | {ca:.1f} | {cb:.1f} | {abs(ca - cb):.1f} |")

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for g in order:
        s = stats(groups[g])
        print(f"  {g:16s} n={s['n']:>4}  r={s['r']:+.2f}  exact={s['exact']:.0%}  gap={s['gap']:.2f}")
    print(f"\n-> {report_out}")

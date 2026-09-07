"""Generic SFT dataset assembly mechanics.

Generic SFT dataset assembly. These functions know
nothing about a specific game's prompt shape or schema -- they take a
`build_messages_fn(question, answer, context=None) -> list[dict]` supplied by
the game module (typically wrapping that game's own `build_rag_messages`
equivalent), and operate on generic `question`/`answer`/`category` fields.
"""

import random


def truncate_context(context: str, budget_chars: int) -> str:
    """Trim grounding context to a char budget, on paragraph/line boundaries.

    Sizing this matters: a grounded training example puts context before the
    answer, so any example longer than the trainer's max sequence length gets
    its answer truncated away entirely -- leaving no target tokens and
    producing a NaN loss. Long contexts also blow up attention memory
    quadratically, so an unbounded retrieval budget can exhaust memory during
    dataset generation, independent of training itself. The budget must leave
    room for the question and answer within the model's max sequence length.
    """
    if len(context) <= budget_chars:
        return context
    kept: list[str] = []
    used = 0
    for para in context.split("\n\n"):
        if used + len(para) > budget_chars:
            break
        kept.append(para)
        used += len(para) + 2
    if not kept:  # single oversized paragraph -- hard cut on a line boundary
        lines = context[:budget_chars].split("\n")
        return "\n".join(lines[:-1]) if len(lines) > 1 else context[:budget_chars]
    return "\n\n".join(kept)


def build_messages(record: dict, build_messages_fn) -> dict:
    """Bare question -> answer training example, in the caller's message shape."""
    return {"messages": build_messages_fn(record["question"]) + [
        {"role": "assistant", "content": record["answer"]}]}


def build_grounded_messages(record: dict, context: str, build_messages_fn) -> dict:
    """Training example in the SAME shape RAG inference uses.

    A dataset trained only on bare question -> answer, while the intended
    inference architecture puts retrieved context in the prompt, is a
    train/inference mismatch that can make a fine-tuned model ignore correct
    retrieved context at inference time -- it was never shown an example
    where the answer was supposed to come from text in the prompt. Delegate
    message shape to the SAME builder eval-time inference uses, so the two
    cannot drift apart.
    """
    return {"messages": build_messages_fn(record["question"], context) + [
        {"role": "assistant", "content": record["answer"]}]}


def stratified_split(records: list[dict], train_frac: float, valid_frac: float, seed: int,
                      category_key: str = "category"):
    """Split records into (train, valid, eval) with per-category proportions
    preserved, rather than a flat random split that could leave a rare
    category entirely out of one split."""
    by_category: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r.get(category_key), []).append(r)

    train, valid, eval_ = [], [], []
    rng = random.Random(seed)
    for cat_records in by_category.values():
        rng.shuffle(cat_records)
        n = len(cat_records)
        n_train = round(n * train_frac)
        n_valid = round(n * valid_frac)
        train += cat_records[:n_train]
        valid += cat_records[n_train:n_train + n_valid]
        eval_ += cat_records[n_train + n_valid:]
    return train, valid, eval_

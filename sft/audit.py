"""Generic text-overlap/duplicate-detection primitives used to audit a
training set before training on it.

Game-agnostic: operates on plain question/target strings extracted from
training records. The caller (a game-specific `audit_sft.py`) supplies the
gold/eval question set, the training records, and any game-specific fields
(refusal regex, citation-id regex, etc.) it additionally wants to check.
"""

import re

# Content words only. English stopwords -- override/extend per game/corpus if
# training data is in a different language.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "do",
    "does", "did", "have", "has", "had", "i", "you", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "as", "into", "my", "me", "we", "can", "will",
    "would", "what", "when", "how", "why", "which", "who", "so", "not",
}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokens(text: str, stopwords: set[str] = STOPWORDS) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in stopwords and len(t) > 1}


def normalize(text: str) -> str:
    """Case-, whitespace-, and punctuation-insensitive form.

    Equality on this is what should trigger a contamination assertion, so it
    is deliberately conservative: it forgives reformatting and nothing else.
    """
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def question_of(rec: dict, marker: str = "\n\nQuestion: ") -> str:
    """The question embedded inside a training record's user message.

    Some prompt shapes emit a bare question and others emit
    "<context>{marker}<question>". Splitting on the last occurrence of
    `marker` recovers the question in both shapes; a context block that
    happens to contain the marker text itself would otherwise take the wrong
    half, so pass a marker string unlikely to appear in retrieved context.
    """
    msgs = rec.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    return user.rpartition(marker)[2] if marker in user else user


def target_of(rec: dict) -> str:
    msgs = rec.get("messages") or []
    return next((m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), "")


def near_duplicate_pairs(texts: list[str], threshold: float, rare_max: int = 40,
                          stopwords: set[str] = STOPWORDS):
    """Pairs above `threshold`, without an O(n^2) comparison.

    An inverted index over RARE tokens (appearing in at most `rare_max`
    records) supplies the candidate pairs: two texts similar enough to matter
    necessarily share an uncommon word, and two that share only common words
    cannot reach a high Jaccard score. This is a prune for a REPORT path only
    -- an exact-match contamination check should never use it, since a prune
    that is wrong even once would be a missed contamination.
    """
    toks = [tokens(t, stopwords=stopwords) for t in texts]
    index: dict[str, list[int]] = {}
    for i, ts in enumerate(toks):
        for t in ts:
            index.setdefault(t, []).append(i)

    seen, pairs = set(), []
    for t, ids in index.items():
        if len(ids) > rare_max:
            continue
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                key = (ids[a_i], ids[b_i])
                if key in seen:
                    continue
                seen.add(key)
                s = jaccard(toks[key[0]], toks[key[1]])
                if s >= threshold:
                    pairs.append((s, key[0], key[1]))
    pairs.sort(reverse=True)
    return pairs

"""Group community decklists into archetypes, and say how much to trust them.

An "archetype" here is an empirical claim -- these decks are built alike --
and clustering is very good at producing such claims whether or not they are
true. So this module ships the machinery to check them alongside the
machinery to make them:

  - `similarity` is a WEIGHTED Jaccard over copy counts, not a set overlap.
    Three copies of a card and one copy are different decisions, and a
    set-based measure calls two decks identical when their card lists match
    and their counts do not. Four pairs in the launch corpus have identical
    card SETS; all four also have identical counts, but that had to be
    checked rather than assumed.
  - `cluster` is average-linkage agglomerative with a similarity THRESHOLD,
    not k-means with a chosen k. There is no reason to believe this format
    has a particular number of archetypes, and a method that requires that
    number as input invents it. Average linkage rather than single linkage
    because single linkage chains: A-B and B-C merge A with C however
    unalike they are, which on this corpus produces one blob.
  - `null_decks` builds the comparison every clustering claim needs.

## The null model, and what it showed

A cluster of decks that share a colour identity is not a finding: decks of
one colour can only draw from that colour's cards, so they overlap by
construction. `null_decks` replaces each deck's cards with a random legal
selection from the SAME colour identity, preserving the deck's size and its
copy-count shape, and destroys nothing else. Any structure that survives that
is structure colour and size do not explain.

Measured on the 272-deck launch corpus, at similarity >= 0.5:

    real decks:  339 pairs
    null decks:    1 pair

and above 0.6 the null has none at all. That is the number that licenses
calling these archetypes rather than colour groupings, and
`scripts/cluster_decks.py --null` recomputes it rather than citing this
docstring.

## What is deliberately NOT claimed

The threshold is a knob, and different values give different archetype
counts -- 20 clusters of three or more decks at 0.40, 16 at 0.50. There is no
principled "correct" value, so the CLI prints the whole sensitivity curve
instead of one number, and the caller picks. Cluster COUNT is a property of
the threshold; cluster CONTENT, at any threshold in that range, is what the
corpus actually supports.
"""

import random
from collections import Counter

DEFAULT_THRESHOLD = 0.40
DEFAULT_CORE_SHARE = 0.8

LEGEND_ZONE = "legends"
DECK_ZONE = "deck"


def _zone(deck: dict, code: str) -> list[dict]:
    for zone in deck.get("zones") or []:
        if zone.get("zone_code") == code:
            return zone.get("cards") or []
    return []


def deck_counts(deck: dict) -> Counter:
    """Card slug -> copies, for the main deck only.

    Legends are excluded on purpose: every deck has exactly three, they are
    the deck's identity rather than its content, and including them would add
    a constant similarity floor to every pair sharing a Legend.
    """
    counts: Counter = Counter()
    for entry in _zone(deck, DECK_ZONE):
        counts[entry["card_slug"]] += entry.get("quantity", 0)
    return counts


def legend_slugs(deck: dict) -> list[str]:
    return [e["card_slug"] for e in _zone(deck, LEGEND_ZONE)
            for _ in range(e.get("quantity", 1))]


def colour_identity(deck: dict, card_index: dict) -> frozenset:
    """The colours this deck's Legends supply RAM in."""
    return frozenset(
        card_index[s]["color"] for s in legend_slugs(deck) if s in card_index)


def similarity(a: Counter, b: Counter) -> float:
    """Weighted Jaccard: shared copies over total copies. 0.0 to 1.0."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    intersection = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return intersection / union if union else 0.0


def similarity_matrix(counts: list[Counter]) -> list[list[float]]:
    n = len(counts)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[j][i] = similarity(counts[i], counts[j])
    return matrix


def cluster(matrix: list[list[float]], threshold: float = DEFAULT_THRESHOLD) -> list[list[int]]:
    """Average-linkage agglomerative clustering down to `threshold`.

    The threshold is INCLUSIVE: two clusters exactly `threshold` similar do
    merge, so "threshold 0.4" reads as "at least 0.4 alike".

    That boundary is not a technicality. Weighted Jaccard is a ratio of two
    integer copy counts, so round values are common rather than rare: in the
    272-deck launch corpus 22 pairs sit exactly at 0.40 and 35 exactly at
    0.50, and flipping this comparison to `>` moves the cluster count at 0.50
    from 198 to 200. The first version of this docstring asserted the opposite
    -- that a float would "almost never land on a round number" -- which was
    plausible, untested, and wrong.

    Returns lists of indices into `matrix`, each sorted, the whole list
    ordered by size then first member -- deterministic, so two runs on the
    same input give the same archetype ids.

    Merging tracks the SUM of cross-cluster similarities and divides by the
    product of sizes, rather than recomputing every pair each round: the sum
    of a merged cluster is just the sum of its parts, which keeps this cheap
    enough to run the sensitivity curve and a null in the same command.
    """
    n = len(matrix)
    if n == 0:
        return []
    members = {i: [i] for i in range(n)}
    totals = {i: dict(enumerate(matrix[i])) for i in range(n)}
    for i in range(n):
        totals[i].pop(i, None)

    while len(members) > 1:
        # `best is None` until something clears the threshold, so the
        # threshold test and the best-so-far test stay separate and the
        # inclusive `>=` boundary does not have to be encoded twice.
        #
        # `> best` keeps the first of several exactly-tied pairs and `>= best`
        # would keep the last. Both are deterministic -- the ordering comes
        # from `keys`, not from this comparison -- and exact ties do not occur
        # on weighted-Jaccard floats, so this direction is arbitrary and no
        # test pins it. Flipping it is not a bug worth catching.
        best = None
        pair = None
        keys = list(members)
        for idx, a in enumerate(keys):
            for b in keys[idx + 1:]:
                avg = totals[a][b] / (len(members[a]) * len(members[b]))
                if avg >= threshold and (best is None or avg > best):
                    best = avg
                    pair = (a, b)
        if pair is None:
            break
        a, b = pair
        members[a] = members[a] + members[b]
        for other in members:
            if other in (a, b):
                continue
            totals[a][other] = totals[other][a] = totals[a][other] + totals[b][other]
        del members[b]
        for other in totals:
            totals[other].pop(b, None)
        totals.pop(b, None)

    out = [sorted(v) for v in members.values()]
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def cohesion(members: list[int], matrix: list[list[float]]) -> float:
    """Mean pairwise similarity inside a cluster. 1.0 for a singleton."""
    if len(members) < 2:
        return 1.0
    pairs = [matrix[i][j] for x, i in enumerate(members) for j in members[x + 1:]]
    return sum(pairs) / len(pairs)


def core_cards(members: list[int], counts: list[Counter],
               share: float = DEFAULT_CORE_SHARE) -> list[tuple[str, float]]:
    """Cards appearing in at least `share` of a cluster's decks.

    This is what actually names an archetype: the cards its members agree on.
    Returned as `(slug, share)` sorted by share then slug, so the ordering is
    stable rather than dependent on dict iteration.
    """
    if not members:
        return []
    seen: Counter = Counter()
    for i in members:
        for slug in counts[i]:
            seen[slug] += 1
    threshold = len(members) * share
    out = [(slug, n / len(members)) for slug, n in seen.items() if n >= threshold]
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def duplicate_groups(counts: list[Counter]) -> list[list[int]]:
    """Decks whose card lists AND copy counts are identical.

    Reported rather than removed. Two people uploading the same 42 cards is a
    fact about the format -- a list worth copying -- and silently deduplicating
    would hide it while also changing every cluster's cohesion.
    """
    by_signature: dict = {}
    for i, c in enumerate(counts):
        key = tuple(sorted(c.items()))
        by_signature.setdefault(key, []).append(i)
    return sorted((v for v in by_signature.values() if len(v) > 1),
                  key=lambda g: (-len(g), g[0]))


def null_decks(decks: list[dict], card_index: dict, rng: random.Random) -> list[Counter]:
    """Random legal decks matching each real deck's colour identity and shape.

    Preserves what is not in question -- which colours a deck may draw from,
    how many cards it holds, and how those copies are distributed -- and
    randomizes the only thing that is: WHICH cards. Similarity structure that
    survives this is not explained by colour or size.
    """
    pools: dict = {}
    for card in card_index.values():
        pools.setdefault(card["color"], []).append(card["id"])
    for pool in pools.values():
        pool.sort()

    out = []
    for deck in decks:
        real = deck_counts(deck)
        colours = colour_identity(deck, card_index)
        pool = sorted({slug for colour in colours for slug in pools.get(colour, [])})
        shape = sorted(real.values(), reverse=True)
        if not pool:
            out.append(Counter())
            continue
        picks = rng.sample(pool, min(len(shape), len(pool)))
        out.append(Counter(dict(zip(picks, shape))))
    return out


def pairs_above(matrix: list[list[float]], threshold: float) -> int:
    n = len(matrix)
    return sum(1 for i in range(n) for j in range(i + 1, n) if matrix[i][j] >= threshold)

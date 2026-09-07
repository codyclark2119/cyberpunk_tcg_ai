"""Game-agnostic statistics helpers used across the eval/calibration harness."""


def pearson_r(pairs: list[tuple[float, float]]) -> float:
    """Correlation over (judge A, judge B) score pairs. NaN under 3 pairs.

    Kept as a single definition so every caller computing inter-judge
    correlation reports the same number for the same input -- two divergent
    copies of this formula have historically been a source of silently
    mismatched headline statistics.
    """
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")

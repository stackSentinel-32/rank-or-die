"""
ranker/fusion.py

Score fusion module. Combines Signal A (keyword), Signal B (TF-IDF),
and Signal C (semantic) into a single ranked score using Reciprocal
Rank Fusion (RRF) or a weighted fallback.
"""


def _rank_list(scores: list[float]) -> list[int]:
    """
    Return 1-indexed ranks for each candidate based on descending score order.
    Ties share the same rank (dense ranking not needed — order-of-sort is stable).
    """
    n = len(scores)
    # argsort descending: positions sorted by score high→low
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
    ranks = [0] * n
    for rank_position, candidate_idx in enumerate(sorted_indices, start=1):
        ranks[candidate_idx] = rank_position
    return ranks


def _normalize(scores: list[float]) -> list[float]:
    """Normalize a list to [0.0, 1.0]. Returns all 0.0 if range is zero."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s
    if denom == 0.0:
        return [0.0] * len(scores)
    return [(s - min_s) / denom for s in scores]


def rrf_fusion(
    keyword_scores: list[float],
    tfidf_scores: list[float],
    semantic_scores: list[float],
    k: int = 60,
) -> list[float]:
    """
    Reciprocal Rank Fusion across three signals.

    If semantic_scores is all 0.0 (model unavailable), falls back to a
    two-signal weighted RRF: keyword × 0.65, TF-IDF × 0.35.

    Returns a normalized list of floats in [0.0, 1.0].
    """
    n = len(keyword_scores)
    if n == 0:
        return []

    # Detect if Signal C is active (any non-zero value present)
    semantic_active = any(s > 0.0 for s in semantic_scores)

    # Compute per-signal rank lists
    rank_kw = _rank_list(keyword_scores)
    rank_tf = _rank_list(tfidf_scores)

    if semantic_active:
        rank_se = _rank_list(semantic_scores)
        rrf_scores = [
            1.0 / (k + rank_kw[i])
            + 1.0 / (k + rank_tf[i])
            + 1.0 / (k + rank_se[i])
            for i in range(n)
        ]
    else:
        # Two-signal weighted RRF fallback
        rrf_scores = [
            (1.0 / (k + rank_kw[i])) * 0.65
            + (1.0 / (k + rank_tf[i])) * 0.35
            for i in range(n)
        ]

    return _normalize(rrf_scores)


def weighted_fusion(
    keyword_scores: list[float],
    tfidf_scores: list[float],
    semantic_scores: list[float],
    weights: tuple = (0.50, 0.30, 0.20),
) -> list[float]:
    """
    Simple weighted average of the three signals, normalized to [0.0, 1.0].
    Use as a fallback if RRF produces unexpected results.
    """
    n = len(keyword_scores)
    if n == 0:
        return []

    w_kw, w_tf, w_se = weights
    raw = [
        w_kw * keyword_scores[i]
        + w_tf * tfidf_scores[i]
        + w_se * semantic_scores[i]
        for i in range(n)
    ]
    return _normalize(raw)

"""
ranker/output.py

Reasoning generation and CSV output for the final top-100 ranked candidates.
"""
import csv

from ranker.constants import (
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    TIER2_RECSYS,
    TIER3_LLM,
    TIER3_MLOPS,
    WITCH_COMPANIES,
)

# Ordered tiers for top-skill selection (highest weight first)
_TIER_WEIGHT_ORDER = [
    (TIER1_RETRIEVAL, 4.0),
    (TIER2_NLP_IR, 3.0),
    (TIER2_RECSYS, 3.0),
    (TIER3_LLM, 2.5),
    (TIER3_MLOPS, 2.0),
]

# Keywords that indicate ML work in a role description
_ML_DESC_SIGNALS = TIER1_RETRIEVAL | TIER2_NLP_IR | TIER2_RECSYS


def _top_skills(skill_set: set, n: int = 3) -> str:
    """Return top-n JD-relevant skills as a comma-separated string, sorted by tier weight."""
    selected = []
    for tier_set, _ in _TIER_WEIGHT_ORDER:
        for skill in skill_set:
            if skill in tier_set and skill not in selected:
                selected.append(skill)
            if len(selected) >= n:
                break
        if len(selected) >= n:
            break
    return ", ".join(selected) if selected else "none"


def _assessment_note(assessment_scores: dict) -> str:
    """Return 'assessed X%' based on best assessment score, or empty string."""
    if not assessment_scores:
        return ""
    best = max(assessment_scores.values())
    return f"assessed {best:.0%}"


def _company_type(current_company: str) -> str:
    """Classify current company into consulting / unknown."""
    company_lower = current_company.lower().strip()
    if any(w in company_lower for w in WITCH_COMPANIES):
        return "Consulting"
    if not company_lower:
        return "Unknown co"
    return "Product/startup"


def _compute_ml_yoe(career_roles: list) -> float:
    """Sum months of roles with ML signals in description, return as years."""
    ml_months = 0.0
    for role in career_roles:
        desc = role.get("description", "").lower()
        if any(kw in desc for kw in _ML_DESC_SIGNALS):
            ml_months += role.get("actual_months", 0.0)
    return ml_months / 12.0


def _geo_status(geo_bucket: str, willing_relocate: bool) -> str:
    if geo_bucket == "preferred_city":
        return "India-preferred"
    elif geo_bucket == "india_other":
        return "India"
    elif willing_relocate:
        return "International+willing"
    else:
        return "International"


def _github_status(github_score: float) -> str:
    if github_score > 0:
        return f"active({github_score:.0f})"
    return "not linked"


def generate_reasoning(
    features: dict,
    keyword_score: float,
    tfidf_score: float,
    semantic_score: float,
    availability: float,
    geo_bonus: float,
    final_score: float,
    is_honeypot: bool,
) -> str:
    """
    Generate a human-readable reasoning string for a ranked candidate.
    Every slot is filled from actual computed values — no hallucination.
    """
    yoe         = features.get("years_exp", 0.0)
    skill_set   = features.get("skill_set", set())
    assess      = features.get("assessment_scores", {})
    company     = features.get("current_company", "")
    career_roles = features.get("career_roles", [])
    notice      = features.get("notice_days", 999)
    geo_bucket  = features.get("geo_bucket", "international")
    willing     = features.get("willing_relocate", False)
    github      = features.get("github_score", -1)

    top3        = _top_skills(skill_set, n=3)
    assess_note = _assessment_note(assess)
    co_type     = _company_type(company)
    ml_yoe      = _compute_ml_yoe(career_roles)
    geo_stat    = _geo_status(geo_bucket, willing)
    gh_stat     = _github_status(github)
    hp_note     = " [HONEYPOT]" if is_honeypot else ""

    # Build assess segment only if non-empty
    assess_seg  = f" | {assess_note}" if assess_note else ""

    reasoning = (
        f"{yoe:.1f}yr exp | Skills: {top3}{assess_seg} | "
        f"{co_type} at {company} | ML YOE ~{ml_yoe:.1f}yr | "
        f"Notice {notice}d | {geo_stat} | GitHub {gh_stat} | "
        f"Keyword {keyword_score:.2f} TF-IDF {tfidf_score:.2f} Semantic {semantic_score:.2f} -> "
        f"Skill fused × Avail {availability:.2f} + Geo {geo_bonus:.2f} = {final_score:.3f}{hp_note}"
    )
    return reasoning


def write_csv(ranked_candidates: list[dict], output_path: str) -> None:
    """
    Validate the final top-100 list then write it to a CSV file.

    Expects each dict to have: candidate_id, rank, score, reasoning.
    Raises AssertionError with a message if any validation fails.
    """
    # --- Assertions ---
    assert len(ranked_candidates) == 100, (
        f"Expected 100 rows, got {len(ranked_candidates)}"
    )

    assert len({c["candidate_id"] for c in ranked_candidates}) == 100, (
        "Duplicate candidate IDs in top-100"
    )

    scores = [c["score"] for c in ranked_candidates]
    assert all(scores[i] > scores[i + 1] for i in range(99)), (
        "Scores not strictly decreasing"
    )

    assert all(c["rank"] == i + 1 for i, c in enumerate(ranked_candidates)), (
        "Ranks are wrong (expected 1..100 in order)"
    )

    honeypot_count = sum(
        1 for c in ranked_candidates if "[HONEYPOT]" in c.get("reasoning", "")
    )
    assert honeypot_count <= 10, (
        f"Too many honeypots in top-100: {honeypot_count} (limit is 10)"
    )

    # --- Write CSV ---
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for c in ranked_candidates:
            writer.writerow([
                c["candidate_id"],
                c["rank"],
                f"{c['score']:.6f}",
                c["reasoning"],
            ])

    print(f"[output] Wrote {len(ranked_candidates)} rows to {output_path}")
    if honeypot_count:
        print(f"[output] Warning: {honeypot_count} honeypot(s) in top-100")

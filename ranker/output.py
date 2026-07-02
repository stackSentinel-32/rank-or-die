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
    PRODUCT_STARTUPS_INDIA,
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
        return "outsourcing"
    if company_lower in PRODUCT_STARTUPS_INDIA:
        return "startup"
    if not company_lower:
        return "Unknown co"
    return "product"


def _compute_ml_yoe(career_roles: list) -> float:
    """Sum months of roles with ML signals in description, return as years."""
    ml_months = 0.0
    for role in career_roles:
        desc = role.get("description", "").lower()
        if any(kw in desc for kw in _ML_DESC_SIGNALS):
            ml_months += role.get("actual_months", 0.0)
    return ml_months / 12.0


def _geo_status(geo_bucket: str) -> str:
    if geo_bucket == "preferred_city":
        return "India (preferred city)"
    elif geo_bucket == "india_other":
        return "India"
    else:
        return "international"


def _github_status(github_score: float) -> str:
    if github_score > 0:
        return "GitHub active"
    elif github_score == 0:
        return "GitHub inactive"
    return "GitHub not linked"


def _skill_slots(skill_set: set, max_t1: int = 2, max_t2: int = 2) -> tuple[int, str, str]:
    """
    Returns:
        count    – number of TIER1 retrieval skills the candidate has (used for the count label)
        t1_str   – top-max_t1 TIER1 skill names (shown in parentheses after the count)
        t2_str   – top-max_t2 TIER2_NLP_IR skill names (shown separately as NLP/IR:, or "" if none)
    TIER2 skills are NEVER mixed into the TIER1 list.
    """
    tier1_hits = [s for s in skill_set if s in TIER1_RETRIEVAL]
    tier2_hits = [s for s in skill_set if s in TIER2_NLP_IR]
    count  = len(tier1_hits)
    t1_str = ", ".join(tier1_hits[:max_t1])
    t2_str = ", ".join(tier2_hits[:max_t2])
    return count, t1_str, t2_str


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
    # --- Raw features ---
    yoe          = features.get("years_exp", 0.0)
    skill_set    = features.get("skill_set", set())
    assess       = features.get("assessment_scores", {})
    company_raw  = features.get("current_company", "")
    industry     = features.get("current_industry", "")
    company_size = features.get("company_size", "")
    career_roles = features.get("career_roles", [])
    notice       = features.get("notice_days", 999)
    geo_bucket   = features.get("geo_bucket", "international")
    github       = features.get("github_score", -1)
    title_raw    = features.get("current_title", "")

    # --- Slots ---
    title = title_raw.title() if title_raw.strip() else "ML Engineer"
    company = company_raw.title() if company_raw.strip() else "Unknown"

    # company_type
    company_lower = company_raw.lower()
    if any(w in company_lower for w in WITCH_COMPANIES):
        co_type = "outsourcing"
    elif company_lower in PRODUCT_STARTUPS_INDIA:
        co_type = "startup"
    else:
        # parse company_size upper bound
        size_upper = 0
        if company_size:
            try:
                size_upper = int(str(company_size).split("-")[-1].replace("+", "").strip())
            except (ValueError, IndexError):
                size_upper = 0
        if "software" in industry and 0 < size_upper < 1000:
            co_type = "startup"
        elif "software" in industry:
            co_type = "product"
        else:
            co_type = "consulting"

    # assessment note
    if assess:
        best_score = max(assess.values())
        assess_note = f"assessed {best_score:.0%}; "
    else:
        assess_note = ""

    ml_yoe   = _compute_ml_yoe(career_roles)
    geo_stat = _geo_status(geo_bucket)
    gh_stat  = _github_status(github)
    hp_note  = " [HONEYPOT]" if is_honeypot else ""

    # Dynamic length constraint: try max_t1=2 then 1 then 0 to fit within 220 chars.
    # If still too long, truncate title and company.
    reasoning = ""
    for max_t1 in [2, 1, 0]:
        skill_count, t1_str, t2_str = _skill_slots(skill_set, max_t1=max_t1, max_t2=2)
        skill_word  = "retrieval skill" if skill_count == 1 else "retrieval skills"
        nlp_ir_part = f"; NLP/IR: {t2_str}" if t2_str else ""
        reasoning = (
            f"{title} with {yoe:.1f} yrs at {company} ({co_type}); "
            f"{skill_count} {skill_word} ({t1_str}){nlp_ir_part}; "
            f"{assess_note}"
            f"ML YOE {ml_yoe:.1f} yrs; "
            f"notice {notice}d; "
            f"{geo_stat}; "
            f"{gh_stat}; "
            f"score {final_score:.3f}.{hp_note}"
        )
        if len(reasoning) <= 220:
            break

    # If still too long after dropping skills, truncate title and company
    if len(reasoning) > 220:
        if len(title) > 15:
            title = title[:12] + "..."
        if len(company) > 15:
            company = company[:12] + "..."
        skill_count, t1_str, t2_str = _skill_slots(skill_set, max_t1=0, max_t2=0)
        skill_word = "retrieval skill" if skill_count == 1 else "retrieval skills"
        reasoning = (
            f"{title} with {yoe:.1f} yrs at {company} ({co_type}); "
            f"{skill_count} {skill_word} ({t1_str}); "
            f"{assess_note}"
            f"ML YOE {ml_yoe:.1f} yrs; "
            f"notice {notice}d; "
            f"{geo_stat}; "
            f"{gh_stat}; "
            f"score {final_score:.3f}.{hp_note}"
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
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
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

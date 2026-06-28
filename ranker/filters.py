from dataclasses import dataclass, field
from typing import List

from ranker.constants import (
    WITCH_COMPANIES,
    ALL_JD_SKILLS,
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    TIER2_RECSYS,
    NEGATIVE_CV_SPEECH,
)

# ---------------------------------------------------------------------------
# Module-level filter counter — inspect after a full run to debug aggression
# ---------------------------------------------------------------------------
FILTER_COUNTS: dict[str, int] = {
    "witch_only_career": 0,
    "zero_ai_signal": 0,
    "wrong_domain": 0,
    "cv_speech_specialist_no_nlp": 0,
    "honeypot": 0,
    "passed": 0,
}

def print_filter_summary() -> None:
    """Print how many candidates hit each filter. Call after full pipeline run."""
    total = sum(FILTER_COUNTS.values())
    print("\n=== Filter Summary ===")
    for reason, count in FILTER_COUNTS.items():
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {reason:<35} {count:>7,}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<35} {total:>7,}")
    print("=====================\n")


# ---------------------------------------------------------------------------
# Wrong-domain title keywords
# ---------------------------------------------------------------------------
_WRONG_DOMAIN_TITLES = {
    "civil engineer",
    "accountant",
    "hr manager",
    "mechanical engineer",
    "marketing manager",
    "content writer",
    "operations manager",
}

# CS/ML/Software degree keywords — presence in any education entry overrides filter
_CS_DEGREE_KEYWORDS = {
    "computer science", "cs", "information technology", "it",
    "software engineering", "machine learning", "data science",
    "artificial intelligence", "electronics", "electrical engineering",
    "mathematics", "statistics", "mca", "bca", "b.tech", "m.tech",
    "b.e", "m.e", "be", "me"
}


@dataclass
class FilterResult:
    should_discard: bool
    discard_reason: str           # empty string if not discarded
    is_honeypot: bool
    honeypot_reasons: List[str]


def _has_cs_degree(features: dict) -> bool:
    """True if any education entry hints at a CS/tech background."""
    for edu in (features.get("education") or []):
        degree = str(edu.get("degree") or "").lower()
        field_of_study = str(edu.get("field_of_study") or "").lower()
        combined = degree + " " + field_of_study
        if any(kw in combined for kw in _CS_DEGREE_KEYWORDS):
            return True
    return False


def apply_filters(features: dict) -> FilterResult:
    """
    Applies four hard discard filters in order, then runs honeypot detection.
    Returns a FilterResult. Discard stops at the first matching filter.
    """
    skill_set: set = features.get("skill_set", set())
    description_text: str = features.get("description_text", "")
    current_title: str = features.get("current_title", "")
    career_roles: list = features.get("career_roles", [])
    years_exp: float = features.get("years_exp", 0.0)
    honeypot_flags: list = features.get("honeypot_flags", [])

    # -----------------------------------------------------------------------
    # FILTER 1 — WITCH-only career
    # -----------------------------------------------------------------------
    if years_exp > 4 and career_roles:
        companies_in_roles = {
            r.get("company", "").lower().strip()
            for r in career_roles
        }
        # Remove empty strings
        companies_in_roles.discard("")

        all_witch = bool(companies_in_roles) and all(
            any(w in company for w in WITCH_COMPANIES)
            for company in companies_in_roles
        )

        if all_witch:
            # Exception: 3+ TIER1 keywords anywhere in ALL descriptions
            all_descriptions = " ".join(
                r.get("description", "").lower() for r in career_roles
            )
            tier1_hits_in_desc = sum(
                1 for kw in TIER1_RETRIEVAL if kw in all_descriptions
            )
            if tier1_hits_in_desc < 3:
                FILTER_COUNTS["witch_only_career"] += 1
                hp_flags, is_hp = _check_honeypot(honeypot_flags)
                return FilterResult(
                    should_discard=True,
                    discard_reason="witch_only_career",
                    is_honeypot=is_hp,
                    honeypot_reasons=hp_flags,
                )

    # -----------------------------------------------------------------------
    # FILTER 2 — Zero AI signal
    # -----------------------------------------------------------------------
    skill_jd_overlap = skill_set & ALL_JD_SKILLS
    desc_has_jd_word = any(kw in description_text for kw in ALL_JD_SKILLS)

    if len(skill_jd_overlap) == 0 and not desc_has_jd_word:
        FILTER_COUNTS["zero_ai_signal"] += 1
        hp_flags, is_hp = _check_honeypot(honeypot_flags)
        return FilterResult(
            should_discard=True,
            discard_reason="zero_ai_signal",
            is_honeypot=is_hp,
            honeypot_reasons=hp_flags,
        )

    # -----------------------------------------------------------------------
    # FILTER 3 — Wrong domain
    # -----------------------------------------------------------------------
    title_is_wrong_domain = any(wt in current_title for wt in _WRONG_DOMAIN_TITLES)
    if title_is_wrong_domain:
        jd_skill_count = len(skill_jd_overlap)
        if jd_skill_count < 2 and not _has_cs_degree(features):
            FILTER_COUNTS["wrong_domain"] += 1
            hp_flags, is_hp = _check_honeypot(honeypot_flags)
            return FilterResult(
                should_discard=True,
                discard_reason="wrong_domain",
                is_honeypot=is_hp,
                honeypot_reasons=hp_flags,
            )

    # -----------------------------------------------------------------------
    # FILTER 4 — CV/Speech specialist with no NLP background
    # -----------------------------------------------------------------------
    negative_hits = len(skill_set & NEGATIVE_CV_SPEECH)
    tier1_hits = len(skill_set & TIER1_RETRIEVAL)
    tier2_hits = len(skill_set & (TIER2_NLP_IR | TIER2_RECSYS))

    if negative_hits > 4 and tier1_hits == 0 and tier2_hits == 0:
        FILTER_COUNTS["cv_speech_specialist_no_nlp"] += 1
        hp_flags, is_hp = _check_honeypot(honeypot_flags)
        return FilterResult(
            should_discard=True,
            discard_reason="cv_speech_specialist_no_nlp",
            is_honeypot=is_hp,
            honeypot_reasons=hp_flags,
        )

    # -----------------------------------------------------------------------
    # PASSED all filters
    # -----------------------------------------------------------------------
    hp_flags, is_hp = _check_honeypot(honeypot_flags)
    if is_hp:
        FILTER_COUNTS["honeypot"] += 1
    FILTER_COUNTS["passed"] += 1
    return FilterResult(
        should_discard=False,
        discard_reason="",
        is_honeypot=is_hp,
        honeypot_reasons=hp_flags,
    )


def _check_honeypot(honeypot_flags: list) -> tuple[list, bool]:
    """
    Determine honeypot status from the flags list.
    Returns (honeypot_flags, is_honeypot).

    Rules:
    - >= 2 flags of ANY kind → honeypot
    - == 1 flag and it's "salary_inverted" → honeypot
    - Single "instant_expert" alone → NOT honeypot (too common)
    """
    flags = list(honeypot_flags)
    n = len(flags)

    if n == 0:
        return flags, False
    if n >= 2:
        return flags, True
    # exactly 1 flag
    if flags[0] == "salary_inverted":
        return flags, True
    # single "instant_expert" or any other single flag → not honeypot
    return flags, False

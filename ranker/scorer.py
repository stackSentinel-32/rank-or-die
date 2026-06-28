import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ranker.constants import (
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    TIER2_RECSYS,
    TIER3_LLM,
    TIER3_MLOPS,
    NEGATIVE_CV_SPEECH,
    WITCH_COMPANIES,
    ALL_JD_SKILLS,
)

# ---------------------------------------------------------------------------
# JD reference text for TF-IDF Signal B
# ---------------------------------------------------------------------------
JD_TEXT = (
    "Senior AI Engineer embedding retrieval vector search semantic search "
    "FAISS Pinecone Qdrant sentence-transformers dense retrieval NLP "
    "information retrieval ranking recommendation systems NDCG MRR A/B testing "
    "Python production ML applied machine learning product company "
    "learning to rank passage retrieval bi-encoder cross-encoder"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_weight(skill_name: str) -> float:
    if skill_name in TIER1_RETRIEVAL:
        return 4.0
    if skill_name in TIER2_NLP_IR:
        return 3.0
    if skill_name in TIER2_RECSYS:
        return 3.0
    if skill_name in TIER3_LLM:
        return 2.5
    if skill_name in TIER3_MLOPS:
        return 2.0
    return 0.0


def _proficiency_mult(proficiency: str) -> float:
    if proficiency == "advanced":
        return 1.0
    if proficiency == "intermediate":
        return 0.7
    if proficiency == "beginner":
        return 0.3
    # unknown / missing → treat as beginner
    return 0.3


def _duration_mult(duration_months: int) -> float:
    if duration_months >= 36:
        return 1.0
    if duration_months >= 18:
        return 0.85
    if duration_months >= 6:
        return 0.70
    return 0.50


def _assessment_override(score_01: float) -> float:
    """Return proficiency_mult based on assessment score (0-1)."""
    if score_01 >= 0.80:
        return 1.0
    if score_01 >= 0.60:
        return 0.75
    if score_01 >= 0.40:
        return 0.45
    return 0.20


def _company_ml_mult(company: str, industry: str, company_size: str) -> float:
    if any(w in company for w in WITCH_COMPANIES):
        return 0.25
    if "it services" in industry:
        return 0.40
    if "software" in industry and company_size in {"11-50", "51-200", "201-500"}:
        return 1.0
    if "software" in industry:
        return 0.85
    return 0.60


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

MAX_POSSIBLE = 50.0


def score_keywords(features: dict) -> float:
    """
    Compute a keyword/skill score from feature dict.
    Returns a float in [0.0, 1.0].
    """
    skill_map: dict = features.get("skill_map", {})
    assessment_scores: dict = features.get("assessment_scores", {})
    description_text: str = features.get("description_text", "")
    skill_set: set = features.get("skill_set", set())
    career_roles: list = features.get("career_roles", [])
    years_exp: float = features.get("years_exp", 0.0)

    # -----------------------------------------------------------------------
    # Step 1 — Per-skill points
    # -----------------------------------------------------------------------
    skill_points = 0.0
    for skill_name, sdata in skill_map.items():
        tw = _tier_weight(skill_name)
        if tw == 0.0:
            continue  # not a JD-relevant skill, skip

        dm = _duration_mult(sdata.get("duration_months") or 0)

        # Assessment override takes priority over self-reported proficiency
        if skill_name in assessment_scores:
            pm = _assessment_override(assessment_scores[skill_name])
        else:
            pm = _proficiency_mult(sdata.get("proficiency") or "")

        skill_points += tw * pm * dm

    # -----------------------------------------------------------------------
    # Step 2 — Description text bonus (partial credit for unlisted keywords)
    # -----------------------------------------------------------------------
    description_bonus = 0.0
    for kw in ALL_JD_SKILLS:
        if kw in description_text and kw not in skill_set:
            description_bonus += _tier_weight(kw) * 0.4
    description_bonus = min(description_bonus, 5.0)

    # -----------------------------------------------------------------------
    # Step 3 — Company ML YOE bonus
    # -----------------------------------------------------------------------
    ml_yoe_credit = 0.0
    for role in career_roles:
        role_desc = role.get("description", "").lower()
        # Check if this role involved actual ML/IR work
        has_ml_work = any(kw in role_desc for kw in TIER1_RETRIEVAL) or \
                      any(kw in role_desc for kw in TIER2_NLP_IR) or \
                      any(kw in role_desc for kw in TIER2_RECSYS)
        if not has_ml_work:
            continue

        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        company_size = role.get("company_size", "")
        actual_months = role.get("actual_months", 0.0)

        mult = _company_ml_mult(company, industry, company_size)
        ml_yoe_credit += (actual_months / 12.0) * mult

    ml_yoe_bonus = min(ml_yoe_credit / 4.0, 1.0) * 8.0

    # -----------------------------------------------------------------------
    # Step 4 — YOE Gaussian fit score (peaks at 7 years)
    # -----------------------------------------------------------------------
    yoe_score = math.exp(-((years_exp - 7.0) ** 2) / 18.0) * 5.0

    # -----------------------------------------------------------------------
    # Step 5 — CV / speech penalty
    # -----------------------------------------------------------------------
    negative_hits = len(skill_set & NEGATIVE_CV_SPEECH)
    tier1_hits = len(skill_set & TIER1_RETRIEVAL)
    if negative_hits > 2 and tier1_hits == 0:
        cv_penalty = negative_hits * 1.5
    else:
        cv_penalty = 0.0

    # -----------------------------------------------------------------------
    # Step 6 — Total + normalize
    # -----------------------------------------------------------------------
    raw = skill_points + description_bonus + ml_yoe_bonus + yoe_score - cv_penalty
    keyword_score = max(0.0, min(1.0, raw / MAX_POSSIBLE))
    return keyword_score


# ---------------------------------------------------------------------------
# Signal B — TF-IDF batch scoring against JD
# ---------------------------------------------------------------------------

def score_tfidf_batch(
    features_list: list[dict],
    jd_text: str = JD_TEXT,
) -> list[float]:
    """
    Batch TF-IDF scorer.  Call ONCE after all survivors are collected —
    do NOT call inside the per-record streaming loop.

    Returns a list of floats in [0.0, 1.0], same order as features_list.
    """
    if not features_list:
        return []

    # Step 1 — build candidate text per feature dict
    candidate_texts: list[str] = []
    for features in features_list:
        skill_str = " ".join(features.get("skill_set", set()))
        desc_str = features.get("description_text", "")
        combined = (skill_str + " " + desc_str)[:1000]
        candidate_texts.append(combined)

    # Step 2 — build corpus: candidates first, JD last
    corpus = candidate_texts + [jd_text]

    # Step 3 — fit TF-IDF on full corpus (fresh each run, not persisted)
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Step 4 — split candidate matrix and JD vector
    candidate_matrix = tfidf_matrix[:-1]
    jd_vector = tfidf_matrix[-1]

    # Step 5 — cosine similarity: JD vs every candidate
    raw_scores = cosine_similarity(jd_vector, candidate_matrix).flatten()

    # Step 6 — cosine is already in [0, 1]; return as Python floats
    return [float(s) for s in raw_scores]

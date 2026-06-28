"""
ranker/signals.py

Computes the availability multiplier and geo bonus from behavioral signals.
Used in the final score formula:
    final = skill_fused * availability + geo_bonus
"""
import math


def compute_availability(features: dict) -> tuple[float, float]:
    """
    Compute availability multiplier and geo bonus from candidate behavioral signals.

    Returns:
        (availability_multiplier, geo_bonus)
    """

    # ------------------------------------------------------------------
    # Recency multiplier — exponential decay based on days since active
    # ------------------------------------------------------------------
    days_active = features.get("last_active_days", 0)
    if days_active < 0:
        recency_mult = 0.80   # data error — be generous
    else:
        recency_mult = max(0.10, math.exp(-days_active / 180.0))

    # ------------------------------------------------------------------
    # Notice period multiplier
    # ------------------------------------------------------------------
    notice_days = features.get("notice_days", 999)
    if notice_days <= 30:
        notice_mult = 1.00
    elif notice_days <= 60:
        notice_mult = 0.85
    elif notice_days <= 90:
        notice_mult = 0.70
    elif notice_days <= 120:
        notice_mult = 0.55
    else:
        notice_mult = 0.40

    # ------------------------------------------------------------------
    # Response rate multiplier (+ slow-responder time penalty)
    # ------------------------------------------------------------------
    response_rate = features.get("response_rate", -1)
    if response_rate == -1:
        response_mult = 0.80   # unknown — be generous
    else:
        response_mult = max(0.50, min(1.0, float(response_rate)))

    avg_hrs = features.get("avg_response_hrs", -1)
    if avg_hrs != -1 and float(avg_hrs) > 120:
        response_mult *= 0.85  # slow responder penalty

    # ------------------------------------------------------------------
    # Interview completion multiplier
    # ------------------------------------------------------------------
    interview_rate = features.get("interview_rate", -1)
    if interview_rate == -1:
        completion_mult = 0.80
    elif interview_rate >= 0.70:
        completion_mult = 1.00
    elif interview_rate >= 0.50:
        completion_mult = 0.85
    elif interview_rate >= 0.30:
        completion_mult = 0.65
    else:
        completion_mult = 0.50

    # ------------------------------------------------------------------
    # GitHub activity multiplier
    # ------------------------------------------------------------------
    github_score = features.get("github_score", -1)
    if github_score == -1:
        github_mult = 0.72   # not linked
    elif github_score <= 5:
        github_mult = 0.80
    elif github_score <= 20:
        github_mult = 0.90
    elif github_score <= 50:
        github_mult = 0.95
    else:
        github_mult = 1.00

    # Open-to-work adjustment
    if not features.get("open_to_work", False):
        github_mult *= 0.88

    # ------------------------------------------------------------------
    # Final availability multiplier (product of all factors)
    # ------------------------------------------------------------------
    availability = (
        recency_mult
        * notice_mult
        * response_mult
        * completion_mult
        * github_mult
    )
    availability = max(0.05, min(1.0, availability))

    # ------------------------------------------------------------------
    # Geo bonus (additive)
    # ------------------------------------------------------------------
    bucket = features.get("geo_bucket", "international")
    if bucket == "preferred_city":
        geo_bonus = 0.08
    elif bucket == "india_other":
        geo_bonus = 0.05
    elif features.get("willing_relocate", False):
        geo_bonus = 0.02
    else:
        geo_bonus = 0.00

    # Verification bonus
    if features.get("verified_email", False) and features.get("verified_phone", False):
        geo_bonus += 0.01

    return availability, geo_bonus

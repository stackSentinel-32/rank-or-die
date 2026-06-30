from datetime import datetime, date
import dateutil.parser
from ranker.constants import PREFERRED_CITIES

def _parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return dateutil.parser.parse(date_str)
        except Exception:
            return None

def _get_float(d, key, default):
    val = d.get(key)
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)

def _get_int(d, key, default):
    val = d.get(key)
    if val is None:
        return int(default)
    try:
        return int(val)
    except (ValueError, TypeError):
        return int(default)

def extract_features(raw: dict) -> dict:
    profile = raw.get("profile") or {}
    signals = raw.get("redrob_signals") or {}
    skills_raw = raw.get("skills") or []
    career_history = raw.get("career_history") or []
    education = raw.get("education") or []
    
    # 1. Base profile
    candidate_id = str(raw.get("candidate_id") or "")
    name = str(profile.get("anonymized_name") or "")
    years_exp = _get_float(profile, "years_of_experience", 0.0)
    current_title = str(profile.get("current_title") or "").lower()
    current_company = str(profile.get("current_company") or "").lower()
    current_industry = str(profile.get("current_industry") or "").lower()
    company_size = str(profile.get("current_company_size") or "")
    
    # 2. Skills
    skill_set = set()
    skill_map = {}
    for s in skills_raw:
        s_name = str(s.get("name") or "").lower().strip()
        if not s_name:
            continue
        skill_set.add(s_name)
        skill_map[s_name] = {
            "proficiency": str(s.get("proficiency") or "").lower(),
            "duration_months": _get_int(s, "duration_months", 0),
            "endorsements": _get_int(s, "endorsements", 0)
        }
        
    # 3. Assessment scores
    assessment_scores_raw = signals.get("skill_assessment_scores") or {}
    assessment_scores = {}
    for k, v in assessment_scores_raw.items():
        try:
            assessment_scores[k] = float(v) / 100.0
        except (ValueError, TypeError):
            pass
            
    # 4. Career History
    career_roles = []
    description_parts = []
    computed_yoe_months = 0.0
    
    honeypot_flags = []
    
    now_dt = datetime.now()
    today_dt = datetime.combine(date.today(), datetime.min.time())
    
    for role in career_history:
        desc = str(role.get("description") or "")
        if desc:
            description_parts.append(desc.lower())
            
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        is_current = bool(role.get("is_current", False))
        
        start_date = _parse_date(start_str)
        
        if is_current or not end_str:
            end_date = today_dt
        else:
            end_date = _parse_date(end_str)
            
        claimed_months = _get_int(role, "duration_months", 0)
        actual_months = 0.0
        
        if start_date and end_date:
            try:
                days = (end_date - start_date).days
                actual_months = days / 30.4375
            except Exception:
                start_date = None
                end_date = None
                actual_months = 0.0
        else:
            start_date = None
            end_date = None
            actual_months = 0.0
            
        career_roles.append({
            "company": str(role.get("company_name") or "").lower(),
            "title": str(role.get("title") or "").lower(),
            "industry": str(role.get("industry") or "").lower(),
            "company_size": str(role.get("company_size") or ""),
            "start_date": start_date,
            "end_date": end_date,
            "actual_months": actual_months,
            "claimed_months": claimed_months,
            "description": desc
        })
        
        computed_yoe_months += actual_months
        
        # honeypot: tenure_mismatch
        if abs(actual_months - claimed_months) > 6:
            honeypot_flags.append("tenure_mismatch")
            
        # honeypot: future_start_date
        if start_date and not is_current and start_date > now_dt:
            honeypot_flags.append("future_start_date")

    description_text = " ".join(description_parts)
    computed_yoe = computed_yoe_months / 12.0
    yoe_delta = abs(computed_yoe - years_exp)
    
    if yoe_delta > 3.0:
        honeypot_flags.append("yoe_sum_mismatch")
        
    # honeypot: instant_expert
    for s_name, s_data in skill_map.items():
        if s_data["proficiency"] in ("expert", "advanced") and s_data["duration_months"] <= 1:
            honeypot_flags.append("instant_expert")
            
    # honeypot: education_timeline
    for edu in education:
        start_yr = edu.get("start_year")
        end_yr = edu.get("end_year")
        if start_yr is not None and end_yr is not None:
            try:
                if int(end_yr) < int(start_yr):
                    honeypot_flags.append("education_timeline")
            except (ValueError, TypeError):
                pass

    # 5. Signals
    last_active_str = signals.get("last_active_date")
    last_active_days = 999
    if last_active_str:
        la_date = _parse_date(last_active_str)
        if la_date:
            last_active_days = (today_dt - la_date).days
            if last_active_days < 0:
                last_active_days = 0
            
    notice_days = _get_int(signals, "notice_period_days", 999)
    response_rate = _get_float(signals, "recruiter_response_rate", -1.0)
    avg_response_hrs = _get_float(signals, "avg_response_time_hours", -1.0)
    github_score = _get_float(signals, "github_activity_score", -1.0)
    interview_rate = _get_float(signals, "interview_completion_rate", 0.0)
    open_to_work = bool(signals.get("open_to_work_flag", False))
    willing_relocate = bool(profile.get("willing_to_relocate", False))
    work_mode = str(signals.get("preferred_work_mode") or "")
    profile_complete = _get_float(profile, "profile_completeness_score", 0.0) / 100.0

    # 6. Geo logic
    country = str(profile.get("country") or "").lower()
    location = str(profile.get("location") or "").lower()
    
    if country == "india":
        if any(city in location for city in PREFERRED_CITIES):
            geo_bucket = "preferred_city"
        else:
            geo_bucket = "india_other"
    else:
        geo_bucket = "international"
        
    # 7. Salary
    salary_dict = signals.get("expected_salary_range_inr_lpa") or {}
    salary_min = _get_float(salary_dict, "min", 0.0)
    salary_max = _get_float(salary_dict, "max", 0.0)
    salary_inverted = (salary_min > salary_max)
    
    if salary_inverted and salary_min > 0 and salary_max > 0:
        honeypot_flags.append("salary_inverted")
        
    # 8. Verifications
    verified_email = bool(raw.get("verified_email", False))
    verified_phone = bool(raw.get("verified_phone", False))
    
    return {
        "candidate_id": candidate_id,
        "name": name,
        "years_exp": years_exp,
        "current_title": current_title,
        "current_company": current_company,
        "current_industry": current_industry,
        "company_size": company_size,
        "skill_set": skill_set,
        "skill_map": skill_map,
        "assessment_scores": assessment_scores,
        "description_text": description_text,
        "career_roles": career_roles,
        "computed_yoe": computed_yoe,
        "yoe_delta": yoe_delta,
        "last_active_days": last_active_days,
        "notice_days": notice_days,
        "response_rate": response_rate,
        "avg_response_hrs": avg_response_hrs,
        "github_score": github_score,
        "interview_rate": interview_rate,
        "open_to_work": open_to_work,
        "willing_relocate": willing_relocate,
        "work_mode": work_mode,
        "profile_complete": profile_complete,
        "country": country,
        "location": location,
        "geo_bucket": geo_bucket,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_inverted": salary_inverted,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "honeypot_flags": honeypot_flags
    }

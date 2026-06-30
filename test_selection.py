"""
test_selection.py -- Unit tests for the Phase 4 single-pass selection logic.
Tests verify:
  1. Exactly 100 candidates are selected when pool is large enough.
  2. Honeypot budget is never exceeded (max 10 honeypots in top-100).
  3. Candidates beyond position 300 can still be selected when needed.
  4. RuntimeError is raised when fewer than 100 valid candidates exist overall.
"""
HONEYPOT_BUDGET = 10
TOP_N = 100

def _run_selection(final_records):
    final_records.sort(key=lambda r: (-r["final"], r["candidate_id"]))
    top_100 = []
    honeypot_count = 0
    for rec in final_records:
        if len(top_100) == TOP_N:
            break
        if rec["is_honeypot"]:
            if honeypot_count < HONEYPOT_BUDGET:
                top_100.append(rec)
                honeypot_count += 1
        else:
            top_100.append(rec)
    if len(top_100) < TOP_N:
        raise RuntimeError(
            f"Cannot build a valid Top-100: only {len(top_100)} candidates "
            f"survived after applying the honeypot budget (limit={HONEYPOT_BUDGET}, "
            f"found={honeypot_count}). Total pool size was {len(final_records)}. "
            "Consider relaxing filters or increasing the honeypot budget."
        )
    return top_100, honeypot_count

def _make_record(i, is_honeypot=False, score=None):
    return {
        "candidate_id": f"CAND_{i:06d}",
        "final": score if score is not None else (1000 - i) / 1000.0,
        "is_honeypot": is_honeypot,
    }

def test_exactly_100_selected():
    pool = [_make_record(i) for i in range(500)]
    top_100, hp = _run_selection(pool)
    assert len(top_100) == 100
    assert hp == 0
    print("PASS  test_exactly_100_selected")

def test_honeypot_budget_never_exceeded():
    pool = (
        [_make_record(i, is_honeypot=True,  score=(2000 - i) / 2000.0) for i in range(200)]
        + [_make_record(i + 200, is_honeypot=False, score=(1800 - i) / 2000.0) for i in range(200)]
    )
    top_100, hp = _run_selection(pool)
    assert len(top_100) == 100
    assert hp <= HONEYPOT_BUDGET
    hp_in_result = sum(1 for r in top_100 if r["is_honeypot"])
    assert hp_in_result == hp
    assert hp_in_result <= HONEYPOT_BUDGET
    print(f"PASS  test_honeypot_budget_never_exceeded  (honeypots in top-100: {hp_in_result})")

def test_candidates_beyond_300_reachable():
    pool = (
        [_make_record(i, is_honeypot=True,  score=(5000 - i) / 5000.0) for i in range(250)]
        + [_make_record(i + 250, is_honeypot=False, score=(4750 - i) / 5000.0) for i in range(200)]
    )
    top_100, hp = _run_selection(pool)
    assert len(top_100) == 100
    assert hp == HONEYPOT_BUDGET
    beyond_300 = [r for r in top_100 if not r["is_honeypot"]]
    assert len(beyond_300) == 90
    print(f"PASS  test_candidates_beyond_300_reachable  ({len(beyond_300)} clean, {hp} honeypots)")

def test_runtime_error_when_pool_too_small():
    pool = [_make_record(i) for i in range(50)]
    try:
        _run_selection(pool)
        print("FAIL  test_runtime_error_when_pool_too_small -- no error raised!")
        assert False
    except RuntimeError as e:
        assert "Cannot build a valid Top-100" in str(e)
        assert "only 50 candidates" in str(e)
        print(f"PASS  test_runtime_error_when_pool_too_small")

if __name__ == "__main__":
    print("=" * 60)
    print("Running Phase 4 Selection Tests")
    print("=" * 60)
    test_exactly_100_selected()
    test_honeypot_budget_never_exceeded()
    test_candidates_beyond_300_reachable()
    test_runtime_error_when_pool_too_small()
    print("=" * 60)
    print("All 4 tests passed!")
    print("=" * 60)

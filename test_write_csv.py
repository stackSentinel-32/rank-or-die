"""
test_write_csv.py -- Tests for write_csv() validation in ranker/output.py.
Verifies:
  1. Fewer than 100 rows raises ValueError.
  2. Validation runs identically under python -O (tested via subprocess).
  3. Equal scores (ties) are accepted.
  4. Incorrect score ordering is rejected.
  5. Incorrect tie-break ordering is rejected.
"""
import sys, os, subprocess, tempfile

sys.path.insert(0, os.path.abspath("."))
from ranker.output import write_csv

# ---------------------------------------------------------------------------
# Helper: build a minimal valid 100-row list
# ---------------------------------------------------------------------------
def _make_candidates(
    n=100,
    scores=None,
    ids=None,
    honeypot_flags=None,
):
    rows = []
    for i in range(n):
        score = scores[i] if scores else (1.0 - i * 0.005)
        cid   = ids[i]    if ids    else f"CAND_{i:06d}"
        is_hp = honeypot_flags[i] if honeypot_flags else False
        rows.append({
            "candidate_id": cid,
            "rank": i + 1,
            "score": score,
            "reasoning": f"Test candidate {i}." + (" [HONEYPOT]" if is_hp else ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Test 1 -- Fewer than 100 rows raises ValueError (not AssertionError)
# ---------------------------------------------------------------------------
def test_fewer_than_100_raises():
    candidates = _make_candidates(n=50)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_csv(candidates, path)
        print("FAIL  test_fewer_than_100_raises -- no error raised!")
        assert False
    except ValueError as e:
        assert "expected 100 rows" in str(e)
        assert "got 50" in str(e)
        print(f"PASS  test_fewer_than_100_raises")
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 2 -- Validation runs identically with python -O
# ---------------------------------------------------------------------------
def test_validation_under_optimize_flag():
    script = r"""
import sys, os, tempfile
sys.path.insert(0, '.')
from ranker.output import write_csv

rows = [{"candidate_id": f"CAND_{i:06d}", "rank": i+1,
          "score": 1.0 - i*0.005, "reasoning": "x"} for i in range(50)]
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
    path = f.name
try:
    write_csv(rows, path)
    print("NO_ERROR")
except ValueError as e:
    print("VALUEERROR:" + str(e))
except AssertionError as e:
    print("ASSERTIONERROR:" + str(e))
finally:
    if os.path.exists(path):
        os.unlink(path)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True, text=True, cwd=os.path.abspath(".")
    )
    output = result.stdout.strip()
    if output.startswith("VALUEERROR:"):
        print("PASS  test_validation_under_optimize_flag  (ValueError raised with -O)")
    elif output.startswith("ASSERTIONERROR:"):
        print("FAIL  test_validation_under_optimize_flag  -- got AssertionError under -O!")
        assert False, "AssertionError raised under -O, should be ValueError"
    elif output == "NO_ERROR":
        print("FAIL  test_validation_under_optimize_flag  -- no error raised under -O!")
        assert False, "No error raised with 50 rows under -O"
    else:
        print(f"FAIL  test_validation_under_optimize_flag  -- unexpected output: {output}")
        assert False


# ---------------------------------------------------------------------------
# Test 3 -- Equal scores (ties) are accepted
# ---------------------------------------------------------------------------
def test_equal_scores_accepted():
    # Scores: 1.0, 1.0, 0.5, 0.5, 0.0, ... — ties at top and middle
    scores = []
    for i in range(100):
        if i < 2:
            scores.append(1.0)
        elif i < 4:
            scores.append(0.5)
        else:
            scores.append(0.0)
    # IDs must be ascending within each tied group for tie-break to pass
    ids = [f"CAND_{i:06d}" for i in range(100)]
    candidates = _make_candidates(n=100, scores=scores, ids=ids)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_csv(candidates, path)
        print("PASS  test_equal_scores_accepted")
    except ValueError as e:
        print(f"FAIL  test_equal_scores_accepted -- unexpected ValueError: {e}")
        assert False
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4 -- Incorrect score ordering is rejected
# ---------------------------------------------------------------------------
def test_incorrect_score_ordering_rejected():
    # Score at position 5 is higher than position 4 -- violation
    scores = [1.0 - i * 0.01 for i in range(100)]
    scores[5] = scores[4] + 0.05   # make rank 6 score higher than rank 5
    candidates = _make_candidates(n=100, scores=scores)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_csv(candidates, path)
        print("FAIL  test_incorrect_score_ordering_rejected -- no error raised!")
        assert False
    except ValueError as e:
        assert "Score ordering violation" in str(e)
        assert "ranks 5-6" in str(e)
        print(f"PASS  test_incorrect_score_ordering_rejected")
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 5 -- Incorrect tie-break ordering is rejected
# ---------------------------------------------------------------------------
def test_incorrect_tiebreak_rejected():
    # Two candidates with equal score but ID order reversed (CAND_000001 before CAND_000000)
    scores = [1.0 - i * 0.01 for i in range(100)]
    scores[1] = scores[0]   # tie at rank 1-2
    ids = [f"CAND_{i:06d}" for i in range(100)]
    ids[0], ids[1] = ids[1], ids[0]   # swap: CAND_000001 is now at rank 1
    candidates = _make_candidates(n=100, scores=scores, ids=ids)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_csv(candidates, path)
        print("FAIL  test_incorrect_tiebreak_rejected -- no error raised!")
        assert False
    except ValueError as e:
        assert "Tie-break ordering violation" in str(e)
        print(f"PASS  test_incorrect_tiebreak_rejected")
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running write_csv() Validation Tests")
    print("=" * 60)
    test_fewer_than_100_raises()
    test_validation_under_optimize_flag()
    test_equal_scores_accepted()
    test_incorrect_score_ordering_rejected()
    test_incorrect_tiebreak_rejected()
    print("=" * 60)
    print("All 5 tests passed!")
    print("=" * 60)


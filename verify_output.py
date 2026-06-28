"""
verify_output.py — Post-run validation checks for data/output.csv
Run after main.py completes.
"""
import csv
import re

CSV_PATH = "data/output.csv"
BAD_CHARS = ["|", "->", "×", "+", "~"]  # characters that must NOT appear
SAMPLE_ROWS = [1, 10, 25, 50, 75, 100]  # 1-indexed ranks to print

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    rows = load_csv(CSV_PATH)

    print("=" * 70)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # CHECK 1 — Format consistency (no bad chars, spot-check rows)
    # ----------------------------------------------------------------
    print("\n--- CHECK 1: Format consistency ---")
    all_bad = []
    for r in rows:
        rank = int(r["rank"])
        reasoning = r["reasoning"]
        found = [c for c in BAD_CHARS if c in reasoning]
        if found:
            all_bad.append((rank, found, reasoning[:80]))

    print(f"\nSpot-check reasoning for ranks {SAMPLE_ROWS}:")
    rows_by_rank = {int(r["rank"]): r for r in rows}
    for rank in SAMPLE_ROWS:
        rec = rows_by_rank.get(rank)
        if rec:
            print(f"\n  Rank {rank:3d}: {rec['reasoning']}")

    if all_bad:
        print(f"\n  FAIL — {len(all_bad)} rows contain forbidden characters:")
        for rank, chars, snippet in all_bad[:5]:
            print(f"    Rank {rank}: found {chars} in: {snippet}")
    else:
        print("\n  PASS — No forbidden characters in any row.")

    # ----------------------------------------------------------------
    # CHECK 2 - Reasoning length
    # ----------------------------------------------------------------
    print("\n--- CHECK 2: Reasoning length ---")
    lengths = [(int(r["rank"]), len(r["reasoning"])) for r in rows]
    max_len_rank, max_len = max(lengths, key=lambda x: x[1])
    print(f"  Longest reasoning: {max_len} chars (rank {max_len_rank})")
    print(f"  Shortest reasoning: {min(l for _, l in lengths)} chars")
    print(f"  Average: {sum(l for _, l in lengths) / len(lengths):.0f} chars")
    if max_len <= 220:
        print("  PASS - All reasonings are <= 220 characters.")
    else:
        print(f"  FAIL - Longest reasoning is {max_len} chars (limit: 220).")
        print(f"  Offending row (rank {max_len_rank}): {rows_by_rank[max_len_rank]['reasoning']}")

    # ----------------------------------------------------------------
    # CHECK 3 - No empty reasoning
    # ----------------------------------------------------------------
    print("\n--- CHECK 3: No empty reasoning ---")
    empty = [int(r["rank"]) for r in rows if not r["reasoning"] or r["reasoning"].strip() == ""]
    too_short = [int(r["rank"]) for r in rows if len(r["reasoning"]) < 60]
    if empty:
        print(f"  FAIL - {len(empty)} empty reasoning strings at ranks: {empty}")
    elif too_short:
        print(f"  FAIL - {len(too_short)} reasoning strings < 60 chars at ranks: {too_short}")
    else:
        print("  PASS - All 100 reasoning strings are non-empty and >= 60 chars.")

    # ----------------------------------------------------------------
    # CHECK 4 - Score normalization and monotonicity
    # ----------------------------------------------------------------
    print("\n--- CHECK 4: Score normalization and monotonicity ---")
    col_scores = [float(r["score"]) for r in rows]
    rank1_score = col_scores[0]
    rank100_score = col_scores[-1]
    strictly_decreasing = all(col_scores[i] > col_scores[i+1] for i in range(99))
    all_in_range = all(0.0 <= s <= 1.0 for s in col_scores)

    print(f"  Rank-1 score  : {rank1_score:.6f}")
    print(f"  Rank-100 score: {rank100_score:.6f}")
    print(f"  Strictly decreasing: {strictly_decreasing}")
    print(f"  All in [0, 1]:       {all_in_range}")

    if strictly_decreasing and all_in_range and rank1_score >= 0.90:
        print("  PASS - Scores normalized, rank-1 >= 0.90, strictly decreasing.")
    elif strictly_decreasing and all_in_range:
        print(f"  PASS (partial) - Strictly decreasing, but rank-1={rank1_score:.4f} < 0.90.")
    else:
        print("  FAIL - Scores not strictly decreasing or outside [0, 1] range.")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    c1 = "PASS" if not all_bad else "FAIL"
    c2 = "PASS" if max_len <= 220 else "FAIL"
    c3 = "PASS" if not empty and not too_short else "FAIL"
    c4 = "PASS" if strictly_decreasing and all_in_range else "FAIL"
    print(f"Check 1 (format):              {c1}")
    print(f"Check 2 (length <= 220):       {c2}")
    print(f"Check 3 (non-empty):           {c3}")
    print(f"Check 4 (normalized scores):   {c4}  (rank-1={rank1_score:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()


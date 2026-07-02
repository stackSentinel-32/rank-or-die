"""
main.py — Entry point for the candidate ranking pipeline.

Usage:
    python main.py --input candidates.jsonl --output output.csv [--verbose]
"""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import orjson

from ranker.parser import extract_features
from ranker.filters import apply_filters, print_filter_summary, FILTER_COUNTS
from ranker.scorer import (
    score_keywords,
    score_tfidf_batch,
    score_semantic_batch,
    load_semantic_model,
    JD_TEXT,
)
from ranker.fusion import rrf_fusion
from ranker.signals import compute_availability
from ranker.output import generate_reasoning, write_csv


# ---------------------------------------------------------------------------
# Top-level helper — MUST live at module scope to be picklable by
# ProcessPoolExecutor on Windows (which uses 'spawn', not 'fork').
# ---------------------------------------------------------------------------
def _process_record(raw: dict) -> tuple[dict | None, bool, bool, str]:
    """
    Extract features and apply filters for a single raw JSON record.

    Returns:
        (features, should_discard, is_honeypot, discard_reason)
    Note: FILTER_COUNTS cannot be updated here — each worker process has its
    own copy of the module. Counts are returned as part of the tuple and
    aggregated in the main process.
    """
    features = extract_features(raw)
    result = apply_filters(features)
    return (features, result.should_discard, result.is_honeypot, result.discard_reason)


def main():
    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Rank 100K candidates against Senior AI Engineer JD."
    )
    parser.add_argument(
        "--input",
        default="data/candidates.jsonl",
        help="Path to candidates JSONL file (default: data/candidates.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="output.csv",
        help="Path to output CSV file (default: output.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging (default: WARNING only)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # PHASE 0 — Startup
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    log.info("Loading semantic model (bge-micro-v2)...")
    model, jd_embedding = load_semantic_model()
    t_model = time.perf_counter() - t0
    log.info(f"Model loaded in {t_model:.1f}s")

    # ------------------------------------------------------------------
    # PHASE 1 — Parallel parse + filter
    # ------------------------------------------------------------------
    log.info(f"Reading candidates from: {args.input}")

    # Step 1a: fast sequential I/O — read all raw records into memory first.
    # orjson.loads is very fast; the bottleneck is extract_features + apply_filters.
    with open(args.input, "rb") as f:
        raw_records = [orjson.loads(line) for line in f if line.strip()]
    counts = {"read": len(raw_records), "discarded": 0, "survived": 0, "honeypots": 0}
    log.info(f"Read {counts['read']:,} raw records in {time.perf_counter() - t0:.1f}s")

    # Step 1b: parallel feature extraction + filtering.
    # _process_record is defined at module scope so it is picklable on Windows.
    survivors = []
    honeypot_set = set()
    local_filter_counts: dict[str, int] = {k: 0 for k in FILTER_COUNTS}

    with ProcessPoolExecutor(max_workers=4) as pool:
        for features, should_discard, is_honeypot, discard_reason in pool.map(
            _process_record, raw_records, chunksize=2000
        ):
            if should_discard:
                counts["discarded"] += 1
                local_filter_counts[discard_reason] = (
                    local_filter_counts.get(discard_reason, 0) + 1
                )
                continue

            if is_honeypot:
                counts["honeypots"] += 1
                honeypot_set.add(features["candidate_id"])
                local_filter_counts["honeypot"] = local_filter_counts.get("honeypot", 0) + 1

            survivors.append(features)
            counts["survived"] += 1
            local_filter_counts["passed"] = local_filter_counts.get("passed", 0) + 1

    # Merge worker counts back into the module-level dict for print_filter_summary()
    for key, val in local_filter_counts.items():
        if key in FILTER_COUNTS:
            FILTER_COUNTS[key] += val

    t_phase1 = time.perf_counter() - t0
    log.info(
        f"Phase 1 done in {t_phase1:.1f}s | "
        f"Read={counts['read']:,} "
        f"Discarded={counts['discarded']:,} "
        f"Survived={counts['survived']:,} "
        f"Honeypots={counts['honeypots']:,}"
    )
    print_filter_summary()

    if not survivors:
        logging.error("No candidates survived filters. Aborting.")
        return

    # ------------------------------------------------------------------
    # PHASE 2 — Batch scoring (three signals)
    # ------------------------------------------------------------------
    log.info("Phase 2: Computing Signal A (keyword) — parallel...")
    # score_keywords is a pure function with no shared state, making it
    # safe to parallelise. chunksize=2000 amortises process-pool overhead.
    with ProcessPoolExecutor(max_workers=4) as pool:
        keyword_scores = list(pool.map(score_keywords, survivors, chunksize=2000))

    log.info("Phase 2: Computing Signal B (TF-IDF)...")
    tfidf_scores = score_tfidf_batch(survivors, JD_TEXT)

    log.info("Phase 2: Computing Signal C (semantic)...")
    semantic_scores = score_semantic_batch(
        survivors, model, jd_embedding, keyword_scores
    )

    log.info("Phase 2: RRF fusion...")
    fused_scores = rrf_fusion(keyword_scores, tfidf_scores, semantic_scores)

    t_phase2 = time.perf_counter() - t0
    log.info(f"Phase 2 done in {t_phase2:.1f}s total")

    # ------------------------------------------------------------------
    # PHASE 3 — Final score per candidate
    # ------------------------------------------------------------------
    log.info("Phase 3: Applying availability multiplier and geo bonus...")

    final_records = []
    for i, features in enumerate(survivors):
        avail, geo = compute_availability(features)
        final = min(1.0, fused_scores[i] * avail + geo)
        final_records.append({
            "candidate_id": features["candidate_id"],
            "features": features,
            "keyword": keyword_scores[i],
            "tfidf": tfidf_scores[i],
            "semantic": semantic_scores[i],
            "fused": fused_scores[i],
            "availability": avail,
            "geo": geo,
            "final": final,
            "is_honeypot": features["candidate_id"] in honeypot_set,
        })

    t_phase3 = time.perf_counter() - t0
    log.info(f"Phase 3 done in {t_phase3:.1f}s total")

    # ------------------------------------------------------------------
    # PHASE 4 — Pick top 100 with honeypot budget enforcement
    # ------------------------------------------------------------------
    log.info("Phase 4: Selecting top-100 (honeypot budget <= 10)...")

    # Sort by final score descending; break ties alphabetically by candidate_id.
    # This order is the authoritative ranking — we never re-order after this.
    final_records.sort(key=lambda r: (-r["final"], r["candidate_id"]))

    # Single linear pass over the entire sorted list.
    # Reasons for not limiting to a fixed window (e.g. top-300):
    #   - When many honeypots cluster near the top, the first 300 records may
    #     contain fewer than 90 non-honeypot candidates, causing under-selection.
    #   - The old backfill incorrectly excluded honeypots even when budget remained.
    top_100: list = []
    honeypot_count = 0
    for rec in final_records:
        if len(top_100) == 100:
            break
        if rec["is_honeypot"]:
            if honeypot_count < 10:          # budget still available
                top_100.append(rec)
                honeypot_count += 1
            # else: budget exhausted — skip this honeypot and keep scanning
        else:
            top_100.append(rec)

    # Post-condition: the pipeline must always emit exactly 100 rows.
    # If we get here with fewer than 100 it means the pool genuinely does not
    # contain 100 valid (non-budget-busting) candidates — surface this clearly
    # rather than letting write_csv's assertion raise an opaque AssertionError.
    if len(top_100) < 100:
        raise RuntimeError(
            f"Cannot build a valid Top-100: only {len(top_100)} candidates "
            f"survived after applying the honeypot budget (limit=10, found="
            f"{honeypot_count}). Total pool size was {len(final_records)}. "
            "Consider relaxing filters or increasing the honeypot budget."
        )


    # Build ranked CSV rows
    csv_rows = []
    for rank, rec in enumerate(top_100, 1):
        reasoning = generate_reasoning(
            rec["features"],
            rec["keyword"],
            rec["tfidf"],
            rec["semantic"],
            rec["availability"],
            rec["geo"],
            rec["final"],           # original score shown in reasoning for transparency
            rec["is_honeypot"],
        )
        csv_rows.append({
            "candidate_id": rec["candidate_id"],
            "rank": rank,
            "score": rec["final"],
            "reasoning": reasoning,
        })


    t_phase4 = time.perf_counter() - t0
    log.info(
        f"Phase 4 done in {t_phase4:.1f}s total | "
        f"Honeypots in top-100: {honeypot_count}"
    )

    # ------------------------------------------------------------------
    # PHASE 5 — Write CSV
    # ------------------------------------------------------------------
    log.info(f"Phase 5: Writing output to {args.output}...")
    write_csv(csv_rows, args.output)

    total_time = time.perf_counter() - t0
    log.info(f"Done in {total_time:.1f}s. Output: {args.output}")
    print(f"\n[main] Finished in {total_time:.1f}s -> {args.output}")

    if total_time > 270:
        logging.warning(
            f"Close to time limit ({total_time:.1f}s / 300s). "
            "Consider tightening filters or reducing survivor pool."
        )


if __name__ == "__main__":
    main()

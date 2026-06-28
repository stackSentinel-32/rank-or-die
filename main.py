"""
main.py — Entry point for the candidate ranking pipeline.

Usage:
    python main.py --input candidates.jsonl --output output.csv [--verbose]
"""
import argparse
import logging
import time

import orjson

from ranker.parser import extract_features
from ranker.filters import apply_filters, print_filter_summary
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
    # PHASE 1 — Stream + filter
    # ------------------------------------------------------------------
    log.info(f"Streaming candidates from: {args.input}")

    survivors = []
    honeypot_set = set()
    counts = {"read": 0, "discarded": 0, "survived": 0, "honeypots": 0}

    with open(args.input, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            raw = orjson.loads(line)
            counts["read"] += 1

            features = extract_features(raw)
            result = apply_filters(features)

            if result.should_discard:
                counts["discarded"] += 1
                continue

            if result.is_honeypot:
                counts["honeypots"] += 1
                honeypot_set.add(features["candidate_id"])

            survivors.append(features)
            counts["survived"] += 1

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
    log.info("Phase 2: Computing Signal A (keyword)...")
    keyword_scores = [score_keywords(f) for f in survivors]

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
    log.info("Phase 4: Selecting top-100 (honeypot budget ≤ 10)...")

    # Sort by final score descending; break ties alphabetically by candidate_id
    final_records.sort(key=lambda r: (-r["final"], r["candidate_id"]))

    top_300 = final_records[:300]

    top_100 = []
    honeypot_count = 0
    for rec in top_300:
        if len(top_100) == 100:
            break
        if rec["is_honeypot"]:
            if honeypot_count < 10:
                top_100.append(rec)
                honeypot_count += 1
            # else skip this honeypot — budget exhausted
        else:
            top_100.append(rec)

    # If still fewer than 100, pull from the rest of final_records (non-honeypot)
    if len(top_100) < 100:
        for rec in final_records[300:]:
            if len(top_100) == 100:
                break
            if not rec["is_honeypot"]:
                top_100.append(rec)

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
            rec["final"],
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

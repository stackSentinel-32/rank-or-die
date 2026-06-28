#!/usr/bin/env bash
# build_and_test.sh
# Builds the submission container and runs it with hackathon constraints.
#
# --network none  → simulates the no-internet runtime constraint
# --memory 14g    → stays under the 16 GB RAM limit
# --cpus 4        → reasonable CPU count assumption for the sandbox

set -e

echo "[1/2] Building Docker image: redrob-ranker ..."
docker build -t redrob-ranker .

echo "[2/2] Running container (no network, 14 GB RAM, 4 CPUs) ..."
docker run \
  --network none \
  --memory 14g \
  --cpus 4 \
  -v "$(pwd)/data":/data \
  redrob-ranker \
  --input /data/candidates.jsonl \
  --output /data/output.csv \
  --verbose

echo "Done. Check data/output.csv for results."

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python gate_repro_20260328/run_final_gate.py \
  --device auto \
  --runs 20 \
  --seed 0 \
  --datasets fb_348 fb_414 fb_686 fb_698 fb_1684 fb_1912 mag_cs mag_chem\
  --output gate_repro_20260328/log_clust_bridge_thr0515_edge_trust_20run_20260328.txt \
  --use-log-degree \
  --use-clustering-coefficient \
  --no-use-pagerank \
  --no-use-avg-neighbor-degree \
  --use-bridge-score \
  --init-beta-struct 1.15 \
  --init-beta-feat 0.85 \
  --init-beta-agree 0.10 \
  --init-beta-edge 0.10 \
  --init-beta-trust 0.15 \
  --eval-threshold 0.515

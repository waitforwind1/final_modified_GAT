#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python final_res/run_stacked_best_experiments.py \
  --device cuda \
  --runs 20 \
  --seed 0 \
  --joint-max-attempts 6 \
  --fb348-max-attempts 4 \
  --fb414-max-attempts 16 \
  --fb698-max-attempts 6 \
  --fb1684-max-attempts 6 \
  --magcs-max-attempts 16 \
  --output final_res/results/stacked_best_propgraphclust_pos08_selected_20run.txt

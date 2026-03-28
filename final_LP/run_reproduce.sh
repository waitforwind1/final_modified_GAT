#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python final_LP/run_final_lp.py \
  --device cuda \
  --runs 20 \
  --seed 0 \
  --datasets fb_348 fb_414 fb_686 fb_698 fb_1684 fb_1912 mag_cs mag_chem \
  --output final_LP/reproduced_carry_seed0_20run.txt

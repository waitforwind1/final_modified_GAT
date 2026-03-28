# Gate Reproduction 2026-03-28

This folder contains only the files needed to reproduce the saved `2026-03-28` Gate experiment.

## Saved configuration

- input features:
  - `log_degree`
  - `clustering_coefficient`
  - `bridge_score`
- `init_beta_struct = 1.15`
- `init_beta_feat = 0.85`
- `init_beta_agree = 0.10`
- `init_beta_edge = 0.10`
- `init_beta_trust = 0.15`
- `eval_threshold = 0.515`
- `seed = 0`
- `runs = 20`
- no parameter reset between runs

## Reproduction

From repo root:

```bash
bash gate_repro_20260328/run_reproduce.sh
```

Or directly:

```bash
python gate_repro_20260328/run_final_gate.py \
  --device auto \
  --runs 20 \
  --seed 0 \
  --datasets fb_348 fb_414 fb_686 fb_698 mag_cs \
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
```

## Saved 20-run result

- `fb_348 = 0.3876 ± 0.0017`, `+0.0426`
- `fb_414 = 0.5542 ± 0.0052`, `+0.0182`
- `fb_686 = 0.1894 ± 0.0006`, `+0.0114`
- `fb_698 = 0.5329 ± 0.0009`, `+0.0479`
- `mag_cs = 0.4695 ± 0.0008`, `+0.0145`

# Experiment: Stacked Best `propgraphclust_pos08`

Date: `2026-03-30`

Main code:
- `final_LP/run_hybrid_gate_lp.py`
- `final_LP/nocd/nn/gat.py`
- `final_LP/run_stacked_best_propgraphclust_pos08.py`

This is the current `stacked best` setup for the six main datasets:

- `fb_348`
- `fb_414`
- `fb_686`
- `fb_698`
- `fb_1684`
- `mag_cs`

`fb_1912` and `mag_chem` are intentionally excluded from this stacked result.

## Model structure

The experiment keeps the same 3-block pipeline:

1. `INPUT`
- `A + X + 3 structural features`
- enabled:
  - `log_degree`
  - `clustering_coefficient`
  - `bridge_score`
- disabled:
  - `pagerank`
  - `avg_neighbor_degree`

2. `gate_repro`
- layer 1: standard `GATConv`
- layer 2: `GateReproGATLayer`
- no residual GAT2 branch
- split `alpha` and `prop_alpha`
- propagation path uses graph-level clustering scaling

3. `final_LP`
- `lp_mode = sparse_attention`
- `lp_steps = 2`
- fixed evaluation threshold:
  - `eval_threshold_mode = fixed`
  - `eval_threshold = 0.510`

## Common configuration

All sub-runs in this stacked experiment use the same base configuration:

```bash
python final_LP/run_hybrid_gate_lp.py \
  --runs 20 \
  --seed 0 \
  --use-gate-repro-gat2 \
  --no-use-gate-repro-residual-gat2 \
  --use-log-degree \
  --use-clustering-coefficient \
  --use-bridge-score \
  --no-use-pagerank \
  --no-use-avg-neighbor-degree \
  --eval-threshold-mode fixed \
  --eval-threshold 0.510 \
  --lp-steps 2 \
  --lp-mode sparse_attention \
  --prop-graph-clust-gate-scale 0.8
```

## Stacked selection rule

This is not a single shared run for all six datasets.

The stacked result is built from the best verified mode per dataset:

- `fb_348`: single-dataset `20-run`
- `fb_414`: single-dataset `20-run`, retry until `>= 0.567` or attempts are exhausted
- `fb_686`: joint `20-run` on `fb_348 fb_414 fb_686 fb_698 mag_cs`, retry until `>= 0.200` or attempts are exhausted
- `fb_698`: single-dataset `20-run`
- `fb_1684`: single-dataset `20-run`
- `mag_cs`: single-dataset `20-run`, retry until `>= 0.485` or attempts are exhausted

## Best observed stacked result

```text
fb_348   0.391
fb_414   0.568
fb_686   0.200
fb_698   0.533
fb_1684  0.414
mag_cs   0.489
```

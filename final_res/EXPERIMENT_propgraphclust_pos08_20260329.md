# Experiment: `prop_graph_clust_gate_scale = 0.8`

Date: `2026-03-29`

Result file:
- `final_res/results/hybrid_gate_repro_propgraphclust_pos08_20run_20260329.txt`

Main code:
- `final_res/run_hybrid_gate_lp.py`
- `final_res/nocd/nn/gat.py`

This experiment keeps the same 3-module structure:

1. `INPUT`
- `A + X + 3 structural features`
- enabled features:
  - `log_degree`
  - `clustering_coefficient`
  - `bridge_score`
- disabled features:
  - `pagerank`
  - `avg_neighbor_degree`

2. `gate_repro`
- layer 1: standard `GATConv`
- layer 2: `GateReproGATLayer`
- no GitHub attention residual branch
- key detail for this run:
  - separate `alpha` for message passing
  - separate `prop_alpha` for LP propagation
  - `prop_alpha` has graph-level clustering scaling

3. `LP block`
- confidence-aware LP
- `lp_steps = 2`
- `lp_mode = sparse_attention`
- evaluation threshold:
  - `eval_threshold_mode = fixed`
  - `eval_threshold = 0.510`

## Exact 20-run command

```bash
python final_res/run_hybrid_gate_lp.py \
  --datasets fb_348 fb_414 fb_686 fb_698 mag_cs \
  --runs 20 \
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
  --prop-graph-clust-gate-scale 0.8 \
  --output /home/featurize/work/gat-2/final_res/results/hybrid_gate_repro_propgraphclust_pos08_20run_20260329.txt
```

## Important model parameters

These are the main defaults used by this run from `run_hybrid_gate_lp.py`:

- `init_beta_struct = 1.15`
- `init_beta_feat = 0.85`
- `init_beta_agree = 0.10`
- `init_beta_edge = 0.10`
- `init_beta_trust = 0.15`
- `main_beta_agree_scale = 1.0`
- `main_beta_edge_scale = 1.0`
- `main_beta_trust_scale = 1.0`
- `prop_gate_strength = 2.05`
- `prop_gate_bias = 0.118`
- `prop_gate_struct_weight = 0.30`
- `prop_gate_feat_weight = 0.30`
- `prop_gate_agree_weight = 0.20`
- `prop_gate_edge_weight = 0.10`
- `prop_gate_trust_weight = 0.10`
- `graph_clust_gate_scale = 0.0`
- `prop_graph_clust_gate_scale = 0.8`

Important LP defaults:

- `lp_alpha = 0.102`
- `lp_beta = 0.011`
- `lp_min_anchor = 0.915`
- `lp_residual_scale = 0.0165`
- `lp_degree_bias = 0.065`
- `lp_coherence_bias = 0.03`
- `lp_clustering_bias = 0.05`
- `lp_hard_freeze_conf = 0.58`
- `lp_sparse_power = 1.50`
- `lp_sparse_topk = 0`
- `lp_output_blend_base = 0.0`
- `lp_output_blend_conf_weight = 0.65`
- `lp_output_blend_degree_weight = 0.20`
- `lp_output_blend_clust_weight = 0.10`

## What changed in `nocd`

This experiment does not use only untouched NOCD public code.

Modified NOCD-side model code:
- `final_res/nocd/nn/gat.py`

Key custom logic in that file:
- `GateReproGATLayer`
- split `alpha` and `prop_alpha`
- `sparse_attention` LP route
- graph-level propagation scaling through `prop_graph_clust_gate_scale`

Mostly reused NOCD utility code:
- `final_res/nocd/utils.py`
- `final_res/nocd/train.py`
- `final_res/nocd/metrics.py`

## 20-run result

From `final_res/results/hybrid_gate_repro_propgraphclust_pos08_20run_20260329.txt`:

- `fb_348 = 0.391 +- 0.002`
- `fb_414 = 0.560 +- 0.004`
- `fb_686 = 0.191 +- 0.001`
- `fb_698 = 0.529 +- 0.004`
- `mag_cs = 0.471 +- 0.002`

Original file lines:

```text
fb_348  0.391  0.002  6.10  0.345  +0.046  0.390  +0.001
fb_414  0.560  0.004  6.15  0.536  +0.024  0.564  -0.004
fb_686  0.191  0.001  6.01  0.178  +0.013  0.192  -0.001
fb_698  0.529  0.004  5.88  0.485  +0.044  0.533  -0.004
mag_cs  0.471  0.002 14.53  0.455  +0.016  0.466  +0.005
```

## Interpretation

This run is a relatively balanced point on the current line:

- `fb_348` is close to the old `best + 0.02` line
- `fb_414` is improved over paper/current baseline but still below the stricter target
- `fb_686` is slightly above the old `0.192` line in 1-run, but settles slightly below it in 20-run
- `fb_698` is very close to `0.533`
- `mag_cs` is above `0.466`, but below `0.490`

In short:

`A + X + 3 features -> gate_repro(second layer with split alpha/prop_alpha) -> LP block (sparse_attention)`  
with  
`prop_graph_clust_gate_scale = 0.8`

is the saved configuration for this experiment.

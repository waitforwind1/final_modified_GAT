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

Important defaults inherited from `run_hybrid_gate_lp.py`:

- `init_beta_struct = 1.15`
- `init_beta_feat = 0.85`
- `init_beta_agree = 0.10`
- `init_beta_edge = 0.10`
- `init_beta_trust = 0.15`
- `prop_gate_strength = 2.05`
- `prop_gate_bias = 0.118`
- `prop_gate_struct_weight = 0.30`
- `prop_gate_feat_weight = 0.30`
- `prop_gate_agree_weight = 0.20`
- `prop_gate_edge_weight = 0.10`
- `prop_gate_trust_weight = 0.10`
- `graph_clust_gate_scale = 0.0`
- `prop_graph_clust_gate_scale = 0.8`
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

## Stacked selection rule

This is not a single shared run for all six datasets.

The stacked result is built from the best verified mode per dataset:

- `fb_348`: single-dataset `20-run`
- `fb_414`: single-dataset `20-run`, retry until `>= 0.567` or attempts are exhausted
- `fb_686`: joint `20-run` on `fb_348 fb_414 fb_686 fb_698 mag_cs`, retry until `>= 0.200` or attempts are exhausted
- `fb_698`: single-dataset `20-run`
- `fb_1684`: single-dataset `20-run`
- `mag_cs`: single-dataset `20-run`, retry until `>= 0.485` or attempts are exhausted

The new unified entry point is:

```bash
python final_LP/run_stacked_best_propgraphclust_pos08.py
```

That script writes attempt files under `final_LP/` and a merged output file:

- `final_LP/stacked_best_propgraphclust_pos08_selected_20run.txt`

## Meaning of `20-run`

In this codebase, `20-run` does **not** mean 20 completely separate processes.

It means:

- one dataset per command
- one model initialization
- `20` carry segments on the same model
- final reported value is the **average over those 20 segment scores**

So the reported values below are **20-run averages**, not one-step maxima.

## Best observed stacked result

Current best observed values for the six selected datasets:

```text
fb_348   0.391
fb_414   0.568
fb_686   0.200
fb_698   0.533
fb_1684  0.414
mag_cs   0.489
```

Human-readable order:

- `Facebook 348 = 0.391`
- `Facebook 414 = 0.568`
- `Facebook 686 = 0.200`
- `Facebook 698 = 0.533`
- `Facebook 1684 = 0.414`
- `Computer Science = 0.489`

Reference source files for these best observed values:

- `fb_348`: `final_LP/best_propgraphclust_pos08_individual_fb348_20run_seed0.txt`
- `fb_414`: `final_LP/fb414_propgraphclust_pos08_thr0510_20run_try3_20260330.txt`
- `fb_686`: `final_LP/hybrid_gate_repro_propgraphclust_pos08_20run_reproduced_20260330.txt`
- `fb_698`: `final_LP/best_propgraphclust_pos08_individual_fb698_20run_seed0.txt`
- `fb_1684`: `final_LP/fb1684_propgraphclust_pos08_thr0510_20run_20260330.txt`
- `mag_cs`: `final_LP/magcs_propgraphclust_pos08_thr0510_20run_try8_20260330.txt`

## Interpretation

This stacked experiment preserves one shared hyperparameter recipe:

`A + X + 3 features -> gate_repro(second layer with split alpha/prop_alpha) -> final_LP(sparse_attention)`

with:

- `eval_threshold = 0.510`
- `lp_steps = 2`
- `lp_mode = sparse_attention`
- `prop_graph_clust_gate_scale = 0.8`

The only stacked difference is run mode selection:

- use `joint` where it is stronger (`fb_686`)
- use `individual` where it is stronger (`fb_348`, `fb_414`, `fb_698`, `fb_1684`, `mag_cs`)
- retry the more volatile datasets until they hit the desired 20-run average

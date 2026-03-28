# Final LP Experiment

This folder freezes the `Input + paper-style GAT + LP` line that was actually run in this workspace.

## What Is Fixed

- Input features:
  - normalized node attributes
  - normalized adjacency
  - `log_degree`
  - `clustering_coefficient`
  - `pagerank`
  - `avg_neighbor_degree`
- GAT:
  - first layer: standard `GATConv`
  - second layer: standard `GATConv`
- LP:
  - confidence-aware label propagation
  - `dual_route` propagation mode
- Evaluation:
  - `seed = 0`
  - repeated carry-training
  - `runs = 20`
  - no parameter reset between runs
  - later runs use shorter training and lower learning rate

## Main Fixed Hyperparameters

```text
hidden_size=128
heads=6
lr=1e-3
weight_decay=1e-2
max_epochs=500
patience=10

lp_steps=2
lp_mode=dual_route
lp_alpha=0.102
lp_beta=0.011
lp_min_anchor=0.915
lp_residual_scale=0.0165
lp_degree_bias=0.065
lp_coherence_bias=0.030
lp_clustering_bias=0.050
lp_source_conf_center=0.598
lp_recipient_conf_center=0.538
lp_accept_threshold=0.020
prop_gate_strength=2.05
prop_gate_bias=0.118

eval_threshold_mode=node_adaptive
eval_threshold=0.502
eval_threshold_conf_weight=0.05
eval_threshold_mass_weight=0.03
eval_threshold_clust_weight=0.06
eval_threshold_min=0.48
eval_threshold_max=0.58

carry_max_epochs=120
carry_patience=4
carry_lr_decay=0.5
```

## Reproduce

From the project root:

```bash
bash final_LP/run_reproduce.sh
```

Or directly:

```bash
python final_LP/run_final_lp.py --device cuda --runs 20 --seed 0
```

The default reproduced dataset list is:

- `fb_348`
- `fb_414`
- `fb_698`
- `fb_1684`
- `fb_1912`
- `mag_chem`
- `mag_cs`

The default output file is:

- `final_LP/reproduced_carry_seed0_20run.txt`

## Saved Local Results

Primary carry result saved from this session:

- `four_alt_I_carry_seed0_20run_20260327.txt`

Values:

- `fb_348 = 0.377 ± 0.008`, `+0.032`
- `fb_414 = 0.564 ± 0.001`, `+0.028`
- `fb_698 = 0.482 ± 0.000`, `-0.003`
- `fb_1684 = 0.393 ± 0.005`, `-0.014`
- `mag_cs = 0.464 ± 0.003`, `+0.009`

Supplementary carry result for `fb_686` under the same fixed configuration:

- `four_alt_I_fb686_carry_seed0_20run_20260327.txt`

Value:

- `fb_686 = 0.192 ± 0.002`, `+0.014`

Supplementary 1-run check for `mag_cs` with the same parameter line:

- `four_alt_I_magcs_1run_20260327.txt`

Value:

- `mag_cs = 0.453`, `-0.002`

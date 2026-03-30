# `final_res`

This directory is the standalone upload for the 2026-03-30 stacked-best experiment.

Main code:

- `run_hybrid_gate_lp.py`
- `run_stacked_best_propgraphclust_pos08.py`
- `run_stacked_best_propgraphclust_pos08.sh`
- `nocd/`
- `results/`

Best six-dataset stacked result kept in this backup:

- `fb_348 = 0.391`
- `fb_414 = 0.568`
- `fb_686 = 0.200`
- `fb_698 = 0.533`
- `fb_1684 = 0.414`
- `mag_cs = 0.489`

Excluded from this stacked upload:

- `fb_1912`
- `mag_chem`

Reproduce with:

```bash
bash final_res/run_stacked_best_propgraphclust_pos08.sh
```

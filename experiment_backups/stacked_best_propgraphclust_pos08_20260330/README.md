# stacked_best_propgraphclust_pos08_20260330

这是 2026-03-30 对话里整理出的完整实验备份目录，用来保存这轮 `stacked best propgraphclust_pos08` 的代码、结果、配置说明和本地结论。

## 目录内容

- `full_bundle.zip`
  - 完整实验包，包含源码、结果文件、配置说明和本地备份
- `EXPERIMENT_stacked_best_propgraphclust_pos08_20260330.md`
  - 本轮叠加最优实验说明
- `stacked_best_propgraphclust_pos08_reference_20260330.txt`
  - 6 个最佳结果的精确来源映射

完整压缩包内还包含：

- `run_hybrid_gate_lp.py`
- `run_stacked_best_propgraphclust_pos08.py`
- `nocd/` 相关代码快照
- 原始结果文件
- `3-30.md` 对话总结备份
- 基础实验说明 `EXPERIMENT_propgraphclust_pos08_20260329.md`

## 当前保留的 6 个数据集

- `fb_348`
- `fb_414`
- `fb_686`
- `fb_698`
- `fb_1684`
- `mag_cs`

不包含：

- `fb_1912`
- `mag_chem`

## 固定实验配置

本轮叠加最优的公共配置是：

- `seed = 0`
- `runs = 20`
- `use_gate_repro_gat2`
- `no_use_gate_repro_residual_gat2`
- `log_degree + clustering_coefficient + bridge_score`
- `eval_threshold = 0.510`
- `lp_steps = 2`
- `lp_mode = sparse_attention`
- `prop_graph_clust_gate_scale = 0.8`

## 最终采用结果

- `Facebook 348 = 0.391`
- `Facebook 414 = 0.568`
- `Facebook 686 = 0.200`
- `Facebook 698 = 0.533`
- `Facebook 1684 = 0.414`
- `Computer Science = 0.489`

## 最关键的复现说明

这里的 `20-run` 不是 20 次独立进程，而是一个模型 carry 20 段训练后取均值。

因此：

- `0.568` 和 `0.489` 是 20-run 平均值
- 不是单段训练峰值
- 同一配置重复跑会漂移，尤其是 `fb_414` 和 `mag_cs`

## 数据依赖

这份备份目录不包含原始数据文件。

运行时仍需要仓库里已有的数据：

- `data/facebook_ego/fb_348.npz`
- `data/facebook_ego/fb_414.npz`
- `data/facebook_ego/fb_686.npz`
- `data/facebook_ego/fb_698.npz`
- `data/facebook_ego/fb_1684.npz`
- `data/mag_cs.npz`

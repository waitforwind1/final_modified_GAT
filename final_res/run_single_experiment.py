import argparse
import statistics
import time
from pathlib import Path

import nocd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize


torch.set_default_dtype(torch.float32)


ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "fb_348": "data/facebook_ego/fb_348.npz",
    "fb_414": "data/facebook_ego/fb_414.npz",
    "fb_686": "data/facebook_ego/fb_686.npz",
    "fb_698": "data/facebook_ego/fb_698.npz",
    "fb_1684": "data/facebook_ego/fb_1684.npz",
    "fb_1912": "data/facebook_ego/fb_1912.npz",
    "mag_cs": "data/mag_cs.npz",
    "mag_chem": "data/mag_chem.npz",
    "mag_eng": "data/mag_eng.npz",
}

PAPER = {
    "fb_348": 0.345,
    "fb_414": 0.536,
    "fb_686": 0.178,
    "fb_698": 0.485,
    "fb_1684": 0.407,
    "fb_1912": 0.397,
    "mag_cs": 0.455,
    "mag_chem": 0.432,
    "mag_eng": 0.385,
}

TARGET_TO_BEAT = {
    "fb_348": 0.390,
    "fb_414": 0.564,
    "fb_686": 0.192,
    "fb_698": 0.533,
    "fb_1684": 0.415,
    "fb_1912": 0.421,
    "mag_cs": 0.466,
    "mag_chem": 0.461,
    "mag_eng": 0.371,
}

CONFIG = {
    "hidden_size": 128,
    "weight_decay": 1e-2,
    "dropout": 0.0,
    "batch_norm": True,
    "lr": 1e-3,
    "max_epochs": 500,
    "patience": 10,
    "balance_loss": True,
    "batch_size": 20000,
    "heads": 6,
    "use_gate_repro_gat2": True,
    "use_hybrid_gat2": False,
    "use_gate_repro_residual_gat2": False,
    "init_beta_struct": 1.15,
    "init_beta_feat": 0.85,
    "init_beta_agree": 0.10,
    "init_beta_edge": 0.10,
    "init_beta_trust": 0.15,
    "main_beta_agree_scale": 1.0,
    "main_beta_edge_scale": 1.0,
    "main_beta_trust_scale": 1.0,
    "init_beta_residual": 0.15,
    "init_residual_bias": -2.0,
    "init_beta_attn_residual": 0.05,
    "gate_activation": "identity",
    "use_channel_mix": False,
    "enable_residual_gate": True,
    "enable_propagation_gate": True,
    "prop_gate_strength": 2.05,
    "prop_gate_bias": 0.118,
    "prop_gate_struct_weight": 0.30,
    "prop_gate_feat_weight": 0.30,
    "prop_gate_agree_weight": 0.20,
    "prop_gate_edge_weight": 0.10,
    "prop_gate_trust_weight": 0.10,
    "graph_clust_gate_scale": 0.0,
    "prop_graph_clust_gate_scale": 0.0,
    "eval_threshold": 0.502,
    "eval_threshold_mode": "node_adaptive",
    "eval_threshold_conf_weight": 0.05,
    "eval_threshold_mass_weight": 0.03,
    "eval_threshold_clust_weight": 0.06,
    "eval_threshold_min": 0.48,
    "eval_threshold_max": 0.58,
    "num_workers": 0,
    "lp_steps": 2,
    "lp_alpha": 0.102,
    "lp_beta": 0.011,
    "lp_min_anchor": 0.915,
    "lp_residual_scale": 0.0165,
    "lp_degree_bias": 0.065,
    "lp_coherence_bias": 0.03,
    "lp_clustering_bias": 0.05,
    "lp_graph_scale_bias": 1.0,
    "lp_node_attn_base": 0.10,
    "lp_node_attn_cluster_weight": 0.70,
    "lp_node_attn_conf_weight": 0.20,
    "lp_node_attn_degree_weight": 0.25,
    "lp_source_conf_center": 0.598,
    "lp_source_conf_sharpness": 10.0,
    "lp_recipient_conf_center": 0.538,
    "lp_recipient_conf_sharpness": 10.0,
    "lp_hard_source_min_conf": 0.60,
    "lp_hard_freeze_conf": 0.58,
    "lp_accept_sharpness": 14.0,
    "lp_accept_threshold": 0.020,
    "lp_accept_quality_weight": 0.75,
    "lp_accept_margin_weight": 0.20,
    "lp_accept_struct_weight": 0.05,
    "lp_accept_graph_clust_weight": 0.0,
    "lp_accept_graph_degree_weight": 0.0,
    "lp_accept_change_weight": 0.0,
    "lp_accept_conf_penalty_weight": 0.0,
    "lp_train_graph_clust_bias": 0.0,
    "contrastive_enabled": False,
    "contrastive_projection_dim": 64,
    "lp_mode": "dual_route",
    "lp_attn_blend": 0.5,
    "lp_auto_blend_base": 0.5,
    "lp_auto_blend_scale": 1.0,
    "lp_sparse_scale": 1.25,
    "lp_sparse_power": 1.50,
    "lp_sparse_topk": 0,
    "lp_dual_route_graph_clust_damp": 0.0,
    "lp_output_blend_enabled": False,
    "lp_output_blend_base": 0.0,
    "lp_output_blend_conf_weight": 0.65,
    "lp_output_blend_degree_weight": 0.20,
    "lp_output_blend_clust_weight": 0.10,
    "carry_max_epochs": 120,
    "carry_patience": 4,
    "carry_lr_decay": 0.5,
    "use_log_degree": True,
    "use_clustering_coefficient": True,
    "use_pagerank": False,
    "use_avg_neighbor_degree": False,
    "use_bridge_score": True,
}

PREPARED = {}


def compute_log_degree_feature(adj):
    deg = np.asarray(adj.sum(axis=1)).reshape(-1, 1).astype(np.float32)
    feat = np.log1p(deg)
    if feat.max() > 0:
        feat = feat / feat.max()
    return sp.csr_matrix(feat)


def compute_clustering_coefficient_feature(adj):
    adj_bin = adj.copy().astype(np.float32).tocsr()
    adj_bin.data[:] = 1.0
    adj_bin.setdiag(0)
    adj_bin.eliminate_zeros()
    triangles = adj_bin.multiply(adj_bin @ adj_bin).sum(axis=1).A1 / 2.0
    deg = np.asarray(adj_bin.sum(axis=1)).reshape(-1).astype(np.float32)
    feat = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 1
    feat[valid] = 2.0 * triangles[valid] / (deg[valid] * (deg[valid] - 1.0))
    return sp.csr_matrix(feat.reshape(-1, 1))


def compute_pagerank_feature(adj, damping=0.85, max_iter=50, tol=1e-6):
    adj_bin = adj.copy().astype(np.float32).tocsr()
    adj_bin.data[:] = 1.0
    adj_bin.setdiag(0)
    adj_bin.eliminate_zeros()
    n = adj_bin.shape[0]
    deg = np.asarray(adj_bin.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_deg = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 0
    inv_deg[valid] = 1.0 / deg[valid]
    pr = np.full(n, 1.0 / max(n, 1), dtype=np.float32)
    teleport = np.full(n, (1.0 - damping) / max(n, 1), dtype=np.float32)
    dangling = (~valid).astype(np.float32)
    for _ in range(max_iter):
        prev = pr.copy()
        walk = adj_bin.transpose().dot(prev * inv_deg)
        dangling_mass = prev[dangling > 0].sum() / max(n, 1)
        pr = teleport + damping * (walk + dangling_mass)
        if np.abs(pr - prev).sum() < tol:
            break
    if pr.max() > 0:
        pr = pr / pr.max()
    return sp.csr_matrix(pr.reshape(-1, 1))


def compute_avg_neighbor_degree_feature(adj):
    adj_bin = adj.copy().astype(np.float32).tocsr()
    adj_bin.data[:] = 1.0
    adj_bin.setdiag(0)
    adj_bin.eliminate_zeros()
    deg = np.asarray(adj_bin.sum(axis=1)).reshape(-1).astype(np.float32)
    neighbor_deg_sum = adj_bin.dot(deg.reshape(-1, 1)).reshape(-1)
    feat = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 0
    feat[valid] = neighbor_deg_sum[valid] / deg[valid]
    if feat.max() > 0:
        feat = feat / feat.max()
    return sp.csr_matrix(feat.reshape(-1, 1))


def compute_bridge_score_feature(adj):
    deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    clust = compute_clustering_coefficient_feature(adj).toarray().reshape(-1).astype(np.float32)
    feat = deg * (1.0 - clust)
    if feat.max() > 0:
        feat = feat / feat.max()
    return sp.csr_matrix(feat.reshape(-1, 1))


def build_input_features(adj, attr):
    attr_norm = normalize(attr)
    adj_norm = normalize(adj)
    struct_blocks = []
    if CONFIG["use_log_degree"]:
        struct_blocks.append(compute_log_degree_feature(adj))
    if CONFIG["use_clustering_coefficient"]:
        struct_blocks.append(compute_clustering_coefficient_feature(adj))
    if CONFIG["use_pagerank"]:
        struct_blocks.append(compute_pagerank_feature(adj))
    if CONFIG["use_avg_neighbor_degree"]:
        struct_blocks.append(compute_avg_neighbor_degree_feature(adj))
    if CONFIG["use_bridge_score"]:
        struct_blocks.append(compute_bridge_score_feature(adj))
    if len(struct_blocks) > 3:
        raise ValueError("At most 3 extra structural features are allowed beyond A and X.")
    x_input = sp.hstack([attr_norm, adj_norm] + struct_blocks).tocsr()
    struct_feat = np.hstack([x.toarray() for x in struct_blocks]).astype(np.float32)
    return x_input, struct_feat


def get_prepared_dataset(path):
    if path in PREPARED:
        return PREPARED[path]
    loaded = nocd.data.load_dataset(str(ROOT / path))
    adj, attr, labels = loaded["A"], loaded["X"], loaded["Z"]
    x_input, struct_feat = build_input_features(adj, attr)
    PREPARED[path] = (adj, x_input, struct_feat, labels)
    return PREPARED[path]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_prediction_mask(scores, struct_feat):
    if CONFIG["eval_threshold_mode"] == "fixed":
        return (scores > CONFIG["eval_threshold"]).detach().cpu().numpy()
    eps = 1e-8
    probs = scores / (scores.sum(dim=1, keepdim=True) + eps)
    if scores.size(1) > 1:
        max_entropy = float(np.log(scores.size(1)))
        entropy = -(probs * torch.log(probs + eps)).sum(dim=1, keepdim=True)
        confidence = 1.0 - entropy / max_entropy
    else:
        confidence = torch.ones((scores.size(0), 1), device=scores.device, dtype=scores.dtype)
    mass = torch.tanh(scores.sum(dim=1, keepdim=True) / scores.sum(dim=1, keepdim=True).mean().clamp_min(eps))
    clustering = struct_feat[:, 1:2]
    threshold = CONFIG["eval_threshold"]
    threshold = threshold + CONFIG["eval_threshold_conf_weight"] * (confidence - 0.5)
    threshold = threshold + CONFIG["eval_threshold_mass_weight"] * (mass - 0.5)
    threshold = threshold + CONFIG["eval_threshold_clust_weight"] * (clustering - 0.5)
    threshold = torch.clamp(threshold, min=CONFIG["eval_threshold_min"], max=CONFIG["eval_threshold_max"])
    return (scores > threshold).detach().cpu().numpy()


def evaluate_nmi(model, x_norm, adj_norm, struct_feat, labels):
    model.eval()
    with torch.no_grad():
        scores = F.relu(model(x_norm, adj_norm, struct_feat))
        pred = build_prediction_mask(scores, struct_feat)
        return nocd.metrics.overlapping_nmi(pred, labels)


def collect_scores(model, x_norm, adj_norm, struct_feat):
    model.eval()
    with torch.no_grad():
        return F.relu(model(x_norm, adj_norm, struct_feat))


def build_model(adj, x_input, struct_feat_np, labels, device):
    x_norm = nocd.utils.to_sparse_tensor(x_input, cuda=(device.type == "cuda")).to(device)
    struct_feat = torch.tensor(struct_feat_np, dtype=torch.float32, device=device)
    model = nocd.nn.GAT(
        x_norm.shape[1],
        [CONFIG["hidden_size"]],
        labels.shape[1],
        struct_dim=struct_feat.shape[1],
        batch_norm=CONFIG["batch_norm"],
        dropout=CONFIG["dropout"],
        heads=CONFIG["heads"],
        use_gate_repro_gat2=CONFIG["use_gate_repro_gat2"],
        use_hybrid_gat2=CONFIG["use_hybrid_gat2"],
        use_gate_repro_residual_gat2=CONFIG["use_gate_repro_residual_gat2"],
        init_beta_struct=CONFIG["init_beta_struct"],
        init_beta_feat=CONFIG["init_beta_feat"],
        init_beta_agree=CONFIG["init_beta_agree"],
        init_beta_edge=CONFIG["init_beta_edge"],
        init_beta_trust=CONFIG["init_beta_trust"],
        main_beta_agree_scale=CONFIG["main_beta_agree_scale"],
        main_beta_edge_scale=CONFIG["main_beta_edge_scale"],
        main_beta_trust_scale=CONFIG["main_beta_trust_scale"],
        init_beta_residual=CONFIG["init_beta_residual"],
        init_residual_bias=CONFIG["init_residual_bias"],
        init_beta_attn_residual=CONFIG["init_beta_attn_residual"],
        gate_activation=CONFIG["gate_activation"],
        enable_residual_gate=CONFIG["enable_residual_gate"],
        enable_propagation_gate=CONFIG["enable_propagation_gate"],
        prop_gate_strength=CONFIG["prop_gate_strength"],
        prop_gate_bias=CONFIG["prop_gate_bias"],
        prop_gate_struct_weight=CONFIG["prop_gate_struct_weight"],
        prop_gate_feat_weight=CONFIG["prop_gate_feat_weight"],
        prop_gate_agree_weight=CONFIG["prop_gate_agree_weight"],
        prop_gate_edge_weight=CONFIG["prop_gate_edge_weight"],
        prop_gate_trust_weight=CONFIG["prop_gate_trust_weight"],
        graph_clust_gate_scale=CONFIG["graph_clust_gate_scale"],
        prop_graph_clust_gate_scale=CONFIG["prop_graph_clust_gate_scale"],
        use_channel_mix=CONFIG["use_channel_mix"],
        lp_steps=CONFIG["lp_steps"],
        lp_alpha=CONFIG["lp_alpha"],
        lp_beta=CONFIG["lp_beta"],
        lp_min_anchor=CONFIG["lp_min_anchor"],
        lp_residual_scale=CONFIG["lp_residual_scale"],
        lp_degree_bias=CONFIG["lp_degree_bias"],
        lp_coherence_bias=CONFIG["lp_coherence_bias"],
        lp_clustering_bias=CONFIG["lp_clustering_bias"],
        lp_graph_scale_bias=CONFIG["lp_graph_scale_bias"],
        lp_node_attn_base=CONFIG["lp_node_attn_base"],
        lp_node_attn_cluster_weight=CONFIG["lp_node_attn_cluster_weight"],
        lp_node_attn_conf_weight=CONFIG["lp_node_attn_conf_weight"],
        lp_node_attn_degree_weight=CONFIG["lp_node_attn_degree_weight"],
        lp_source_conf_center=CONFIG["lp_source_conf_center"],
        lp_source_conf_sharpness=CONFIG["lp_source_conf_sharpness"],
        lp_recipient_conf_center=CONFIG["lp_recipient_conf_center"],
        lp_recipient_conf_sharpness=CONFIG["lp_recipient_conf_sharpness"],
        lp_hard_source_min_conf=CONFIG["lp_hard_source_min_conf"],
        lp_hard_freeze_conf=CONFIG["lp_hard_freeze_conf"],
        lp_accept_sharpness=CONFIG["lp_accept_sharpness"],
        lp_accept_threshold=CONFIG["lp_accept_threshold"],
        lp_accept_quality_weight=CONFIG["lp_accept_quality_weight"],
        lp_accept_margin_weight=CONFIG["lp_accept_margin_weight"],
        lp_accept_struct_weight=CONFIG["lp_accept_struct_weight"],
        lp_accept_graph_clust_weight=CONFIG["lp_accept_graph_clust_weight"],
        lp_accept_graph_degree_weight=CONFIG["lp_accept_graph_degree_weight"],
        lp_accept_change_weight=CONFIG["lp_accept_change_weight"],
        lp_accept_conf_penalty_weight=CONFIG["lp_accept_conf_penalty_weight"],
        lp_train_graph_clust_bias=CONFIG["lp_train_graph_clust_bias"],
        contrastive_enabled=CONFIG["contrastive_enabled"],
        contrastive_projection_dim=CONFIG["contrastive_projection_dim"],
        lp_mode=CONFIG["lp_mode"],
        lp_attn_blend=CONFIG["lp_attn_blend"],
        lp_auto_blend_base=CONFIG["lp_auto_blend_base"],
        lp_auto_blend_scale=CONFIG["lp_auto_blend_scale"],
        lp_sparse_scale=CONFIG["lp_sparse_scale"],
        lp_sparse_power=CONFIG["lp_sparse_power"],
        lp_sparse_topk=CONFIG["lp_sparse_topk"],
        lp_dual_route_graph_clust_damp=CONFIG["lp_dual_route_graph_clust_damp"],
        lp_output_blend_enabled=CONFIG["lp_output_blend_enabled"],
        lp_output_blend_base=CONFIG["lp_output_blend_base"],
        lp_output_blend_conf_weight=CONFIG["lp_output_blend_conf_weight"],
        lp_output_blend_degree_weight=CONFIG["lp_output_blend_degree_weight"],
        lp_output_blend_clust_weight=CONFIG["lp_output_blend_clust_weight"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    return x_norm, struct_feat, model, opt


def train_segment(adj, labels, x_norm, struct_feat, model, opt, segment_idx, device):
    run_max_epochs = CONFIG["max_epochs"] if segment_idx == 0 else CONFIG["carry_max_epochs"]
    run_patience = CONFIG["patience"] if segment_idx == 0 else CONFIG["carry_patience"]
    if segment_idx > 0:
        for group in opt.param_groups:
            group["lr"] = CONFIG["lr"] * (CONFIG["carry_lr_decay"] ** segment_idx)

    sampler = nocd.sampler.get_edge_sampler(
        adj,
        CONFIG["batch_size"],
        CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    adj_norm = model.normalize_adj(adj, cuda=(device.type == "cuda")).to(device)
    decoder = nocd.nn.BerpoDecoder(labels.shape[0], adj.nnz, balance_loss=CONFIG["balance_loss"])
    val_loss = np.inf
    early = nocd.train.NoImprovementStopping(lambda: val_loss, patience=run_patience)
    saver = nocd.train.ModelSaver(model)
    saver.save()
    start = time.time()

    for epoch, batch in enumerate(sampler):
        if epoch > run_max_epochs:
            break
        if epoch % 25 == 0:
            with torch.no_grad():
                model.eval()
                scores = F.relu(model(x_norm, adj_norm, struct_feat))
                val_loss = decoder.loss_full(scores, adj)
                early.next_step()
                if early.should_save():
                    saver.save()
                if early.should_stop():
                    break
        model.train()
        opt.zero_grad()
        scores = F.relu(model(x_norm, adj_norm, struct_feat))
        ones_idx, zeros_idx = batch
        ones_idx = ones_idx.to(device)
        zeros_idx = zeros_idx.to(device)
        loss = decoder.loss_batch(scores, ones_idx, zeros_idx)
        loss = loss + nocd.utils.l2_reg_loss(model, scale=CONFIG["weight_decay"])
        loss.backward()
        opt.step()

    saver.restore()
    nmi = evaluate_nmi(model, x_norm, adj_norm, struct_feat, labels)
    return nmi, time.time() - start


def run_dataset(dataset_name, device, runs, seed, output_path, dump_dir=None):
    path = DATASETS[dataset_name]
    set_seed(seed)
    adj, x_input, struct_feat_np, labels = get_prepared_dataset(path)
    x_norm, struct_feat, model, opt = build_model(adj, x_input, struct_feat_np, labels, device)
    values = []
    times = []

    for segment_idx in range(runs):
        nmi, elapsed = train_segment(adj, labels, x_norm, struct_feat, model, opt, segment_idx, device)
        values.append(nmi)
        times.append(elapsed)

    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    avg_time = statistics.mean(times)
    delta_paper = avg - PAPER[dataset_name]
    delta_target = avg - TARGET_TO_BEAT[dataset_name]
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(
            f"{dataset_name}\t{avg:.3f}\t{std:.3f}\t{avg_time:.2f}\t"
            f"{PAPER[dataset_name]:.3f}\t{delta_paper:+.3f}\t"
            f"{TARGET_TO_BEAT[dataset_name]:.3f}\t{delta_target:+.3f}\n"
        )
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        adj_norm = model.normalize_adj(adj, cuda=(device.type == "cuda")).to(device)
        scores = collect_scores(model, x_norm, adj_norm, struct_feat).detach().cpu().numpy()
        np.savez_compressed(
            dump_dir / f"{dataset_name}.npz",
            scores=scores,
            struct_feat=struct_feat.detach().cpu().numpy(),
            labels=labels.astype(np.int8),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run hybrid Input + Gate-GAT + LP carry experiment.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(Path("final_res") / "results" / "hybrid_gate_lp_seed0_20run.txt"))
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=sorted(DATASETS))
    parser.add_argument("--init-beta-struct", type=float, default=CONFIG["init_beta_struct"])
    parser.add_argument("--init-beta-feat", type=float, default=CONFIG["init_beta_feat"])
    parser.add_argument("--init-beta-agree", type=float, default=CONFIG["init_beta_agree"])
    parser.add_argument("--init-beta-edge", type=float, default=CONFIG["init_beta_edge"])
    parser.add_argument("--init-beta-trust", type=float, default=CONFIG["init_beta_trust"])
    parser.add_argument(
        "--main-beta-agree-scale",
        type=float,
        default=CONFIG["main_beta_agree_scale"],
    )
    parser.add_argument(
        "--main-beta-edge-scale",
        type=float,
        default=CONFIG["main_beta_edge_scale"],
    )
    parser.add_argument(
        "--main-beta-trust-scale",
        type=float,
        default=CONFIG["main_beta_trust_scale"],
    )
    parser.add_argument("--init-beta-residual", type=float, default=CONFIG["init_beta_residual"])
    parser.add_argument("--init-residual-bias", type=float, default=CONFIG["init_residual_bias"])
    parser.add_argument(
        "--init-beta-attn-residual",
        type=float,
        default=CONFIG["init_beta_attn_residual"],
    )
    parser.add_argument("--prop-gate-strength", type=float, default=CONFIG["prop_gate_strength"])
    parser.add_argument("--prop-gate-bias", type=float, default=CONFIG["prop_gate_bias"])
    parser.add_argument(
        "--prop-gate-struct-weight",
        type=float,
        default=CONFIG["prop_gate_struct_weight"],
    )
    parser.add_argument(
        "--prop-gate-feat-weight",
        type=float,
        default=CONFIG["prop_gate_feat_weight"],
    )
    parser.add_argument(
        "--prop-gate-agree-weight",
        type=float,
        default=CONFIG["prop_gate_agree_weight"],
    )
    parser.add_argument(
        "--prop-gate-edge-weight",
        type=float,
        default=CONFIG["prop_gate_edge_weight"],
    )
    parser.add_argument(
        "--prop-gate-trust-weight",
        type=float,
        default=CONFIG["prop_gate_trust_weight"],
    )
    parser.add_argument(
        "--graph-clust-gate-scale",
        type=float,
        default=CONFIG["graph_clust_gate_scale"],
    )
    parser.add_argument(
        "--prop-graph-clust-gate-scale",
        type=float,
        default=CONFIG["prop_graph_clust_gate_scale"],
    )
    parser.add_argument(
        "--gate-activation",
        choices=["identity", "tanh", "softplus"],
        default=CONFIG["gate_activation"],
    )
    parser.add_argument("--use-channel-mix", dest="use_channel_mix", action="store_true")
    parser.add_argument("--no-use-channel-mix", dest="use_channel_mix", action="store_false")
    parser.add_argument("--enable-residual-gate", dest="enable_residual_gate", action="store_true")
    parser.add_argument("--disable-residual-gate", dest="enable_residual_gate", action="store_false")
    parser.add_argument("--enable-propagation-gate", dest="enable_propagation_gate", action="store_true")
    parser.add_argument("--disable-propagation-gate", dest="enable_propagation_gate", action="store_false")
    parser.add_argument("--use-gate-repro-gat2", dest="use_gate_repro_gat2", action="store_true")
    parser.add_argument("--no-use-gate-repro-gat2", dest="use_gate_repro_gat2", action="store_false")
    parser.add_argument(
        "--use-gate-repro-residual-gat2",
        dest="use_gate_repro_residual_gat2",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-gate-repro-residual-gat2",
        dest="use_gate_repro_residual_gat2",
        action="store_false",
    )
    parser.add_argument("--eval-threshold", type=float, default=CONFIG["eval_threshold"])
    parser.add_argument(
        "--eval-threshold-mode",
        choices=["node_adaptive", "fixed"],
        default=CONFIG["eval_threshold_mode"],
    )
    parser.add_argument(
        "--eval-threshold-conf-weight",
        type=float,
        default=CONFIG["eval_threshold_conf_weight"],
    )
    parser.add_argument(
        "--eval-threshold-mass-weight",
        type=float,
        default=CONFIG["eval_threshold_mass_weight"],
    )
    parser.add_argument(
        "--eval-threshold-clust-weight",
        type=float,
        default=CONFIG["eval_threshold_clust_weight"],
    )
    parser.add_argument("--eval-threshold-min", type=float, default=CONFIG["eval_threshold_min"])
    parser.add_argument("--eval-threshold-max", type=float, default=CONFIG["eval_threshold_max"])
    parser.add_argument("--lp-alpha", type=float, default=CONFIG["lp_alpha"])
    parser.add_argument("--lp-steps", type=int, default=CONFIG["lp_steps"])
    parser.add_argument("--lp-beta", type=float, default=CONFIG["lp_beta"])
    parser.add_argument(
        "--lp-mode",
        choices=["adj", "attention", "sparse_attention", "blend", "auto_blend", "node_mix", "dual_route"],
        default=CONFIG["lp_mode"],
    )
    parser.add_argument("--lp-attn-blend", type=float, default=CONFIG["lp_attn_blend"])
    parser.add_argument("--lp-auto-blend-base", type=float, default=CONFIG["lp_auto_blend_base"])
    parser.add_argument("--lp-auto-blend-scale", type=float, default=CONFIG["lp_auto_blend_scale"])
    parser.add_argument("--lp-sparse-scale", type=float, default=CONFIG["lp_sparse_scale"])
    parser.add_argument("--lp-sparse-power", type=float, default=CONFIG["lp_sparse_power"])
    parser.add_argument("--lp-sparse-topk", type=int, default=CONFIG["lp_sparse_topk"])
    parser.add_argument(
        "--lp-dual-route-graph-clust-damp",
        type=float,
        default=CONFIG["lp_dual_route_graph_clust_damp"],
    )
    parser.add_argument("--enable-lp-output-blend", dest="lp_output_blend_enabled", action="store_true")
    parser.add_argument("--disable-lp-output-blend", dest="lp_output_blend_enabled", action="store_false")
    parser.add_argument("--lp-output-blend-base", type=float, default=CONFIG["lp_output_blend_base"])
    parser.add_argument(
        "--lp-output-blend-conf-weight",
        type=float,
        default=CONFIG["lp_output_blend_conf_weight"],
    )
    parser.add_argument(
        "--lp-output-blend-degree-weight",
        type=float,
        default=CONFIG["lp_output_blend_degree_weight"],
    )
    parser.add_argument(
        "--lp-output-blend-clust-weight",
        type=float,
        default=CONFIG["lp_output_blend_clust_weight"],
    )
    parser.add_argument("--lp-min-anchor", type=float, default=CONFIG["lp_min_anchor"])
    parser.add_argument("--lp-residual-scale", type=float, default=CONFIG["lp_residual_scale"])
    parser.add_argument("--lp-degree-bias", type=float, default=CONFIG["lp_degree_bias"])
    parser.add_argument("--lp-coherence-bias", type=float, default=CONFIG["lp_coherence_bias"])
    parser.add_argument("--lp-clustering-bias", type=float, default=CONFIG["lp_clustering_bias"])
    parser.add_argument("--lp-source-conf-center", type=float, default=CONFIG["lp_source_conf_center"])
    parser.add_argument(
        "--lp-source-conf-sharpness",
        type=float,
        default=CONFIG["lp_source_conf_sharpness"],
    )
    parser.add_argument(
        "--lp-recipient-conf-center",
        type=float,
        default=CONFIG["lp_recipient_conf_center"],
    )
    parser.add_argument(
        "--lp-recipient-conf-sharpness",
        type=float,
        default=CONFIG["lp_recipient_conf_sharpness"],
    )
    parser.add_argument(
        "--lp-hard-source-min-conf",
        type=float,
        default=CONFIG["lp_hard_source_min_conf"],
    )
    parser.add_argument(
        "--lp-hard-freeze-conf",
        type=float,
        default=CONFIG["lp_hard_freeze_conf"],
    )
    parser.add_argument(
        "--lp-accept-sharpness",
        type=float,
        default=CONFIG["lp_accept_sharpness"],
    )
    parser.add_argument(
        "--lp-accept-threshold",
        type=float,
        default=CONFIG["lp_accept_threshold"],
    )
    parser.add_argument(
        "--lp-accept-quality-weight",
        type=float,
        default=CONFIG["lp_accept_quality_weight"],
    )
    parser.add_argument(
        "--lp-accept-margin-weight",
        type=float,
        default=CONFIG["lp_accept_margin_weight"],
    )
    parser.add_argument(
        "--lp-accept-struct-weight",
        type=float,
        default=CONFIG["lp_accept_struct_weight"],
    )
    parser.add_argument(
        "--lp-accept-graph-clust-weight",
        type=float,
        default=CONFIG["lp_accept_graph_clust_weight"],
    )
    parser.add_argument(
        "--lp-accept-graph-degree-weight",
        type=float,
        default=CONFIG["lp_accept_graph_degree_weight"],
    )
    parser.add_argument(
        "--lp-accept-change-weight",
        type=float,
        default=CONFIG["lp_accept_change_weight"],
    )
    parser.add_argument(
        "--lp-accept-conf-penalty-weight",
        type=float,
        default=CONFIG["lp_accept_conf_penalty_weight"],
    )
    parser.add_argument(
        "--lp-train-graph-clust-bias",
        type=float,
        default=CONFIG["lp_train_graph_clust_bias"],
    )
    parser.add_argument("--use-log-degree", dest="use_log_degree", action="store_true")
    parser.add_argument("--no-use-log-degree", dest="use_log_degree", action="store_false")
    parser.add_argument(
        "--use-clustering-coefficient",
        dest="use_clustering_coefficient",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-clustering-coefficient",
        dest="use_clustering_coefficient",
        action="store_false",
    )
    parser.add_argument("--use-pagerank", dest="use_pagerank", action="store_true")
    parser.add_argument("--no-use-pagerank", dest="use_pagerank", action="store_false")
    parser.add_argument(
        "--use-avg-neighbor-degree",
        dest="use_avg_neighbor_degree",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-avg-neighbor-degree",
        dest="use_avg_neighbor_degree",
        action="store_false",
    )
    parser.add_argument("--use-bridge-score", dest="use_bridge_score", action="store_true")
    parser.add_argument("--no-use-bridge-score", dest="use_bridge_score", action="store_false")
    parser.set_defaults(
        use_log_degree=None,
        use_clustering_coefficient=None,
        use_pagerank=None,
        use_avg_neighbor_degree=None,
        use_bridge_score=None,
        use_channel_mix=None,
        enable_residual_gate=None,
        enable_propagation_gate=None,
        use_gate_repro_gat2=None,
        use_gate_repro_residual_gat2=None,
        lp_output_blend_enabled=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["init_beta_struct"] = args.init_beta_struct
    CONFIG["init_beta_feat"] = args.init_beta_feat
    CONFIG["init_beta_agree"] = args.init_beta_agree
    CONFIG["init_beta_edge"] = args.init_beta_edge
    CONFIG["init_beta_trust"] = args.init_beta_trust
    CONFIG["main_beta_agree_scale"] = args.main_beta_agree_scale
    CONFIG["main_beta_edge_scale"] = args.main_beta_edge_scale
    CONFIG["main_beta_trust_scale"] = args.main_beta_trust_scale
    CONFIG["init_beta_residual"] = args.init_beta_residual
    CONFIG["init_residual_bias"] = args.init_residual_bias
    CONFIG["init_beta_attn_residual"] = args.init_beta_attn_residual
    CONFIG["prop_gate_strength"] = args.prop_gate_strength
    CONFIG["prop_gate_bias"] = args.prop_gate_bias
    CONFIG["prop_gate_struct_weight"] = args.prop_gate_struct_weight
    CONFIG["prop_gate_feat_weight"] = args.prop_gate_feat_weight
    CONFIG["prop_gate_agree_weight"] = args.prop_gate_agree_weight
    CONFIG["prop_gate_edge_weight"] = args.prop_gate_edge_weight
    CONFIG["prop_gate_trust_weight"] = args.prop_gate_trust_weight
    CONFIG["graph_clust_gate_scale"] = args.graph_clust_gate_scale
    CONFIG["prop_graph_clust_gate_scale"] = args.prop_graph_clust_gate_scale
    CONFIG["gate_activation"] = args.gate_activation
    CONFIG["eval_threshold"] = args.eval_threshold
    CONFIG["eval_threshold_mode"] = args.eval_threshold_mode
    CONFIG["eval_threshold_conf_weight"] = args.eval_threshold_conf_weight
    CONFIG["eval_threshold_mass_weight"] = args.eval_threshold_mass_weight
    CONFIG["eval_threshold_clust_weight"] = args.eval_threshold_clust_weight
    CONFIG["eval_threshold_min"] = args.eval_threshold_min
    CONFIG["eval_threshold_max"] = args.eval_threshold_max
    CONFIG["lp_alpha"] = args.lp_alpha
    CONFIG["lp_steps"] = args.lp_steps
    CONFIG["lp_beta"] = args.lp_beta
    CONFIG["lp_mode"] = args.lp_mode
    CONFIG["lp_attn_blend"] = args.lp_attn_blend
    CONFIG["lp_auto_blend_base"] = args.lp_auto_blend_base
    CONFIG["lp_auto_blend_scale"] = args.lp_auto_blend_scale
    CONFIG["lp_sparse_scale"] = args.lp_sparse_scale
    CONFIG["lp_sparse_power"] = args.lp_sparse_power
    CONFIG["lp_sparse_topk"] = args.lp_sparse_topk
    CONFIG["lp_dual_route_graph_clust_damp"] = args.lp_dual_route_graph_clust_damp
    CONFIG["lp_output_blend_base"] = args.lp_output_blend_base
    CONFIG["lp_output_blend_conf_weight"] = args.lp_output_blend_conf_weight
    CONFIG["lp_output_blend_degree_weight"] = args.lp_output_blend_degree_weight
    CONFIG["lp_output_blend_clust_weight"] = args.lp_output_blend_clust_weight
    CONFIG["lp_min_anchor"] = args.lp_min_anchor
    CONFIG["lp_residual_scale"] = args.lp_residual_scale
    CONFIG["lp_degree_bias"] = args.lp_degree_bias
    CONFIG["lp_coherence_bias"] = args.lp_coherence_bias
    CONFIG["lp_clustering_bias"] = args.lp_clustering_bias
    CONFIG["lp_source_conf_center"] = args.lp_source_conf_center
    CONFIG["lp_source_conf_sharpness"] = args.lp_source_conf_sharpness
    CONFIG["lp_recipient_conf_center"] = args.lp_recipient_conf_center
    CONFIG["lp_recipient_conf_sharpness"] = args.lp_recipient_conf_sharpness
    CONFIG["lp_hard_source_min_conf"] = args.lp_hard_source_min_conf
    CONFIG["lp_hard_freeze_conf"] = args.lp_hard_freeze_conf
    CONFIG["lp_accept_sharpness"] = args.lp_accept_sharpness
    CONFIG["lp_accept_threshold"] = args.lp_accept_threshold
    CONFIG["lp_accept_quality_weight"] = args.lp_accept_quality_weight
    CONFIG["lp_accept_margin_weight"] = args.lp_accept_margin_weight
    CONFIG["lp_accept_struct_weight"] = args.lp_accept_struct_weight
    CONFIG["lp_accept_graph_clust_weight"] = args.lp_accept_graph_clust_weight
    CONFIG["lp_accept_graph_degree_weight"] = args.lp_accept_graph_degree_weight
    CONFIG["lp_accept_change_weight"] = args.lp_accept_change_weight
    CONFIG["lp_accept_conf_penalty_weight"] = args.lp_accept_conf_penalty_weight
    CONFIG["lp_train_graph_clust_bias"] = args.lp_train_graph_clust_bias
    if args.use_channel_mix is not None:
        CONFIG["use_channel_mix"] = args.use_channel_mix
    if args.enable_residual_gate is not None:
        CONFIG["enable_residual_gate"] = args.enable_residual_gate
    if args.enable_propagation_gate is not None:
        CONFIG["enable_propagation_gate"] = args.enable_propagation_gate
    if args.lp_output_blend_enabled is not None:
        CONFIG["lp_output_blend_enabled"] = args.lp_output_blend_enabled
    if args.use_gate_repro_gat2 is not None:
        CONFIG["use_gate_repro_gat2"] = args.use_gate_repro_gat2
    if args.use_gate_repro_residual_gat2 is not None:
        CONFIG["use_gate_repro_residual_gat2"] = args.use_gate_repro_residual_gat2
    if CONFIG["use_gate_repro_residual_gat2"]:
        CONFIG["use_gate_repro_gat2"] = False
        CONFIG["use_hybrid_gat2"] = False
    if args.use_log_degree is not None:
        CONFIG["use_log_degree"] = args.use_log_degree
    if args.use_clustering_coefficient is not None:
        CONFIG["use_clustering_coefficient"] = args.use_clustering_coefficient
    if args.use_pagerank is not None:
        CONFIG["use_pagerank"] = args.use_pagerank
    if args.use_avg_neighbor_degree is not None:
        CONFIG["use_avg_neighbor_degree"] = args.use_avg_neighbor_degree
    if args.use_bridge_score is not None:
        CONFIG["use_bridge_score"] = args.use_bridge_score
    enabled_struct = [
        CONFIG["use_log_degree"],
        CONFIG["use_clustering_coefficient"],
        CONFIG["use_pagerank"],
        CONFIG["use_avg_neighbor_degree"],
        CONFIG["use_bridge_score"],
    ]
    if sum(bool(x) for x in enabled_struct) > 3:
        raise ValueError("Only up to 3 extra structural features are allowed beyond A and X.")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(torch.__file__)
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    # print(f"Hybrid target thresholds: {TARGET_TO_BEAT}")
    # print(
    #     "Enabled extra features:",
    #     {
    #         "log_degree": CONFIG["use_log_degree"],
    #         "clustering_coefficient": CONFIG["use_clustering_coefficient"],
    #         "pagerank": CONFIG["use_pagerank"],
    #         "avg_neighbor_degree": CONFIG["use_avg_neighbor_degree"],
    #         "bridge_score": CONFIG["use_bridge_score"],
    #     },
    # )
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    dump_dir = None
    if args.dump_dir is not None:
        dump_dir = ROOT / args.dump_dir if not Path(args.dump_dir).is_absolute() else Path(args.dump_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    for name in args.datasets:
        run_dataset(name, device, args.runs, args.seed, out, dump_dir=dump_dir)


if __name__ == "__main__":
    main()

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
}

PAPER = {
    "fb_348": 0.345,
    "fb_414": 0.536,
    "fb_686": 0.178,
    "fb_698": 0.485,
    "fb_1684": 0.407,
    "fb_1912":0.397,
    "mag_cs": 0.455,
    "mag_chem":0.432
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
    "stochastic_loss": True,
    "batch_size": 20000,
    "heads": 6,
    "init_beta_struct": 1.0,
    "init_beta_feat": 1.0,
    "gate_activation": "identity",
    "enable_residual_gate": True,
    "enable_propagation_gate": True,
    "prop_gate_strength": 2.05,
    "prop_gate_bias": 0.118,
    "eval_threshold": 0.502,
    "eval_threshold_mode": "node_adaptive",
    "eval_threshold_conf_weight": 0.05,
    "eval_threshold_mass_weight": 0.03,
    "eval_threshold_margin_weight": 0.0,
    "eval_threshold_degree_weight": 0.0,
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
    "carry_max_epochs": 120,
    "carry_patience": 4,
    "carry_lr_decay": 0.5,
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


def build_input_features(adj, attr):
    attr_norm = normalize(attr)
    adj_norm = normalize(adj)
    struct_blocks = [
        compute_log_degree_feature(adj),
        compute_clustering_coefficient_feature(adj),
        compute_pagerank_feature(adj),
        compute_avg_neighbor_degree_feature(adj),
    ]
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
    threshold = torch.clamp(
        threshold,
        min=CONFIG["eval_threshold_min"],
        max=CONFIG["eval_threshold_max"],
    )
    return (scores > threshold).detach().cpu().numpy()


def evaluate_nmi(model, x_norm, adj_norm, struct_feat, labels):
    model.eval()
    with torch.no_grad():
        scores = F.relu(model(x_norm, adj_norm, struct_feat))
        pred = build_prediction_mask(scores, struct_feat)
        return nocd.metrics.overlapping_nmi(pred, labels)


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
        init_beta_struct=CONFIG["init_beta_struct"],
        init_beta_feat=CONFIG["init_beta_feat"],
        gate_activation=CONFIG["gate_activation"],
        enable_residual_gate=CONFIG["enable_residual_gate"],
        enable_propagation_gate=CONFIG["enable_propagation_gate"],
        prop_gate_strength=CONFIG["prop_gate_strength"],
        prop_gate_bias=CONFIG["prop_gate_bias"],
        use_channel_mix=False,
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
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    return x_norm, struct_feat, model, opt


def train_segment(dataset_name, adj, labels, x_norm, struct_feat, model, opt, segment_idx, device):
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


def run_dataset(dataset_name, device, runs, seed, output_path):
    path = DATASETS[dataset_name]
    set_seed(seed)
    adj, x_input, struct_feat_np, labels = get_prepared_dataset(path)
    x_norm, struct_feat, model, opt = build_model(adj, x_input, struct_feat_np, labels, device)
    values = []
    times = []

    for segment_idx in range(runs):
        nmi, elapsed = train_segment(dataset_name, adj, labels, x_norm, struct_feat, model, opt, segment_idx, device)
        values.append(nmi)
        times.append(elapsed)
        if len(values) >= 3:
            print(f"[{dataset_name} carry] interim avg={statistics.mean(values):.4f} std={statistics.stdev(values):.4f}")

    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    avg_time = statistics.mean(times)
    delta = avg - PAPER[dataset_name]
    print(f"[{dataset_name}] paper={PAPER[dataset_name]:.4f} current={avg:.4f} delta={delta:+.4f}")
    print(f"[{dataset_name}] final avg={avg:.4f} std={std:.4f} time={avg_time:.2f}s")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"{dataset_name}\t{avg:.3f}\t{std:.3f}\t{avg_time:.2f}\t{PAPER[dataset_name]:.3f}\t{delta:+.3f}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the final LP carry experiment.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fb_348", "fb_414", "fb_686","fb_698", "fb_1684", "fb_1912", "mag_cs", "mag_chem"],
        choices=sorted(DATASETS),
    )
    parser.add_argument(
        "--output",
        default=str(Path("final_LP") / "reproduced_carry_seed0_20run.txt"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(torch.__file__)
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    for name in args.datasets:
        run_dataset(name, device, args.runs, args.seed, out)


if __name__ == "__main__":
    main()

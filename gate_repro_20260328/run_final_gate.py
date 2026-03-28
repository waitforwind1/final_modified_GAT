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
    'fb_348': ROOT / 'data/facebook_ego/fb_348.npz',
    'fb_414': ROOT / 'data/facebook_ego/fb_414.npz',
    'fb_686': ROOT / 'data/facebook_ego/fb_686.npz',
    'fb_698': ROOT / 'data/facebook_ego/fb_698.npz',
    'fb_1684': ROOT / 'data/facebook_ego/fb_1684.npz',
    'fb_1912': ROOT / 'data/facebook_ego/fb_1912.npz',
    'mag_chem': ROOT / 'data/mag_chem.npz',
    'mag_cs': ROOT / 'data/mag_cs.npz',
    'mag_eng': ROOT / 'data/mag_eng.npz',
}


PAPER_BASELINE_NMI = {
    'fb_348': 0.345,
    'fb_414': 0.536,
    'fb_686': 0.178,
    'fb_698': 0.485,
    'fb_1684': 0.407,
    'fb_1912': 0.397,
    'mag_chem': 0.432,
    'mag_cs': 0.455,
    'mag_eng': 0.385,
}


DEFAULTS = {
    'hidden_size': 128,
    'weight_decay': 1e-2,
    'dropout': 0.0,
    'batch_norm': True,
    'lr': 1e-3,
    'max_epochs': 500,
    'patience': 10,
    'batch_size': 20000,
    'heads': 6,
    'init_beta_struct': 1.15,
    'init_beta_feat': 0.85,
    'init_beta_agree': 0.10,
    'init_beta_edge': 0.08,
    'init_beta_trust': 0.12,
    'eval_threshold': 0.54,
    'carry_max_epochs': 120,
    'carry_patience': 4,
    'carry_lr_decay': 0.5,
    'use_log_degree': True,
    'use_clustering_coefficient': True,
    'use_pagerank': True,
    'use_avg_neighbor_degree': True,
    'use_bridge_score': True,
}


PREPARED_DATA_CACHE = {}


def compute_log_degree_feature(A):
    deg = np.asarray(A.sum(axis=1)).reshape(-1, 1).astype(np.float32)
    log_deg = np.log1p(deg)
    max_val = log_deg.max()
    if max_val > 0:
        log_deg = log_deg / max_val
    return sp.csr_matrix(log_deg)


def compute_clustering_coefficient_feature(A):
    A_bin = A.copy().astype(np.float32).tocsr()
    A_bin.data[:] = 1.0
    A_bin.setdiag(0)
    A_bin.eliminate_zeros()

    triangles = A_bin.multiply(A_bin @ A_bin).sum(axis=1).A1 / 2.0
    deg = np.asarray(A_bin.sum(axis=1)).reshape(-1).astype(np.float32)

    clust = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 1
    clust[valid] = 2.0 * triangles[valid] / (deg[valid] * (deg[valid] - 1.0))
    return sp.csr_matrix(clust.reshape(-1, 1))


def compute_pagerank_feature(A, damping=0.85, max_iter=50, tol=1e-6):
    A_bin = A.copy().astype(np.float32).tocsr()
    A_bin.data[:] = 1.0
    A_bin.setdiag(0)
    A_bin.eliminate_zeros()

    n = A_bin.shape[0]
    deg = np.asarray(A_bin.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_deg = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 0
    inv_deg[valid] = 1.0 / deg[valid]

    pr = np.full(n, 1.0 / max(n, 1), dtype=np.float32)
    teleport = np.full(n, (1.0 - damping) / max(n, 1), dtype=np.float32)
    dangling = (~valid).astype(np.float32)

    for _ in range(max_iter):
        prev = pr.copy()
        walk = A_bin.transpose().dot(prev * inv_deg)
        dangling_mass = prev[dangling > 0].sum() / max(n, 1)
        pr = teleport + damping * (walk + dangling_mass)
        if np.abs(pr - prev).sum() < tol:
            break

    max_val = pr.max()
    if max_val > 0:
        pr = pr / max_val
    return sp.csr_matrix(pr.reshape(-1, 1))


def compute_avg_neighbor_degree_feature(A):
    A_bin = A.copy().astype(np.float32).tocsr()
    A_bin.data[:] = 1.0
    A_bin.setdiag(0)
    A_bin.eliminate_zeros()

    deg = np.asarray(A_bin.sum(axis=1)).reshape(-1).astype(np.float32)
    neighbor_deg_sum = A_bin.dot(deg.reshape(-1, 1)).reshape(-1)

    avg_neighbor_deg = np.zeros_like(deg, dtype=np.float32)
    valid = deg > 0
    avg_neighbor_deg[valid] = neighbor_deg_sum[valid] / deg[valid]

    max_val = avg_neighbor_deg.max()
    if max_val > 0:
        avg_neighbor_deg = avg_neighbor_deg / max_val
    return sp.csr_matrix(avg_neighbor_deg.reshape(-1, 1))


def compute_bridge_score_feature(A):
    A_bin = A.copy().astype(np.float32).tocsr()
    A_bin.data[:] = 1.0
    A_bin.setdiag(0)
    A_bin.eliminate_zeros()

    deg = np.asarray(A_bin.sum(axis=1)).reshape(-1).astype(np.float32)
    clust = compute_clustering_coefficient_feature(A).toarray().reshape(-1).astype(np.float32)

    bridge_score = deg * (1.0 - clust)
    max_val = bridge_score.max()
    if max_val > 0:
        bridge_score = bridge_score / max_val
    return sp.csr_matrix(bridge_score.reshape(-1, 1))


def build_input_features(A, X, feature_config):
    X_norm = normalize(X)
    A_norm = normalize(A)
    struct_blocks = []
    if feature_config['use_log_degree']:
        struct_blocks.append(compute_log_degree_feature(A))
    if feature_config['use_clustering_coefficient']:
        struct_blocks.append(compute_clustering_coefficient_feature(A))
    if feature_config['use_pagerank']:
        struct_blocks.append(compute_pagerank_feature(A))
    if feature_config['use_avg_neighbor_degree']:
        struct_blocks.append(compute_avg_neighbor_degree_feature(A))
    if feature_config['use_bridge_score']:
        struct_blocks.append(compute_bridge_score_feature(A))

    if not struct_blocks:
        raise ValueError('At least one structural feature must be enabled.')

    x_input = sp.hstack([X_norm, A_norm] + struct_blocks).tocsr()
    struct_feat = np.hstack([feat.toarray() for feat in struct_blocks]).astype(np.float32)
    return x_input, struct_feat


def get_feature_key(feature_config):
    return (
        feature_config['use_log_degree'],
        feature_config['use_clustering_coefficient'],
        feature_config['use_pagerank'],
        feature_config['use_avg_neighbor_degree'],
        feature_config['use_bridge_score'],
    )


def get_prepared_dataset(path, feature_config):
    path = str(path)
    cache_key = (path, get_feature_key(feature_config))
    cached = PREPARED_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loader = nocd.data.load_dataset(path)
    A, X, Z_gt = loader['A'], loader['X'], loader['Z']
    x_input, struct_feat_np = build_input_features(A, X, feature_config)
    prepared = (A, x_input, struct_feat_np, Z_gt)
    PREPARED_DATA_CACHE[cache_key] = prepared
    return prepared


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_nmi(gnn, x_norm, adj_norm, struct_feat, z_gt, threshold):
    gnn.eval()
    with torch.no_grad():
        z = F.relu(gnn(x_norm, adj_norm, struct_feat))
        z_pred = z.detach().cpu().numpy() > threshold
        return nocd.metrics.overlapping_nmi(z_pred, z_gt)


def build_model(A, x_input, struct_feat_np, z_gt, device):
    x_norm = nocd.utils.to_sparse_tensor(x_input, cuda=(device.type == 'cuda')).to(device)
    struct_feat = torch.tensor(struct_feat_np, dtype=torch.float32, device=device)
    _, K = z_gt.shape
    gnn = nocd.nn.GAT(
        x_norm.shape[1],
        [DEFAULTS['hidden_size']],
        K,
        struct_dim=struct_feat.shape[1],
        batch_norm=DEFAULTS['batch_norm'],
        dropout=DEFAULTS['dropout'],
        heads=DEFAULTS['heads'],
        init_beta_struct=DEFAULTS['init_beta_struct'],
        init_beta_feat=DEFAULTS['init_beta_feat'],
        init_beta_agree=DEFAULTS['init_beta_agree'],
        init_beta_edge=DEFAULTS['init_beta_edge'],
        init_beta_trust=DEFAULTS['init_beta_trust'],
    ).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=DEFAULTS['lr'])
    return x_norm, struct_feat, gnn, opt


def train_segment(dataset_name, segment_idx, device, A, z_gt, x_norm, struct_feat, gnn, opt):
    N = z_gt.shape[0]
    sampler = nocd.sampler.get_edge_sampler(
        A,
        DEFAULTS['batch_size'],
        DEFAULTS['batch_size'],
        num_workers=0,
    )
    adj_norm = gnn.normalize_adj(A, cuda=(device.type == 'cuda')).to(device)
    decoder = nocd.nn.BerpoDecoder(N, A.nnz, balance_loss=True)

    max_epochs = DEFAULTS['max_epochs'] if segment_idx == 0 else DEFAULTS['carry_max_epochs']
    patience = DEFAULTS['patience'] if segment_idx == 0 else DEFAULTS['carry_patience']
    if segment_idx > 0:
        for group in opt.param_groups:
            group['lr'] = DEFAULTS['lr'] * (DEFAULTS['carry_lr_decay'] ** segment_idx)

    val_loss = np.inf
    early_stopping = nocd.train.NoImprovementStopping(lambda: val_loss, patience=patience)
    model_saver = nocd.train.ModelSaver(gnn)
    start_time = time.time()

    for epoch, batch in enumerate(sampler):
        if epoch > max_epochs:
            break

        if epoch % 25 == 0:
            with torch.no_grad():
                gnn.eval()
                z_eval = F.relu(gnn(x_norm, adj_norm, struct_feat))
                val_loss = decoder.loss_full(z_eval, A)
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    break

        gnn.train()
        opt.zero_grad()
        z = F.relu(gnn(x_norm, adj_norm, struct_feat))
        ones_idx, zeros_idx = batch
        ones_idx = ones_idx.to(device)
        zeros_idx = zeros_idx.to(device)
        loss = decoder.loss_batch(z, ones_idx, zeros_idx)
        loss = loss + nocd.utils.l2_reg_loss(gnn, scale=DEFAULTS['weight_decay'])
        loss.backward()
        opt.step()

    model_saver.restore()
    final_nmi = evaluate_nmi(gnn, x_norm, adj_norm, struct_feat, z_gt, DEFAULTS['eval_threshold'])
    return final_nmi, time.time() - start_time


def run_dataset(dataset_name, device, runs, seed, output, feature_config):
    set_random_seed(seed)
    A, x_input, struct_feat_np, z_gt = get_prepared_dataset(DATASETS[dataset_name], feature_config)
    x_norm, struct_feat, gnn, opt = build_model(A, x_input, struct_feat_np, z_gt, device)

    results = []
    times = []
    for segment_idx in range(runs):
        nmi, elapsed = train_segment(
            dataset_name,
            segment_idx,
            device,
            A,
            z_gt,
            x_norm,
            struct_feat,
            gnn,
            opt,
        )
        results.append(nmi)
        times.append(elapsed)
        if len(results) >= 3:
            avg = statistics.mean(results)
            sd = statistics.stdev(results)
            print(f'[{dataset_name} carry] interim avg={avg:.4f} std={sd:.4f}')

    avg = statistics.mean(results)
    sd = statistics.stdev(results) if len(results) > 1 else 0.0
    avg_time = statistics.mean(times)
    baseline = PAPER_BASELINE_NMI[dataset_name]
    print(f'[{dataset_name}] paper={baseline:.4f} current={avg:.4f} delta={avg - baseline:+.4f}')
    print(f'[{dataset_name}] final avg={avg:.4f} std={sd:.4f} time={avg_time:.2f}s')

    with output.open('a', encoding='utf-8') as resultfile:
        resultfile.write(
            f'{dataset_name}\t{avg:.3f}\t{sd:.3f}\t{avg_time:.2f}\t{baseline:.3f}\t{(avg - baseline):+.3f}\n'
        )


def main():
    parser = argparse.ArgumentParser(description='Reproduce the final Gate carry-training experiment.')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['fb_348', 'fb_414', 'fb_686', 'fb_698', 'fb_1684', 'fb_1912', 'mag_cs', 'mag_chem'],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument(
        '--output',
        default=str(Path(__file__).resolve().parent / 'final_gate_results.txt'),
    )
    parser.add_argument('--heads', type=int)
    parser.add_argument('--init-beta-struct', type=float)
    parser.add_argument('--init-beta-feat', type=float)
    parser.add_argument('--init-beta-agree', type=float)
    parser.add_argument('--init-beta-edge', type=float)
    parser.add_argument('--init-beta-trust', type=float)
    parser.add_argument('--eval-threshold', type=float)
    parser.add_argument('--use-log-degree', dest='use_log_degree', action='store_true')
    parser.add_argument('--no-use-log-degree', dest='use_log_degree', action='store_false')
    parser.add_argument(
        '--use-clustering-coefficient',
        dest='use_clustering_coefficient',
        action='store_true',
    )
    parser.add_argument(
        '--no-use-clustering-coefficient',
        dest='use_clustering_coefficient',
        action='store_false',
    )
    parser.add_argument('--use-pagerank', dest='use_pagerank', action='store_true')
    parser.add_argument('--no-use-pagerank', dest='use_pagerank', action='store_false')
    parser.add_argument(
        '--use-avg-neighbor-degree',
        dest='use_avg_neighbor_degree',
        action='store_true',
    )
    parser.add_argument(
        '--no-use-avg-neighbor-degree',
        dest='use_avg_neighbor_degree',
        action='store_false',
    )
    parser.add_argument('--use-bridge-score', dest='use_bridge_score', action='store_true')
    parser.add_argument('--no-use-bridge-score', dest='use_bridge_score', action='store_false')
    parser.set_defaults(
        use_log_degree=None,
        use_clustering_coefficient=None,
        use_pagerank=None,
        use_avg_neighbor_degree=None,
        use_bridge_score=None,
    )
    args = parser.parse_args()

    if args.heads is not None:
        DEFAULTS['heads'] = args.heads
    if args.init_beta_struct is not None:
        DEFAULTS['init_beta_struct'] = args.init_beta_struct
    if args.init_beta_feat is not None:
        DEFAULTS['init_beta_feat'] = args.init_beta_feat
    if args.init_beta_agree is not None:
        DEFAULTS['init_beta_agree'] = args.init_beta_agree
    if args.init_beta_edge is not None:
        DEFAULTS['init_beta_edge'] = args.init_beta_edge
    if args.init_beta_trust is not None:
        DEFAULTS['init_beta_trust'] = args.init_beta_trust
    if args.eval_threshold is not None:
        DEFAULTS['eval_threshold'] = args.eval_threshold

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device)
    output = Path(args.output)
    if output.exists():
        output.unlink()

    feature_config = {
        'use_log_degree': DEFAULTS['use_log_degree'] if args.use_log_degree is None else args.use_log_degree,
        'use_clustering_coefficient': (
            DEFAULTS['use_clustering_coefficient']
            if args.use_clustering_coefficient is None
            else args.use_clustering_coefficient
        ),
        'use_pagerank': DEFAULTS['use_pagerank'] if args.use_pagerank is None else args.use_pagerank,
        'use_avg_neighbor_degree': (
            DEFAULTS['use_avg_neighbor_degree']
            if args.use_avg_neighbor_degree is None
            else args.use_avg_neighbor_degree
        ),
        'use_bridge_score': (
            DEFAULTS['use_bridge_score'] if args.use_bridge_score is None else args.use_bridge_score
        ),
    }

    print(torch.__file__)
    print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')
    print(f'Running final Gate carry experiment with seed={args.seed}, runs={args.runs}')
    print(f'Feature config: {feature_config}')

    for dataset_name in args.datasets:
        print(f'\nDataset: {dataset_name}')
        print(f'Path: {DATASETS[dataset_name]}')
        print(f'Device: {device}')
        print(f'Config: {DEFAULTS}')
        run_dataset(dataset_name, device, args.runs, args.seed, output, feature_config)


if __name__ == '__main__':
    main()

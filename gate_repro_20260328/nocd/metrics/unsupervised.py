import numpy as np


__all__ = [
    'evaluate_unsupervised',
    'clustering_coef',
    'coverage',
    'density',
    'conductance',
]


def evaluate_unsupervised(Z_pred, adj):
    return {'coverage': coverage(Z_pred, adj), 'density': density(Z_pred, adj), 'conductance': conductance(Z_pred, adj), 'clustering_coef': clustering_coef(Z_pred, adj)}


def clustering_coef(Z_pred, adj):
    def clustering_coef_community(ind, adj):
        adj_com = adj[ind][:, ind]
        n = ind.sum()
        if n < 3:
            return 0
        possible = (n - 2) * (n - 1) * n / 6
        existing = (adj_com @ adj_com @ adj_com).diagonal().sum() / 6
        return existing / possible

    Z_pred = Z_pred.astype(bool)
    com_sizes = Z_pred.sum(0)
    clust_coefs = np.array([clustering_coef_community(Z_pred[:, c], adj) for c in range(Z_pred.shape[1])])
    return clust_coefs @ com_sizes / com_sizes.sum()


def coverage(Z_pred, adj):
    u, v = adj.nonzero()
    return ((Z_pred[u] * Z_pred[v]).sum(1) > 0).sum() / adj.nnz


def density(Z_pred, adj):
    def density_community(ind, adj):
        ind = ind.astype(bool)
        n = ind.sum()
        if n < 2:
            return 0.0
        else:
            return adj[ind][:, ind].nnz / (n**2 - n)
    Z_pred = Z_pred.astype(bool)
    com_sizes = Z_pred.sum(0) / Z_pred.sum()
    densities = np.array([density_community(Z_pred[:, c], adj) for c in range(Z_pred.shape[1])])
    return densities @ com_sizes


def conductance(Z_pred, adj):
    def conductance_community(ind, adj):
        ind = ind.astype(bool)
        inside = adj[ind, :][:, ind].nnz
        outside = adj[~ind, :][:, ind].nnz
        if inside + outside == 0:
            return 1
        return outside / (inside + outside)

    Z_pred = Z_pred.astype(bool)
    com_sizes = Z_pred.sum(0)
    conductances = np.array([conductance_community(Z_pred[:, c], adj) for c in range(Z_pred.shape[1])])
    return conductances @ com_sizes / com_sizes.sum()

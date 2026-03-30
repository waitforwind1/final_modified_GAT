import numpy as np
import scipy.sparse as sp


def _get_matrix(loader, prefix):
    if f"{prefix}.data" in loader:
        return sp.csr_matrix(
            (loader[f"{prefix}.data"], loader[f"{prefix}.indices"], loader[f"{prefix}.indptr"]),
            shape=loader[f"{prefix}.shape"],
        )
    if f"{prefix}_data" in loader:
        return sp.csr_matrix(
            (loader[f"{prefix}_data"], loader[f"{prefix}_indices"], loader[f"{prefix}_indptr"]),
            shape=loader[f"{prefix}_shape"],
        )
    if prefix in loader:
        obj = loader[prefix]
        if obj.dtype == "O" and len(obj.shape) == 0:
            return obj.item()
        return obj
    return None


def load_dataset(file_name):
    if not file_name.endswith(".npz"):
        file_name += ".npz"

    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)

        adj = _get_matrix(loader, "adj_matrix")
        if adj is None:
            adj = _get_matrix(loader, "adj")
        if adj is None:
            raise ValueError(f"Missing adjacency matrix in {file_name}.")

        attr = _get_matrix(loader, "attr_matrix")
        if attr is None:
            attr = _get_matrix(loader, "attr")
        if attr is not None and not sp.issparse(attr):
            attr = sp.csr_matrix(attr)

        labels = _get_matrix(loader, "labels")
        if labels is None:
            raise ValueError(f"Missing labels in {file_name}.")

        if isinstance(labels, np.ndarray) and labels.ndim == 1:
            labels = sp.csr_matrix(np.eye(np.max(labels) + 1)[labels])

        if sp.issparse(adj):
            adj = adj.tolil()
            adj.setdiag(0)
            adj = adj.tocsr()
            adj.eliminate_zeros()

        if sp.issparse(labels):
            labels = labels.toarray().astype(np.float32)
        else:
            labels = labels.astype(np.float32)

        return {"A": adj, "X": attr, "Z": labels}

import numpy as np
import scipy.sparse as sp


def get_matrix(loader, prefix):
    if f'{prefix}.data' in loader:
        return sp.csr_matrix((loader[f'{prefix}.data'], loader[f'{prefix}.indices'], loader[f'{prefix}.indptr']), shape=loader[f'{prefix}.shape'])
    elif f'{prefix}_data' in loader:
        return sp.csr_matrix((loader[f'{prefix}_data'], loader[f'{prefix}_indices'], loader[f'{prefix}_indptr']), shape=loader[f'{prefix}_shape'])
    elif prefix in loader:
        obj = loader[prefix]
        if obj.dtype == 'O' and len(obj.shape) == 0:
            return obj.item()
        return obj
    return None


def load_dataset(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)

        A = get_matrix(loader, 'adj_matrix')
        if A is None:
            A = get_matrix(loader, 'adj')
        if A is None:
            raise ValueError(f'无法在 {file_name} 中找到邻接矩阵 (Adjacency Matrix)。')

        X = get_matrix(loader, 'attr_matrix')
        if X is None:
            X = get_matrix(loader, 'attr')
        if X is not None and not sp.issparse(X):
            X = sp.csr_matrix(X)

        Z = get_matrix(loader, 'labels')
        if Z is None:
            raise ValueError(f'无法在 {file_name} 中找到标签 (Labels)。')

        if isinstance(Z, np.ndarray) and Z.ndim == 1:
            num_classes = np.max(Z) + 1
            Z = np.eye(num_classes)[Z]
            Z = sp.csr_matrix(Z)

        if sp.issparse(A):
            A = A.tolil()
            A.setdiag(0)
            A = A.tocsr()
            A.eliminate_zeros()

        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)
        else:
            Z = Z.astype(np.float32)

        graph = {
            'A': A,
            'X': X,
            'Z': Z,
        }

        if 'node_names' in loader:
            graph['node_names'] = loader['node_names'].tolist()
        if 'attr_names' in loader:
            graph['attr_names'] = loader['attr_names'].tolist()
        if 'class_names' in loader:
            graph['class_names'] = loader['class_names'].tolist()

        return graph

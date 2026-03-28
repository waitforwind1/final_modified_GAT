import numpy as np
import scipy.sparse as sp
from nocd.utils import to_sparse_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import softmax


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.sparse_coo_tensor(x.indices(), new_values, x.size(), device=x.device)
    return F.dropout(x, p=p, training=training)


class HybridGateGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        struct_dim,
        heads=6,
        dropout=0.0,
        init_beta_struct=1.15,
        init_beta_feat=0.85,
        init_beta_agree=0.10,
        init_beta_edge=0.08,
        init_beta_trust=0.12,
    ):
        super().__init__(node_dim=0, aggr='add')
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.node_trust_lin = nn.Linear(struct_dim, 1, bias=True)
        self.edge_mlp_lin1 = nn.Linear(4, 8, bias=True)
        self.edge_mlp_lin2 = nn.Linear(8, 1, bias=True)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.beta_struct = nn.Parameter(torch.tensor(init_beta_struct))
        self.beta_feat = nn.Parameter(torch.tensor(init_beta_feat))
        self.beta_agree = nn.Parameter(torch.tensor(init_beta_agree))
        self.beta_edge = nn.Parameter(torch.tensor(init_beta_edge))
        self.beta_trust = nn.Parameter(torch.tensor(init_beta_trust))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.node_trust_lin.weight)
        nn.init.zeros_(self.node_trust_lin.bias)
        nn.init.xavier_uniform_(self.edge_mlp_lin1.weight)
        nn.init.zeros_(self.edge_mlp_lin1.bias)
        nn.init.zeros_(self.edge_mlp_lin2.weight)
        nn.init.zeros_(self.edge_mlp_lin2.bias)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, struct_feat):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        node_trust = torch.sigmoid(self.node_trust_lin(struct_feat)).squeeze(-1)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)

        alpha = self.edge_updater(
            edge_index,
            alpha_src=alpha_src,
            alpha_dst=alpha_dst,
            x=x,
            struct_feat=struct_feat,
            node_trust=node_trust,
        )

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.mean(dim=1)
        out = out + self.bias
        return out

    def edge_update(
        self,
        alpha_src_j,
        alpha_dst_i,
        x_j,
        x_i,
        struct_feat_j,
        struct_feat_i,
        node_trust_j,
        node_trust_i,
        index,
        ptr,
        size_i,
    ):
        alpha_raw = F.leaky_relu(alpha_src_j + alpha_dst_i, negative_slope=0.2)

        struct_sim = F.cosine_similarity(struct_feat_j, struct_feat_i, dim=-1, eps=1e-8)
        feat_sim = F.cosine_similarity(
            x_j.reshape(x_j.size(0), -1),
            x_i.reshape(x_i.size(0), -1),
            dim=-1,
            eps=1e-8,
        )
        agreement = struct_sim * feat_sim
        conflict = torch.abs(struct_sim - feat_sim)

        edge_cues = torch.stack([struct_sim, feat_sim, agreement, conflict], dim=-1)
        edge_gate = self.edge_mlp_lin2(F.relu(self.edge_mlp_lin1(edge_cues))).squeeze(-1)
        edge_bonus = torch.sigmoid(edge_gate) * F.relu(agreement)

        edge_trust = torch.sqrt(torch.clamp(node_trust_j * node_trust_i, min=0.0))
        trust_bonus = edge_trust * F.relu(0.5 * (struct_sim + feat_sim))

        gate_logit = (
            self.beta_struct * struct_sim
            + self.beta_feat * feat_sim
            + self.beta_agree * agreement
            + self.beta_edge * edge_bonus
            + self.beta_trust * trust_bonus
        )

        alpha = softmax(alpha_raw + gate_logit.unsqueeze(-1), index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)


class GAT(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        struct_dim,
        dropout=0.0,
        batch_norm=True,
        heads=6,
        init_beta_struct=1.15,
        init_beta_feat=0.85,
        init_beta_agree=0.10,
        init_beta_edge=0.08,
        init_beta_trust=0.12,
    ):
        super().__init__()
        self.dropout = dropout

        hidden_dim = hidden_dims[0]
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, add_self_loops=True, concat=False)
        self.gat2 = HybridGateGATLayer(
            hidden_dim,
            output_dim,
            struct_dim=struct_dim,
            heads=heads,
            dropout=dropout,
            init_beta_struct=init_beta_struct,
            init_beta_feat=init_beta_feat,
            init_beta_agree=init_beta_agree,
            init_beta_edge=init_beta_edge,
            init_beta_trust=init_beta_trust,
        )

        if batch_norm:
            self.batch_norm = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False)
            ])
        else:
            self.batch_norm = None

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        if self.batch_norm is not None:
            for bn in self.batch_norm:
                bn.reset_parameters()

    def _extract_edge_index(self, adj):
        if hasattr(adj, 'indices'):
            return adj.indices()
        if hasattr(adj, '_indices'):
            return adj._indices()
        if isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.size(0) == 2:
            return adj
        return adj.nonzero().t()

    def forward(self, x, adj, struct_feat):
        edge_index = self._extract_edge_index(adj)
        if self.dropout != 0:
            x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)

        h = self.gat1(x, edge_index)
        h = F.relu(h)

        if self.batch_norm is not None:
            h = self.batch_norm[0](h)

        if self.dropout != 0:
            h = sparse_or_dense_dropout(h, p=self.dropout, training=self.training)

        return self.gat2(h, edge_index, struct_feat)

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        return [w for n, w in self.named_parameters() if 'bias' in n]

    @staticmethod
    def normalize_adj(adj: sp.csr_matrix, cuda: bool = True):
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm, cuda=cuda)

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


def sparse_or_dense_matmul(adj, x):
    if isinstance(adj, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)) or (
        torch.is_tensor(adj) and adj.layout == torch.sparse_coo
    ):
        return torch.sparse.mm(adj, x)
    return adj @ x


class ConfidenceLabelPropagation(nn.Module):
    def __init__(
        self,
        prop_steps=0,
        alpha=0.20,
        global_beta=0.05,
        min_anchor=0.60,
        residual_scale=0.15,
        degree_bias=0.25,
        coherence_bias=0.20,
        clustering_bias=0.20,
        graph_scale_bias=1.0,
        node_attn_base=0.10,
        node_attn_cluster_weight=0.70,
        node_attn_conf_weight=0.20,
        node_attn_degree_weight=0.25,
        source_conf_center=0.55,
        source_conf_sharpness=8.0,
        recipient_conf_center=0.50,
        recipient_conf_sharpness=8.0,
        hard_source_min_conf=1.1,
        hard_freeze_conf=1.1,
        accept_sharpness=12.0,
        accept_threshold=0.0,
        accept_quality_weight=0.70,
        accept_margin_weight=0.20,
        accept_struct_weight=0.10,
        accept_graph_clust_weight=0.0,
        accept_graph_degree_weight=0.0,
        accept_change_weight=0.0,
        accept_conf_penalty_weight=0.0,
        train_graph_clust_bias=0.0,
    ):
        super().__init__()
        self.prop_steps = prop_steps
        self.alpha = alpha
        self.global_beta = global_beta
        self.min_anchor = min_anchor
        self.residual_scale = residual_scale
        self.degree_bias = degree_bias
        self.coherence_bias = coherence_bias
        self.clustering_bias = clustering_bias
        self.graph_scale_bias = graph_scale_bias
        self.node_attn_base = node_attn_base
        self.node_attn_cluster_weight = node_attn_cluster_weight
        self.node_attn_conf_weight = node_attn_conf_weight
        self.node_attn_degree_weight = node_attn_degree_weight
        self.source_conf_center = source_conf_center
        self.source_conf_sharpness = source_conf_sharpness
        self.recipient_conf_center = recipient_conf_center
        self.recipient_conf_sharpness = recipient_conf_sharpness
        self.hard_source_min_conf = hard_source_min_conf
        self.hard_freeze_conf = hard_freeze_conf
        self.accept_sharpness = accept_sharpness
        self.accept_threshold = accept_threshold
        self.accept_quality_weight = accept_quality_weight
        self.accept_margin_weight = accept_margin_weight
        self.accept_struct_weight = accept_struct_weight
        self.accept_graph_clust_weight = accept_graph_clust_weight
        self.accept_graph_degree_weight = accept_graph_degree_weight
        self.accept_change_weight = accept_change_weight
        self.accept_conf_penalty_weight = accept_conf_penalty_weight
        self.train_graph_clust_bias = train_graph_clust_bias

    def _local_context(self, prop_adj, state, eps):
        if isinstance(prop_adj, tuple):
            adj_context = sparse_or_dense_matmul(prop_adj[0], state)
            attn_context = sparse_or_dense_matmul(prop_adj[1], state)
            if len(prop_adj) > 2:
                route_bias = prop_adj[2]
                if torch.is_tensor(route_bias) and route_bias.dim() == 2:
                    node_attn_mix = torch.clamp(route_bias, 0.0, 1.0)
                else:
                    node_attn_mix = torch.tensor(route_bias, device=state.device, dtype=state.dtype)
            else:
                node_attn_mix = 0.5
            return (1.0 - node_attn_mix) * adj_context + node_attn_mix * attn_context
        return sparse_or_dense_matmul(prop_adj, state)

    def _quality_score(self, state, local_context, struct_feat, eps):
        local_quality = F.cosine_similarity(state, local_context, dim=1, eps=1e-8).unsqueeze(-1)
        local_quality = torch.clamp((local_quality + 1.0) * 0.5, 0.0, 1.0)

        probs = state / (state.sum(dim=1, keepdim=True) + eps)
        topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
        if probs.size(1) > 1:
            margin = topk[:, :1] - topk[:, 1:2]
        else:
            margin = torch.ones_like(local_quality)

        if struct_feat is not None and struct_feat.size(1) > 1:
            clustering = struct_feat[:, 1:2]
        else:
            clustering = torch.full_like(local_quality, 0.5)

        return (
            self.accept_quality_weight * local_quality
            + self.accept_margin_weight * margin
            + self.accept_struct_weight * clustering
        )

    def _confidence_score(self, state, eps):
        score_mass = state.sum(dim=1, keepdim=True)
        norm_scores = state / (score_mass + eps)
        if state.size(1) > 1:
            max_entropy = float(np.log(state.size(1)))
            entropy = -(norm_scores * torch.log(norm_scores + eps)).sum(dim=1, keepdim=True)
            certainty = 1.0 - entropy / max_entropy
        else:
            certainty = torch.ones_like(score_mass)

        mass_scale = score_mass.mean().detach().clamp_min(eps)
        magnitude = torch.tanh(score_mass / mass_scale)
        return torch.clamp(0.5 * certainty + 0.5 * magnitude, 0.0, 1.0)

    def forward(self, logits, prop_adj, struct_feat=None):
        if self.prop_steps <= 0:
            return logits

        seed = F.relu(logits)
        if seed.numel() == 0:
            return seed

        eps = 1e-8
        confidence = self._confidence_score(seed, eps)

        weighted_seed = confidence * seed
        global_prior = weighted_seed.sum(dim=0, keepdim=True) / confidence.sum().clamp_min(eps)
        global_prior = global_prior.expand_as(seed)

        anchor_strength = torch.clamp(self.min_anchor + self.alpha * confidence, 0.0, 0.995)
        uncertainty = 1.0 - confidence
        if struct_feat is not None and struct_feat.size(1) > 0:
            log_degree = struct_feat[:, :1]
            low_degree = torch.clamp(1.0 - log_degree, 0.0, 1.0)
            if struct_feat.size(1) > 1:
                clustering = struct_feat[:, 1:2]
                low_clustering = torch.clamp(1.0 - clustering, 0.0, 1.0)
                graph_scale = torch.clamp(1.0 - clustering.mean(), 0.2, 1.0)
            else:
                low_clustering = torch.ones_like(confidence)
                graph_scale = seed.new_tensor(1.0)
        else:
            low_degree = torch.ones_like(confidence)
            low_clustering = torch.ones_like(confidence)
            graph_scale = seed.new_tensor(1.0)
        propagated = seed
        base_state = seed
        for _ in range(self.prop_steps):
            source_gate = torch.sigmoid(self.source_conf_sharpness * (confidence - self.source_conf_center))
            recipient_gate = torch.sigmoid(self.recipient_conf_sharpness * (self.recipient_conf_center - confidence))
            if self.hard_source_min_conf <= 1.0:
                source_gate = source_gate * (confidence >= self.hard_source_min_conf).to(source_gate.dtype)
            if self.hard_freeze_conf <= 1.0:
                recipient_gate = recipient_gate * (confidence < self.hard_freeze_conf).to(recipient_gate.dtype)
            if isinstance(prop_adj, tuple):
                weighted_state = source_gate * propagated
                adj_num = sparse_or_dense_matmul(prop_adj[0], weighted_state)
                adj_den = sparse_or_dense_matmul(prop_adj[0], source_gate).clamp_min(eps)
                adj_context = adj_num / adj_den

                attn_num = sparse_or_dense_matmul(prop_adj[1], weighted_state)
                attn_den = sparse_or_dense_matmul(prop_adj[1], source_gate).clamp_min(eps)
                attn_context = attn_num / attn_den
                if struct_feat is not None and struct_feat.size(1) > 1:
                    clustering = struct_feat[:, 1:2]
                else:
                    clustering = torch.full_like(confidence, 0.5)
                node_attn_mix = torch.clamp(
                    self.node_attn_base
                    + self.node_attn_cluster_weight * clustering
                    + self.node_attn_conf_weight * confidence
                    - self.node_attn_degree_weight * low_degree,
                    0.0,
                    1.0,
                )
                if len(prop_adj) > 2:
                    route_bias = prop_adj[2]
                    node_attn_mix = torch.clamp(node_attn_mix * route_bias, 0.0, 1.0)
                local_context = (1.0 - node_attn_mix) * adj_context + node_attn_mix * attn_context
            else:
                weighted_state = source_gate * propagated
                local_num = sparse_or_dense_matmul(prop_adj, weighted_state)
                local_den = sparse_or_dense_matmul(prop_adj, source_gate).clamp_min(eps)
                local_context = local_num / local_den
            fused_context = (1.0 - self.global_beta) * local_context + self.global_beta * global_prior
            agreement = F.cosine_similarity(propagated, fused_context, dim=1, eps=1e-8).unsqueeze(-1)
            agreement = torch.clamp((agreement + 1.0) * 0.5, 0.0, 1.0)
            seed_agreement = F.cosine_similarity(seed, fused_context, dim=1, eps=1e-8).unsqueeze(-1)
            seed_agreement = torch.clamp((seed_agreement + 1.0) * 0.5, 0.0, 1.0)
            selective_gate = torch.clamp(
                uncertainty
                + self.degree_bias * low_degree
                + self.coherence_bias * seed_agreement
                + self.clustering_bias * low_clustering,
                0.0,
                1.0,
            )
            update_gate = recipient_gate * selective_gate * agreement * (1.0 - anchor_strength)
            residual = self.residual_scale * self.graph_scale_bias * graph_scale * update_gate * (fused_context - propagated)
            propagated = anchor_strength * seed + (1.0 - anchor_strength) * propagated + residual
            propagated = torch.clamp(propagated, min=0.0)

        base_context = self._local_context(prop_adj, base_state, eps)
        prop_context = self._local_context(prop_adj, propagated, eps)
        base_quality = self._quality_score(base_state, base_context, struct_feat, eps)
        prop_quality = self._quality_score(propagated, prop_context, struct_feat, eps)
        base_confidence = self._confidence_score(base_state, eps)
        prop_confidence = self._confidence_score(propagated, eps)
        change_penalty = (propagated - base_state).abs().mean(dim=1, keepdim=True)
        conf_penalty = F.relu(base_confidence - prop_confidence)
        quality_gain = prop_quality - base_quality
        quality_gain = quality_gain - self.accept_change_weight * change_penalty
        quality_gain = quality_gain - self.accept_conf_penalty_weight * conf_penalty

        graph_bias = 0.0
        graph_clustering_mean = None
        if struct_feat is not None and struct_feat.size(1) > 0:
            log_degree_mean = struct_feat[:, :1].mean()
            graph_bias = graph_bias + self.accept_graph_degree_weight * (log_degree_mean - 0.5)
            if struct_feat.size(1) > 1:
                graph_clustering_mean = struct_feat[:, 1:2].mean()
                graph_bias = graph_bias + self.accept_graph_clust_weight * (graph_clustering_mean - 0.5)

        effective_threshold = self.accept_threshold + graph_bias
        if self.training and self.train_graph_clust_bias != 0.0 and graph_clustering_mean is not None:
            effective_threshold = effective_threshold - self.train_graph_clust_bias * (graph_clustering_mean - 0.5)

        low_conf_mask = 1.0 - (confidence >= self.hard_freeze_conf).to(propagated.dtype) if self.hard_freeze_conf <= 1.0 else (1.0 - confidence)
        accept_gate = torch.sigmoid(self.accept_sharpness * (quality_gain - effective_threshold))
        accept_gate = accept_gate * low_conf_mask
        return base_state + accept_gate * (propagated - base_state)


class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        output_dim,
        enabled=True,
        projection_dim=64,
    ):
        super().__init__()
        self.enabled = enabled
        self.proj1 = nn.Linear(output_dim, output_dim, bias=False)
        self.proj2 = nn.Linear(output_dim, projection_dim, bias=False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)

    def forward(self, logits):
        scores = F.relu(logits)
        if (not self.enabled) or scores.numel() == 0:
            return None
        proj = F.relu(self.proj1(scores))
        proj = self.proj2(proj)
        return F.normalize(proj, p=2, dim=1)


class HybridGateGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        struct_dim,
        heads=6,
        dropout=0.0,
        add_self_loops=True,
        init_beta_struct=1.0,
        init_beta_feat=1.0,
        init_beta_agree=0.0,
        init_beta_edge=0.0,
        init_beta_trust=0.0,
        gate_activation='identity',
        use_channel_mix=True,
        prop_gate_strength=2.0,
        prop_gate_bias=0.15,
        enable_residual_gate=True,
        enable_propagation_gate=True,
    ):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.struct_dim = struct_dim
        self.heads = heads
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.init_beta_struct = init_beta_struct
        self.init_beta_feat = init_beta_feat
        self.init_beta_agree = init_beta_agree
        self.init_beta_edge = init_beta_edge
        self.init_beta_trust = init_beta_trust
        self.gate_activation = gate_activation
        self.use_channel_mix = use_channel_mix
        self.prop_gate_strength = prop_gate_strength
        self.prop_gate_bias = prop_gate_bias
        self.enable_residual_gate = enable_residual_gate
        self.enable_propagation_gate = enable_propagation_gate

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.node_trust_lin = nn.Linear(struct_dim, 1, bias=True)
        self.edge_mlp_lin1 = nn.Linear(4, 8, bias=True)
        self.edge_mlp_lin2 = nn.Linear(8, 1, bias=True)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.beta_struct = nn.Parameter(torch.tensor(1.0))
        self.beta_feat = nn.Parameter(torch.tensor(1.0))
        self.beta_agree = nn.Parameter(torch.tensor(0.0))
        self.beta_edge = nn.Parameter(torch.tensor(0.0))
        self.beta_trust = nn.Parameter(torch.tensor(0.0))
        self.beta_centrality = nn.Parameter(torch.tensor(1.0))
        self.cue_mix = nn.Parameter(torch.Tensor(heads, 4))
        self.cue_bias = nn.Parameter(torch.zeros(heads))
        self.residual_gate_scale = nn.Parameter(torch.tensor(0.5))
        self.struct_reliability = nn.Parameter(torch.zeros(heads))
        self.feat_reliability = nn.Parameter(torch.zeros(heads))
        self.centrality_reliability = nn.Parameter(torch.zeros(heads))
        self._cached_attention = None

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
        with torch.no_grad():
            self.beta_struct.fill_(self.init_beta_struct)
            self.beta_feat.fill_(self.init_beta_feat)
            self.beta_agree.fill_(self.init_beta_agree)
            self.beta_edge.fill_(self.init_beta_edge)
            self.beta_trust.fill_(self.init_beta_trust)
            self.beta_centrality.fill_(1.0)
            self.residual_gate_scale.fill_(0.5)
        nn.init.xavier_uniform_(self.cue_mix)
        nn.init.zeros_(self.cue_bias)
        nn.init.zeros_(self.struct_reliability)
        nn.init.zeros_(self.feat_reliability)
        nn.init.zeros_(self.centrality_reliability)

    def _activate_gate(self, beta):
        if self.gate_activation == 'identity':
            return beta
        if self.gate_activation == 'tanh':
            return torch.tanh(beta)
        if self.gate_activation == 'softplus':
            return F.softplus(beta)
        raise ValueError(f'Unsupported gate_activation: {self.gate_activation}')

    def forward(self, x, edge_index, struct_feat, return_attention_weights=False):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        node_trust = torch.sigmoid(self.node_trust_lin(struct_feat)).squeeze(-1)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)

        alpha, prop_alpha = self.edge_updater(
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
        if return_attention_weights:
            return out, (edge_index, alpha, prop_alpha)
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
            eps=1e-8
        )
        agreement = struct_sim * feat_sim
        conflict = torch.abs(struct_sim - feat_sim)
        edge_cues = torch.stack([struct_sim, feat_sim, agreement, conflict], dim=-1)
        edge_gate = self.edge_mlp_lin2(F.relu(self.edge_mlp_lin1(edge_cues))).squeeze(-1)
        edge_bonus = torch.sigmoid(edge_gate) * F.relu(agreement)
        edge_trust = torch.sqrt(torch.clamp(node_trust_j * node_trust_i, min=0.0))
        trust_bonus = edge_trust * F.relu(0.5 * (struct_sim + feat_sim))
        # The last two structural channels are global centrality-style priors.
        if struct_feat_j.size(-1) >= 2:
            centrality_j = struct_feat_j[:, -2:]
            centrality_i = struct_feat_i[:, -2:]
            centrality_sim = F.cosine_similarity(centrality_j, centrality_i, dim=-1, eps=1e-8)
            centrality_gap = -torch.mean(torch.abs(centrality_j - centrality_i), dim=-1)
        else:
            centrality_sim = torch.zeros_like(struct_sim)
            centrality_gap = torch.zeros_like(struct_sim)

        if self.enable_residual_gate:
            if self.use_channel_mix:
                cue_stack = torch.stack([
                    self._activate_gate(self.beta_struct) * struct_sim,
                    self._activate_gate(self.beta_feat) * feat_sim,
                    self._activate_gate(self.beta_centrality) * centrality_sim,
                    self._activate_gate(self.beta_centrality) * centrality_gap,
                ], dim=-1)
                gate_logit = cue_stack @ self.cue_mix.t() + self.cue_bias
            else:
                struct_reliability = torch.sigmoid(self.struct_reliability).unsqueeze(0)
                feat_reliability = torch.sigmoid(self.feat_reliability).unsqueeze(0)
                centrality_reliability = torch.sigmoid(self.centrality_reliability).unsqueeze(0)

                gate_logit = struct_reliability * self._activate_gate(self.beta_struct) * struct_sim.unsqueeze(-1)
                gate_logit = gate_logit + feat_reliability * self._activate_gate(self.beta_feat) * feat_sim.unsqueeze(-1)
                gate_logit = gate_logit + centrality_reliability * self._activate_gate(self.beta_centrality) * (
                    0.5 * centrality_sim.unsqueeze(-1) + 0.5 * centrality_gap.unsqueeze(-1)
                )

                # Keep the gate as a bounded residual correction to stabilize cross-dataset behavior.
                gate_logit = torch.tanh(gate_logit) * torch.sigmoid(self.residual_gate_scale)
            gate_logit = gate_logit + self._activate_gate(self.beta_agree) * agreement.unsqueeze(-1)
            gate_logit = gate_logit + self._activate_gate(self.beta_edge) * edge_bonus.unsqueeze(-1)
            gate_logit = gate_logit + self._activate_gate(self.beta_trust) * trust_bonus.unsqueeze(-1)
            alpha_raw = alpha_raw + gate_logit

        if self.enable_propagation_gate:
            prop_support = (
                0.25 * struct_sim
                + 0.25 * feat_sim
                + 0.15 * agreement
                + 0.10 * edge_bonus
                + 0.10 * trust_bonus
                + 0.10 * centrality_sim
                + 0.05 * (1.0 + centrality_gap)
            )
            prop_gate = torch.sigmoid(self.prop_gate_strength * (prop_support - self.prop_gate_bias))
            prop_alpha_raw = alpha_raw + torch.log(prop_gate.unsqueeze(-1).clamp_min(1e-8))
        else:
            prop_alpha_raw = alpha_raw

        alpha = softmax(alpha_raw, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        prop_alpha = softmax(prop_alpha_raw, index, ptr, size_i)
        return alpha, prop_alpha

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)


class GateReproGATLayer(MessagePassing):
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
        main_beta_agree_scale=1.0,
        main_beta_edge_scale=1.0,
        main_beta_trust_scale=1.0,
        prop_gate_strength=2.0,
        prop_gate_bias=0.15,
        prop_gate_struct_weight=0.30,
        prop_gate_feat_weight=0.30,
        prop_gate_agree_weight=0.20,
        prop_gate_edge_weight=0.10,
        prop_gate_trust_weight=0.10,
        graph_clust_gate_scale=0.0,
        prop_graph_clust_gate_scale=0.0,
        enable_propagation_gate=True,
    ):
        super().__init__(node_dim=0, aggr='add')
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.prop_gate_strength = prop_gate_strength
        self.prop_gate_bias = prop_gate_bias
        self.prop_gate_struct_weight = prop_gate_struct_weight
        self.prop_gate_feat_weight = prop_gate_feat_weight
        self.prop_gate_agree_weight = prop_gate_agree_weight
        self.prop_gate_edge_weight = prop_gate_edge_weight
        self.prop_gate_trust_weight = prop_gate_trust_weight
        self.graph_clust_gate_scale = graph_clust_gate_scale
        self.prop_graph_clust_gate_scale = prop_graph_clust_gate_scale
        self.enable_propagation_gate = enable_propagation_gate
        self._graph_gate_scale = None
        self._prop_graph_gate_scale = None

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
        self.main_beta_agree_scale = main_beta_agree_scale
        self.main_beta_edge_scale = main_beta_edge_scale
        self.main_beta_trust_scale = main_beta_trust_scale

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

    def forward(self, x, edge_index, struct_feat, return_attention_weights=False):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        node_trust = torch.sigmoid(self.node_trust_lin(struct_feat)).squeeze(-1)
        if struct_feat.size(1) > 1:
            graph_clustering = torch.clamp(struct_feat[:, 1:2].mean(), 0.0, 1.0)
            graph_gate_scale = 1.0 - self.graph_clust_gate_scale * (graph_clustering - 0.5)
            self._graph_gate_scale = torch.clamp(graph_gate_scale, 0.7, 1.3)
            prop_graph_gate_scale = 1.0 - self.prop_graph_clust_gate_scale * (graph_clustering - 0.5)
            self._prop_graph_gate_scale = torch.clamp(prop_graph_gate_scale, 0.7, 1.3)
        else:
            self._graph_gate_scale = x.new_tensor(1.0)
            self._prop_graph_gate_scale = x.new_tensor(1.0)

        alpha, prop_alpha = self.edge_updater(
            edge_index,
            alpha_src=(x * self.att_src).sum(dim=-1),
            alpha_dst=(x * self.att_dst).sum(dim=-1),
            x=x,
            struct_feat=struct_feat,
            node_trust=node_trust,
        )

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.mean(dim=1)
        out = out + self.bias
        self._graph_gate_scale = None
        self._prop_graph_gate_scale = None
        if return_attention_weights:
            return out, (edge_index, alpha, prop_alpha)
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
        base_gate_logit = (
            self.beta_struct * struct_sim
            + self.beta_feat * feat_sim
            + self.beta_agree * agreement
            + self.beta_edge * edge_bonus
            + self.beta_trust * trust_bonus
        )
        alpha_gate_logit = (
            self.beta_struct * struct_sim
            + self.beta_feat * feat_sim
            + (self.beta_agree * self.main_beta_agree_scale * self._graph_gate_scale) * agreement
            + (self.beta_edge * self.main_beta_edge_scale * self._graph_gate_scale) * edge_bonus
            + (self.beta_trust * self.main_beta_trust_scale * self._graph_gate_scale) * trust_bonus
        )
        base_alpha_raw = alpha_raw + alpha_gate_logit.unsqueeze(-1)
        if self.enable_propagation_gate:
            prop_support = (
                self.prop_gate_struct_weight * struct_sim
                + self.prop_gate_feat_weight * feat_sim
                + (self.prop_gate_agree_weight * self._prop_graph_gate_scale) * agreement
                + (self.prop_gate_edge_weight * self._prop_graph_gate_scale) * edge_bonus
                + (self.prop_gate_trust_weight * self._prop_graph_gate_scale) * trust_bonus
            )
            prop_gate = torch.sigmoid(self.prop_gate_strength * (prop_support - self.prop_gate_bias))
            prop_alpha_raw = alpha_raw + base_gate_logit.unsqueeze(-1) + torch.log(prop_gate.unsqueeze(-1).clamp_min(1e-8))
        else:
            prop_alpha_raw = alpha_raw + base_gate_logit.unsqueeze(-1)
        alpha = softmax(base_alpha_raw, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        prop_alpha = softmax(prop_alpha_raw, index, ptr, size_i)
        return alpha, prop_alpha

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)


class GateReproResidualGATLayer(MessagePassing):
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
        init_beta_residual=0.15,
        init_residual_bias=-2.0,
        init_beta_attn_residual=0.05,
    ):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.init_beta_residual = init_beta_residual
        self.init_residual_bias = init_residual_bias
        self.init_beta_attn_residual = init_beta_attn_residual

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.node_trust_lin = nn.Linear(struct_dim, 1, bias=True)
        self.residual_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.residual_gate_lin = nn.Linear(struct_dim, 1, bias=True)
        self.attn_residual_lin = nn.Linear(struct_dim, 1, bias=True)
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
        self.beta_residual = nn.Parameter(torch.tensor(init_beta_residual))
        self.beta_attn_residual = nn.Parameter(torch.tensor(init_beta_attn_residual))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.node_trust_lin.weight)
        nn.init.zeros_(self.node_trust_lin.bias)
        nn.init.xavier_uniform_(self.residual_lin.weight)
        nn.init.zeros_(self.residual_gate_lin.weight)
        nn.init.zeros_(self.residual_gate_lin.bias)
        nn.init.zeros_(self.attn_residual_lin.weight)
        nn.init.zeros_(self.attn_residual_lin.bias)
        nn.init.xavier_uniform_(self.edge_mlp_lin1.weight)
        nn.init.zeros_(self.edge_mlp_lin1.bias)
        nn.init.zeros_(self.edge_mlp_lin2.weight)
        nn.init.zeros_(self.edge_mlp_lin2.bias)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.beta_residual.fill_(self.init_beta_residual)
            self.residual_gate_lin.bias.fill_(self.init_residual_bias)
            self.beta_attn_residual.fill_(self.init_beta_attn_residual)

    def forward(self, x, edge_index, struct_feat, return_attention_weights=False):
        residual = self.residual_lin(x)
        residual_gate = torch.sigmoid(self.residual_gate_lin(struct_feat))
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        node_trust = torch.sigmoid(self.node_trust_lin(struct_feat)).squeeze(-1)

        alpha = self.edge_updater(
            edge_index,
            alpha_src=(x * self.att_src).sum(dim=-1),
            alpha_dst=(x * self.att_dst).sum(dim=-1),
            x=x,
            struct_feat=struct_feat,
            node_trust=node_trust,
        )

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.mean(dim=1)
        out = out + self.beta_residual * residual_gate * residual
        out = out + self.bias
        if return_attention_weights:
            return out, (edge_index, alpha, alpha)
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
        attn_residual = 0.5 * (
            self.attn_residual_lin(struct_feat_j).squeeze(-1)
            + self.attn_residual_lin(struct_feat_i).squeeze(-1)
        )
        gate_logit = (
            self.beta_struct * struct_sim
            + self.beta_feat * feat_sim
            + self.beta_agree * agreement
            + self.beta_edge * edge_bonus
            + self.beta_trust * trust_bonus
        )
        alpha = softmax(
            alpha_raw
            + gate_logit.unsqueeze(-1)
            + self.beta_attn_residual * attn_residual.unsqueeze(-1),
            index,
            ptr,
            size_i,
        )
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
        dropout=0.5,
        batch_norm=False,
        heads=6,
        use_gate_repro_gat2=False,
        use_hybrid_gat2=False,
        use_gate_repro_residual_gat2=False,
        init_beta_struct=1.0,
        init_beta_feat=1.0,
        init_beta_agree=0.0,
        init_beta_edge=0.0,
        init_beta_trust=0.0,
        main_beta_agree_scale=1.0,
        main_beta_edge_scale=1.0,
        main_beta_trust_scale=1.0,
        init_beta_residual=0.15,
        init_residual_bias=-2.0,
        init_beta_attn_residual=0.05,
        gate_activation='identity',
        use_channel_mix=True,
        prop_gate_strength=2.0,
        prop_gate_bias=0.15,
        prop_gate_struct_weight=0.30,
        prop_gate_feat_weight=0.30,
        prop_gate_agree_weight=0.20,
        prop_gate_edge_weight=0.10,
        prop_gate_trust_weight=0.10,
        graph_clust_gate_scale=0.0,
        prop_graph_clust_gate_scale=0.0,
        enable_residual_gate=True,
        enable_propagation_gate=True,
        lp_steps=0,
        lp_alpha=0.20,
        lp_beta=0.05,
        lp_min_anchor=0.60,
        lp_residual_scale=0.15,
        lp_degree_bias=0.25,
        lp_coherence_bias=0.20,
        lp_clustering_bias=0.20,
        lp_graph_scale_bias=1.0,
        lp_node_attn_base=0.10,
        lp_node_attn_cluster_weight=0.70,
        lp_node_attn_conf_weight=0.20,
        lp_node_attn_degree_weight=0.25,
        lp_source_conf_center=0.55,
        lp_source_conf_sharpness=8.0,
        lp_recipient_conf_center=0.50,
        lp_recipient_conf_sharpness=8.0,
        lp_hard_source_min_conf=1.1,
        lp_hard_freeze_conf=1.1,
        lp_accept_sharpness=12.0,
        lp_accept_threshold=0.0,
        lp_accept_quality_weight=0.70,
        lp_accept_margin_weight=0.20,
        lp_accept_struct_weight=0.10,
        lp_accept_graph_clust_weight=0.0,
        lp_accept_graph_degree_weight=0.0,
        lp_accept_change_weight=0.0,
        lp_accept_conf_penalty_weight=0.0,
        lp_train_graph_clust_bias=0.0,
        contrastive_enabled=True,
        contrastive_projection_dim=64,
        lp_mode='adj',
        lp_attn_blend=0.5,
        lp_auto_blend_base=0.5,
        lp_auto_blend_scale=1.0,
        lp_sparse_scale=1.25,
        lp_sparse_power=1.50,
        lp_sparse_topk=0,
        lp_dual_route_graph_clust_damp=0.0,
        lp_output_blend_enabled=False,
        lp_output_blend_base=0.0,
        lp_output_blend_conf_weight=0.65,
        lp_output_blend_degree_weight=0.20,
        lp_output_blend_clust_weight=0.10,
    ):
        super().__init__()
        self.dropout = dropout
        self.lp_mode = lp_mode
        self.lp_attn_blend = lp_attn_blend
        self.lp_auto_blend_base = lp_auto_blend_base
        self.lp_auto_blend_scale = lp_auto_blend_scale
        self.lp_sparse_scale = lp_sparse_scale
        self.lp_sparse_power = lp_sparse_power
        self.lp_sparse_topk = lp_sparse_topk
        self.lp_dual_route_graph_clust_damp = lp_dual_route_graph_clust_damp
        self.lp_output_blend_enabled = lp_output_blend_enabled
        self.lp_output_blend_base = lp_output_blend_base
        self.lp_output_blend_conf_weight = lp_output_blend_conf_weight
        self.lp_output_blend_degree_weight = lp_output_blend_degree_weight
        self.lp_output_blend_clust_weight = lp_output_blend_clust_weight
        self.use_gate_repro_gat2 = use_gate_repro_gat2
        self.use_hybrid_gat2 = use_hybrid_gat2
        self.use_gate_repro_residual_gat2 = use_gate_repro_residual_gat2

        hidden_dim = hidden_dims[0]

        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            add_self_loops=True,
            concat=False
        )

        if self.use_gate_repro_residual_gat2:
            self.gat2 = GateReproResidualGATLayer(
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
                init_beta_residual=init_beta_residual,
                init_residual_bias=init_residual_bias,
                init_beta_attn_residual=init_beta_attn_residual,
            )
        elif self.use_gate_repro_gat2:
            self.gat2 = GateReproGATLayer(
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
                main_beta_agree_scale=main_beta_agree_scale,
                main_beta_edge_scale=main_beta_edge_scale,
                main_beta_trust_scale=main_beta_trust_scale,
                prop_gate_strength=prop_gate_strength,
                prop_gate_bias=prop_gate_bias,
                prop_gate_struct_weight=prop_gate_struct_weight,
                prop_gate_feat_weight=prop_gate_feat_weight,
                prop_gate_agree_weight=prop_gate_agree_weight,
                prop_gate_edge_weight=prop_gate_edge_weight,
                prop_gate_trust_weight=prop_gate_trust_weight,
                graph_clust_gate_scale=graph_clust_gate_scale,
                prop_graph_clust_gate_scale=prop_graph_clust_gate_scale,
                enable_propagation_gate=enable_propagation_gate,
            )
        elif self.use_hybrid_gat2:
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
                gate_activation=gate_activation,
                use_channel_mix=use_channel_mix,
                prop_gate_strength=prop_gate_strength,
                prop_gate_bias=prop_gate_bias,
                enable_residual_gate=enable_residual_gate,
                enable_propagation_gate=enable_propagation_gate,
            )
        else:
            self.gat2 = GATConv(
                hidden_dim,
                output_dim,
                heads=heads,
                add_self_loops=True,
                concat=False,
                dropout=dropout,
            )
        self.lp = ConfidenceLabelPropagation(
            prop_steps=lp_steps,
            alpha=lp_alpha,
            global_beta=lp_beta,
            min_anchor=lp_min_anchor,
            residual_scale=lp_residual_scale,
            degree_bias=lp_degree_bias,
            coherence_bias=lp_coherence_bias,
            clustering_bias=lp_clustering_bias,
            graph_scale_bias=lp_graph_scale_bias,
            node_attn_base=lp_node_attn_base,
            node_attn_cluster_weight=lp_node_attn_cluster_weight,
            node_attn_conf_weight=lp_node_attn_conf_weight,
            node_attn_degree_weight=lp_node_attn_degree_weight,
            source_conf_center=lp_source_conf_center,
            source_conf_sharpness=lp_source_conf_sharpness,
            recipient_conf_center=lp_recipient_conf_center,
            recipient_conf_sharpness=lp_recipient_conf_sharpness,
            hard_source_min_conf=lp_hard_source_min_conf,
            hard_freeze_conf=lp_hard_freeze_conf,
            accept_sharpness=lp_accept_sharpness,
            accept_threshold=lp_accept_threshold,
            accept_quality_weight=lp_accept_quality_weight,
            accept_margin_weight=lp_accept_margin_weight,
            accept_struct_weight=lp_accept_struct_weight,
            accept_graph_clust_weight=lp_accept_graph_clust_weight,
            accept_graph_degree_weight=lp_accept_graph_degree_weight,
            accept_change_weight=lp_accept_change_weight,
            accept_conf_penalty_weight=lp_accept_conf_penalty_weight,
            train_graph_clust_bias=lp_train_graph_clust_bias,
        )
        self.contrastive_head = ContrastiveProjectionHead(
            output_dim=output_dim,
            enabled=contrastive_enabled,
            projection_dim=contrastive_projection_dim,
        )
        self.contrastive_head.reset_parameters()

        if batch_norm:
            self.batch_norm = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False)
            ])
        else:
            self.batch_norm = None

        self.layers = nn.ModuleList([self.gat1, self.gat2])

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.contrastive_head.reset_parameters()

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

    def _build_attention_prop_adj(self, edge_index, alpha, num_nodes):
        edge_weight = alpha.mean(dim=1)
        attn_adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            (num_nodes, num_nodes),
            device=edge_weight.device,
        ).coalesce()
        transpose_adj = torch.sparse_coo_tensor(
            attn_adj.indices().flip(0),
            attn_adj.values(),
            attn_adj.size(),
            device=edge_weight.device,
        ).coalesce()
        merged = (attn_adj + transpose_adj).coalesce()
        row_sum = torch.sparse.sum(merged, dim=1).to_dense().clamp_min(1e-8)
        row_inv = 1.0 / row_sum
        norm_values = merged.values() * row_inv[merged.indices()[0]]
        return torch.sparse_coo_tensor(
            merged.indices(),
            norm_values,
            merged.size(),
            device=edge_weight.device,
        ).coalesce()

    def _sparsify_attention_adj(self, attn_adj):
        attn_adj = attn_adj.coalesce()
        indices = attn_adj.indices()
        values = attn_adj.values()
        row = indices[0]
        col = indices[1]

        row_deg = torch.bincount(row, minlength=attn_adj.size(0)).clamp_min(1)
        mean_weight = 1.0 / row_deg[row].to(values.dtype)
        keep_mask = values >= (self.lp_sparse_scale * mean_weight)
        keep_mask = keep_mask | (row == col)

        kept_indices = indices[:, keep_mask]
        kept_values = values[keep_mask]
        if kept_values.numel() == 0:
            return attn_adj

        if self.lp_sparse_topk > 0:
            topk_mask = torch.zeros_like(kept_values, dtype=torch.bool)
            kept_row = kept_indices[0]
            kept_col = kept_indices[1]
            num_nodes = attn_adj.size(0)
            for node in range(num_nodes):
                node_mask = kept_row == node
                if not torch.any(node_mask):
                    continue
                node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
                node_cols = kept_col[node_idx]
                self_idx = node_idx[node_cols == node]
                other_idx = node_idx[node_cols != node]
                if self_idx.numel() > 0:
                    topk_mask[self_idx] = True
                if other_idx.numel() > self.lp_sparse_topk:
                    _, order = torch.topk(kept_values[other_idx], k=self.lp_sparse_topk)
                    topk_mask[other_idx[order]] = True
                else:
                    topk_mask[other_idx] = True
            kept_indices = kept_indices[:, topk_mask]
            kept_values = kept_values[topk_mask]
            if kept_values.numel() == 0:
                return attn_adj

        kept_values = kept_values.pow(self.lp_sparse_power)
        sparse_adj = torch.sparse_coo_tensor(
            kept_indices,
            kept_values,
            attn_adj.size(),
            device=attn_adj.device,
        ).coalesce()
        row_sum = torch.sparse.sum(sparse_adj, dim=1).to_dense().clamp_min(1e-8)
        row_inv = 1.0 / row_sum
        norm_values = sparse_adj.values() * row_inv[sparse_adj.indices()[0]]
        return torch.sparse_coo_tensor(
            sparse_adj.indices(),
            norm_values,
            sparse_adj.size(),
            device=sparse_adj.device,
        ).coalesce()

    def _blend_sparse_adj(self, adj, attn_adj, blend):
        adj = adj.coalesce()
        merged = (adj * (1.0 - blend) + attn_adj * blend).coalesce()
        row_sum = torch.sparse.sum(merged, dim=1).to_dense().clamp_min(1e-8)
        row_inv = 1.0 / row_sum
        norm_values = merged.values() * row_inv[merged.indices()[0]]
        return torch.sparse_coo_tensor(
            merged.indices(),
            norm_values,
            merged.size(),
            device=merged.device,
        ).coalesce()

    def _resolve_lp_adj(self, adj, edge_index, alpha, num_nodes, struct_feat):
        if self.lp_mode == 'adj':
            return adj
        attn_adj = self._build_attention_prop_adj(edge_index, alpha, num_nodes)
        if self.lp_mode == 'attention':
            return attn_adj
        if self.lp_mode == 'sparse_attention':
            return self._sparsify_attention_adj(attn_adj)
        if self.lp_mode == 'blend':
            blend = torch.clamp(torch.tensor(self.lp_attn_blend, device=attn_adj.device), 0.0, 1.0)
            return self._blend_sparse_adj(adj, attn_adj, blend)
        if self.lp_mode == 'auto_blend':
            if struct_feat is not None and struct_feat.size(1) > 1:
                clustering_mean = struct_feat[:, 1].mean()
                blend = self.lp_auto_blend_base + self.lp_auto_blend_scale * (clustering_mean - 0.5)
            else:
                blend = torch.tensor(self.lp_attn_blend, device=attn_adj.device)
            blend = torch.clamp(blend, 0.0, 1.0)
            return self._blend_sparse_adj(adj, attn_adj, blend)
        if self.lp_mode == 'node_mix':
            return adj, attn_adj
        if self.lp_mode == 'dual_route':
            sparse_attn_adj = self._sparsify_attention_adj(attn_adj)
            if struct_feat is not None and struct_feat.size(1) > 1:
                clustering = struct_feat[:, 1:2]
                route_bias = torch.clamp(0.35 + 1.10 * clustering, 0.25, 1.0)
                if self.lp_dual_route_graph_clust_damp > 0:
                    graph_clustering = torch.clamp(clustering.mean(), 0.0, 1.0)
                    route_bias = route_bias * (
                        1.0 - self.lp_dual_route_graph_clust_damp * graph_clustering
                    )
                    route_bias = torch.clamp(route_bias, 0.25, 1.0)
            else:
                route_bias = torch.tensor(1.0, device=attn_adj.device)
            return adj, sparse_attn_adj, route_bias
        raise ValueError(f'Unsupported lp_mode: {self.lp_mode}')

    def _blend_lp_output(self, pre_lp_out, post_lp_out, struct_feat):
        if not self.lp_output_blend_enabled:
            return post_lp_out, None

        pre_lp_score = F.relu(pre_lp_out)
        post_lp_score = F.relu(post_lp_out)
        eps = 1e-8
        score_mass = pre_lp_score.sum(dim=1, keepdim=True)
        probs = pre_lp_score / (score_mass + eps)
        if pre_lp_score.size(1) > 1:
            max_entropy = float(np.log(pre_lp_score.size(1)))
            entropy = -(probs * torch.log(probs + eps)).sum(dim=1, keepdim=True)
            confidence = 1.0 - entropy / max_entropy
        else:
            confidence = torch.ones_like(score_mass)

        if struct_feat is not None and struct_feat.size(1) > 0:
            log_degree = struct_feat[:, :1]
            if struct_feat.size(1) > 1:
                clustering = struct_feat[:, 1:2]
            else:
                clustering = torch.full_like(log_degree, 0.5)
        else:
            log_degree = torch.full_like(confidence, 0.5)
            clustering = torch.full_like(confidence, 0.5)

        lp_mix = (
            self.lp_output_blend_base
            + self.lp_output_blend_conf_weight * (1.0 - confidence)
            + self.lp_output_blend_degree_weight * (1.0 - log_degree)
            + self.lp_output_blend_clust_weight * (1.0 - clustering)
        )
        lp_mix = torch.clamp(lp_mix, 0.0, 1.0)
        blended = pre_lp_score + lp_mix * (post_lp_score - pre_lp_score)
        return blended, lp_mix

    def forward(self, x, adj, struct_feat):
        out, _ = self.forward_with_aux(x, adj, struct_feat)
        return out

    def forward_with_aux(self, x, adj, struct_feat):
        edge_index = self._extract_edge_index(adj)

        if self.dropout != 0:
            x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)

        h = self.gat1(x, edge_index)
        h = F.relu(h)

        if self.batch_norm is not None:
            h = self.batch_norm[0](h)

        if self.dropout != 0:
            h = sparse_or_dense_dropout(h, p=self.dropout, training=self.training)

        if self.use_gate_repro_residual_gat2:
            out, (lp_edge_index, lp_alpha, lp_prop_alpha) = self.gat2(
                h,
                edge_index,
                struct_feat,
                return_attention_weights=True,
            )
        elif self.use_gate_repro_gat2:
            out, (lp_edge_index, lp_alpha, lp_prop_alpha) = self.gat2(
                h,
                edge_index,
                struct_feat,
                return_attention_weights=True,
            )
        elif self.use_hybrid_gat2:
            out, (lp_edge_index, lp_alpha, lp_prop_alpha) = self.gat2(
                h,
                edge_index,
                struct_feat,
                return_attention_weights=True,
            )
        else:
            out, (lp_edge_index, lp_alpha) = self.gat2(h, edge_index, return_attention_weights=True)
            lp_prop_alpha = lp_alpha
        pre_lp_out = out
        contrastive_embed = self.contrastive_head(out)
        lp_adj = self._resolve_lp_adj(adj, lp_edge_index, lp_prop_alpha, out.size(0), struct_feat)
        out = self.lp(out, lp_adj, struct_feat)
        out, lp_output_mix = self._blend_lp_output(pre_lp_out, out, struct_feat)
        aux = {
            'pre_lp_out': pre_lp_out,
            'lp_adj': lp_adj,
            'edge_index': lp_edge_index,
            'alpha': lp_alpha,
            'prop_alpha': lp_prop_alpha,
            'lp_output_mix': lp_output_mix,
            'contrastive_embed': contrastive_embed,
        }
        return out, aux

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        return [w for n, w in self.named_parameters() if 'bias' in n]

    @staticmethod
    def get_adj(adj: sp.csr_matrix, cuda: bool = True):
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj = adj.tocsr()
        return to_sparse_tensor(adj, cuda=cuda)

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

    @staticmethod
    def nor_edge(adj: sp.csr_matrix):
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
        adj = adj_norm.tocoo()
        newedge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
        return newedge_index

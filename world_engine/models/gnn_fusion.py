# /world_engine/models/gnn_fusion.py
# ======================================================================================
# World Discovery Engine (WDE)
# GNNFusion — Multi-sensor encoder + Graph Neural Network fusion model (pure PyTorch)
# --------------------------------------------------------------------------------------
# This module implements a self-contained, dependency-light GNN for the WDE pipeline:
#   • Modality encoders (per-sensor MLPs) for Sentinel-2, Sentinel-1, Landsat, LiDAR,
#     soils/climate, and arbitrary custom modalities supplied at runtime.
#   • Fusion block (concat / gated-sum / attention) to combine modality embeddings.
#   • Message passing GNN (GCN or GAT) implemented WITHOUT torch-geometric, so it runs
#     in standard Kaggle runtimes and plain servers. Uses PyTorch sparse ops for GCN
#     and a simple edge-softmax loop for GAT.
#   • Graph builders (kNN / radius / temporal window) to construct edges on the fly.
#   • Heads for classification (logits), regression (scores), and optional uncertainty
#     (log variance) for NASA-grade diagnostics in WDE candidate scoring.
#
# Design Goals
# ------------
# - Zero hard dependency on torch-geometric: everything is implemented with
#   torch, numpy, and (optionally) scikit-learn for kNN graph construction.
# - Reproducible by default (global RNG seeding).
# - Clear, well-documented API with save/load helpers for stateful training.
# - Flexible I/O: pass a dict of modality feature tensors; only present modalities
#   are encoded and fused.
#
# Typical Usage
# -------------
#   from world_engine.models.gnn_fusion import (
#       GNNFusion, GNNFusionConfig, build_knn_graph
#   )
#
#   N = 2048
#   x_s2   = torch.randn(N, 18)    # Sentinel-2 features
#   x_s1   = torch.randn(N, 8)     # Sentinel-1 features
#   coords = torch.rand(N, 2)      # lon/lat or projected (meters)
#   edge_index = build_knn_graph(coords, k=8)  # [2, E] (destination-source format)
#
#   cfg = GNNFusionConfig(
#       modalities={"sentinel2": 18, "sentinel1": 8},
#       fusion_type="gate",
#       gnn_type="gcn",
#       hidden_dim=128,
#       num_layers=3,
#       num_classes=2,          # optional classification head
#       regression_targets=1,   # optional regression head
#       predict_log_var=True,   # optional uncertainty for regression
#   )
#   model = GNNFusion(cfg)
#
#   out = model({"sentinel2": x_s2, "sentinel1": x_s1}, edge_index=edge_index)
#   # out = {"emb": [N, H], "logits": [N, C]?, "reg": [N, R]?, "log_var": [N, R]?}
#
# Notes
# -----
# - Edge format: edge_index is shape [2, E] with rows [dst, src]. This matches the
#   "message from src → dst" mental model and is convenient for per-destination softmax
#   in GAT. Utilities below build edges in this convention.
# - All tensors are expected as torch.float32 on the same device as the model.
# - For large graphs, consider mini-batching by spatial tiles or clusters upstream.
#
# License
# -------
# MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: fast kNN via scikit-learn (falls back to NumPy if unavailable)
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


# ======================================================================================
# Global utilities
# ======================================================================================

def set_global_seed(seed: int = 42) -> None:
    """
    Set global RNG seeds for PyTorch and NumPy to promote reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _ensure_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class GNNFusionConfig:
    """
    Configuration for the GNNFusion model.

    Modalities
    ----------
    modalities : Dict[str, int]
        Mapping modality name → feature dimension. Only provided modalities at
        forward-time are used (missing ones are ignored gracefully).
        Common keys: "sentinel2", "sentinel1", "landsat", "lidar", "soil", "climate", "custom_*".

    Modality encoders
    -----------------
    enc_hidden : int
        Hidden width of per-modality MLPs.
    enc_depth : int
        Number of layers in each modality MLP (>=1). Depth=1 → Linear only.

    Fusion
    ------
    fusion_type : str
        One of {"concat", "gate", "attn"}. "attn" computes per-modality attention weights.
    fusion_hidden : int
        Hidden width inside the fusion gate/attention scorer.
    fusion_dropout : float
        Dropout applied within fusion scorer.

    Graph neural network
    --------------------
    gnn_type : str
        One of {"gcn", "gat"}.
    hidden_dim : int
        Node embedding dimension inside GNN stack (post-fusion).
    num_layers : int
        Number of GNN layers.
    dropout : float
        Dropout applied after each GNN layer.
    layernorm : bool
        If True, applies LayerNorm after each GNN layer.
    residual : bool
        If True, adds residual connections across GNN layers (dim must match).

    Heads
    -----
    num_classes : int
        If > 0, enables classification head producing logits [N, num_classes].
    regression_targets : int
        If > 0, enables regression head producing values [N, regression_targets].
    predict_log_var : bool
        If True with regression enabled, also predict log-variance [N, regression_targets]
        for heteroscedastic uncertainty modeling.

    Misc
    ----
    seed : int
        RNG seed for reproducibility.
    """
    # Modalities (fill with known defaults; override in constructor as needed)
    modalities: Dict[str, int] = field(
        default_factory=lambda: {
            "sentinel2": 18,   # example: indices or stats
            "sentinel1": 8,
            "landsat": 12,
            "lidar": 6,
            "soil": 10,
            "climate": 8,
        }
    )

    # Encoders
    enc_hidden: int = 128
    enc_depth: int = 2

    # Fusion
    fusion_type: str = "gate"       # {"concat", "gate", "attn"}
    fusion_hidden: int = 128
    fusion_dropout: float = 0.0

    # GNN
    gnn_type: str = "gcn"           # {"gcn", "gat"}
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    layernorm: bool = True
    residual: bool = True

    # Heads
    num_classes: int = 0
    regression_targets: int = 1
    predict_log_var: bool = True

    # Misc
    seed: int = 42


# ======================================================================================
# Building blocks — encoders, fusion, and GNN layers
# ======================================================================================

class MLP(nn.Module):
    """
    Simple MLP with configurable depth, ReLU activations, LayerNorm option, and dropout.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        depth: int = 2,
        dropout: float = 0.0,
        layernorm: bool = False,
        final_activation: Optional[nn.Module] = nn.ReLU(inplace=True),
    ):
        super().__init__()
        assert depth >= 1, "MLP depth must be >= 1"
        layers: List[nn.Module] = []
        d_in = in_dim
        if depth == 1:
            layers.append(nn.Linear(d_in, out_dim))
        else:
            for i in range(depth - 1):
                layers.append(nn.Linear(d_in, hidden))
                if layernorm:
                    layers.append(nn.LayerNorm(hidden))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d_in = hidden
            layers.append(nn.Linear(d_in, out_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoder(nn.Module):
    """
    Per-modality encoder: lightweight MLP projecting raw features to a shared width.
    """

    def __init__(self, in_dim: int, out_dim: int, depth: int, hidden: int):
        super().__init__()
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            depth=depth,
            dropout=0.0,
            layernorm=True,
            final_activation=nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FusionBlock(nn.Module):
    """
    Combine a dict of modality embeddings into a single node embedding.

    Modes
    -----
    - "concat":    Concatenate along feature dim and project to hidden_dim.
    - "gate":      Compute scalar gates per modality (sigmoid) and sum(g_i * e_i).
    - "attn":      Compute attention per modality (softmax) and sum(a_i * e_i).

    All modes return a tensor of shape [N, hidden_dim].
    """

    def __init__(self, fusion_type: str, hidden_dim: int, fusion_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projection for concat
        self.concat_proj: Optional[nn.Linear] = None

        # Scorers for gate/attn (shared across modalities; fed with each modality emb)
        self.score_mlp = MLP(
            in_dim=hidden_dim, out_dim=1, hidden=fusion_hidden, depth=2,
            dropout=dropout, layernorm=True, final_activation=None
        )

    def _stack(self, embs: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack modality embeddings into [M, N, H]; returns (stack, N, H, M)
        """
        M = len(embs)
        assert M > 0, "No modality embeddings provided to FusionBlock"
        N, H = embs[0].shape
        for e in embs:
            assert e.shape == (N, H), "All modality embeddings must have shape [N, hidden_dim]"
        stack = torch.stack(embs, dim=0)  # [M, N, H]
        return stack  # [M, N, H]

    def forward(self, emb_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Gather present modality embeddings preserving deterministic key order
        keys = sorted(emb_dict.keys())
        embs = [emb_dict[k] for k in keys]
        if self.fusion_type == "concat":
            z = torch.cat(embs, dim=-1)  # [N, M*H]
            if self.concat_proj is None:
                self.concat_proj = nn.Linear(z.shape[-1], self.hidden_dim).to(z.device)
            out = self.concat_proj(self.dropout(z))
            return F.relu(out, inplace=True)

        stack = self._stack(embs)                  # [M, N, H]
        M, N, H = stack.shape
        if self.fusion_type == "gate":
            # independent sigmoid gate per modality
            scores = []
            for m in range(M):
                s = self.score_mlp(self.dropout(stack[m]))  # [N, 1]
                scores.append(torch.sigmoid(s))             # [N, 1]
            gates = torch.stack(scores, dim=0)              # [M, N, 1]
            fused = (gates * stack).sum(dim=0)              # [N, H]
            return F.relu(fused, inplace=True)

        if self.fusion_type == "attn":
            # softmax weights across modalities
            scores = []
            for m in range(M):
                s = self.score_mlp(self.dropout(stack[m]))  # [N, 1]
                scores.append(s)
            scores = torch.stack(scores, dim=0)             # [M, N, 1]
            attn = torch.softmax(scores, dim=0)             # [M, N, 1]
            fused = (attn * stack).sum(dim=0)               # [N, H]
            return F.relu(fused, inplace=True)

        raise ValueError(f"Unknown fusion_type: {self.fusion_type}")


# ----------------------------- GCN (symmetric norm) ----------------------------------

def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Add self-loops to edge_index (dst, src) format. Returns new edge_index [2, E+N].
    """
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    self_loops[0, :] = torch.arange(num_nodes, device=device)  # dst
    self_loops[1, :] = torch.arange(num_nodes, device=device)  # src
    return torch.cat([edge_index, self_loops], dim=1)


def _symmetrize(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Ensure undirected edges by adding reversed edges (dst, src) -> (src, dst).
    Assumes edge_index shape [2, E].
    """
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)


def _build_gcn_norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Build normalized adjacency A_hat = D^{-1/2} (A + I) D^{-1/2} as a sparse COO tensor.
    edge_index is [2, E] (dst, src). Returns a coalesced torch.sparse_coo_tensor shape [N, N].
    """
    # Add reversed edges to force symmetry and add self loops
    ei = _symmetrize(edge_index)
    ei = _add_self_loops(ei, num_nodes)  # [2, E’]

    # Values all ones initially (unweighted)
    values = torch.ones(ei.shape[1], device=ei.device, dtype=torch.float32)

    # Compute degree: sum of incoming weights per node (row-sum of A)
    # deg[i] = sum_j A[i, j] (because index 0 is dst)
    deg = torch.zeros(num_nodes, device=ei.device, dtype=torch.float32)
    deg.index_add_(0, ei[0], values)  # accumulate by destination index

    # Normalization for each edge: 1 / sqrt(deg[dst] * deg[src])
    d_dst = deg[ei[0]]
    d_src = deg[ei[1]]
    norm = values / torch.sqrt(torch.clamp(d_dst * d_src, min=1e-12))

    A = torch.sparse_coo_tensor(indices=ei, values=norm, size=(num_nodes, num_nodes))
    A = A.coalesce()
    return A


class GCNLayer(nn.Module):
    """
    One GCN layer with symmetric normalization, implemented via sparse matmul.

    y = ReLU( A_hat @ (xW) )
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        A_hat = _build_gcn_norm_adj(edge_index, num_nodes)  # sparse [N, N]
        xw = self.lin(x)                                    # [N, out_dim]
        y = torch.sparse.mm(A_hat, xw)                      # [N, out_dim]
        return F.relu(y, inplace=True)


# ----------------------------- GAT (single head) -------------------------------------

class GATLayer(nn.Module):
    """
    Single-head Graph Attention Layer (GAT) without torch-geometric.

    Implementation notes
    --------------------
    - Input edge_index is [2, E] with rows [dst, src] (messages src → dst).
    - We compute attention coefficients α_{i←j} using:
        e_ij = LeakyReLU( a^T [W x_i || W x_j] )
        α_ij = softmax_over_j(e_ij) for fixed i (destination).
      and output:
        y_i = Σ_j α_ij * (W x_j)
    - For simplicity/compatibility, we implement the softmax per destination node
      using a Python loop over unique destinations. This is acceptable for mid-sized
      graphs typical in WDE tiles; for very large graphs upstream batching is advised.
    """

    def __init__(self, in_dim: int, out_dim: int, neg_slope: float = 0.2, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(2 * out_dim))  # a vector for [Wh_i || Wh_j]
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.leaky_relu = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        # Xavier init for attention as in GAT
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        Wh = self.lin(x)  # [N, out_dim]
        dst, src = edge_index  # [E], [E]

        # Compute raw attention scores e_ij per edge
        Wh_i = Wh[dst]  # [E, out_dim]
        Wh_j = Wh[src]  # [E, out_dim]
        cat = torch.cat([Wh_i, Wh_j], dim=-1)  # [E, 2*out_dim]
        e = self.leaky_relu(torch.mv(cat, self.att))  # [E]

        # Softmax over incoming edges for each destination i
        y = torch.zeros_like(Wh)
        # Group edges by destination
        unique_dst, inverse = torch.unique(dst, return_inverse=True)
        # Loop over unique destinations (acceptable for mid-sized E)
        for g, i_node in enumerate(unique_dst.tolist()):
            mask = (inverse == g)
            e_i = e[mask]                           # [E_i]
            src_i = src[mask]                       # [E_i]
            # softmax
            e_max = torch.max(e_i)
            alpha = torch.exp(e_i - e_max)
            alpha = alpha / torch.clamp(alpha.sum(), min=1e-12)
            # aggregate
            msg = (alpha.unsqueeze(1) * Wh[src_i]).sum(dim=0)  # [out_dim]
            y[i_node] = msg

        if self.bias is not None:
            y = y + self.bias
        return F.elu(y, inplace=True)


# ----------------------------- GNN stack wrapper -------------------------------------

class GNNStack(nn.Module):
    """
    Stack of GCN or GAT layers with optional dropout, LayerNorm, and residuals.
    """

    def __init__(self, gnn_type: str, in_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.0, layernorm: bool = True, residual: bool = True):
        super().__init__()
        assert num_layers >= 1
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layernorm = layernorm
        self.residual = residual

        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            a, b = dims[i], dims[i + 1]
            if self.gnn_type == "gcn":
                layers.append(GCNLayer(a, b))
            elif self.gnn_type == "gat":
                layers.append(GATLayer(a, b))
            else:
                raise ValueError(f"Unknown gnn_type: {self.gnn_type}")
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)]) if layernorm else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        for li, layer in enumerate(self.layers):
            h = layer(x, edge_index, num_nodes=n)
            if self.layernorm:
                h = self.norms[li](h)
            h = self.dropout(h)
            if self.residual and h.shape == x.shape:
                x = x + h
            else:
                x = h
        return x


# ======================================================================================
# Full model
# ======================================================================================

class GNNFusion(nn.Module):
    """
    Multi-sensor → fusion → GNN → heads.

    Inputs (forward)
    ----------------
    features : Dict[str, Tensor]
        Mapping modality → tensor of shape [N, dim]. Only modalities present are used.
    edge_index : Tensor
        Long tensor of shape [2, E] in (dst, src) format.
    return_embeddings : bool
        If True, include "emb" key in outputs (post-GNN embeddings).

    Outputs (dict)
    --------------
    {
      "emb":       [N, H]                (always returned, unless return_embeddings=False)
      "logits":    [N, C]                (if num_classes > 0)
      "reg":       [N, R]                (if regression_targets > 0)
      "log_var":   [N, R]                (if predict_log_var and regression head is enabled)
    }
    """

    def __init__(self, cfg: GNNFusionConfig):
        super().__init__()
        self.cfg = cfg
        set_global_seed(cfg.seed)

        # 1) Build per-modality encoders
        self.hidden_dim = cfg.hidden_dim
        self.encoders = nn.ModuleDict()
        for name, in_dim in cfg.modalities.items():
            self.encoders[name] = ModalityEncoder(
                in_dim=in_dim, out_dim=cfg.hidden_dim, depth=cfg.enc_depth, hidden=cfg.enc_hidden
            )

        # 2) Fusion
        self.fusion = FusionBlock(
            fusion_type=cfg.fusion_type,
            hidden_dim=cfg.hidden_dim,
            fusion_hidden=cfg.fusion_hidden,
            dropout=cfg.fusion_dropout,
        )

        # 3) GNN stack
        self.gnn = GNNStack(
            gnn_type=cfg.gnn_type,
            in_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            layernorm=cfg.layernorm,
            residual=cfg.residual,
        )

        # 4) Heads (optional)
        self.cls_head = None
        if cfg.num_classes and cfg.num_classes > 0:
            self.cls_head = MLP(
                in_dim=cfg.hidden_dim, out_dim=cfg.num_classes,
                hidden=cfg.hidden_dim, depth=2, dropout=cfg.dropout,
                layernorm=True, final_activation=None
            )
        self.reg_head = None
        self.logvar_head = None
        if cfg.regression_targets and cfg.regression_targets > 0:
            self.reg_head = MLP(
                in_dim=cfg.hidden_dim, out_dim=cfg.regression_targets,
                hidden=cfg.hidden_dim, depth=2, dropout=cfg.dropout,
                layernorm=True, final_activation=None
            )
            if cfg.predict_log_var:
                self.logvar_head = MLP(
                    in_dim=cfg.hidden_dim, out_dim=cfg.regression_targets,
                    hidden=cfg.hidden_dim, depth=2, dropout=cfg.dropout,
                    layernorm=True, final_activation=None
                )

    # ----------------------------- Forward --------------------------------------------

    def encode_modalities(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode available modality features → shared hidden_dim. Missing modalities are ignored.
        """
        emb: Dict[str, torch.Tensor] = {}
        for name, enc in self.encoders.items():
            if name in features:
                x = features[name]
                if x.dim() != 2 or x.shape[1] != enc.mlp.net[0].in_features:
                    raise ValueError(
                        f"Modality '{name}' expected shape [N, {enc.mlp.net[0].in_features}], "
                        f"got {tuple(x.shape)}."
                    )
                emb[name] = enc(x)
        if not emb:
            raise ValueError("No known modalities present in 'features'.")
        return emb

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        return_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # 1) Encode each available modality
        emb_dict = self.encode_modalities(features)  # {name: [N, H]}

        # 2) Fuse to a single embedding per node
        z0 = self.fusion(emb_dict)                  # [N, H]

        # 3) Graph propagation
        z = self.gnn(z0, edge_index)                # [N, H]

        # 4) Heads
        out: Dict[str, torch.Tensor] = {}
        if return_embeddings:
            out["emb"] = z
        if self.cls_head is not None:
            out["logits"] = self.cls_head(z)
        if self.reg_head is not None:
            out["reg"] = self.reg_head(z)
            if self.logvar_head is not None:
                out["log_var"] = self.logvar_head(z)
        return out

    # ----------------------------- Save / Load ----------------------------------------

    def save(self, dir_path: str) -> None:
        """
        Persist config and model weights. Layout:
          dir/
            config.json
            model.pt
        """
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        torch.save(self.state_dict(), os.path.join(dir_path, "model.pt"))

    @classmethod
    def load(cls, dir_path: str, map_location: Optional[str] = None) -> "GNNFusion":
        """
        Load config and model weights from a directory created by save().
        """
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = GNNFusionConfig(**cfg_dict)
        model = cls(cfg)
        state = torch.load(os.path.join(dir_path, "model.pt"), map_location=map_location or "cpu")
        model.load_state_dict(state)
        return model


# ======================================================================================
# Graph builders — kNN, radius, temporal
# ======================================================================================

def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_knn_graph(coords: Union[torch.Tensor, np.ndarray], k: int = 8) -> torch.Tensor:
    """
    Build a directed k-NN graph (dst, src) with mutualized edges (adds both directions).

    Parameters
    ----------
    coords : [N, D]
        Node coordinates (e.g., lon/lat in projected space or pixel centers).
    k : int
        Number of neighbors per node (excluding self). Effective edges ~= 2 * N * k.

    Returns
    -------
    edge_index : LongTensor [2, E]
        Edges in (dst, src) format.
    """
    X = _to_numpy(coords)
    N = X.shape[0]
    if _SKLEARN_AVAILABLE:
        nn = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm="auto", metric="euclidean")
        nn.fit(X)
        dists, inds = nn.kneighbors(X, return_distance=True)  # inds: [N, k+1]
        # Exclude self (first neighbor) and create edges i <- j
        src_list: List[int] = []
        dst_list: List[int] = []
        for i in range(N):
            for j in inds[i, 1:]:
                dst_list.append(i)
                src_list.append(int(j))
                # mutual edge
                dst_list.append(int(j))
                src_list.append(i)
    else:
        # Fallback: O(N^2) distance (fine for moderate N)
        d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)  # [N, N]
        src_list, dst_list = [], []
        for i in range(N):
            idx = np.argpartition(d2[i], kth=min(k + 1, N - 1))[: k + 1]  # includes i
            idx = idx[idx != i]
            idx = idx[:k]
            for j in idx:
                dst_list.append(i)
                src_list.append(int(j))
                dst_list.append(int(j))
                src_list.append(i)

    edge_index = torch.tensor([dst_list, src_list], dtype=torch.long)
    return edge_index


def build_radius_graph(coords: Union[torch.Tensor, np.ndarray], radius: float) -> torch.Tensor:
    """
    Build undirected radius graph with edges between nodes within 'radius' (Euclidean).

    Returns
    -------
    edge_index : LongTensor [2, E]
    """
    X = _to_numpy(coords)
    N = X.shape[0]
    src_list, dst_list = [], []
    # Naive O(N^2) radius query (acceptable for tiles); upstream batching recommended if needed.
    for i in range(N):
        d2 = np.sum((X[i] - X) ** 2, axis=-1)
        idx = np.where((d2 > 0) & (d2 <= radius * radius))[0]
        for j in idx:
            dst_list.append(i); src_list.append(int(j))
            dst_list.append(int(j)); src_list.append(i)
    return torch.tensor([dst_list, src_list], dtype=torch.long)


def build_temporal_edges(timestamps: Union[torch.Tensor, np.ndarray], dt_max: float) -> torch.Tensor:
    """
    Connect nodes whose absolute time difference is <= dt_max (undirected).

    Parameters
    ----------
    timestamps : [N] (float or int)
        Per-node timestamps (e.g., days since epoch).
    dt_max : float
        Maximum allowed separation for an edge.

    Returns
    -------
    edge_index : LongTensor [2, E]
    """
    t = _to_numpy(timestamps).reshape(-1)
    N = t.shape[0]
    src_list, dst_list = [], []
    for i in range(N):
        dt = np.abs(t[i] - t)
        idx = np.where((dt > 0) & (dt <= dt_max))[0]
        for j in idx:
            dst_list.append(i); src_list.append(int(j))
            dst_list.append(int(j)); src_list.append(i)
    return torch.tensor([dst_list, src_list], dtype=torch.long)


# ======================================================================================
# Optional: losses for quick experimentation (not required for inference)
# ======================================================================================

def heteroscedastic_gaussian_nll(y_pred: torch.Tensor, y_true: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Bin-wise heteroscedastic Gaussian negative log-likelihood:
      L = 0.5 * [ log σ^2 + (y - μ)^2 / σ^2 ]
    where σ^2 = exp(log_var). Reduction: mean over batch and targets.
    """
    var = torch.exp(torch.clamp(log_var, min=-10.0, max=10.0))
    return 0.5 * (log_var + (y_true - y_pred) ** 2 / torch.clamp(var, min=1e-12)).mean()


# ======================================================================================
# Self-test (CPU) — safe to remove if undesired
# ======================================================================================

if __name__ == "__main__":
    # Minimal smoke test on synthetic data
    set_global_seed(123)

    device = torch.device("cpu")
    N = 512
    feats = {
        "sentinel2": torch.randn(N, 18, device=device),
        "sentinel1": torch.randn(N, 8, device=device),
        "soil":      torch.randn(N, 10, device=device),
    }
    coords = torch.rand(N, 2, device=device)
    edge_index = build_knn_graph(coords, k=6).to(device)

    cfg = GNNFusionConfig(
        modalities={"sentinel2": 18, "sentinel1": 8, "soil": 10},
        fusion_type="gate",
        gnn_type="gcn",
        hidden_dim=128,
        num_layers=2,
        dropout=0.05,
        num_classes=3,
        regression_targets=1,
        predict_log_var=True,
        seed=123,
    )
    model = GNNFusion(cfg).to(device)
    model.eval()

    with torch.no_grad():
        out = model(feats, edge_index=edge_index)
        print("emb:", tuple(out["emb"].shape))
        print("logits:", tuple(out["logits"].shape) if "logits" in out else None)
        print("reg:", tuple(out["reg"].shape) if "reg" in out else None)
        print("log_var:", tuple(out["log_var"].shape) if "log_var" in out else None)

    # Quick training-like step (optional)
    model.train()
    y_cls = torch.randint(0, cfg.num_classes, (N,), device=device)
    y_reg = torch.randn(N, cfg.regression_targets, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(3):
        optim.zero_grad(set_to_none=True)
        out = model(feats, edge_index=edge_index)
        loss = torch.tensor(0.0, device=device)

        if "logits" in out:
            loss = loss + F.cross_entropy(out["logits"], y_cls)

        if "reg" in out:
            if cfg.predict_log_var and "log_var" in out:
                loss = loss + heteroscedastic_gaussian_nll(out["reg"], y_reg, out["log_var"])
            else:
                loss = loss + F.mse_loss(out["reg"], y_reg)

        loss.backward()
        optim.step()
        print(f"[step {step}] loss = {loss.item():.4f}")

    # Save / load sanity check
    tmp_dir = "./_gnn_fusion_tmp"
    model.save(tmp_dir)
    model2 = GNNFusion.load(tmp_dir)
    model2.eval()
    with torch.no_grad():
        out2 = model2({k: v.cpu() for k, v in feats.items()}, edge_index=edge_index.cpu())
        print("Reload OK:", tuple(out2["emb"].shape))
```

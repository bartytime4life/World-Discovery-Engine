```python
# /models/uncertainty_bgnn.py
# ======================================================================================
# World Discovery Engine (WDE)
# UncertaintyBGNN — Bayesian Graph Neural Network with Aleatoric + Epistemic Uncertainty
# --------------------------------------------------------------------------------------
# This module implements a lightweight, dependency-minimal Bayesian GNN in pure PyTorch,
# suitable for Kaggle/CI environments without torch-geometric. It supports:
#
#   • Graph Convolutional layers (GCN) with either deterministic Linear or Bayesian
#     mean-field (variational) Linear (BayesLinear) weights.
#   • Heteroscedastic regression head: predicts μ and log σ² (aleatoric uncertainty).
#   • Classification head: Monte Carlo (MC) sampling for predictive probabilities,
#     with Expected Calibration Error (ECE) utility.
#   • Epistemic uncertainty via MC dropout and/or weight sampling from BayesLinear.
#   • ELBO training with KL regularization (β-annealing optional).
#   • kNN / radius graph builders and sparse normalized adjacency without PyG.
#
# Design goals
# ------------
# - Pure PyTorch, no torch-geometric; graphs represented by (dst, src) edge_index.
# - Practical defaults, reproducibility by seed, clear save/load helpers.
# - Works out of the box for node-level classification or regression.
#
# Typical usage
# -------------
#   from models.uncertainty_bgnn import (
#       UncertaintyBGNN, UncertaintyBGNNConfig, build_knn_graph,
#       heteroscedastic_gaussian_nll
#   )
#
#   X = torch.randn(N, D)
#   edge_index = build_knn_graph(coords, k=8)    # [2, E], (dst, src)
#
#   cfg = UncertaintyBGNNConfig(
#       in_dim=D, hidden_dim=128, num_layers=2,
#       task="regression", regression_targets=1, bayes_layers=True
#   )
#   model = UncertaintyBGNN(cfg).to(device)
#
#   # Training step (ELBO for Bayes layers + NLL for task)
#   outs = model(X, edge_index=edge_index, sample=True)  # sample weights for ELBO
#   nll = outs["nll"]  # computed only if y_true passed to loss helpers (see below)
#   kl  = model.kl_divergence()
#   loss = nll + beta * kl
#
# License
# -------
# MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# Global utilities
# ======================================================================================

def set_global_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ======================================================================================
# Simple graph utilities (no torch-geometric)
# ======================================================================================

def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
    idx = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([idx, idx], dim=0)  # [2, N] (dst=i, src=i)
    return torch.cat([edge_index, self_loops], dim=1)


def _symmetrize(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def build_norm_adj(edge_index: torch.Tensor, num_nodes: int, add_loops: bool = True, symmetrize: bool = True) -> torch.Tensor:
    """
    Return A_hat = D^{-1/2} A D^{-1/2} as a coalesced sparse COO tensor [N, N].
    edge_index format is [2, E] with rows [dst, src].
    """
    ei = edge_index
    if symmetrize:
        ei = _symmetrize(ei)
    if add_loops:
        ei = _add_self_loops(ei, num_nodes)
    values = torch.ones(ei.shape[1], device=ei.device, dtype=torch.float32)

    deg = torch.zeros(num_nodes, device=ei.device, dtype=torch.float32)
    deg.index_add_(0, ei[0], values)  # sum incoming per dst
    norm = values / torch.sqrt(torch.clamp(deg[ei[0]] * deg[ei[1]], min=1e-12))
    A = torch.sparse_coo_tensor(ei, norm, size=(num_nodes, num_nodes)).coalesce()
    return A


def build_knn_graph(coords: Union[torch.Tensor, np.ndarray], k: int = 8) -> torch.Tensor:
    """
    Build an undirected kNN graph (mutualized) in (dst, src) format.
    """
    X = _to_numpy(coords)
    N = X.shape[0]
    # Fast path with sklearn if present
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn_s = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm="auto", metric="euclidean")
        nn_s.fit(X)
        _, inds = nn_s.kneighbors(X, return_distance=True)
        src, dst = [], []
        for i in range(N):
            for j in inds[i, 1:]:
                dst.append(i); src.append(int(j))
                dst.append(int(j)); src.append(i)
    except Exception:
        # O(N^2) fallback
        d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        src, dst = [], []
        for i in range(N):
            idx = np.argpartition(d2[i], kth=min(k + 1, N - 1))[: k + 1]
            idx = idx[idx != i][:k]
            for j in idx:
                dst.append(i); src.append(int(j))
                dst.append(int(j)); src.append(i)

    return torch.tensor([dst, src], dtype=torch.long)


def build_radius_graph(coords: Union[torch.Tensor, np.ndarray], radius: float) -> torch.Tensor:
    X = _to_numpy(coords)
    N = X.shape[0]
    src, dst = [], []
    r2 = radius * radius
    for i in range(N):
        d2 = np.sum((X[i] - X) ** 2, axis=-1)
        idx = np.where((d2 > 0) & (d2 <= r2))[0]
        for j in idx:
            dst.append(i); src.append(int(j))
            dst.append(int(j)); src.append(i)
    return torch.tensor([dst, src], dtype=torch.long)


# ======================================================================================
# Variational Bayesian Linear layer (mean-field)
# ======================================================================================

class BayesLinear(nn.Module):
    """
    Mean-field Bayesian Linear layer with Gaussian prior N(0, σ_p^2 I) and
    variational posterior N(μ, σ^2) with σ = softplus(ρ).

    During forward:
      - If 'sample=True', draws weight & bias samples via reparameterization.
      - Else uses the mean parameters (μ) for deterministic forward (MAP).

    KL divergence to the prior is accumulated and can be retrieved via .kl.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.prior_var = prior_sigma ** 2

        # Posterior parameters
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.b_mu = nn.Parameter(torch.empty(out_features))
            self.b_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("b_mu", None)
            self.register_parameter("b_rho", None)

        self.reset_parameters()
        self._kl = None  # cached on last forward when sample=True

    def reset_parameters(self) -> None:
        # He initialization for μ; ρ initialized so σ ~ small
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        nn.init.constant_(self.w_rho, -5.0)
        if self.b_mu is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_mu, -bound, bound)
            nn.init.constant_(self.b_rho, -5.0)

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w_sigma = self._softplus(self.w_rho)
            eps_w = torch.randn_like(self.w_mu)
            W = self.w_mu + w_sigma * eps_w

            if self.b_mu is not None:
                b_sigma = self._softplus(self.b_rho)
                eps_b = torch.randn_like(self.b_mu)
                b = self.b_mu + b_sigma * eps_b
            else:
                b = None

            # KL divergence: KL(q||p) for factorized Gaussians
            # KL(N(μ, σ^2) || N(0, σ_p^2)) = log(σ_p/σ) + (σ^2 + μ^2)/(2σ_p^2) - 1/2
            kl_w = torch.log(self.prior_sigma / w_sigma) + (w_sigma.pow(2) + self.w_mu.pow(2)) / (2 * self.prior_var) - 0.5
            kl = kl_w.sum()
            if self.b_mu is not None:
                kl_b = torch.log(self.prior_sigma / b_sigma) + (b_sigma.pow(2) + self.b_mu.pow(2)) / (2 * self.prior_var) - 0.5
                kl = kl + kl_b.sum()
            self._kl = kl
        else:
            W = self.w_mu
            b = self.b_mu
            self._kl = None

        return F.linear(x, W, b)

    def kl(self) -> torch.Tensor:
        if self._kl is None:
            return torch.tensor(0.0, device=self.w_mu.device)
        return self._kl


# ======================================================================================
# GCN layers (deterministic and Bayesian)
# ======================================================================================

class GCNLayerDet(nn.Module):
    """
    Deterministic GCN layer: y = ReLU( A_hat @ (X W) ).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        A = build_norm_adj(edge_index, N)  # sparse
        xw = self.lin(x)
        y = torch.sparse.mm(A, xw)
        return F.relu(y, inplace=True)


class GCNLayerBayes(nn.Module):
    """
    Bayesian GCN layer: y = ReLU( A_hat @ (X W̃) ), where W̃ ~ q(W) (mean-field).
    """

    def __init__(self, in_dim: int, out_dim: int, prior_sigma: float = 0.1, bias: bool = True):
        super().__init__()
        self.blin = BayesLinear(in_dim, out_dim, prior_sigma=prior_sigma, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sample: bool = True) -> torch.Tensor:
        N = x.shape[0]
        A = build_norm_adj(edge_index, N)  # sparse
        xw = self.blin(x, sample=sample)
        y = torch.sparse.mm(A, xw)
        return F.relu(y, inplace=True)

    def kl(self) -> torch.Tensor:
        return self.blin.kl()


# ======================================================================================
# Config & Model
# ======================================================================================

@dataclass
class UncertaintyBGNNConfig:
    """
    Configuration for UncertaintyBGNN.

    Core
    ----
    in_dim : int
        Node feature dimension.
    hidden_dim : int
        GNN hidden dimension.
    num_layers : int
        Number of GCN layers.

    Bayesian
    --------
    bayes_layers : bool
        If True, uses Bayesian GCN layers (mean-field). Else deterministic GCN.
    prior_sigma : float
        Prior std-dev for BayesLinear in Bayesian layers.

    Dropout
    -------
    dropout : float
        Dropout after each GCN layer (also acts as MC dropout at inference if mc_dropout=True).
    mc_dropout : bool
        If True, keeps dropout active in eval mode during MC inference.

    Task
    ----
    task : str
        "classification" or "regression".
    num_classes : int
        Number of classes (classification). Ignored otherwise.
    regression_targets : int
        Number of regression targets (regression). Ignored otherwise.
    predict_log_var : bool
        If True in regression, predicts log σ² (aleatoric uncertainty).

    Training helpers
    ----------------
    seed : int
        RNG seed for reproducibility.
    """
    # Core
    in_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2

    # Bayesian
    bayes_layers: bool = True
    prior_sigma: float = 0.1

    # Dropout
    dropout: float = 0.0
    mc_dropout: bool = True

    # Task
    task: str = "regression"         # {"classification", "regression"}
    num_classes: int = 2
    regression_targets: int = 1
    predict_log_var: bool = True

    # Training helpers
    seed: int = 42


class UncertaintyBGNN(nn.Module):
    """
    Bayesian Graph Neural Network with optional aleatoric head.

    Forward outputs
    ---------------
    For task="classification":
        {
          "emb":      [N, H],           # node embeddings
          "logits":   [N, C],           # class logits (one MC sample or mean-of-samples if avg=True)
        }
    For task="regression":
        {
          "emb":      [N, H],
          "mu":       [N, R],           # predicted mean
          "log_var":  [N, R]?           # predicted log-variance if predict_log_var=True
        }

    KL divergence
    -------------
    - If Bayesian layers are used, call .kl_divergence() after a forward (with sample=True)
      to obtain the sum of KL terms for the current stochastic forward pass.
    """

    def __init__(self, cfg: UncertaintyBGNNConfig):
        super().__init__()
        self.cfg = cfg
        set_global_seed(cfg.seed)

        layers: List[nn.Module] = []
        dims = [cfg.in_dim] + [cfg.hidden_dim] * cfg.num_layers
        for i in range(cfg.num_layers):
            in_d, out_d = dims[i], dims[i + 1]
            if cfg.bayes_layers:
                layers.append(GCNLayerBayes(in_d, out_d, prior_sigma=cfg.prior_sigma, bias=True))
            else:
                layers.append(GCNLayerDet(in_d, out_d))
        self.gnn = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_layers)])
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        # Heads
        if cfg.task == "classification":
            self.cls_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
            self.reg_head = None
            self.logvar_head = None
        elif cfg.task == "regression":
            self.cls_head = None
            self.reg_head = nn.Linear(cfg.hidden_dim, cfg.regression_targets)
            self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.regression_targets) if cfg.predict_log_var else None
        else:
            raise ValueError("cfg.task must be 'classification' or 'regression'.")

    # ----- Core forward ---------------------------------------------------------------

    def _forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor, sample: bool) -> torch.Tensor:
        h = x
        for li, layer in enumerate(self.gnn):
            if isinstance(layer, GCNLayerBayes):
                h = layer(h, edge_index, sample=sample)
            else:
                h = layer(h, edge_index)
            h = self.norms[li](h)
            h = self.dropout(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sample: bool = True,
        return_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Single stochastic forward pass (if sample=True) or deterministic (sample=False).

        Note: For MC inference, call `predict_mc()` which averages T samples.
        """
        # Enable MC dropout during eval if requested
        if self.cfg.mc_dropout:
            self.train()  # keeps dropout active
        else:
            self.eval()

        z = self._forward_gnn(x, edge_index, sample=sample)
        out: Dict[str, torch.Tensor] = {}
        if return_embeddings:
            out["emb"] = z

        if self.cfg.task == "classification":
            out["logits"] = self.cls_head(z)
        else:
            mu = self.reg_head(z)
            out["mu"] = mu
            if self.logvar_head is not None:
                out["log_var"] = self.logvar_head(z)
        return out

    # ----- KL divergence --------------------------------------------------------------

    def kl_divergence(self) -> torch.Tensor:
        """
        Sum of KL divergences for all Bayesian layers for the last stochastic forward.
        Call this after a forward(sample=True). If sample=False or deterministic layers,
        returns 0.
        """
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.gnn:
            if isinstance(layer, GCNLayerBayes):
                kl = kl + layer.kl()
        return kl

    # ----- Monte Carlo prediction -----------------------------------------------------

    @torch.no_grad()
    def predict_mc(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        T: int = 30,
        reduce: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run T stochastic passes and aggregate predictions.

        For classification:
          Returns {"probs": [N, C], "probs_std": [N, C], "logits_mean": [N, C]}

        For regression:
          Returns {"mu": [N, R], "std": [N, R], "aleatoric_std": [N, R], "epistemic_std": [N, R]}
            where:
              - total variance = epistemic_var + aleatoric_var
              - aleatoric_var = mean(exp(log_var)) if enabled, else 0
              - epistemic_var = var(μ) across samples
        """
        device = next(self.parameters()).device
        preds = []

        if self.cfg.task == "classification":
            for _ in range(T):
                out = self.forward(x, edge_index, sample=True, return_embeddings=False)
                probs = F.softmax(out["logits"], dim=-1)
                preds.append(probs.unsqueeze(0))  # [1, N, C]
            P = torch.cat(preds, dim=0)  # [T, N, C]
            probs_mean = P.mean(dim=0)
            probs_std = P.std(dim=0, unbiased=False)
            logits_mean = torch.log(torch.clamp(probs_mean, 1e-12, 1.0))
            return {"probs": probs_mean, "probs_std": probs_std, "logits_mean": logits_mean}

        # Regression
        mu_list, var_list = [], []
        for _ in range(T):
            out = self.forward(x, edge_index, sample=True, return_embeddings=False)
            mu_list.append(out["mu"].unsqueeze(0))  # [1, N, R]
            if self.logvar_head is not None and "log_var" in out:
                var_list.append(torch.exp(torch.clamp(out["log_var"], min=-10.0, max=10.0)).unsqueeze(0))
        MU = torch.cat(mu_list, dim=0)  # [T, N, R]
        mu_mean = MU.mean(dim=0)
        mu_var = MU.var(dim=0, unbiased=False)  # epistemic variance
        if var_list:
            V = torch.cat(var_list, dim=0)      # [T, N, R]
            aleatoric_var = V.mean(dim=0)
        else:
            aleatoric_var = torch.zeros_like(mu_mean)
        total_var = mu_var + aleatoric_var
        return {
            "mu": mu_mean,
            "std": torch.sqrt(torch.clamp(total_var, min=1e-12)),
            "aleatoric_std": torch.sqrt(torch.clamp(aleatoric_var, min=1e-12)),
            "epistemic_std": torch.sqrt(torch.clamp(mu_var, min=1e-12)),
        }

    # ----- Save / Load ----------------------------------------------------------------

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        torch.save(self.state_dict(), os.path.join(dir_path, "model.pt"))

    @classmethod
    def load(cls, dir_path: str, map_location: Optional[str] = None) -> "UncertaintyBGNN":
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            cfg = UncertaintyBGNNConfig(**json.load(f))
        model = cls(cfg)
        state = torch.load(os.path.join(dir_path, "model.pt"), map_location=map_location or "cpu")
        model.load_state_dict(state)
        return model


# ======================================================================================
# Losses & metrics
# ======================================================================================

def heteroscedastic_gaussian_nll(mu: torch.Tensor, y: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Gaussian NLL with learned per-target log variance:
      L = 0.5 * [ log σ^2 + (y - μ)^2 / σ^2 ]
    """
    var = torch.exp(torch.clamp(log_var, min=-10.0, max=10.0))
    return 0.5 * (log_var + (y - mu) ** 2 / torch.clamp(var, min=1e-12)).mean()


def expected_calibration_error(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """
    Expected Calibration Error (ECE) for multi-class probabilities.
      probs: [N, C], y_true: [N] with integer labels.
    """
    with torch.no_grad():
        conf, pred = probs.max(dim=1)  # [N]
        correct = (pred == y_true).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.zeros([], device=probs.device)
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.any():
                acc = correct[m].mean()
                avg_conf = conf[m].mean()
                w = m.float().mean()
                ece = ece + w * torch.abs(avg_conf - acc)
        return ece


# ======================================================================================
# Training helpers (optional)
# ======================================================================================

def elbo_loss_classification(
    model: UncertaintyBGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1.0,
    sample: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    ELBO = NLL + β * KL
      - NLL: Cross-entropy on a single stochastic forward.
      - KL: sum of KL terms across Bayesian layers.
    """
    out = model(x, edge_index, sample=sample, return_embeddings=False)
    nll = F.cross_entropy(out["logits"], y)
    kl = model.kl_divergence()
    loss = nll + beta * kl
    return loss, {"nll": float(nll.item()), "kl": float(kl.item())}


def elbo_loss_regression(
    model: UncertaintyBGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1.0,
    sample: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    ELBO = NLL + β * KL
      - NLL: Gaussian NLL with learned log variance if enabled, else MSE.
    """
    out = model(x, edge_index, sample=sample, return_embeddings=False)
    if "log_var" in out:
        nll = heteroscedastic_gaussian_nll(out["mu"], y, out["log_var"])
    else:
        nll = F.mse_loss(out["mu"], y)
    kl = model.kl_divergence()
    loss = nll + beta * kl
    return loss, {"nll": float(nll.item()), "kl": float(kl.item())}


# ======================================================================================
# Self-test (CPU) — synthetic demo
# ======================================================================================

if __name__ == "__main__":
    set_global_seed(123)
    device = torch.device("cpu")

    # Synthetic graph
    N, D = 400, 16
    coords = torch.rand(N, 2)
    edge_index = build_knn_graph(coords, k=6)

    # Node features & targets
    X = torch.randn(N, D)
    # Create latent class structure from first two dims + spatial proximity
    latent = X[:, 0] + 0.5 * X[:, 1] + 2.0 * (coords[:, 0] - coords[:, 1])
    y_cls = (latent > latent.median()).long()
    y_reg = latent.unsqueeze(1) + 0.1 * torch.randn(N, 1)

    # ---------------- Classification demo ----------------
    cfg_c = UncertaintyBGNNConfig(
        in_dim=D, hidden_dim=64, num_layers=2,
        bayes_layers=True, prior_sigma=0.1,
        dropout=0.1, mc_dropout=True,
        task="classification", num_classes=2,
        seed=7,
    )
    model_c = UncertaintyBGNN(cfg_c).to(device)
    opt = torch.optim.Adam(model_c.parameters(), lr=3e-3, weight_decay=1e-4)

    model_c.train()
    for step in range(50):
        opt.zero_grad(set_to_none=True)
        loss, logs = elbo_loss_classification(model_c, X, edge_index, y_cls, beta=1e-3, sample=True)
        loss.backward()
        opt.step()
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                pmc = model_c.predict_mc(X, edge_index, T=30)
                acc = (pmc["probs"].argmax(dim=1) == y_cls).float().mean().item()
                ece = expected_calibration_error(pmc["probs"], y_cls, n_bins=15).item()
            print(f"[cls step {step+1:03d}] loss={loss.item():.4f} nll={logs['nll']:.4f} kl={logs['kl']:.4f} acc={acc:.3f} ece={ece:.3f}")

    # ---------------- Regression demo --------------------
    cfg_r = UncertaintyBGNNConfig(
        in_dim=D, hidden_dim=64, num_layers=2,
        bayes_layers=True, prior_sigma=0.1,
        dropout=0.1, mc_dropout=True,
        task="regression", regression_targets=1, predict_log_var=True,
        seed=11,
    )
    model_r = UncertaintyBGNN(cfg_r).to(device)
    opt = torch.optim.Adam(model_r.parameters(), lr=3e-3, weight_decay=1e-4)

    model_r.train()
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        loss, logs = elbo_loss_regression(model_r, X, edge_index, y_reg, beta=1e-3, sample=True)
        loss.backward()
        opt.step()
        if (step + 1) % 12 == 0:
            with torch.no_grad():
                pmc = model_r.predict_mc(X, edge_index, T=30)
                rmse = torch.sqrt(F.mse_loss(pmc["mu"], y_reg)).item()
                # Check uncertainty decomposition
                estd = pmc["epistemic_std"].mean().item()
                astd = pmc["aleatoric_std"].mean().item()
            print(f"[reg step {step+1:03d}] loss={loss.item():.4f} nll={logs['nll']:.4f} kl={logs['kl']:.4f} rmse={rmse:.4f} epi_std={estd:.4f} alea_std={astd:.4f}")

    # ---------------- Save / Load smoke test -----------
    out_dir = "./_ubgnn_tmp"
    model_r.save(out_dir)
    reloaded = UncertaintyBGNN.load(out_dir)
    with torch.no_grad():
        pmc2 = reloaded.predict_mc(X, edge_index, T=10)
    print("Reload OK; μ shape:", tuple(pmc2["mu"].shape))
```

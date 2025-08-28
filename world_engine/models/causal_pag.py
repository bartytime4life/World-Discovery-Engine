```python
# models/causal_pag.py
# ======================================================================================
# World Discovery Engine (WDE)
# CausalPAG — Constraint-based causal discovery to a Partial Ancestral Graph (PAG)
# --------------------------------------------------------------------------------------
# What this is
# ------------
# A dependency-light implementation (NumPy-only, SciPy optional) of a small,
# production-friendly causal discovery routine that estimates a **PAG-style**
# graph from observational tabular data using conditional-independence (CI)
# tests with a Fisher-Z/partial-correlation test under (approx.) Gaussian
# assumptions.
#
# Key features
# ------------
# • PC/FCI-inspired skeleton discovery with sepsets (up to max_k conditioning size)
# • Collider orientation (A o-> B <-o C) using sepset(B ∉ S_{A,C})
# • Meek-style propagation (subset of rules) adapted to PAG marks
# • PAG edge marks with endpoints in {'o', '-', '>'} meaning:
#       '>'  : arrowhead at this endpoint (into the node)
#       '-'  : tail at this endpoint (out of the node)
#       'o'  : circle (uncertain endpoint; could be tail or arrowhead)
#   Examples:
#       A o-o B   (undetermined)
#       A o-> B   (arrowhead at B)
#       A -o B    (tail at A, circle at B)
#       A -> B    (tail at A, arrowhead at B)
# • Pluggable CI tester; default is Fisher-Z partial correlation
# • Background knowledge: forbidden/oriented edges can be supplied
# • DOT export for quick visualization
#
# What this is NOT
# ----------------
# • A full FCI implementation with all orientation rules (graph equivalence class
#   completeness in the presence of latent/confounding selection bias is very
#   involved). We implement a **conservative core** that works well in practice
#   on mid-sized problems and is safe-by-default for scientific pipelines.
#
# Typical usage
# -------------
#   import numpy as np
#   from models.causal_pag import CausalPAG
#
#   X = ...  # [N, D] continuous data (rows = samples, cols = variables)
#   names = ["NDVI","SOC","P","Slope","ADE"]
#
#   pag = CausalPAG(alpha=0.01, max_k=2, verbose=True)
#   pag.fit(X, var_names=names)
#   print(pag.edges_str())
#   print(pag.to_dot())
#
#   # Query: Is X -> Y oriented? What is the (conservative) adjustment suggestion?
#   print(pag.orientation("SOC", "ADE"))     # "SOC -> ADE" / "SOC o-> ADE" / etc.
#   print(pag.suggest_adjustment_set("SOC", "ADE"))
#
# CI Testing
# ----------
# • Default Fisher-Z partial correlation test (Gaussian; works surprisingly well).
# • You can supply a custom tester with signature:
#       def ci_test(i, j, cond_set_indices) -> (is_independent: bool, p_value: float)
#
# Background knowledge
# --------------------
# • Supply sets of forbidden edges or required arrowheads via `prior_knowledge=...`
#   when calling `fit()`. See docstring on `fit`.
#
# License
# -------
# MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    # SciPy is optional; if present we use the survival function for Fisher-Z p-values.
    from scipy.stats import norm as _scipy_norm  # type: ignore
    _SCIPY = True
except Exception:
    _SCIPY = False


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _standardize(X: np.ndarray) -> np.ndarray:
    """
    Z-score columns (variables). Avoids division-by-zero by adding small eps to std.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def _least_squares_residual(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Return residuals of regressing Y on Z (no intercept; Y and Z assumed standardized).
    Shapes:
      Y : [N]          (vector)
      Z : [N, k]       (matrix of covariates); if k == 0, returns Y.
    Uses numpy.linalg.lstsq with rcond=None for numerical stability.
    """
    if Z.size == 0:
        return Y
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ beta


def _partial_corr_fisher_z(
    Xz: np.ndarray,
    i: int,
    j: int,
    cond: Sequence[int],
) -> Tuple[float, float]:
    """
    Compute partial correlation rho_{ij·cond} and two-sided p-value using Fisher Z.

    Parameters
    ----------
    Xz : [N, D] standardized data
    i, j : int variable indices
    cond : list[int] conditioning set indices

    Returns
    -------
    (rho, p_value)
    """
    Yi = Xz[:, i]
    Yj = Xz[:, j]
    if len(cond) == 0:
        # Simple correlation
        r = float(np.clip(np.corrcoef(Yi, Yj)[0, 1], -0.999999, 0.999999))
    else:
        Z = Xz[:, cond]
        ri = _least_squares_residual(Yi, Z)
        rj = _least_squares_residual(Yj, Z)
        # Pearson on residuals
        denom = (np.linalg.norm(ri) * np.linalg.norm(rj))
        if denom < 1e-12:
            r = 0.0
        else:
            r = float(np.clip(ri.dot(rj) / denom, -0.999999, 0.999999))

    # Fisher-Z transform; under H0: r=0, Z ~ N(0, 1/sqrt(N - |S| - 3))
    # Two-sided p-value
    n = Xz.shape[0]
    dof = max(1, n - len(cond) - 3)
    if abs(r) >= 1.0:
        return r, 0.0
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    if _SCIPY:
        p = 2.0 * _scipy_norm.sf(abs(z))
    else:
        # Normal tail approximation (good enough without SciPy)
        # sf(x) ≈ (1/(x*sqrt(2π))) * exp(-x^2/2) for x>0; clamp for small x.
        x = abs(z)
        if x < 1e-8:
            p = 1.0
        else:
            tail = (1.0 / (x * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * x * x)
            p = min(1.0, 2.0 * tail)
    return r, float(p)


# --------------------------------------------------------------------------------------
# Graph representation (PAG with endpoint marks)
# --------------------------------------------------------------------------------------

Mark = str  # 'o', '-', '>'

@dataclass(frozen=True)
class Edge:
    """
    PAG edge with marks at both endpoints.
      u (m_u) --- (m_v) v
    Conventions:
      • m_u is the mark at the u-end; m_v at the v-end.
      • marks are in {'o','-','>'}
    """
    u: int
    v: int
    mu: Mark
    mv: Mark

    def as_tuple(self) -> Tuple[int, int, Mark, Mark]:
        return (self.u, self.v, self.mu, self.mv)

    def reversed(self) -> "Edge":
        return Edge(self.v, self.u, self.mv, self.mu)

    def __str__(self) -> str:
        # e.g., "A o-> B" / "A -> B" / "A o-o B"
        def seg(m: Mark) -> str:
            if m == '>':
                return '>'
            if m == '-':
                return '-'
            return 'o'
        left = seg(self.mu)
        right = seg(self.mv)
        # Compose a small ASCII:
        # tail '-' from u; arrow '>' into v; circle 'o' for uncertain
        # We show "u <mark> <mark> v"
        arrow = f"{left}{right}"
        # Map two-mark pair to a pretty connector (purely cosmetic)
        pair_to_str = {
            'oo': 'o-o',
            'o>': 'o->',
            '->': '->',
            '-o': '-o',
            '>o': '<-o',  # (rare to print)
            '>-': '<-',
        }
        conn = pair_to_str.get(arrow, f"{left}{right}")
        return f"{self.u} {conn} {self.v}"


class PAG:
    """
    Lightweight PAG structure over D variables (0..D-1).
    Internally stores edges in a dict with ordered (u < v) key.
    """

    def __init__(self, D: int):
        self.D = D
        # key: (min(u,v), max(u,v)) -> Edge with canonical ordering (u<v)
        self._edges: Dict[Tuple[int, int], Edge] = {}

    # ---- basic ops ----

    def has_edge(self, a: int, b: int) -> bool:
        u, v = (a, b) if a < b else (b, a)
        return (u, v) in self._edges

    def get_edge(self, a: int, b: int) -> Optional[Edge]:
        u, v = (a, b) if a < b else (b, a)
        e = self._edges.get((u, v))
        if e is None:
            return None
        return e if (a == u and b == v) else e.reversed()

    def add_undetermined(self, a: int, b: int) -> None:
        if a == b:
            return
        u, v = (a, b) if a < b else (b, a)
        if (u, v) not in self._edges:
            self._edges[(u, v)] = Edge(u, v, 'o', 'o')

    def remove_edge(self, a: int, b: int) -> None:
        u, v = (a, b) if a < b else (b, a)
        self._edges.pop((u, v), None)

    def orient(self, a: int, b: int, mark_a: Mark, mark_b: Mark) -> None:
        """
        Set endpoint marks for edge a—b. Creates edge if missing.
        Use only marks in {'o','-','>'}.
        """
        if a == b:
            return
        u, v = (a, b) if a < b else (b, a)
        mu, mv = (mark_a, mark_b) if (a == u and b == v) else (mark_b, mark_a)
        e = self._edges.get((u, v))
        if e is None:
            e = Edge(u, v, mu, mv)
        else:
            # combine marks "monotonically":
            # once an endpoint becomes '>' or '-', it should not be downgraded to 'o'.
            mu = _dominant_mark(e.mu, mu)
            mv = _dominant_mark(e.mv, mv)
            e = Edge(u, v, mu, mv)
        self._edges[(u, v)] = e

    def neighbors(self, a: int) -> List[int]:
        out: List[int] = []
        for (u, v), e in self._edges.items():
            if u == a:
                out.append(v)
            elif v == a:
                out.append(u)
        return sorted(set(out))

    def edges(self) -> List[Edge]:
        return list(self._edges.values())

    def copy(self) -> "PAG":
        g = PAG(self.D)
        g._edges = dict(self._edges)
        return g

    # ---- pretty ----

    def __str__(self) -> str:
        parts = []
        for e in self.edges():
            parts.append(str(e))
        return "\n".join(parts)

    def to_dot(self, names: Optional[List[str]] = None) -> str:
        """
        Export to Graphviz DOT. Arrowheads and tails are approximated:
          - tail '-' rendered as normal edge with dir=forward
          - arrow '>' rendered with arrowhead=normal
          - circle 'o' rendered with arrowhead=dot (as a visual cue)
        """
        if names is None:
            names = [f"X{i}" for i in range(self.D)]

        def end_attrs(mark: Mark) -> Dict[str, str]:
            if mark == '>':
                return {"arrowhead": "normal"}
            if mark == '-':
                return {"arrowhead": "none"}
            return {"arrowhead": "dot"}  # 'o' circle

        lines = ["digraph PAG {", '  graph [rankdir=LR];', '  node [shape=ellipse];']
        for i in range(self.D):
            lines.append(f'  {i} [label="{names[i]}"];')

        for e in self.edges():
            a, b, ma, mb = e.u, e.v, e.mu, e.mv
            # represent as two half-edges using dir=forward with proper arrowheads
            attr_ab = end_attrs(mb)
            attr_ba = end_attrs(ma)
            # We draw a single directed edge a -> b with head attributes = mark at b,
            # and decorate tail using label (Graphviz limitation). For better fidelity,
            # we draw two parallel edges with different arrowheads but low penwidth.
            lines.append(f'  {a} -> {b} [arrowhead={attr_ab["arrowhead"]}, color="#444444"];')
            lines.append(f'  {b} -> {a} [arrowhead={attr_ba["arrowhead"]}, color="#444444"];')
        lines.append("}")
        return "\n".join(lines)


def _dominant_mark(old: Mark, new: Mark) -> Mark:
    """
    Escalate endpoint marks monotonically: '>' and '-' dominate 'o'.
    If conflicting hard marks appear ('>' vs '-'), keep existing hard mark
    (conservative) and ignore the new one.
    """
    if old == new:
        return old
    if old == 'o':
        return new
    # old is hard, new is soft 'o' -> keep old
    if new == 'o':
        return old
    # both are hard but conflicting: keep old (conservative)
    return old


# --------------------------------------------------------------------------------------
# Causal discovery
# --------------------------------------------------------------------------------------

CITest = Callable[[int, int, Sequence[int]], Tuple[bool, float]]
# convention for CITest return if you implement your own:
#   returns (is_independent, p_value)

class CausalPAG:
    """
    Learn a (conservative) PAG from data using constraint-based discovery.

    Parameters
    ----------
    alpha : float
        Significance level for CI tests (Fisher-Z). If p >= alpha, treat as independent.
    max_k : int
        Maximum conditioning set size for skeleton search.
    ci_method : {"fisher_z"} or callable
        CI testing method. If callable, must implement signature of CITest but return
        (is_independent: bool, p_value: float). The provided callable will be wrapped
        to align with the internal convention.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_k: int = 2,
        ci_method: str | Callable[[int, int, Sequence[int]], Tuple[float, float]] = "fisher_z",
        verbose: bool = False,
    ):
        self.alpha = float(alpha)
        self.max_k = int(max_k)
        self.verbose = bool(verbose)

        self._Xz: Optional[np.ndarray] = None  # standardized data
        self.D: int = 0
        self.var_names: List[str] = []
        self.pag: Optional[PAG] = None
        # separating sets: key (i,j) with i<j -> set/list of indices S
        self.sepset: Dict[Tuple[int, int], Tuple[int, ...]] = {}

        if isinstance(ci_method, str):
            if ci_method.lower() != "fisher_z":
                raise ValueError("Only 'fisher_z' is built-in.")
            self._ci = self._ci_fisher_wrapper
        else:
            # user supplied returns (rho, p) or (is_independent, p)
            def _wrap(i: int, j: int, S: Sequence[int]) -> Tuple[bool, float]:
                out = ci_method(i, j, S)
                if not isinstance(out, tuple) or len(out) != 2:
                    raise ValueError("Custom CI must return (metric, p_value) or (bool, p_value).")
                metric, p = out
                if isinstance(metric, bool):
                    return metric, float(p)
                # interpret as correlation-like magnitude: independence if p>=alpha
                return (float(p) >= self.alpha, float(p))
            self._ci = _wrap

    # ------------------------------ Public API ----------------------------------------

    def fit(
        self,
        X: np.ndarray,
        var_names: Optional[Sequence[str]] = None,
        prior_knowledge: Optional[Dict[str, Iterable[Tuple[str, str]]]] = None,
    ) -> "CausalPAG":
        """
        Estimate a PAG from data.

        Parameters
        ----------
        X : np.ndarray [N, D]
            Continuous data (rows = samples).
        var_names : list[str], optional
            Names for variables; defaults to X0..X{D-1}.
        prior_knowledge : dict, optional
            Background constraints:
              {
                "forbid": [("A","B"), ...],      # forbid any edge between A and B
                "orient": [("A","B"), ...],      # enforce A -> B (tail at A, head at B)
                "no_head": [("A","B"), ...],     # forbid arrowhead into B from A (i.e., A ?- B)
                "no_tail": [("A","B"), ...],     # forbid tail at A towards B (i.e., A ?-> B disallowed)
              }
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D].")
        self._Xz = _standardize(X)
        self.D = X.shape[1]
        self.var_names = list(var_names) if var_names is not None else [f"X{i}" for i in range(self.D)]
        if len(self.var_names) != self.D:
            raise ValueError("var_names length must equal number of columns in X.")

        # 1) Build initial complete undirected PAG with 'o-o' marks
        G = PAG(self.D)
        for i in range(self.D):
            for j in range(i + 1, self.D):
                G.add_undetermined(i, j)

        # 2) Apply prior-knowledge hard constraints (forbid edges)
        if prior_knowledge and "forbid" in prior_knowledge:
            for a_name, b_name in prior_knowledge["forbid"]:
                a = self._idx(a_name); b = self._idx(b_name)
                G.remove_edge(a, b)

        # 3) Skeleton discovery (PC-like) with sepsets
        self.sepset = {}
        l = 0
        # adjacency list will be updated; we iterate by conditioning size l
        while True:
            changed = False
            # snapshot neighbor sets (undirected)
            adj = {v: [w for w in G.neighbors(v)] for v in range(self.D)}
            # For each pair (i, j) adjacent, try to find a separating set S with |S|=l
            for i in range(self.D):
                nbrs = adj[i]
                if len(nbrs) < 1:
                    continue
                for j in list(nbrs):
                    if j <= i:
                        continue
                    # If fewer than l neighbors (excluding j), skip
                    cand = [k for k in nbrs if k != j]
                    if len(cand) < l:
                        continue
                    found_sep = False
                    for S in itertools.combinations(cand, l):
                        indep, p = self._ci(i, j, S)
                        if indep:
                            # Remove edge i—j; record S as separating set
                            if self.verbose:
                                self._log(f"sep({self._n(i)},{self._n(j)}) with S={self._names(S)}  (p={p:.3g})")
                            self.sepset[(min(i, j), max(i, j))] = tuple(sorted(S))
                            G.remove_edge(i, j)
                            changed = True
                            found_sep = True
                            break
                    # If removed, no need to test other sets for (i, j)
                # end for j
            # end for i
            if not changed or l >= self.max_k:
                break
            l += 1

        # 4) Collider orientation: A *-* B *-* C, A and C nonadjacent, and B ∉ S_{A,C}
        #    Orient A o-> B <-o C (add arrowheads into B).
        for b in range(self.D):
            nbrs = G.neighbors(b)
            if len(nbrs) < 2:
                continue
            for a, c in itertools.combinations(nbrs, 2):
                if G.has_edge(a, c):
                    continue  # only unshielded triples
                S = self.sepset.get((min(a, c), max(a, c)))
                if S is None or (b not in S):
                    # orient a ?-> b <-? c
                    if G.has_edge(a, b):
                        G.orient(a, b, 'o', '>')  # arrowhead at b
                    if G.has_edge(c, b):
                        G.orient(c, b, 'o', '>')

        # 5) Meek-style propagation (subset of rules) adapted to PAG marks.
        #    We conservatively apply:
        #    R1: Orient v-structure chains: if A -> B and B o- C and A, C nonadjacent, orient B -o C as B -> C
        #    R2: A -> B and B -> C and A o- C  => orient A -> C  (avoid new collider at B)
        #    R3: A -o B, and there exists a directed path A -> ... -> B  => orient A -> B
        #        (we approximate with 2-hop lookahead for practicality)
        #    We iterate to closure or max iterations.
        self._propagate_orientations(G, max_iter=20)

        # 6) Apply additional prior knowledge (orients / endpoint forbids)
        if prior_knowledge:
            for key, pairs in prior_knowledge.items():
                if key == "orient":
                    for a_name, b_name in pairs:
                        a = self._idx(a_name); b = self._idx(b_name)
                        if G.has_edge(a, b):
                            G.orient(a, b, '-', '>')
                elif key == "no_head":
                    for a_name, b_name in pairs:
                        a = self._idx(a_name); b = self._idx(b_name)
                        e = G.get_edge(a, b)
                        if e is not None and ((a < b and e.mv == '>') or (b < a and e.mu == '>')):
                            # replace head with circle conservatively
                            G.orient(a, b, e.mu if a < b else 'o', 'o' if a < b else e.mv)
                elif key == "no_tail":
                    for a_name, b_name in pairs:
                        a = self._idx(a_name); b = self._idx(b_name)
                        e = G.get_edge(a, b)
                        if e is not None and ((a < b and e.mu == '-') or (b < a and e.mv == '-')):
                            G.orient(a, b, 'o' if a < b else e.mu, e.mv if a < b else 'o')

        self.pag = G
        if self.verbose:
            self._log("Discovery complete.")
        return self

    def orientation(self, a_name: str, b_name: str) -> str:
        """
        Human-readable orientation between a and b, e.g., "A -> B" / "A o-> B" / "A ⟂ B".
        """
        self._assert_fitted()
        a, b = self._idx(a_name), self._idx(b_name)
        e = self.pag.get_edge(a, b)  # type: ignore
        if e is None:
            return f"{a_name} ⟂ {b_name}"
        def mark_to_str(m: Mark, left: bool) -> str:
            if m == '>':
                return '<-' if left else '->'
            if m == '-':
                return '-' if left else '-'
            return 'o'
        # Compose: a <mark b-end> b with adjustment for printability
        left = 'o' if e.mu == 'o' else ('-' if e.mu == '-' else '<')
        right = 'o' if e.mv == 'o' else ('-' if e.mv == '-' else '>')
        pair = f"{left}{right}"
        mapping = {
            'oo': 'o-o',
            'o>': 'o->',
            '->': '->',
            '-o': '-o',
            '<o': '<-o',
            '<-': '<-',
        }
        s = mapping.get(pair, pair)
        return f"{a_name} {s} {b_name}"

    def edges_str(self) -> str:
        """
        Pretty multiline string of all edges with variable names.
        """
        self._assert_fitted()
        lines = []
        for e in self.pag.edges():  # type: ignore
            a, b = self._n(e.u), self._n(e.v)
            lines.append(self._edge_str_named(a, b, e.mu, e.mv))
        return "\n".join(lines)

    def to_dot(self) -> str:
        """
        Export current PAG to Graphviz DOT with variable names.
        """
        self._assert_fitted()
        return self.pag.to_dot(self.var_names)  # type: ignore

    def suggest_adjustment_set(self, treat: str, outcome: str) -> Tuple[Set[str], str]:
        """
        Heuristic, **conservative** adjustment suggestion.

        Rules (conservative):
        - If any incident circle at treat or ambiguous arcs on the treat→... path
          exist that could indicate latent confounding, return (∅, "ambiguous").
        - Else (treat has only tails out and arrowheads in), suggest adjusting for
          all current parents of treat (nodes with arrowhead into treat) **excluding**
          any descendants of treat (as far as we can infer with current orientations).

        Returns
        -------
        (Z, status)
          Z : set of variable names proposed for backdoor adjustment (may be empty).
          status : "ok" (confident), "ambiguous" (uncertain due to circles), or
                   "no_edge" (no path treat—outcome in current skeleton).
        """
        self._assert_fitted()
        G: PAG = self.pag  # type: ignore
        t = self._idx(treat); y = self._idx(outcome)
        # Quick reachability check in the undirected skeleton
        if not self._connected(G, t, y):
            return set(), "no_edge"

        # If endpoint marks touching treat have any circle, we deem ambiguous
        for n in G.neighbors(t):
            e = G.get_edge(t, n)
            if e is None:
                continue
            mark_at_t = e.mu if e.u == t else e.mv
            if mark_at_t == 'o':
                return set(), "ambiguous"

        # Collect parents of treat (arrowheads into treat)
        parents = set()
        for n in G.neighbors(t):
            e = G.get_edge(n, t)
            if e is None:
                continue
            # e is oriented from n -> t if mark at t is '>' and mark at n is '-' (or 'o' allowed?)
            mark_at_t = e.mv if (e.u == n and e.v == t) else e.mu
            if mark_at_t == '>':
                parents.add(self._n(n))

        # Exclude nodes that are already descendants of treat (if we have hard '->' paths).
        # With partial orientation we approximate using 2-hop search for t -> ... -> node.
        desc = self._descendants_hard(G, t, max_hops=3)
        Z = {p for p in parents if p not in desc}
        return Z, "ok"

    # ------------------------------ Internals -----------------------------------------

    def _ci_fisher_wrapper(self, i: int, j: int, S: Sequence[int]) -> Tuple[bool, float]:
        assert self._Xz is not None
        _, p = _partial_corr_fisher_z(self._Xz, i, j, S)
        return (p >= self.alpha), p

    def _propagate_orientations(self, G: PAG, max_iter: int = 20) -> None:
        """
        Apply a small set of Meek-style orientation rules adapted for PAG marks.
        Conservative: never flip a hard mark once set.
        """
        for _ in range(max_iter):
            changed = False

            # R1: Orient B—C if A -> B, B o- C, and A, C nonadjacent (unshielded), as B -> C.
            for B in range(self.D):
                # collect A s.t. A -> B
                Aset = []
                for A in G.neighbors(B):
                    eAB = G.get_edge(A, B)
                    if eAB is None:
                        continue
                    # A -> B if mark at B is '>' and at A is '-' (or 'o' is allowed on A for partial?)
                    if ((eAB.u == A and eAB.mv == '>') or (eAB.v == A and eAB.mu == '>')):
                        Aset.append(A)
                if not Aset:
                    continue
                for C in G.neighbors(B):
                    if any(C == a for a in Aset):
                        continue
                    eBC = G.get_edge(B, C)
                    if eBC is None:
                        continue
                    if G.has_edge(Aset[0], C):
                        continue  # shielded; skip strict R1
                    # if B o- C (mark at B is 'o' or '-', and at C is 'o')
                    # conservative: if B endpoint not yet head into C, set tail at B, head at C
                    mark_B_to_C = eBC.mu if eBC.u == B else eBC.mv
                    mark_C_to_B = eBC.mv if eBC.u == B else eBC.mu
                    if mark_C_to_B != '>':  # avoid creating C -> B
                        # set B -> C
                        mb_new_B = '-'  # tail at B
                        mb_new_C = '>'
                        # apply
                        pre = G.get_edge(B, C)
                        G.orient(B, C, mb_new_B, mb_new_C)
                        post = G.get_edge(B, C)
                        if pre is None or post is None or pre.as_tuple() != post.as_tuple():
                            changed = True

            # R2: If A -> B and B -> C and A o- C, orient A -> C
            for A in range(self.D):
                for B in G.neighbors(A):
                    eAB = G.get_edge(A, B)
                    if eAB is None:
                        continue
                    if not _is_directed(G, A, B):
                        continue
                    for C in G.neighbors(B):
                        if C == A:
                            continue
                        eBC = G.get_edge(B, C)
                        if eBC is None or not _is_directed(G, B, C):
                            continue
                        if not G.has_edge(A, C):
                            continue
                        eAC = G.get_edge(A, C)
                        if eAC is None:
                            continue
                        # A o- C or A -o C (not yet arrowhead at A or C)
                        markA = eAC.mu if eAC.u == A else eAC.mv
                        markC = eAC.mv if eAC.u == A else eAC.mu
                        if markA != '>' and markC != '<':  # approximate: no head into A
                            pre = eAC.as_tuple()
                            G.orient(A, C, '-', '>')
                            post = G.get_edge(A, C).as_tuple()  # type: ignore
                            if pre != post:
                                changed = True

            # R3 (approx): If A -o B and there exists D with A -> D and D -> B, orient A -> B
            for A in range(self.D):
                for B in G.neighbors(A):
                    eAB = G.get_edge(A, B)
                    if eAB is None:
                        continue
                    # Check A -o B (tail at A; circle at B allowed)
                    mA = eAB.mu if eAB.u == A else eAB.mv
                    mB = eAB.mv if eAB.u == A else eAB.mu
                    if mA == '-' and mB in ('o', '>'):
                        # look for A -> D -> B
                        found_chain = False
                        for D in G.neighbors(A):
                            if D == B:
                                continue
                            if not _is_directed(G, A, D):
                                continue
                            eDB = G.get_edge(D, B)
                            if eDB is None:
                                continue
                            if _is_directed(G, D, B):
                                found_chain = True
                                break
                        if found_chain:
                            pre = eAB.as_tuple()
                            G.orient(A, B, '-', '>')
                            post = G.get_edge(A, B).as_tuple()  # type: ignore
                            if pre != post:
                                changed = True

            if not changed:
                break

    # ------------------------------ Helpers -------------------------------------------

    def _idx(self, name: str) -> int:
        try:
            return self.var_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown variable name: {name}")

    def _n(self, idx: int) -> str:
        return self.var_names[idx]

    def _names(self, idxs: Iterable[int]) -> List[str]:
        return [self._n(i) for i in idxs]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CausalPAG] {msg}")

    def _assert_fitted(self) -> None:
        if self.pag is None:
            raise RuntimeError("Call fit() first.")

    def _connected(self, G: PAG, s: int, t: int) -> bool:
        # undirected connectivity in skeleton
        seen = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            if u == t:
                return True
            for v in G.neighbors(u):
                if v not in seen:
                    seen.add(v); stack.append(v)
        return False

    def _descendants_hard(self, G: PAG, s: int, max_hops: int = 3) -> Set[str]:
        """
        Traverse only hard-directed edges (A -> B) from s up to max_hops.
        """
        out: Set[int] = set()
        frontier = {s}
        for _ in range(max_hops):
            nxt: Set[int] = set()
            for u in frontier:
                for v in G.neighbors(u):
                    if _is_directed(G, u, v) and v not in out:
                        out.add(v); nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return {self._n(i) for i in out}

    def _edge_str_named(self, a: str, b: str, mu: Mark, mv: Mark) -> str:
        mapping = {
            ('o','o'): 'o-o',
            ('o','>'): 'o->',
            ('-','>'): '->',
            ('-','o'): '-o',
            ('>','o'): '<-o',
            ('>','-'): '<-',
        }
        conn = mapping.get((mu, mv), f"{mu}{mv}")
        return f"{a} {conn} {b}"


def _is_directed(G: PAG, a: int, b: int) -> bool:
    """
    Return True if edge a—b exists and is oriented a -> b (tail at a, arrowhead at b).
    """
    e = G.get_edge(a, b)
    if e is None:
        return False
    ma = e.mu if e.u == a else e.mv
    mb = e.mv if e.u == a else e.mu
    return (ma == '-') and (mb == '>')


# --------------------------------------------------------------------------------------
# Self-test (CPU) — synthetic example
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic linear-Gaussian SCM:
    #   X0 ~ N(0,1)
    #   X1 = 0.8*X0 + eps1
    #   X2 = 0.6*X0 + 0.7*X1 + eps2
    #   X3 = -0.9*X1 + eps3
    #   X4 = 0.5*X2 - 0.4*X3 + eps4
    # True DAG (edges):
    #   X0 -> X1, X0 -> X2, X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4
    np.random.seed(7)
    N = 3000
    X0 = np.random.randn(N)
    e1 = 0.7 * np.random.randn(N)
    X1 = 0.8 * X0 + e1
    e2 = 0.7 * np.random.randn(N)
    X2 = 0.6 * X0 + 0.7 * X1 + e2
    e3 = 0.7 * np.random.randn(N)
    X3 = -0.9 * X1 + e3
    e4 = 0.7 * np.random.randn(N)
    X4 = 0.5 * X2 - 0.4 * X3 + e4

    X = np.stack([X0, X1, X2, X3, X4], axis=1)
    names = [f"X{i}" for i in range(X.shape[1])]

    pag = CausalPAG(alpha=1e-3, max_k=2, verbose=True)
    pag.fit(X, var_names=names)

    print("\nEstimated PAG edges:")
    print(pag.edges_str())

    print("\nDOT:")
    print(pag.to_dot())

    print("\nOrientation queries:")
    for (a, b) in [("X0","X1"), ("X1","X2"), ("X3","X4")]:
        print("  ", pag.orientation(a, b))

    print("\nAdjustment suggestion (X2 -> X4):")
    Z, status = pag.suggest_adjustment_set("X2", "X4")
    print("  Z =", Z, "status =", status)
```

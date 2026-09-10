# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Clean-room EmbedOR + ORC-ManL for frozen-embedding class-separation analysis.

This is an independent reimplementation, written from the two papers and their
public reference code, of:

  * **EmbedOR** -- Saidi, Wang, Hu, Mishne & Cloninger,
    "EmbedOR: Provable Cluster-Preserving Visualizations with Curvature-Based
    Stochastic Neighbor Embeddings", arXiv:2509.03703 (2025).
    Reference implementation: https://github.com/TristanSaidi/embedor
  * **ORC-ManL** -- Saidi, Youssef, Cloninger & Mishne,
    "Recovering Manifold Structure Using Ollivier-Ricci Curvature",
    ICLR 2025 / arXiv:2410.01149.
    Reference implementation: https://github.com/TristanSaidi/orcml

Neither upstream package declares a license, so nothing here is copied from
them; the algorithms are re-derived from the papers. Ollivier-Ricci curvature is
computed directly from its definition with ``POT`` (optimal transport) and
``scipy`` (unweighted all-pairs shortest paths) -- no ``GraphRicciCurvature``
dependency (its ``scipy<=1.13.1`` pin and ``networkit`` build make it awkward on
Python 3.13 / Fortran-less hosts).

Pipeline (EmbedOR)
-----------------
1. Build a symmetric k-NN graph on the embeddings (edge weight = Euclidean
   distance).
2. Ollivier-Ricci curvature ``kappa(x, y) = 1 - W_1(mu_x, mu_y)`` on every edge,
   with uniform 1-hop measures (idleness ``alpha = 0``) and the *unweighted*
   (hop-count) ground metric -- EmbedOR eq. (3).
3. Re-weight each edge by ``w(x, y) = ||x - y||_2 . E(kappa; p)`` with the
   curvature energy ``E(kappa; p) = [-(log(kappa + 2) - log 2) / log(3/2) + 1]^p + 1``
   -- EmbedOR eq. (4)-(5), ``p = 3``. (The paper's global ``1/7`` prefactor is
   omitted: a uniform edge rescaling leaves both the shortest paths and the
   perplexity-calibrated affinities unchanged.)
4. EmbedOR distance ``Delta_E`` = all-pairs shortest path over the re-weighted
   graph.
5. Feed ``Delta_E`` into the standard SNE affinity (perplexity binary search),
   then a UMAP-style attractive/repulsive SGD layout with spectral init.

Pipeline (ORC-ManL pruning, optional preprocessing)
--------------------------------------------------
Given the same k-NN graph and its edge curvatures, flag an edge as a manifold
"shortcut" when (a) its curvature is below ``kappa_thr = -1 + 4(1 - delta)`` and
(b) removing it forces a detour longer than ``lambda_thr . ||x - y||_2`` with
``lambda_thr = (1 - lda) . pi^2 / (2 . sqrt(24 . lda))`` -- ORC-ManL Alg. 1.
Flagged edges are dropped; any node left isolated is re-attached to its nearest
neighbour.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

# --- Curvature energy parameters (EmbedOR, "orc" edge weighting) --------------
# kappa in [k_min, k_max]; k_crit is the curvature that maps to unit energy.
_ORC_K_MIN, _ORC_K_MAX, _ORC_K_CRIT = -2.0, 1.0, 0.0


# ---------------------------------------------------------------------------
# k-NN graph
# ---------------------------------------------------------------------------
def build_knn_graph(X: np.ndarray, n_neighbors: int = 15) -> nx.Graph:
    """Symmetric k-NN graph with Euclidean edge weights.

    An edge (i, j) exists iff j is among the k nearest neighbours of i *or*
    vice versa (the "OR" / mutual-union rule used by EmbedOR). Node labels are
    integer row indices ``0 .. N-1``.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    k = int(min(max(n_neighbors, 1), n - 1))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    dist, idx = nn.kneighbors(X)  # (N, k+1), first column is the point itself

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            j = int(j)
            # OR rule: keep the shorter weight if the reverse edge already set it.
            if G.has_edge(i, j):
                if d < G[i][j]["weight"]:
                    G[i][j]["weight"] = float(d)
            else:
                G.add_edge(i, j, weight=float(d))
    return G


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature
# ---------------------------------------------------------------------------
def _edge_orc(args: tuple) -> tuple[tuple[int, int], float]:
    """Exact Ollivier-Ricci curvature of one edge (module-level for pickling)."""
    import ot

    (u, v), nbr_u, nbr_v, D, alpha = args
    if not nbr_u or not nbr_v:
        return (min(u, v), max(u, v)), 0.0

    # alpha mass stays at the node; (1 - alpha) spreads uniformly over 1-hop
    # neighbours. EmbedOR uses alpha = 0 -> all mass on the neighbours.
    if alpha > 0.0:
        rows = [u, *nbr_u]
        cols = [v, *nbr_v]
        mu = np.array([alpha, *([(1.0 - alpha) / len(nbr_u)] * len(nbr_u))])
        mv = np.array([alpha, *([(1.0 - alpha) / len(nbr_v)] * len(nbr_v))])
    else:
        rows, cols = list(nbr_u), list(nbr_v)
        mu = np.full(len(nbr_u), 1.0 / len(nbr_u))
        mv = np.full(len(nbr_v), 1.0 / len(nbr_v))

    M = np.ascontiguousarray(D[np.ix_(rows, cols)], dtype=np.float64)
    w1 = float(ot.emd2(mu, mv, M))
    return (min(u, v), max(u, v)), 1.0 - w1


def compute_orc(G: nx.Graph, *, alpha: float = 0.0, n_jobs: int = 1) -> dict[tuple[int, int], float]:
    """Ollivier-Ricci curvature per edge: ``kappa(x, y) = 1 - W_1(mu_x, mu_y)``.

    Matches EmbedOR's ``compute_orc`` (``alpha=0``, ``method="OTD"``,
    ``nbrhood_size=1``, *unweighted* ground metric): the probability measures are
    uniform over the 1-hop neighbourhoods and the transport cost between two
    nodes is their hop distance, so curvature reflects graph topology rather than
    the Euclidean edge lengths. ``W_1`` is the exact earth-mover distance
    (POT network simplex).

    Node labels must be ``0 .. N-1`` (as produced by :func:`build_knn_graph`).
    Returns ``{(min(i,j), max(i,j)): kappa}``.
    """
    from scipy.sparse.csgraph import shortest_path

    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight=None, format="csr")
    D = np.asarray(shortest_path(A, method="D", directed=False, unweighted=True), dtype=np.float64)

    nbrs = [list(G.neighbors(i)) for i in range(n)]
    tasks = [((int(u), int(v)), nbrs[u], nbrs[v], D, alpha) for u, v in G.edges()]

    if n_jobs and n_jobs > 1 and len(tasks) > 512:
        # POT's C solver releases the GIL, so threads give real speedup without
        # pickling the (N x N) distance matrix to worker processes.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = pool.map(_edge_orc, tasks, chunksize=max(len(tasks) // (n_jobs * 4), 1))
        return dict(results)

    return dict(_edge_orc(t) for t in tasks)


def curvature_energy(kappa: float | np.ndarray, p: float = 3.0) -> float | np.ndarray:
    """EmbedOR curvature energy ``E(kappa; p)`` (eq. 5).

    Monotone decreasing in ``kappa``: strongly negative curvature (a shortcut
    edge bridging clusters) is stretched, positive curvature (a within-cluster
    edge) is left near its geometric length.
    """
    k = np.clip(kappa, _ORC_K_MIN + 1e-6, _ORC_K_MAX)
    c = 1.0 / math.log((_ORC_K_MAX - _ORC_K_MIN) / (_ORC_K_CRIT - _ORC_K_MIN))
    energy = (-c * np.log(k - _ORC_K_MIN) + c * math.log(_ORC_K_CRIT - _ORC_K_MIN) + 1.0) ** p + 1.0
    return np.clip(energy, 0.0, np.inf)


def curvature_gap(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 15,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Within- vs cross-class Ollivier-Ricci curvature gap on the k-NN graph.

    The layout-free core of the EmbedOR class-separation signal: build the k-NN
    graph, compute edge curvature, and bucket edges by whether they connect
    same-class or different-class points. Cheap enough to run on every training
    snapshot for a Delta-kappa(epoch) trajectory.

    Returns ``orc_within``, ``orc_cross``, ``orc_gap`` (within - cross),
    ``edge_cross_frac``, ``n_edges``, ``n_components``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    G = build_knn_graph(X, n_neighbors)
    orc = compute_orc(G, n_jobs=n_jobs)

    within, cross = [], []
    for u, v in G.edges():
        kappa = orc[(int(min(u, v)), int(max(u, v)))]
        (within if y[u] == y[v] else cross).append(kappa)

    w = float(np.mean(within)) if within else float("nan")
    c = float(np.mean(cross)) if cross else float("nan")
    return {
        "orc_within": w,
        "orc_cross": c,
        "orc_gap": w - c,
        "edge_cross_frac": len(cross) / max(len(within) + len(cross), 1),
        "n_edges": G.number_of_edges(),
        "n_components": nx.number_connected_components(G),
    }


def reweight_graph(
    G: nx.Graph,
    orc: dict[tuple[int, int], float],
    p: float = 3.0,
) -> nx.Graph:
    """Attach the EmbedOR edge length ``||x - y|| . E(kappa; p)`` as ``embedor_weight``."""
    for u, v in G.edges():
        key = (int(min(u, v)), int(max(u, v)))
        kappa = orc.get(key, _ORC_K_CRIT)
        G[u][v]["orc"] = float(kappa)
        G[u][v]["embedor_weight"] = float(G[u][v]["weight"] * curvature_energy(kappa, p))
    return G


# ---------------------------------------------------------------------------
# EmbedOR distance (all-pairs shortest path)
# ---------------------------------------------------------------------------
def geodesic_matrix(G: nx.Graph, weight_attr: str = "embedor_weight") -> np.ndarray:
    """Dense all-pairs shortest-path matrix over ``weight_attr`` (the EmbedOR distance).

    If the graph is disconnected, unreachable pairs (``+inf``) are replaced by
    twice the largest finite distance so the downstream SNE/layout still runs --
    disconnected components then simply land far apart. Callers see the component
    count via ``n_components`` on the result.
    """
    from scipy.sparse.csgraph import shortest_path

    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight=weight_attr, format="csr")
    D = np.asarray(shortest_path(A, method="D", directed=False), dtype=np.float64)
    if not np.isfinite(D).all():
        finite_max = np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 1.0
        D[~np.isfinite(D)] = 2.0 * finite_max
        np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# SNE affinities + layout
# ---------------------------------------------------------------------------
def _joint_probabilities(D: np.ndarray, perplexity: float) -> np.ndarray:
    """Symmetric SNE joint probabilities from a distance matrix.

    Per-row Gaussian bandwidth found by bisection so that ``2 ** entropy`` hits
    the target perplexity (the standard t-SNE calibration), then symmetrised and
    normalised. Distances enter linearly (``exp(-D * beta)``), matching the
    EmbedOR reference code rather than the squared form written in the paper.
    """
    n = D.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    logU = math.log(perplexity)
    for i in range(n):
        d = D[i].copy()
        d[i] = np.inf
        finite = np.isfinite(d)
        beta, lo, hi = 1.0, 0.0, np.inf
        for _ in range(50):
            w = np.zeros(n)
            w[finite] = np.exp(-d[finite] * beta)
            sw = w.sum()
            if sw <= 0.0:
                beta /= 2.0
                continue
            H = math.log(sw) + beta * (d[finite] * w[finite]).sum() / sw
            if abs(H - logU) < 1e-5:
                break
            if H > logU:
                lo = beta
                beta = beta * 2.0 if not np.isfinite(hi) else (beta + hi) / 2.0
            else:
                hi = beta
                beta = (beta + lo) / 2.0
            w = np.zeros(n)
            w[finite] = np.exp(-d[finite] * beta)
            sw = w.sum()
        P[i] = w / max(sw, 1e-12)
    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def _spectral_init(P: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Spectral embedding of the affinity graph (normalised-Laplacian eigvecs)."""
    from scipy.sparse.linalg import eigsh

    n = P.shape[0]
    deg = P.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(n) - (d_inv_sqrt[:, None] * P * d_inv_sqrt[None, :])
    try:
        vals, vecs = eigsh(L, k=dim + 1, which="SM")
        order = np.argsort(vals)
        Y = vecs[:, order[1 : dim + 1]]
    except Exception:  # pragma: no cover - tiny/degenerate graphs
        rng = np.random.default_rng(seed)
        Y = rng.standard_normal((n, dim)) * 1e-4
    span = Y.max(axis=0) - Y.min(axis=0)
    span[span == 0] = 1.0
    return (Y - Y.min(axis=0)) / span - 0.5


def _layout_sgd(
    P: np.ndarray,
    init: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: str | None,
) -> np.ndarray:
    """UMAP-style attractive/repulsive layout, mirroring EmbedOR's torch layout.

    ``f_ij = 1 / (1 + ||y_i - y_j||^2)``; attractive force weighted by the SNE
    affinity ``P``, repulsive by ``1 - P`` and a normalisation ``gamma``.
    """
    import torch

    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = P.shape[0]

    iu, ju = np.triu_indices(n, k=1)
    aff = torch.tensor(P[iu, ju], dtype=torch.float32, device=dev)
    rep = 1.0 - aff
    Z = float(aff.sum())
    n_pairs = n * (n - 1) / 2.0
    gamma = (n_pairs - Z) / (Z * n * n + 1e-12)

    Y = torch.tensor(init, dtype=torch.float32, device=dev, requires_grad=True)
    ii = torch.tensor(iu, dtype=torch.long, device=dev)
    jj = torch.tensor(ju, dtype=torch.long, device=dev)
    opt = torch.optim.SGD([Y], lr=0.1)

    for _ in range(epochs):
        opt.zero_grad()
        d2 = ((Y[ii] - Y[jj]) ** 2).sum(dim=1)
        f = torch.clamp(1.0 / (1.0 + d2), min=1e-4)
        loss_pos = (-torch.log(f) * aff).sum()
        loss_neg = (-torch.log(1.001 - f) * rep).sum()
        (loss_pos + gamma * loss_neg).backward()
        opt.step()

    return Y.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# ORC-ManL pruning
# ---------------------------------------------------------------------------
@dataclass
class PruneInfo:
    """Bookkeeping from an ORC-ManL pruning pass."""

    kappa_threshold: float
    lambda_threshold: float
    n_edges_before: int
    n_candidates: int
    n_pruned: int
    n_reattached: int
    shortcut_edges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def prune_rate(self) -> float:
        return self.n_pruned / max(self.n_edges_before, 1)


def orcml_prune(
    X: np.ndarray,
    G: nx.Graph,
    orc: dict[tuple[int, int], float],
    *,
    delta: float = 0.8,
    lda: float = 0.01,
    reattach: bool = True,
) -> tuple[nx.Graph, PruneInfo]:
    """Drop manifold-shortcut edges (ORC-ManL, Alg. 1).

    Parameters
    ----------
    X       : (N, D) embeddings the graph was built on (for re-attachment).
    G       : symmetric k-NN graph with Euclidean ``weight`` edge attributes.
    orc     : edge -> Ollivier-Ricci curvature (:func:`compute_orc`).
    delta   : curvature-confidence parameter; candidates have
              ``kappa < -1 + 4(1 - delta)``.
    lda     : metric-distortion parameter; the detour bound is
              ``(1 - lda) pi^2 / (2 sqrt(24 lda)) . ||x - y||``.
    reattach: reconnect any node left isolated to its nearest neighbour.
    """
    kappa_thr = -1.0 + 4.0 * (1.0 - delta)
    lambda_thr = ((1.0 - lda) * math.pi**2) / (2.0 * math.sqrt(24.0 * lda))

    def key(u: int, v: int) -> tuple[int, int]:
        return (int(min(u, v)), int(max(u, v)))

    candidates = [(u, v) for u, v in G.edges() if orc.get(key(u, v), 0.0) < kappa_thr]

    # G_prime: the graph with all candidate edges removed up front.
    G_prime = G.copy()
    G_prime.remove_edges_from(candidates)

    G_pruned = G.copy()
    shortcut_edges: list[tuple[int, int]] = []
    for u, v in candidates:
        w = G[u][v]["weight"]
        try:
            detour = nx.shortest_path_length(G_prime, u, v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            detour = math.inf
        # A shortcut has a long *but finite* detour. An infinite detour means the
        # edge is a bridge -- essential manifold structure, never a shortcut --
        # so keep it (pruning bridges is what shatters the graph into components).
        if math.isfinite(detour) and detour > lambda_thr * w:
            G_pruned.remove_edge(u, v)
            shortcut_edges.append(key(u, v))
        else:
            # Not a genuine shortcut: keep it, and restore it into G_prime so
            # later candidates see the true residual graph.
            G_prime.add_edge(u, v, **G[u][v])

    n_reattached = 0
    if reattach:
        isolated = [node for node in G_pruned.nodes() if G_pruned.degree(node) == 0]
        if isolated:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
            dist, idx = nn.kneighbors(X[isolated])
            for node, d, j in zip(isolated, dist[:, 1], idx[:, 1]):
                G_pruned.add_edge(int(node), int(j), weight=float(d), orc=0.0)
                n_reattached += 1

    info = PruneInfo(
        kappa_threshold=kappa_thr,
        lambda_threshold=lambda_thr,
        n_edges_before=G.number_of_edges(),
        n_candidates=len(candidates),
        n_pruned=len(shortcut_edges),
        n_reattached=n_reattached,
        shortcut_edges=shortcut_edges,
    )
    return G_pruned, info


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
@dataclass
class EmbedORResult:
    coords: np.ndarray                       # (N, 2) 2-D layout
    labels: np.ndarray | None               # (N,) integer class labels, if given
    orc: dict[tuple[int, int], float]       # edge -> Ollivier-Ricci curvature
    graph: nx.Graph                          # the (possibly pruned) working graph
    n_components: int                        # connected components of `graph`
    geodesic: np.ndarray                     # (N, N) EmbedOR distance Delta_E
    prune_info: PruneInfo | None
    params: dict[str, Any]
    timings: dict[str, float]


def run_embedor(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    n_neighbors: int = 15,
    perplexity: float = 40.0,
    p: float = 3.0,
    epochs: int = 300,
    seed: int = 0,
    prune: bool = False,
    prune_delta: float = 0.8,
    prune_lda: float = 0.01,
    n_jobs: int = 1,
    device: str | None = None,
    verbose: bool = True,
) -> EmbedORResult:
    """Run the full EmbedOR pipeline (optionally ORC-ManL-pruned) on ``X``.

    ``X`` should already be whatever representation you want visualised (e.g.
    z-scored frozen backbone features). ``y`` is only carried through for the
    downstream metrics/plots.
    """
    t: dict[str, float] = {}
    log = (lambda *a: print("[embedor]", *a)) if verbose else (lambda *a: None)

    X = np.ascontiguousarray(X, dtype=np.float64)
    _t = time.time()
    G = build_knn_graph(X, n_neighbors)
    t["knn_graph"] = time.time() - _t
    log(f"k-NN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({t['knn_graph']:.1f}s)")

    _t = time.time()
    orc = compute_orc(G, n_jobs=n_jobs)
    t["orc"] = time.time() - _t
    log(f"Ollivier-Ricci curvature on {len(orc)} edges ({t['orc']:.1f}s)")

    prune_info: PruneInfo | None = None
    if prune:
        _t = time.time()
        G, prune_info = orcml_prune(X, G, orc, delta=prune_delta, lda=prune_lda)
        t["prune"] = time.time() - _t
        log(f"ORC-ManL: {prune_info.n_pruned}/{prune_info.n_edges_before} edges pruned "
            f"({100 * prune_info.prune_rate:.1f}%), {prune_info.n_reattached} re-attached "
            f"({t['prune']:.1f}s)")
        # Keep the measured curvatures; give any re-attachment edge (new to the
        # graph, absent from `orc`) a neutral 0.
        for u, v in G.edges():
            orc.setdefault((int(min(u, v)), int(max(u, v))), 0.0)

    reweight_graph(G, orc, p)
    n_components = nx.number_connected_components(G)
    if n_components > 1:
        log(f"WARNING: working graph has {n_components} components; "
            "EmbedOR distances across components are infinite. "
            "Increase n_neighbors or disable pruning.")

    _t = time.time()
    D = geodesic_matrix(G, "embedor_weight")
    t["geodesic"] = time.time() - _t
    log(f"EmbedOR distance matrix {D.shape} ({t['geodesic']:.1f}s)")

    _t = time.time()
    P = _joint_probabilities(D, perplexity)
    Y0 = _spectral_init(P, dim=2, seed=seed)
    coords = _layout_sgd(P, Y0, epochs=epochs, seed=seed, device=device)
    t["layout"] = time.time() - _t
    log(f"layout done ({t['layout']:.1f}s)")

    return EmbedORResult(
        coords=coords,
        labels=None if y is None else np.asarray(y).astype(int),
        orc=orc,
        graph=G,
        n_components=n_components,
        geodesic=D,
        prune_info=prune_info,
        params=dict(
            n_neighbors=n_neighbors, perplexity=perplexity, p=p, epochs=epochs,
            seed=seed, prune=prune, prune_delta=prune_delta, prune_lda=prune_lda,
            n_jobs=n_jobs,
        ),
        timings=t,
    )


# ---------------------------------------------------------------------------
# Class-separation metrics
# ---------------------------------------------------------------------------
def separation_metrics(result: EmbedORResult, X: np.ndarray | None = None) -> dict[str, Any]:
    """Quantify how well ``result`` separates the classes in ``result.labels``.

    - ``silhouette_2d``      : silhouette of the 2-D layout (Euclidean).
    - ``silhouette_raw``     : silhouette of ``X`` (cosine), if ``X`` is given.
    - ``orc_within`` / ``orc_cross`` : mean edge curvature for same-class vs
      cross-class edges of the k-NN graph -- the geometric signal EmbedOR/BallJEPA
      predict should diverge (within-class neighbourhoods more positively curved).
    - ``shortcut_cross_frac``: fraction of ORC-ManL-flagged shortcut edges that
      actually cross a class boundary (only when the result was pruned).
    - ``fragmentation``      : mean number of connected components per class in
      the graph restricted to that class (1.0 = each class is one clean piece).
    """
    y = result.labels
    if y is None:
        raise ValueError("separation_metrics needs result.labels (pass y to run_embedor).")

    out: dict[str, Any] = {}

    try:
        from sklearn.metrics import silhouette_score

        if len(np.unique(y)) > 1:
            out["silhouette_2d"] = float(silhouette_score(result.coords, y, metric="euclidean"))
            if X is not None:
                out["silhouette_raw"] = float(silhouette_score(np.asarray(X), y, metric="cosine"))
    except Exception as exc:  # pragma: no cover
        out["silhouette_error"] = repr(exc)

    within, cross = [], []
    for u, v in result.graph.edges():
        kappa = result.graph[u][v].get("orc", 0.0)
        (within if y[u] == y[v] else cross).append(kappa)
    out["orc_within"] = float(np.mean(within)) if within else float("nan")
    out["orc_cross"] = float(np.mean(cross)) if cross else float("nan")
    out["orc_gap"] = out["orc_within"] - out["orc_cross"]
    out["edge_cross_frac"] = len(cross) / max(len(within) + len(cross), 1)

    if result.prune_info is not None and result.prune_info.shortcut_edges:
        crossing = sum(1 for u, v in result.prune_info.shortcut_edges if y[u] != y[v])
        out["shortcut_cross_frac"] = crossing / len(result.prune_info.shortcut_edges)
        out["prune_rate"] = result.prune_info.prune_rate

    frag = []
    for cls in np.unique(y):
        sub = result.graph.subgraph([i for i in result.graph.nodes() if y[i] == cls])
        if sub.number_of_nodes():
            frag.append(nx.number_connected_components(sub))
    out["fragmentation"] = float(np.mean(frag)) if frag else float("nan")

    return out


if __name__ == "__main__":
    # Smoke test: two interleaving half-moons + a thin bridge of noise, so the
    # k-NN graph is one connected component with genuine cross-class edges. The
    # bridge is the manifold "shortcut" ORC-ManL should flag; EmbedOR should
    # still separate the two moons (higher 2-D silhouette than plain t-SNE).
    rng = np.random.default_rng(0)
    n = 300
    t = np.linspace(0, np.pi, n)
    moon_a = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.08, (n, 2))
    moon_b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + rng.normal(0, 0.08, (n, 2))
    bridge = np.c_[np.linspace(0, 1, 40), np.full(40, 0.25)] + rng.normal(0, 0.03, (40, 2))
    Xs = np.vstack([moon_a, moon_b, bridge])
    ys = np.r_[np.zeros(n), np.ones(n), np.full(40, 0)].astype(int)

    for use_prune in (False, True):
        res = run_embedor(Xs, ys, n_neighbors=15, perplexity=30, epochs=200,
                          seed=0, prune=use_prune, device="cpu")
        m = separation_metrics(res, Xs)
        print(f"  prune={use_prune}: components={res.n_components} "
              f"sil2d={m['silhouette_2d']:.3f} orc_gap={m['orc_gap']:.3f} "
              f"frag={m['fragmentation']:.2f} "
              f"n_pruned={0 if res.prune_info is None else res.prune_info.n_pruned}")

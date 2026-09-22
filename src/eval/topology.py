# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Did a representation keep the topology its data came from?

Written for COIL-100, whose 100 objects are each photographed every 5 degrees, so the
ground truth is 100 disjoint circles. A Gaussian target is contractible and has to fill
those loops in; a locally uniform target does not. Nothing here is Lightning-aware, so
the same functions serve the training callback and the tests.

Measurement runs at two scales. Per object the cloud is 72 points, so persistent
homology is instant and asks whether one circle survived. Globally the cloud is
subsampled across every object at once, which is what training actually saw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class TopologyOptions:
    stride: int = 3
    global_maxdim: int = 1
    thresh_quantile: float = 0.4
    h0_scale: float = 2.0
    h1_cut: float = 0.3
    holdout_objects: int = 20


def scaled(x: np.ndarray, cap: int = 2000, seed: int = 0) -> np.ndarray:
    """Rescale so the median pairwise distance is 1, making barcodes comparable across runs."""
    sample = x
    if len(x) > cap:
        sample = x[np.random.default_rng(seed).choice(len(x), cap, replace=False)]
    median = float(np.median(pdist(sample)))
    return x / median if median > 0 else x


def local_unit(x: np.ndarray, objects: np.ndarray, sample: int = 12, seed: int = 0) -> float:
    """Median pairwise distance inside an object.

    The global cloud is measured in these units rather than its own median distance: a
    circle is small next to the gaps between objects, so normalising globally pushes
    every loop below the resolution of the filtration and no H1 bar survives.
    """
    ids = np.unique(objects)
    if len(ids) > sample:
        ids = np.random.default_rng(seed).choice(ids, sample, replace=False)
    spreads = [float(np.median(pdist(x[objects == o]))) for o in ids]
    unit = float(np.median(spreads))
    return unit if unit > 0 else 1.0


def nn_purity(x: np.ndarray, objects: np.ndarray) -> float:
    """Fraction of frames whose nearest neighbour belongs to the same object."""
    distances = squareform(pdist(x))
    np.fill_diagonal(distances, np.inf)
    return float((objects[distances.argmin(axis=1)] == objects).mean())


def pose_r2(x: np.ndarray, angles: np.ndarray, holdout: int = 3) -> float:
    """R^2 of predicting (cos, sin) of the turntable angle, scored on held-out poses.

    Every ``holdout``-th pose is kept back. Scoring in sample would be meaningless: an
    object contributes 72 frames while the encoder returns 512 features, so least
    squares interpolates any target exactly and the number is 1.0 whatever training did.
    """
    test = np.zeros(len(x), dtype=bool)
    test[::holdout] = True
    if test.all() or not test.any():
        return float("nan")

    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    target = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    coef = np.linalg.lstsq(design[~test], target[~test], rcond=None)[0]
    residual = target[test] - design[test] @ coef
    denom = ((target[test] - target[test].mean(0)) ** 2).sum()
    return float(1.0 - (residual**2).sum() / denom) if denom > 0 else float("nan")


def transfer_r2(x: np.ndarray, angles: np.ndarray, objects: np.ndarray, holdout: int, seed: int = 0) -> float:
    """Fit the angle decoder on some objects, score it on objects it never saw.

    Each object is centred first. Without that the decoder would have to absorb every
    object's position, which no shared intercept can do, and a perfectly shared pose
    direction would still score badly.
    """
    ids = np.unique(objects)
    if len(ids) <= holdout:
        return float("nan")
    test_ids = np.random.default_rng(seed).choice(ids, holdout, replace=False)
    test = np.isin(objects, test_ids)

    x = x.copy()
    for obj in ids:
        mask = objects == obj
        x[mask] -= x[mask].mean(0)

    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    target = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    coef = np.linalg.lstsq(design[~test], target[~test], rcond=None)[0]
    residual = target[test] - design[test] @ coef
    denom = ((target[test] - target[test].mean(0)) ** 2).sum()
    return float(1.0 - (residual**2).sum() / denom) if denom > 0 else float("nan")


def longest_h1(x: np.ndarray) -> float:
    bars = ripser(scaled(x), maxdim=1)["dgms"][1]
    return float((bars[:, 1] - bars[:, 0]).max()) if len(bars) else 0.0


def components(deaths: np.ndarray, at: float) -> int:
    """Clusters still separate at radius ``at``: H0 bars that have not merged yet."""
    return int((deaths > at).sum())


def gap_components(deaths: np.ndarray) -> int:
    """Cluster count read off the widest gap in the H0 barcode, with no scale chosen."""
    finite = np.sort(deaths[np.isfinite(deaths)])
    if len(finite) < 2:
        return len(deaths)
    return len(finite) - int(np.argmax(np.diff(finite)))


def analyse(x: np.ndarray, angles: np.ndarray, objects: np.ndarray, options: TopologyOptions) -> dict:
    per_object_r2, per_object_h1 = [], []
    for obj in np.unique(objects):
        mask = objects == obj
        per_object_r2.append(pose_r2(x[mask], angles[mask]))
        per_object_h1.append(longest_h1(x[mask]))

    keep = np.arange(0, len(x), options.stride)
    cloud = x[keep] / local_unit(x, objects)
    reference = cloud if len(cloud) <= 2000 else cloud[:2000]
    thresh = float(np.quantile(pdist(reference), options.thresh_quantile))

    start = time.perf_counter()
    dgms = ripser(cloud, maxdim=options.global_maxdim, thresh=thresh)["dgms"]
    elapsed = time.perf_counter() - start

    bars, censored, longest = 0, 0, 0.0
    if options.global_maxdim >= 1 and len(dgms[1]):
        births, deaths = dgms[1][:, 0], dgms[1][:, 1]
        censored = int(np.isinf(deaths).sum())
        # A bar still alive at the threshold is censored there, not dropped.
        lives = np.minimum(deaths, thresh) - births
        bars = int((lives > options.h1_cut).sum())
        longest = float(lives.max())

    return {
        "pose_r2_median": float(np.median(per_object_r2)),
        "pose_r2_q25": float(np.quantile(per_object_r2, 0.25)),
        "h1_median": float(np.median(per_object_h1)),
        "h1_q25": float(np.quantile(per_object_h1, 0.25)),
        "transfer_r2": transfer_r2(x, angles, objects, options.holdout_objects),
        "nn_purity": nn_purity(cloud, objects[keep]),
        "global_points": len(cloud),
        "global_thresh": thresh,
        "global_components": components(dgms[0][:, 1], options.h0_scale),
        "global_gap_components": gap_components(dgms[0][:, 1]),
        "global_h1_bars": bars,
        "global_h1_longest": longest,
        "global_h1_censored": censored,
        "global_seconds": elapsed,
    }


def report(label: str, stats: dict) -> None:
    print(f"\n{label}")
    print(f"  per object   pose R2  median {stats['pose_r2_median']:.3f}   lower quartile {stats['pose_r2_q25']:.3f}")
    print(f"               H1 life  median {stats['h1_median']:.3f}   lower quartile {stats['h1_q25']:.3f}")
    print(f"  across objects  transfer R2  {stats['transfer_r2']:.3f}")
    print(
        f"  global ({stats['global_points']} pts, thresh {stats['global_thresh']:.2f})"
        f"  nn purity {stats['nn_purity']:.3f}"
        f"   components {stats['global_components']} (gap {stats['global_gap_components']})"
        f"   H1 bars {stats['global_h1_bars']} (longest {stats['global_h1_longest']:.3f},"
        f" {stats['global_h1_censored']} censored)   [{stats['global_seconds']:.1f}s]"
    )


def persistence(x: np.ndarray, objects: np.ndarray, options: TopologyOptions):
    """The diagrams behind the numbers: one global, plus H1 for every object."""
    keep = np.arange(0, len(x), options.stride)
    cloud = x[keep] / local_unit(x, objects)
    reference = cloud if len(cloud) <= 2000 else cloud[:2000]
    thresh = float(np.quantile(pdist(reference), options.thresh_quantile))

    global_dgms = ripser(cloud, maxdim=options.global_maxdim, thresh=thresh)["dgms"]
    per_object = [ripser(scaled(x[objects == o]), maxdim=1)["dgms"][1] for o in np.unique(objects)]
    return global_dgms, per_object, thresh


def plot_persistence(global_dgms, per_object, title: str, path) -> None:
    """Two panels: the global diagram, and every object's H1 overlaid.

    Points far above the diagonal are features that outlive their birth scale: a circle
    sits high above it, a filled blob leaves everything hugging it. The panels carry
    their own axes because the global cloud is measured in within-object units while
    each object's diagram is scaled by its own spread.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ceiling_of(arrays):
        deaths = [d[np.isfinite(d[:, 1]), 1] for d in arrays if len(d)]
        deaths = [d for d in deaths if len(d)]
        return float(max(d.max() for d in deaths)) * 1.05 if deaths else 1.0

    def dress(ax, ceiling, xlabel="birth"):
        ax.plot([0, ceiling], [0, ceiling], color="0.7", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("death")
        ax.set_xlim(-0.02 * ceiling, ceiling)
        ax.set_ylim(-0.02 * ceiling, ceiling * 1.02)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    global_ceiling = ceiling_of(list(global_dgms))
    dress(axes[0], global_ceiling)
    for dim, colour, name in ((0, "tab:blue", "H0"), (1, "tab:red", "H1")):
        if dim >= len(global_dgms) or not len(global_dgms[dim]):
            continue
        bars = global_dgms[dim].astype(float).copy()
        bars[np.isinf(bars[:, 1]), 1] = global_ceiling
        axes[0].scatter(bars[:, 0], bars[:, 1], s=14, alpha=0.6, color=colour, label=name)
    axes[0].axhline(global_ceiling, color="0.5", linestyle=":", linewidth=1)
    axes[0].legend(loc="lower right")
    axes[0].set_title("global cloud, subsampled")

    object_ceiling = ceiling_of(per_object)
    dress(axes[1], object_ceiling)
    for bars in per_object:
        if not len(bars):
            continue
        bars = bars.astype(float).copy()
        bars[np.isinf(bars[:, 1]), 1] = object_ceiling
        axes[1].scatter(bars[:, 0], bars[:, 1], s=12, alpha=0.3, color="tab:red")
    axes[1].set_title(f"H1 of each object, {len(per_object)} overlaid")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

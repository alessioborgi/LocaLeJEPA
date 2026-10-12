# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The three encoder families, each drawn in the geometry its output lives in.

Two rows (global SIGReg / LocaLeJEPA) by three columns (Euclidean ViT-S/8,
hyperbolic H-ViT-S/8, D4-equivariant ResNet-50), all from the ImageNet-1K
validation features of the 100-epoch pretrains:

  (a) Euclidean: 2-d PCA of the l2-normalized 512-d embedding on a flat grid;
      the dashed segment between two class means is the geodesic of the space.
  (b) Hyperbolic: the Poincare ball of curvature -0.1 drawn as a disk, moved by
      a Mobius isometry so the cloud's mean sits at the centre (no hyperbolic
      distance changes, and the ball origin reappears as the open marker). The
      rings have equal hyperbolic radius and the dashed arc is the geodesic
      between the same two class means. The clip at tangent norm 2.3 is active
      for all 50k validation images, so every embedding lies on one shell, at
      hyperbolic distance 4.60 from the ball origin.
  (c) D4-equivariant: the plane of the 2-dimensional irreducible representation
      of D4 inside the regular representation the encoder outputs, where the
      group acts as the symmetries of the square. Filled markers are the orbit
      rho(g) f(x) of one image's embedding; open markers are the *measured*
      embeddings f(g.x) of its eight rotated and reflected copies, which fall
      short of the orbit because strided downsampling breaks exact equivariance
      at even resolutions.

Under each panel is the spread of the embedding in its own metric, and above it
the published k-NN / linear top-1 of that run.

Panels (a) and (b) read the cached val features written by src/eval_indomain.py
(cache_dir of configs/eval/indomain_in1k_ep100.yaml); panel (c) needs forward
passes of the two D4 checkpoints over the eight group views of a few images and
caches them next to the figure:

    sbatch scripts/slurm_encoder_geometries.sh          # GPU: builds the cache + figure
    python visualisations/plot_encoder_geometries.py    # CPU: restyle from the caches
    python visualisations/plot_encoder_geometries.py --extract-only   # cache only
    python visualisations/plot_encoder_geometries.py --panels [DIR]   # + one image per panel

The second form also writes every panel on its own, without headings, row labels
or the measurement line, for assembly with LaTeX subcaptions; it prints the
numbers each panel carries in the assembled figure so they can go in those
subcaptions, and a legend strip to place under them.
"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

INDOMAIN_CONFIG = os.path.join(_ROOT, "configs/eval/indomain_in1k_ep100.yaml")
PATCH_PCA_CONFIG = os.path.join(_ROOT, "configs/eval/patch_pca_in1k.yaml")
VAL_FEATURES = "imagenet1k_val_128_ep99.pt"
VAL_IMAGES = os.path.join(_ROOT, "data/imagenet/imagenet_full_size/061417/val")

HYP_C = 0.1          # Poincare-ball curvature -c, as in configs/model/encoder/hvit_small_p8.yaml
HYP_CLIP = 2.3       # tangent-norm clip of the hyperbolic head
GROUP_ORDER = 8      # |D4|
CROP = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# The two objectives compared in every column, and the three encoder families.
# ``scores`` are the published k-NN (k=20) / linear top-1 of results/Imagenet1K_results.md
# and results/report_plainViT_LastExp.md, i.e. the rows of Table "imagenet_main".
ROWS = ["SIGReg", "LocaLeJEPA"]
ROW_LABEL = {"SIGReg": "LeJEPA\n(global SIGReg)", "LocaLeJEPA": "LocaLeJEPA\n(local BallReg)"}
COLUMNS = [
    dict(
        key="vit", kind="euclidean",
        title="Euclidean", subtitle=r"ViT-S/8  $\rightarrow\ \mathbb{R}^{512}$",
        runs={"SIGReg": "vit_sigreg", "LocaLeJEPA": "vit_localballreg_l01_k4"},
        setting=r"$\lambda{=}0.1,\ K{=}4$",
        scores={"SIGReg": (24.3, 40.7), "LocaLeJEPA": (34.1, 49.5)},
    ),
    dict(
        key="hvit", kind="hyperbolic",
        title="Hyperbolic", subtitle=r"H-ViT-S/8  $\rightarrow\ \mathbb{D}^{512}_{c=-0.1}$",
        runs={"SIGReg": "hvit_sigreg", "LocaLeJEPA": "hvit_localballreg_l01_k2"},
        setting=r"$\lambda{=}0.1,\ K{=}2$",
        scores={"SIGReg": (25.1, 41.1), "LocaLeJEPA": (33.7, 49.6)},
    ),
    dict(
        key="d4rn50", kind="equivariant",
        title=r"$D_4$-equivariant", subtitle=r"ResNet-50  $\rightarrow\ (\rho_{\mathrm{reg}})^{64}$",
        runs={"SIGReg": "d4rn50_sigreg", "LocaLeJEPA": "d4rn50_localballreg_l01_k4"},
        setting=r"$\lambda{=}0.1,\ K{=}4$",
        scores={"SIGReg": (23.3, 35.7), "LocaLeJEPA": (33.0, 42.3)},
    ),
]

# Three classes only: a scatter compares every pair of colours at once, and three
# is the largest set of the house categorical palette that clears the all-pairs
# colour-vision floors. Everything else is folded into one grey "other" cloud.
CLASSES = ["tabby", "zebra", "sports car"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
OTHER = "#c9c8c2"
N_PER_CLASS = 220      # points drawn per class (the val split has 50 per class...)
N_OTHER = 1400         # background cloud
N_ORBIT_IMAGES = 8     # images per class pushed through the eight group views
N_ORBIT_DRAWN = 2      # rosettes drawn per class


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def class_folders(names):
    """Map ImageNet class names to their (zero-padded) label index via the patch-PCA aliases."""
    aliases = yaml.safe_load(open(PATCH_PCA_CONFIG))["class_aliases"]
    missing = [n for n in names if n not in aliases]
    if missing:
        raise SystemExit(f"unknown class name(s) {missing}; see class_aliases in {PATCH_PCA_CONFIG}")
    return {n: str(aliases[n]) for n in names}


def feature_cache_dir():
    cfg = yaml.safe_load(open(INDOMAIN_CONFIG))
    return cfg["cache_dir"]


def load_val_features(run):
    """Cached (50000, 512) val embeddings + labels of one pretrain, as float64."""
    import torch
    path = os.path.join(feature_cache_dir(), run, VAL_FEATURES)
    if not os.path.exists(path):
        raise SystemExit(f"missing feature cache {path}; run src/eval_indomain.py for {run} first")
    blob = torch.load(path, map_location="cpu")
    return blob["X"].float().numpy().astype(np.float64), blob["y"].numpy()


def d4_views(image):
    """The eight D4 views of one (C, H, W) image: rot90^k and the flipped rot90^k."""
    import torch
    views = []
    for flip in (False, True):
        base = torch.flip(image, dims=[-1]) if flip else image
        views += [torch.rot90(base, k, dims=(-2, -1)) for k in range(4)]
    return torch.stack(views)


def extract_orbits(path, labels):
    """Embed the eight group views of a few val images with both D4 checkpoints.

    Writes an NPZ with, per objective, the (n_images, 8, 512) orbit embeddings,
    plus the regular-representation matrices and the change of basis that block-
    diagonalizes them into irreducible representations (both from escnn, so the
    plot never re-derives D4's conventions).
    """
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms
    from escnn import gspaces

    from src.eval.extract import load_backbone

    gspace = gspaces.flipRot2dOnR2(N=4)
    regular = gspace.regular_repr
    elements = list(gspace.fibergroup.elements)
    rho = np.stack([np.asarray(regular(g)) for g in elements])          # (8, 8, 8)
    change_of_basis = np.asarray(regular.change_of_basis)               # (8, 8)
    irrep_sizes = np.array([gspace.fibergroup.irrep(*name).size for name in regular.irreps])

    folders = class_folders(labels)
    transform = transforms.Compose([
        transforms.Resize(CROP), transforms.CenterCrop(CROP),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    images, image_class = [], []
    for index, name in enumerate(labels):
        files = sorted(glob.glob(os.path.join(VAL_IMAGES, folders[name], "*.JPEG")))[:N_ORBIT_IMAGES]
        if not files:
            raise SystemExit(f"no val images under {os.path.join(VAL_IMAGES, folders[name])}")
        for file in files:
            with Image.open(file) as raw:
                images.append(transform(ImageOps.exif_transpose(raw).convert("RGB")))
            image_class.append(index)
    views = torch.stack([d4_views(image) for image in images])          # (n, 8, C, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    column = next(c for c in COLUMNS if c["kind"] == "equivariant")
    config = yaml.safe_load(open(INDOMAIN_CONFIG))
    out = dict(rho=rho, change_of_basis=change_of_basis, irrep_sizes=irrep_sizes,
               elements=np.array([str(g) for g in elements]), image_class=np.array(image_class))
    for row, run in column["runs"].items():
        backbone = next(b for b in config["backbones"] if b["name"] == run)
        extract_fn, _ = load_backbone(
            experiment=backbone["experiment"],
            checkpoint_path=os.path.join(_ROOT, backbone["checkpoint"]),
            device=device, crop_size=CROP, project_root=_ROOT,
            overrides=backbone.get("overrides"),
        )
        orbits = np.stack([extract_fn(view).numpy().astype(np.float64) for view in views])
        out[f"orbits_{row}"] = orbits
        print(f"[orbits] {run}: {orbits.shape}", flush=True)
        del extract_fn

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"[orbits] wrote {path}", flush=True)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def pca2(X, center=True):
    """Project onto the leading two principal directions.

    ``center=False`` keeps the origin fixed, which is what the ball model wants:
    unlike a vector space, the Poincare ball has a distinguished point, and the
    encoder's own tangent space sits at it.
    """
    origin = X.mean(0) if center else np.zeros(X.shape[1])
    _, _, components = np.linalg.svd(X - origin, full_matrices=False)

    def project(vectors):
        return (vectors - origin) @ components[:2].T

    return project(X), project


def mobius_add(x, y, c=HYP_C):
    """Mobius addition on the Poincare ball; ``mobius_add(-m, x)`` is the isometry
    that moves ``m`` to the origin, so it re-centres a cloud without distorting a
    single hyperbolic distance."""
    xy = (x * y).sum(-1, keepdims=True)
    x2 = (x * x).sum(-1, keepdims=True)
    y2 = (y * y).sum(-1, keepdims=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    return numerator / (1 + 2 * c * xy + c * c * x2 * y2)


def logmap0(x, c=HYP_C):
    """Poincare-ball point -> tangent vector at the origin."""
    sqrt_c = np.sqrt(c)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    scaled = np.clip(sqrt_c * norm, 0.0, 1.0 - 1e-9)
    return np.arctanh(scaled) * x / (sqrt_c * np.maximum(norm, 1e-12))


def expmap0(u, c=HYP_C):
    """Tangent vector at the origin -> Poincare-ball point."""
    sqrt_c = np.sqrt(c)
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    return np.tanh(sqrt_c * norm) * u / (sqrt_c * np.maximum(norm, 1e-12))


def hyperbolic_radius(x, c=HYP_C):
    """Hyperbolic distance from the origin of ball points ``x``."""
    sqrt_c = np.sqrt(c)
    norm = np.clip(sqrt_c * np.linalg.norm(x, axis=-1), 0.0, 1.0 - 1e-9)
    return 2.0 * np.arctanh(norm) / sqrt_c


def mobius_geodesic(z1, z2, n=200):
    """The Poincare-disk geodesic between two unit-disk points: a circular arc
    orthogonal to the boundary (a straight chord when the two points and the
    origin are collinear)."""
    x1, y1 = z1
    x2, y2 = z2
    n1, n2 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
    det = 2.0 * (x1 * y2 - x2 * y1)
    if abs(det) < 1e-9:                                   # diameter
        return np.stack([np.linspace(z1[0], z2[0], n), np.linspace(z1[1], z2[1], n)], axis=1)
    cx = (y2 * (n1 + 1) - y1 * (n2 + 1)) / det
    cy = (x1 * (n2 + 1) - x2 * (n1 + 1)) / det
    radius = np.sqrt(max(cx * cx + cy * cy - 1.0, 1e-12))
    a1 = np.arctan2(y1 - cy, x1 - cx)
    a2 = np.arctan2(y2 - cy, x2 - cx)
    if a2 - a1 > np.pi:
        a2 -= 2 * np.pi
    elif a1 - a2 > np.pi:
        a2 += 2 * np.pi
    angles = np.linspace(a1, a2, n)
    return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)


def irrep_plane(orbits, change_of_basis, irrep_sizes, weights=None):
    """Coordinates of D4 regular-representation embeddings in a 2-d irrep plane.

    ``orbits`` is (..., 512) = (..., 64 fields, 8 group channels). The change of
    basis block-diagonalizes the regular representation into its irreducible
    parts; the two-dimensional irrep occurs twice per field, and the group acts
    on each of those copies by the same 2x2 matrix. Taking a fixed linear
    combination of the copies (the leading direction of the multiplicity space,
    which commutes with the group action) therefore gives a plane in which a
    group element acts exactly as a rotation or a reflection, the same plane for
    every sample.
    """
    fields = orbits.reshape(*orbits.shape[:-1], -1, GROUP_ORDER)
    coordinates = fields @ np.linalg.inv(change_of_basis).T
    offsets = np.concatenate([[0], np.cumsum(irrep_sizes)])
    blocks = [slice(offsets[i], offsets[i + 1]) for i, size in enumerate(irrep_sizes) if size == 2]
    if not blocks:
        raise SystemExit("no 2-dimensional irrep in the regular representation")
    pairs = np.concatenate([coordinates[..., block] for block in blocks], axis=-2)   # (..., copies, 2)
    if weights is None:
        flat = pairs.reshape(-1, *pairs.shape[-2:])
        gram = np.einsum("nci,ndi->cd", flat, flat)
        weights = np.linalg.eigh(gram)[1][:, -1]
    return np.einsum("...cd,c->...d", pairs, weights), weights


def group_action(features, rho):
    """Act with every group element on a batch of regular-representation vectors.

    ``features`` is (n, 512) = (n, 64 fields, |G| group channels); the result is
    (n, |G|, 512), the orbit the architecture defines for each embedding.
    """
    fields = features.reshape(len(features), -1, GROUP_ORDER)
    return np.einsum("gij,nfj->ngfi", rho, fields).reshape(len(features), GROUP_ORDER, -1)


def orbit_agreement(orbits, rho):
    """Cosine between the measured embedding of each group view of an image and
    the group acting on the embedding of the untransformed image.

    Exact equivariance means 1 for every view. A strided encoder only reaches it
    for the group elements that preserve the subsampling lattice (here the
    transpose reflection), so this reports how well the architecture's guarantee
    survives discretization at 128 px.
    """
    acted = group_action(orbits[:, 0], rho)
    dot = (orbits * acted).sum(-1)
    return dot / (np.linalg.norm(orbits, axis=-1) * np.linalg.norm(acted, axis=-1))


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def style(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(INK3)
        side.set_linewidth(0.6)


def note(ax, text, color=INK2):
    """A one-line measurement under a panel."""
    ax.text(0.5, -0.035, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.8, color=color)


def draw_euclidean(ax, X, y, class_ids, rng):
    """2-d PCA of the l2-normalized embedding on a flat grid."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    keep = rng.choice(np.flatnonzero(~np.isin(y, class_ids)), N_OTHER, replace=False)
    subset = np.concatenate([keep] + [np.flatnonzero(y == c)[:N_PER_CLASS] for c in class_ids])
    Z, _ = pca2(X[subset])
    Z /= np.abs(Z).max()
    ax.scatter(*Z[:len(keep)].T, s=3.2, c=OTHER, lw=0, alpha=0.55, rasterized=True)
    start = len(keep)
    means = []
    for color, c in zip(SERIES, class_ids):
        n = min(N_PER_CLASS, int((y == c).sum()))
        points = Z[start:start + n]
        start += n
        ax.scatter(*points.T, s=6.5, c=color, lw=0, alpha=0.9, rasterized=True)
        mean = points.mean(0)
        means.append(mean)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
        angle = np.degrees(np.arctan2(*eigenvectors[:, -1][::-1]))
        ax.add_patch(Ellipse(mean, *(3.0 * np.sqrt(eigenvalues[::-1])), angle=angle,
                             fc="none", ec=color, lw=1.0, alpha=0.85))
    # the geodesic of a flat space: the straight segment between two class means
    ax.plot(*np.array(means[:2]).T, color=INK2, lw=1.0, ls=(0, (4, 2)), zorder=4)
    for x in np.linspace(-1, 1, 9):
        ax.axvline(x, color=INK3, lw=0.3, alpha=0.25, zorder=0)
        ax.axhline(x, color=INK3, lw=0.3, alpha=0.25, zorder=0)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    style(ax)
    sample = X[rng.choice(len(X), 2000, replace=False)]
    cosine = sample @ sample.T
    angles = np.degrees(np.arccos(np.clip(cosine[np.triu_indices(len(sample), 1)], -1, 1)))
    return float(np.median(angles))


def draw_hyperbolic(ax, X, y, class_ids, rng):
    """The Poincare disk, re-centred on the cloud by a Mobius isometry.

    The clip makes every embedding sit on one shell around the ball origin, so
    the informative geometry is the arrangement *within* that shell. Moving the
    cloud's mean to the origin is an isometry of the ball: no hyperbolic distance
    changes, the drawn rings stay rings of equal hyperbolic radius, and the ball
    origin itself shows up as the open marker.

    Points are the exponential map of their projection onto the two leading
    tangent directions of the re-centred cloud, so the drawn radius is the
    hyperbolic distance that survives the projection (a lower bound on the real
    one) and the open marker is where the ball origin lands under the same map.
    """
    ball_radius = HYP_C ** -0.5
    keep = rng.choice(np.flatnonzero(~np.isin(y, class_ids)), N_OTHER, replace=False)
    subset = np.concatenate([keep] + [np.flatnonzero(y == c)[:N_PER_CLASS] for c in class_ids])
    centre = expmap0(logmap0(X[subset]).mean(0, keepdims=True))
    recentred = mobius_add(-centre, X[subset])
    projected, project = pca2(logmap0(recentred), center=False)
    Z = expmap0(projected) / ball_radius                   # unit-disk coordinates
    origin_mark = (expmap0(project(logmap0(-centre))) / ball_radius)[0]

    # a hyperbolic polar grid: the diameters are geodesics and the rings have
    # equal *hyperbolic* radius, so their crowding towards the boundary is the
    # curvature of the space
    for angle in np.arange(8) * np.pi / 8:
        direction = np.array([np.cos(angle), np.sin(angle)])
        ax.plot(*np.stack([-direction, direction]).T, color=INK3, lw=0.3, alpha=0.2, zorder=0)
    for distance in range(1, 7):
        radius = np.tanh(np.sqrt(HYP_C) * distance / 2)
        ax.add_patch(Circle((0, 0), radius, fc="none", ec=INK3, lw=0.3, alpha=0.2, zorder=0))
        if distance in (2, 4):
            ax.text(0, radius, f"{distance}", ha="center", va="bottom", fontsize=5.2,
                    color=INK3, zorder=6)
    ax.add_patch(Circle((0, 0), 1.0, fc="none", ec=INK2, lw=1.0, zorder=5))
    ax.scatter(*Z[:len(keep)].T, s=3.2, c=OTHER, lw=0, alpha=0.55, rasterized=True)
    start, means = len(keep), []
    for color, c in zip(SERIES, class_ids):
        n = min(N_PER_CLASS, int((y == c).sum()))
        points = Z[start:start + n]
        start += n
        ax.scatter(*points.T, s=6.5, c=color, lw=0, alpha=0.9, rasterized=True)
        means.append(points.mean(0))
    ax.plot(*mobius_geodesic(means[0], means[1]).T, color=INK2, lw=1.0, ls=(0, (4, 2)), zorder=6)
    ax.plot(*np.stack([[0.0, 0.0], origin_mark]).T, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=5)
    ax.scatter(*origin_mark, s=24, facecolor="white", edgecolor=INK2, lw=0.9, zorder=7)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    style(ax)
    return float(np.median(hyperbolic_radius(recentred))), float(hyperbolic_radius(centre)[0])


def draw_equivariant(ax, orbits, image_class, rho, plane, limit):
    """Group orbits in the 2-d irrep plane of the regular representation.

    Filled markers are the orbit the architecture defines, the group acting on
    the measured embedding of the untransformed image; the group acts on this
    plane exactly as the rotations and reflections of the square. Open markers
    are the measured embeddings of the eight rotated and reflected copies of the
    same image, whose distance from the filled ones is the equivariance that
    strided downsampling costs at 128 px.
    """
    change_of_basis, irrep_sizes, weights = plane
    scale = np.linalg.norm(orbits[:, 0], axis=-1)[:, None, None]
    exact, _ = irrep_plane(group_action(orbits[:, 0], rho) / scale, change_of_basis, irrep_sizes, weights)
    measured, _ = irrep_plane(orbits / scale, change_of_basis, irrep_sizes, weights)
    for angle in np.arange(4) * np.pi / 4:                 # the four mirror axes of D4
        direction = np.array([np.cos(angle), np.sin(angle)]) * limit * 1.12
        ax.plot(*np.stack([-direction, direction]).T, color=INK3, lw=0.3, alpha=0.3, zorder=0)
    for radius in np.linspace(limit / 3, limit, 3):
        ax.add_patch(Circle((0, 0), radius, fc="none", ec=INK3, lw=0.3, alpha=0.25, zorder=0))
    for index, color in enumerate(SERIES):
        for i in np.flatnonzero(image_class == index)[:N_ORBIT_DRAWN]:
            points = exact[i]
            order = np.argsort(np.arctan2(points[:, 1], points[:, 0]))
            ax.plot(*points[np.concatenate([order, order[:1]])].T, color=color, lw=0.7, alpha=0.5, zorder=3)
            ax.scatter(*measured[i].T, s=15, facecolor="none", edgecolor=color, lw=0.7, alpha=0.75, zorder=4)
            ax.scatter(*points.T, s=12, c=color, lw=0, zorder=5)
    ax.set_xlim(-limit * 1.18, limit * 1.18)
    ax.set_ylim(-limit * 1.18, limit * 1.18)
    style(ax)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def panel_context(orbit_cache):
    """Everything the panels share: the orbit cache, one irrep plane and one
    radial scale for both rows, so the two rosettes are drawn to scale."""
    cache = np.load(orbit_cache)
    rho, change_of_basis = cache["rho"], cache["change_of_basis"]
    irrep_sizes, image_class = cache["irrep_sizes"], cache["image_class"]
    orbits = {row: cache[f"orbits_{row}"] for row in ROWS}
    unit = {row: group_action(o[:, 0], rho) / np.linalg.norm(o[:, 0], axis=-1)[:, None, None]
            for row, o in orbits.items()}
    _, weights = irrep_plane(np.concatenate(list(unit.values())), change_of_basis, irrep_sizes)
    plane = (change_of_basis, irrep_sizes, weights)
    limit = max(np.percentile(np.linalg.norm(irrep_plane(u, change_of_basis, irrep_sizes, weights)[0], axis=-1), 98)
                for u in unit.values())
    return dict(rho=rho, plane=plane, limit=limit, orbits=orbits, image_class=image_class,
                ids=[int(folder) for folder in class_folders(CLASSES).values()])


def draw_panel(ax, column, row, context, seed=0):
    """Draw one (encoder family, objective) panel and return its caption line."""
    # a seed per cell, so a panel drawn on its own is the panel of the grid
    rng = np.random.default_rng([seed, COLUMNS.index(column), ROWS.index(row)])
    if column["kind"] == "equivariant":
        draw_equivariant(ax, context["orbits"][row], context["image_class"],
                         context["rho"], context["plane"], context["limit"])
        share = group_channel_energy(load_val_features(column["runs"][row])[0])
        agreement = np.median(orbit_agreement(context["orbits"][row], context["rho"])[:, 1:])
        return f"group energy {100 * share:.0f}%   ·   orbit agreement {agreement:.2f}"
    X, y = load_val_features(column["runs"][row])
    if column["kind"] == "euclidean":
        return r"spread $\angle = %.0f^\circ$" % draw_euclidean(ax, X, y, context["ids"], rng)
    return r"spread $d_{\mathbb{H}} = %.2f$" % draw_hyperbolic(ax, X, y, context["ids"], rng)[0]


def legend_handles():
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=color, mec="none", label=name)
               for color, name in zip(SERIES, CLASSES)]
    return handles + [
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=OTHER, mec="none", label="other classes"),
        Line2D([], [], marker="o", ls="", ms=5.0, mfc="white", mec=INK2, mew=0.9,
               label=r"ball origin ($d_{\mathbb{H}} = 4.60$ from every embedding)"),
    ]


def house_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.facecolor": "white",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(f"{stem}.{extension}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def build(out, orbit_cache, seed=0):
    """The assembled 3 x 2 figure."""
    context = panel_context(orbit_cache)
    house_style()
    fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(5.6, 4.35))
    for column, column_axes in zip(COLUMNS, axes.T):
        for row, ax in zip(ROWS, column_axes):
            note(ax, draw_panel(ax, column, row, context, seed))
            knn, linear = column["scores"][row]
            ax.set_title(f"k-NN {knn:.1f}   ·   linear {linear:.1f}", fontsize=6.4, pad=3.5, color=INK)

    for ax, row in zip(axes[:, 0], ROWS):
        ax.set_ylabel(ROW_LABEL[row], fontsize=7.6, color=INK, labelpad=6)
    for ax, column in zip(axes[1], COLUMNS):
        ax.set_xlabel(column["setting"], fontsize=6.4, color=INK2, labelpad=19)

    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.008), handletextpad=0.3, columnspacing=1.1)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93), h_pad=0.4)
    for ax, column in zip(axes[0], COLUMNS):           # column headings above the panels
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.985, column["title"], ha="center", va="top",
                 fontsize=8.6, color=INK)
        fig.text(box.x0 + box.width / 2, 0.945, column["subtitle"], ha="center", va="top",
                 fontsize=6.4, color=INK2)
    save(fig, out)
    print(f"wrote {out}.pdf / {out}.png")


def build_panels(out_dir, orbit_cache, seed=0):
    """The same six panels as separate, label-free images, plus the legend strip.

    Each file holds nothing but the panel, so the assembling LaTeX supplies the
    column headings, the row labels and the subcaptions; the numbers every panel
    would have carried are printed here for those captions.
    """
    context = panel_context(orbit_cache)
    house_style()
    print(f"panel{'':16s}k-NN   linear   measurement")
    for column in COLUMNS:
        for row in ROWS:
            fig, ax = plt.subplots(figsize=(1.85, 1.85))
            measurement = draw_panel(ax, column, row, context, seed)
            stem = os.path.join(out_dir, f"{column['key']}_{row.lower()}")
            fig.tight_layout(pad=0.12)
            save(fig, stem)
            knn, linear = column["scores"][row]
            plain = measurement.replace("$", "").replace("\\angle", "angle").replace("^\\circ", " deg")
            plain = plain.replace("\\mathbb{H}", "H").replace("d_{H}", "d_H")
            print(f"{column['key'] + '_' + row.lower():20s}{knn:5.1f}  {linear:6.1f}   {plain}")

    fig = plt.figure(figsize=(5.6, 0.3))
    fig.legend(handles=legend_handles(), loc="center", ncol=5, frameon=False, fontsize=6.2,
               handletextpad=0.3, columnspacing=1.1)
    save(fig, os.path.join(out_dir, "legend"))
    print(f"wrote {len(COLUMNS) * len(ROWS)} panels + legend to {out_dir}/ (pdf + png)")


def group_channel_energy(features):
    """Share of the embedding's energy outside the invariant (trivial) component.

    Averaging a field over its group channels is the projection onto the trivial
    representation, i.e. exactly what a D4-*invariant* encoder would keep. The
    remainder is what makes the embedding transform under the group, so this is
    the fraction of the representation that carries orientation.
    """
    fields = features.reshape(len(features), -1, GROUP_ORDER)
    invariant = GROUP_ORDER * fields.mean(-1) ** 2
    return float(np.mean(1 - invariant.sum(-1) / (fields ** 2).sum((-2, -1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(_ROOT, "paper/figs/encoder_geometries"))
    parser.add_argument("--orbit-cache", default=os.path.join(_ROOT, "paper/figs/encoder_geometries_orbits.npz"))
    parser.add_argument("--extract", action="store_true", help="(re)build the D4 orbit cache")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--panels", nargs="?", const=os.path.join(_ROOT, "paper/figs/encoder_geometries_panels"),
                        default=None, metavar="DIR",
                        help="also write the six panels as separate label-free images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.extract or args.extract_only or not os.path.exists(args.orbit_cache):
        extract_orbits(args.orbit_cache, CLASSES)
    if not args.extract_only:
        build(args.out, args.orbit_cache, args.seed)
        if args.panels:
            build_panels(args.panels, args.orbit_cache, args.seed)


if __name__ == "__main__":
    main()

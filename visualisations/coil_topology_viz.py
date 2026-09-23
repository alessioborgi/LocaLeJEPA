"""Per-object views of COIL-100 embeddings: persistence diagrams, PC1/PC2 planes, rotation gifs, frame call-outs."""

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ripser import ripser

from src.data.coil100_datamodule import COIL100Dataset
from src.eval.topology import scaled


def principal_plane(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(0, keepdims=True)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    return centred @ vectors[:2].T


def per_object(features: np.ndarray, objects: np.ndarray) -> tuple[dict[int, float], dict[int, list]]:
    """Longest H1 bar and the full diagrams of every object's own 72 frames."""
    lifetimes, diagrams = {}, {}
    for obj in np.unique(objects):
        dgms = ripser(scaled(features[objects == obj]), maxdim=1)["dgms"]
        diagrams[int(obj)] = dgms
        bars = dgms[1]
        lifetimes[int(obj)] = float((bars[:, 1] - bars[:, 0]).max()) if len(bars) else 0.0
    return lifetimes, diagrams


def geometry(features: np.ndarray, objects: np.ndarray, chosen: list[int]) -> dict[str, float]:
    """How the chosen objects' loops sit relative to each other: size, separation, and the angles between their planes."""
    centroids = np.stack([features[objects == o].mean(0) for o in chosen])
    radii = np.array([np.linalg.norm(features[objects == o] - c, axis=1).mean() for o, c in zip(chosen, centroids)])
    gaps = np.linalg.norm(centroids[:, None] - centroids[None], axis=-1)[np.triu_indices(len(chosen), 1)]
    planes = []
    for o in chosen:
        x = features[objects == o]
        _, _, v = np.linalg.svd(x - x.mean(0), full_matrices=False)
        planes.append(v[:2])
    angles = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            s = np.linalg.svd(planes[i] @ planes[j].T, compute_uv=False)
            angles.append(np.degrees(np.arccos(np.clip(s, -1, 1))))
    angles = np.array(angles)
    return {
        "loop radius": float(radii.mean()),
        "centroid gap": float(gaps.mean()),
        "gap / radius": float(gaps.mean() / radii.mean()),
        "plane angle 1": float(angles[:, 0].mean()),
        "plane angle 2": float(angles[:, 1].mean()),
    }


@dataclass
class Run:
    name: str
    features: np.ndarray
    objects: np.ndarray
    angles: np.ndarray  # radians, dataset order
    lifetimes: dict[int, float]
    diagrams: dict[int, list]

    def frames(self, obj: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Indices, PC1/PC2 plane and degrees of one object's frames, in dataset order."""
        idx = np.where(self.objects == obj)[0]
        return idx, principal_plane(self.features[idx]), np.degrees(self.angles[idx])


def _payload(dataset: COIL100Dataset, embeddings: torch.Tensor, projections: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """The dump layout src/callbacks/coil_topology.py writes and load_run reads."""
    payload = {"embeddings": embeddings, "angles": torch.from_numpy(dataset.angles), "objects": torch.from_numpy(dataset.objects)}
    return payload if projections is None else {**payload, "projections": projections}


def _run(name: str, payload: dict, dataset: COIL100Dataset, head: str) -> Run:
    features = payload[head].numpy().astype(np.float64)
    assert len(features) == len(dataset.paths)  # dump and dataset share the sorted frame order
    objects = payload["objects"].numpy()
    lifetimes, diagrams = per_object(features, objects)
    return Run(name, features, objects, payload["angles"].numpy().astype(np.float64), lifetimes, diagrams)


def load_run(name: str, dump: str | Path, dataset: COIL100Dataset, head: str = "embeddings") -> Run:
    return _run(name, torch.load(dump, map_location="cpu", weights_only=False), dataset, head)


def embed_checkpoint(
    experiment: str,
    ckpt: str | Path | None = None,
    data_dir: str | Path = ROOT / "data" / "coil-100",
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "auto",
) -> dict[str, torch.Tensor]:
    """Embed every COIL-100 frame with a checkpoint's weights, in the layout src/callbacks/coil_topology.py writes.

    The callback only measures when fit completes, so this is the way to score a run that crashed, an intermediate
    checkpoint, or (with ckpt=None) the initialisation as a reference point.
    """
    import os

    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from torch.utils.data import DataLoader

    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])
    GlobalHydra.instance().clear()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    projector = hydra.utils.instantiate(cfg.model.projector)

    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        for prefix, module in (("encoder.", encoder), ("projector.", projector)):
            # Lightning prefixes every key, and torch.compile adds a _orig_mod. wrap inside it
            weights = {k[len(prefix) :].removeprefix("_orig_mod."): v for k, v in state.items() if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(weights, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{prefix[:-1]}: {len(missing)} missing, {len(unexpected)} unexpected keys")

    for module in (encoder, projector):
        module.to(device).eval().requires_grad_(False)

    dataset = COIL100Dataset(data_dir, hydra.utils.instantiate(cfg.data.eval_transform), split="all")
    embeddings, projections = [], []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False):
            images = images.flatten(0, 1) if images.ndim == 5 else images  # the eval transform is multi-view with one view
            features = encoder(images.to(device))
            embeddings.append(features.float().cpu())
            projections.append(projector(features).float().cpu())

    return _payload(dataset, torch.cat(embeddings), torch.cat(projections))


def load_checkpoint(
    name: str, experiment: str, ckpt: str | Path | None, dataset: COIL100Dataset, head: str = "embeddings", **kwargs
) -> Run:
    return _run(name, embed_checkpoint(experiment, ckpt, **kwargs), dataset, head)


def top(run: Run, count: int = 5) -> list[int]:
    return [o for o, _ in sorted(run.lifetimes.items(), key=lambda kv: -kv[1])[:count]]


def dump_pixels(data_dir: str | Path, out: str | Path, size: int = 128) -> Path:
    """Write COIL-100 frames as flat pixel vectors in the layout src/callbacks/coil_topology.py uses, for a raw-pixel baseline."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = COIL100Dataset(data_dir, transform, split="all")
    pixels = torch.stack([dataset[i][0].flatten() for i in range(len(dataset))])
    torch.save(_payload(dataset, pixels), out)
    return Path(out)


def _ceiling(*diagram_lists) -> float:
    """5% above the longest finite death, so censored bars and the diagonal fit in the axes."""
    return 1.05 * max(float(d[np.isfinite(d[:, 1]), 1].max()) for dgms in diagram_lists for d in dgms if len(d))


def diagram_axes(ax, diagrams, ceiling: float, title: str | None = None) -> None:
    ax.plot([0, ceiling], [0, ceiling], color="0.7", linewidth=1)
    for dim, colour, name in ((0, "tab:blue", "H0"), (1, "tab:red", "H1")):
        if dim >= len(diagrams) or not len(diagrams[dim]):
            continue
        bars = diagrams[dim].astype(float).copy()
        bars[np.isinf(bars[:, 1]), 1] = ceiling
        ax.scatter(bars[:, 0], bars[:, 1], s=28, alpha=0.75, color=colour, label=name)
        if dim == 1:
            birth, death = bars[np.argmax(bars[:, 1] - bars[:, 0])]
            foot = (birth + death) / 2  # the perpendicular from the point meets the diagonal here
            ax.plot([birth, foot], [death, foot], color="black", linestyle="--", linewidth=1.2, zorder=3)
            ax.scatter(birth, death, s=28, color="white", zorder=4)  # opaque, so the dash does not show through the marker's alpha
            ax.scatter(birth, death, s=28, alpha=0.75, color=colour, zorder=5)
    ax.axhline(ceiling, color="0.5", linestyle=":", linewidth=1)
    ax.set_xlim(-0.02 * ceiling, ceiling)
    ax.set_ylim(-0.02 * ceiling, ceiling * 1.02)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.legend(loc="lower right")
    if title is not None:
        ax.set_title(title)


def _angle_scatter(ax, plane: np.ndarray, degrees: np.ndarray, size: float, alpha: float = 1.0, zorder: int = 1):
    """The frames of one object in a plane, coloured by turntable angle on the cyclic hsv wheel."""
    return ax.scatter(plane[:, 0], plane[:, 1], c=degrees, cmap="hsv", s=size, alpha=alpha, zorder=zorder)


def plane_axes(ax, run: Run, obj: int, title: str | None = None) -> None:
    _, plane, degrees = run.frames(obj)
    dots = _angle_scatter(ax, plane, degrees, 40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    if title is not None:
        ax.set_title(title)
    ax.figure.colorbar(dots, ax=ax, label="degrees")


def object_figure(run: Run, obj: int, head: str = "embeddings") -> Figure:
    """Persistence diagram of one object's 72 frames next to the same frames in their PC1/PC2 plane."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    diagram_axes(axes[0], run.diagrams[obj], _ceiling(run.diagrams[obj]), "persistence of these 72 frames")
    plane_axes(axes[1], run, obj, "the same frames, coloured by turntable angle")
    fig.suptitle(f"{run.name}: object {obj}, {head}, longest H1 {run.lifetimes[obj]:.3f}")
    fig.tight_layout()
    return fig


def rotation_gif(dataset: COIL100Dataset, run: Run, obj: int, frame_ms: int = 100) -> bytes:
    """Left: the frame. Right: every frame of this object in its PC1/PC2 plane, the current one ringed."""
    idx, plane, degrees = run.frames(obj)
    order = np.argsort(degrees)  # PCA before sorting: SVD signs depend on row order, and object_figure sees dataset order
    plane, degrees, idx = plane[order], degrees[order], idx[order]
    frames = []
    for k, i in enumerate(idx):
        fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3.4))
        left.imshow(Image.open(dataset.paths[i]))
        left.axis("off")
        left.set_title(f"{degrees[k]:.0f}°")
        _angle_scatter(right, plane, degrees, 18, alpha=0.5)
        right.scatter(plane[k, 0], plane[k, 1], s=180, facecolors="none", edgecolors="black", linewidths=2)
        right.set_aspect("equal", adjustable="datalim")
        right.set_xticks([])
        right.set_yticks([])
        right.set_title("PC1 / PC2")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=80)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))
    buffer = io.BytesIO()
    frames[0].save(buffer, format="gif", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
    return buffer.getvalue()


def callout_axes(
    ax,
    dataset: COIL100Dataset,
    run: Run,
    obj: int,
    k: int = 5,
    callouts: tuple[float, ...] | None = None,
    zoom: float = 0.6,
    title: str | None = None,
) -> None:
    """One object's frames in PC1/PC2 with the frames at `callouts` degrees (default: k evenly spaced from 0°) ringed and shown beside the cloud."""
    idx, plane, degrees = run.frames(obj)
    centre = (plane.min(0) + plane.max(0)) / 2
    targets = np.linspace(0, 360, k, endpoint=False) if callouts is None else np.asarray(callouts, dtype=float)
    picked = [int(np.argmin(np.abs((degrees - t + 180) % 360 - 180))) for t in targets]

    _angle_scatter(ax, plane, degrees, 22, alpha=0.6, zorder=2)
    radius = np.abs(plane - centre).max()
    ax.set_xlim(centre[0] - 2.4 * radius, centre[0] + 2.4 * radius)
    ax.set_ylim(centre[1] - 2.4 * radius, centre[1] + 2.4 * radius)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{run.name}: object {obj}," if title is None else title)

    # thumbnails sit on a ring in axes coordinates, in the same angular order as their points, so leaders do not cross;
    # the ring is as wide as the panel allows once a thumbnail box (image + label + frame) fits inside the edge
    box_px = 128 * zoom + 34
    ring = 0.5 - box_px / 2 / ax.get_window_extent().height - 0.02
    polar = np.arctan2(*(plane[picked] - centre).T[::-1])
    order = np.argsort(polar)
    spacing = np.linspace(0, 2 * np.pi, len(picked), endpoint=False)
    # rotate the ring so every thumbnail sits as close to its own point's direction as the even spacing allows
    trial = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    mismatch = np.angle(np.exp(1j * (polar[order][None, :] - spacing[None, :] - trial[:, None])))
    slots = trial[np.argmin((mismatch**2).sum(1))] + spacing
    for j, phi in zip(order, slots):
        i = picked[j]
        ax.scatter(*plane[i], s=200, facecolors="none", edgecolors="black", linewidths=1.8, zorder=3)
        thumb = OffsetImage(Image.open(dataset.paths[idx[i]]), zoom=zoom)
        label = TextArea(f"{degrees[i]:.0f}°", textprops={"size": 9})
        ax.add_artist(
            AnnotationBbox(
                VPacker(children=[thumb, label], align="center", sep=4),
                plane[i],
                xybox=(0.5 + ring * np.cos(phi), 0.5 + ring * np.sin(phi)),
                boxcoords="axes fraction",
                frameon=True,
                pad=0.15,
                bboxprops={"edgecolor": "black", "linewidth": 1.5},
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 1, "shrinkB": 8},
            )
        )


def callout_figure(dataset: COIL100Dataset, run: Run, obj: int, k: int = 5, callouts: tuple[float, ...] | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    callout_axes(ax, dataset, run, obj, k, callouts)
    fig.tight_layout()
    return fig


GRID_COLUMNS = ("frame", "diagram:LocaLeReg", "callout:LocaLeReg", "callout:SIGReg", "diagram:SIGReg")


GROUP_COLOURS = ("#4c72b0", "#dd8452", "#55a868", "#8172b3")


def _column_groups(columns: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Label and column span of each stretch of adjacent columns belonging to one run."""
    groups: list[list] = []
    for j, column in enumerate(columns):
        label = "Image" if column == "frame" else column.partition(":")[2]
        if groups and groups[-1][0] == label:
            groups[-1][2] = j
        else:
            groups.append([label, j, j])
    return [tuple(group) for group in groups]


def _group_bands(fig: Figure, axes, columns: tuple[str, ...], names: list[str], header: float) -> None:
    """Tint each run's columns and write its name in the band above them."""
    colours = {name: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, name in enumerate(names)}
    renderer = fig.canvas.get_renderer()
    tight = [[ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in row] for row in axes]
    left = [min(row[j].x0 for row in tight) for j in range(len(columns))]
    right = [max(row[j].x1 for row in tight) for j in range(len(columns))]
    pad = 0.25 / fig.get_figwidth()
    y0 = min(b.y0 for row in tight for b in row) - pad
    y1 = max(b.y1 for row in tight for b in row) + pad
    top = y1 + header / fig.get_figheight()
    for label, first, last in _column_groups(columns):
        # groups meet halfway across the gap between their columns, so neighbouring boxes never overlap
        x0 = (right[first - 1] + left[first]) / 2 if first > 0 else left[first] - pad
        x1 = (right[last] + left[last + 1]) / 2 if last + 1 < len(columns) else right[last] + pad
        colour = colours.get(label, "0.5")
        fig.add_artist(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                top - y0,
                boxstyle="round,pad=0,rounding_size=0.015",
                transform=fig.transFigure,
                facecolor=colour,
                edgecolor=colour,
                alpha=0.12,
                linewidth=1.5,
                zorder=-1,
            )
        )
        fig.text((x0 + x1) / 2, (y1 + top) / 2, label, ha="center", va="center", fontsize=15, fontweight="bold", color=colour)


def grid_figure(
    dataset: COIL100Dataset,
    runs: dict[str, Run],
    objects: int | list[int] | tuple[int, ...],
    columns: tuple[str, ...] = GRID_COLUMNS,
    k: int = 5,
    callouts: tuple[float, ...] | dict[int, tuple[float, ...]] | None = None,
    panel: float = 5.0,
) -> Figure:
    """One row per object; columns are "frame" or "<diagram|callout>:<run name>", in any order.

    `callouts` gives the call-out frames in degrees, one tuple for every row or a dict per object id; the rest get k evenly spaced from 0°.

    Adjacent columns of one run are tinted together and labelled with the run name above the grid.
    """
    objects = [objects] if isinstance(objects, int) else list(objects)
    header = 0.5  # inches of tinted band above the panels that carries the group name
    fig, axes = plt.subplots(len(objects), len(columns), figsize=(panel * len(columns), panel * len(objects) + header), squeeze=False)
    for row, obj in zip(axes, objects):
        ceiling = _ceiling(*(run.diagrams[obj] for run in runs.values()))
        at = callouts.get(obj) if isinstance(callouts, dict) else callouts
        for ax, column in zip(row, columns):
            kind, _, name = column.partition(":")
            if kind == "frame":
                frame = int(np.flatnonzero((dataset.objects == obj) & (dataset.angles == 0))[0])
                ax.imshow(Image.open(dataset.paths[frame]))
                ax.axis("off")
                ax.set_title(f"object {obj}")
            elif kind == "diagram":
                diagram_axes(ax, runs[name].diagrams[obj], ceiling, f"longest H1 {runs[name].lifetimes[obj]:.3f}")
            elif kind == "callout":
                callout_axes(ax, dataset, runs[name], obj, k, at, zoom=0.3 * panel / 3.4, title="PC1 / PC2")
            else:
                raise ValueError(column)

    fig.tight_layout(rect=(0, 0, 1, 1 - (header + 0.1) / fig.get_figheight()))
    _group_bands(fig, axes, columns, list(runs), header)
    return fig


def overlay_figure(runs: dict[str, Run], chosen: list[int], title: str) -> tuple[Figure, dict[str, dict[str, float]]]:
    """The chosen objects in one shared PC1/PC2 plane, one panel per run, with their loop geometry."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 5.5), squeeze=False)
    rows = {}
    for axis, run in zip(axes[0], runs.values()):
        mask = np.isin(run.objects, chosen)
        plane, sub_objects, sub_angles = principal_plane(run.features[mask]), run.objects[mask], run.angles[mask]
        for k, obj in enumerate(chosen):
            order = np.argsort(sub_angles[sub_objects == obj])
            pts = plane[sub_objects == obj][order]
            pts = np.vstack([pts, pts[:1]])
            axis.plot(pts[:, 0], pts[:, 1], color=f"C{k}", linewidth=1, alpha=0.7)
            axis.scatter(pts[:-1, 0], pts[:-1, 1], color=f"C{k}", s=14, label=f"obj {obj}")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(run.name)
        axis.legend(fontsize=8)
        rows[run.name] = geometry(run.features, run.objects, chosen)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, rows


def geometry_table(rows: dict[str, dict[str, float]]) -> str:
    keys = list(next(iter(rows.values())))
    lines = [f"{'':>16}" + "".join(f"{name:>14}" for name in rows)]
    lines += [f"{key:>16}" + "".join(f"{rows[name][key]:>14.3f}" for name in rows) for key in keys]
    return "\n".join(lines)

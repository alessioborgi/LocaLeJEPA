"""Forward+backward time and peak memory of the regularizers on a synthetic (V, N, D) batch.

    uv run python scripts/benchmark_regularizers.py --sweep batch --values 32 64 128 256 512
    uv run python scripts/benchmark_regularizers.py --sweep clusters --values 1 2 4 8 16 32 --csv out.csv
"""

import argparse
import csv
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import ballreg_loss, local_ballreg_loss, local_sigreg_loss, sigreg_loss  # noqa: E402

BACKENDS = {"torch": "torch", "fast": "fast_pytorch_kmeans"}
METHODS = ["sigreg", "ballreg", "local_sigreg/torch", "local_sigreg/fast", "local_ballreg/torch", "local_ballreg/fast"]


def loss_fn(name, a):
    if name == "sigreg":
        return partial(sigreg_loss, knots=a.knots)
    if name == "ballreg":
        return partial(ballreg_loss, knots=a.knots, n_quad=a.n_quad)
    target, backend = name.split("/")
    local = local_sigreg_loss if target == "local_sigreg" else local_ballreg_loss
    inner = partial(sigreg_loss, knots=a.knots) if target == "local_sigreg" else partial(ballreg_loss, knots=a.knots, n_quad=a.n_quad, center=True)
    return partial(local, loss=inner, num_clusters=a.clusters, kmeans_iters=a.iters, kmeans_backend=BACKENDS[backend])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(fn, a, device):
    torch.manual_seed(0)
    z = torch.randn(a.views, a.batch, a.dim, device=device, requires_grad=True)
    for _ in range(a.warmup):
        fn(z).backward()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(a.steps):
        z.grad = None
        sync(device)
        t0 = time.perf_counter()
        fn(z).backward()
        sync(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return statistics.median(times) * 1e3, peak


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="none", choices=["none", "batch", "clusters", "dim", "views", "iters", "knots"])
    p.add_argument("--values", type=int, nargs="+", default=[])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--knots", type=int, default=17)
    p.add_argument("--n_quad", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", type=Path)
    a = p.parse_args()
    device = torch.device(a.device)

    print(f"{'method':<20}{a.sweep:>9}{'ms':>10}{'MiB':>9}{'vs global':>11}")
    rows = []
    for value in a.values or [None]:
        if value is not None:
            setattr(a, a.sweep, value)
        global_ms = {}
        for name in a.methods:
            ms, peak = measure(loss_fn(name, a), a, device)
            base = name.split("/")[0]
            if "/" not in name:
                global_ms[base] = ms
            ref = global_ms.get(base.removeprefix("local_"))
            ratio = f"{ms / ref:.2f}x" if "/" in name and ref else "-"
            print(f"{name:<20}{str(value or '-'):>9}{ms:>10.3f}{peak:>9.1f}{ratio:>11}")
            rows.append(dict(method=name, sweep=a.sweep, value=value, ms=ms, peak_mib=peak, views=a.views, batch=a.batch, dim=a.dim, clusters=a.clusters, iters=a.iters, knots=a.knots))

    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

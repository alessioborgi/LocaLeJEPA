# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Export sample bobblehead frames for the main-paper figure.

Every frame shows both figurines, which rotate at independent rates, so the
label space is a 2-torus. To make that visible we export two pairs of frames:
one pair in which figurine 0 has turned while figurine 1 is held nearly fixed,
and one pair with the roles exchanged. Each frame is written title-free as PNG
and PDF (subcaptions carry the angles), and the chosen angles are printed.

    python visualisations/plot_yoda_samples.py [out_dir]
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
ZIP = os.path.join(_ROOT, "data", "yoda_dog_images_and_labels.zip")
FIG_DIR = os.path.join(_HERE, "figures", "yoda_samples")


def _circ_dist(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def _closest(a0, a1, want0, want1):
    """Index of the frame nearest to a target angle pair, on the torus."""
    return int(np.argmin(_circ_dist(a0, want0) ** 2 + _circ_dist(a1, want1) ** 2))


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else FIG_DIR
    z = zipfile.ZipFile(ZIP)
    lab = np.load(io.BytesIO(z.read("labels.npy"))).T          # (N, 2): (theta_0, theta_1)
    # The frame array is ~1.9 GB: extract it once next to the archive and memory-map it,
    # so only the handful of frames we export is ever read into RAM.
    cache = os.environ.get("BOBBLEHEAD_CACHE", os.path.join(os.path.dirname(ZIP), "color_image_export.npy"))
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with z.open("color_image_export.npy") as src, open(cache, "wb") as dst:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                dst.write(chunk)
    images = np.load(cache, mmap_mode="r")
    a0, a1 = lab[:, 0], lab[:, 1]
    hold1, hold0 = float(np.median(a1)), float(np.median(a0))

    # two frames sweeping theta_0 at fixed theta_1, then two sweeping theta_1
    picks = [("f0_a", _closest(a0, a1, 2.3, hold1)), ("f0_b", _closest(a0, a1, 5.6, hold1)),
             ("f1_a", _closest(a0, a1, hold0, 3.0)), ("f1_b", _closest(a0, a1, hold0, 5.8))]

    os.makedirs(out_dir, exist_ok=True)
    for name, idx in picks:
        im = Image.fromarray(images[idx].astype(np.uint8))
        base = os.path.join(out_dir, f"frame_{name}")
        im.save(base + ".png")
        im.save(base + ".pdf")            # vector container around the raster frame
        print(f"wrote {base}.png/.pdf  frame {idx}  theta_0={a0[idx]:.2f} rad  theta_1={a1[idx]:.2f} rad")


if __name__ == "__main__":
    main()

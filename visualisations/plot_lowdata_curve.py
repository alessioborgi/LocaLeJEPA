# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plot the Yoda low-data learning curves from results_lowdata.json."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This file lives in <repo>/visualisations/; read source metrics from the repo's
# logs/ and write rendered figures next to this script under figures/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "logs/eval_angle_probe_lowdata")
OUT = os.path.join(_HERE, "figures")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(DATA, "results_lowdata.json")) as f:
    S = json.load(f)

sizes = S["sizes"]
names = list(S["backbones"].keys())
cmap = plt.get_cmap("tab10")
colors = {n: cmap(i) for i, n in enumerate(names)}

for metric, ylabel, fname, better in [
    ("mae", "test MAE (deg)", "lowdata_mae.png", "min"),
    ("r2", "test R²", "lowdata_r2.png", "max"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for a in range(2):
        ax = axes[a]
        for n in names:
            m = [S["backbones"][n]["sizes"][str(ts)]["angles"][f"angle{a}"][f"{metric}_mean"] for ts in sizes]
            s = [S["backbones"][n]["sizes"][str(ts)]["angles"][f"angle{a}"][f"{metric}_std"] for ts in sizes]
            eff = [S["backbones"][n]["sizes"][str(ts)]["eff_train"] for ts in sizes]
            ax.errorbar(eff, m, yerr=s, marker="o", ms=4, lw=1.5, capsize=2,
                        color=colors[n], label=n)
        ax.set_xscale("log")
        ax.set_title(f"figure {a}")
        ax.set_xlabel("effective # training frames (log)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        if metric == "r2":
            ax.set_ylim(-0.2, 1.02)
    axes[0].legend(fontsize=8, loc="lower right" if better == "max" else "upper right")
    fig.suptitle(f"Yoda angle-probe learning curve ({S['n_seeds']} seeds, mean ± std)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    # Always also emit a vector PDF alongside the raster PNG.
    fig.savefig(os.path.join(OUT, os.path.splitext(fname)[0] + ".pdf"))
    print("wrote", os.path.join(OUT, fname), "(+ .pdf)")

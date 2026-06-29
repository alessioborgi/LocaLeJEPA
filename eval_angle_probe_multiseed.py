# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-seed linear-probe eval on the bobblehead (Yoda) rotation angles.

Re-uses the cached frozen features written by ``eval_angle_probe.py`` and runs
the *same* closed-form ridge probe (:func:`circular_linear_probe`) over N
independent train/val/test splits (one per seed). Reports per-backbone, per-angle
mean +/- std of the held-out test R^2 and mean angular error (deg).

    python eval_angle_probe_multiseed.py --config configs/eval_angle_probe_yoda_vit.yaml --n-seeds 10
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml

from src.eval.angle_probe import circular_linear_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg["cache_dir"]
    out_dir = cfg["output_folder"]
    angle_labels = cfg.get("angle_labels", ["figure 0", "figure 1"])
    # Probe kwargs identical to the single-seed eval, minus the seed (we sweep it).
    probe_kwargs = dict(cfg.get("probe", {}))
    probe_kwargs.pop("seed", None)
    seeds = list(range(args.n_seeds))

    summary = {}
    print("=" * 96)
    print(f"Multi-seed linear probe — {args.n_seeds} splits (seeds {seeds[0]}..{seeds[-1]})")
    print("=" * 96)

    for entry in cfg["backbones"]:
        name = entry["name"]
        cache_path = os.path.join(cache_dir, f"{name}_features.pt")
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"Cached features not found: {cache_path}. Run eval_angle_probe.py first."
            )
        blob = torch.load(cache_path, map_location="cpu")
        X, y = blob["X"].numpy(), blob["y"].numpy()
        if y.ndim == 1:
            y = y[:, None]
        n_ang = y.shape[1]

        # collect[k]["r2"|"mae"] -> list over seeds
        collect = [{"r2": [], "mae": []} for _ in range(n_ang)]
        for s in seeds:
            metrics, _ = circular_linear_probe(X, y, seed=s, **probe_kwargs)
            for k in range(n_ang):
                t = metrics[f"angle{k}"]["test"]
                collect[k]["r2"].append(t["r2"])
                collect[k]["mae"].append(t["mae_deg"])

        summary[name] = {"n_seeds": args.n_seeds, "n_frames": int(X.shape[0]), "angles": {}}
        print(f"\n--- {name}  (N={X.shape[0]} frames, D={X.shape[1]}) ---")
        for k in range(n_ang):
            r2 = np.array(collect[k]["r2"])
            mae = np.array(collect[k]["mae"])
            lbl = angle_labels[k] if k < len(angle_labels) else f"angle{k}"
            summary[name]["angles"][f"angle{k}"] = {
                "label": lbl,
                "r2_mean": float(r2.mean()), "r2_std": float(r2.std(ddof=1)),
                "mae_mean": float(mae.mean()), "mae_std": float(mae.std(ddof=1)),
                "r2_per_seed": r2.tolist(), "mae_per_seed": mae.tolist(),
            }
            print(f"  {lbl:10s}  R2 = {r2.mean():.4f} +/- {r2.std(ddof=1):.4f}"
                  f"   MAE = {mae.mean():.2f} +/- {mae.std(ddof=1):.2f} deg")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results_multiseed.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "results_multiseed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["backbone", "n_seeds"]
        for k in range(2):
            head += [f"angle{k}_r2_mean", f"angle{k}_r2_std", f"angle{k}_mae_mean", f"angle{k}_mae_std"]
        w.writerow(head)
        for name, b in summary.items():
            row = [name, b["n_seeds"]]
            for k in range(2):
                a = b["angles"].get(f"angle{k}")
                if a:
                    row += [f'{a["r2_mean"]:.6f}', f'{a["r2_std"]:.6f}',
                            f'{a["mae_mean"]:.4f}', f'{a["mae_std"]:.4f}']
                else:
                    row += ["", "", "", ""]
            w.writerow(row)
    print(f"\nWritten: {out_dir}/results_multiseed.json and results_multiseed.csv")


if __name__ == "__main__":
    main()

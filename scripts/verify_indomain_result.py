# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Independent audit of every in-domain row of the main table (plan gate G7 / V5).

Generalises the plain-ViT audit to every backbone an eval config names. For each row it
re-derives the numbers with code paths that share NOTHING with ``src/eval_indomain.py``, plus
the controls that would catch the usual ways such a result can be a bug:

  1. cache integrity: shapes, NaN/Inf, duplicated rows, feature norms, RankMe;
  2. labels: cached y == ImageFolder targets in loader order; 50 val img/class;
  3. train/val disjointness: no val feature vector present in the train bank;
  4. independent weighted k-NN (argsort + bincount, fp32) vs the reported numbers;
  5. label-shuffle control: permuted bank labels must give ~0.1% top-1;
  6. independent linear probe: sklearn multinomial logistic regression on a stratified
     50 img/class subset of train (different optimiser, no GPU);
  7. cache <-> checkpoint linkage: re-extract the first N val images from the checkpoint with
     a fresh loader and compare to the cached rows (cos ~ 1), and to EVERY OTHER row's cached
     rows (cos << 1) -- the check that rules out a swapped or duplicated checkpoint across the
     whole table, not just between two rows.

    python scripts/verify_indomain_result.py --config configs/eval/indomain_in1k_campaign.yaml \\
        [--only <name>,<name>] [--root <eval run root>] --out logs/verify/indomain.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def backbones_from_config(config_path: Path, only: list[str] | None = None) -> dict[str, tuple[str, str, str]]:
    """name -> (experiment, checkpoint, feature) for the rows to audit. `_best` twins are
    included when named explicitly and skipped otherwise: the headline rows are last.ckpt."""
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    feature = str(cfg.get("feature", "encoder"))
    rows = {}
    for entry in cfg["backbones"]:
        name = entry["name"]
        if only is not None and name not in only:
            continue
        if only is None and name.endswith("_best"):
            continue
        rows[name] = (entry["experiment"], entry["checkpoint"], feature)
    if only is not None and set(only) - set(rows):
        raise ValueError(f"not in {config_path}: {sorted(set(only) - set(rows))}")
    if not rows:
        raise ValueError(f"{config_path} names no backbone to audit")
    return rows


def eval_settings(config_path: Path) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return {"cache_dir": cfg.get("cache_dir"), "output_folder": cfg.get("output_folder"),
            "crop_size": int(cfg.get("crop_size", 128)), "data_dir": cfg.get("data_dir", "data/imagenet"),
            "report_k": int(cfg.get("knn", {}).get("report_k", 20)), "dataset": cfg.get("dataset", "imagenet1k")}


def log(msg, out):
    print(msg, flush=True)
    out.append(msg)


def rankme(X) -> float:
    import torch

    s = torch.linalg.svdvals(X.float() - X.float().mean(0, keepdim=True))
    p = s / s.sum() + 1e-7
    return float(torch.exp(-(p * p.log()).sum()))


def knn_independent(Xb, yb, Xq, yq, k=20, T=0.07, C=1000, device="cuda", chunk=256, shuffle_labels=False):
    """Cosine weighted k-NN written from scratch: fp32 bank, argsort, bincount."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        yb = yb[torch.randperm(len(yb), generator=g)] if shuffle_labels else yb
        bank = F.normalize(Xb.float(), dim=1).to(device)
        ybd = yb.to(device)
        correct = 0
        for s in range(0, len(Xq), chunk):
            q = F.normalize(Xq[s:s + chunk].float(), dim=1).to(device)
            sim = q @ bank.T
            order = sim.argsort(dim=1, descending=True)[:, :k]
            w = (sim.gather(1, order) / T).exp()
            lab = ybd[order]
            votes = torch.zeros(q.shape[0], C, device=device)
            for j in range(k):
                votes[torch.arange(q.shape[0], device=device), lab[:, j]] += w[:, j]
            correct += int((votes.argmax(1).cpu() == yq[s:s + chunk]).sum())
    return 100.0 * correct / len(Xq)


def main(argv=None) -> None:
    import numpy as np  # noqa: F401  (sklearn needs it below)
    import torch
    import torch.nn.functional as F

    from src.eval.extract import load_backbone
    from src.eval.indomain import get_indomain_splits
    from src.eval.io import feature_cache_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--only", default=None, help="comma-separated row names; default: every last.ckpt row")
    ap.add_argument("--root", default=".", help="the eval run root the config's relative paths resolve against")
    ap.add_argument("--out", default="logs/verify/indomain.md")
    ap.add_argument("--n-reextract", type=int, default=2048)
    ap.add_argument("--per-class-logreg", type=int, default=50)
    args = ap.parse_args(argv)
    os.chdir(args.root)

    settings = eval_settings(args.config)
    rows = backbones_from_config(args.config, args.only.split(",") if args.only else None)
    reported_path = Path(settings["output_folder"]) / "results.json"
    reported = json.loads(reported_path.read_text()) if reported_path.is_file() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = [f"# Independent audit of {args.config} ({time.strftime('%Y-%m-%d %H:%M')}), rows: {', '.join(rows)}", ""]

    train_ds, val_ds, C = get_indomain_splits(settings["dataset"], settings["data_dir"], settings["crop_size"])
    val_targets = torch.tensor(val_ds.targets)
    train_targets = torch.tensor(train_ds.targets)

    feats = {}
    for name, (exp, ckpt, feature) in rows.items():
        log(f"\n## {name}", out)
        epoch = torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch")

        def _cached(split, _name=name, _ckpt=ckpt, _epoch=epoch, _feature=feature):
            path = feature_cache_path(settings["cache_dir"], _name, settings["dataset"], split,
                                      settings["crop_size"], _epoch, _ckpt, feature=_feature)
            blob = torch.load(path, map_location="cpu")
            return blob["X"], blob["y"]

        Xtr, ytr = _cached("train")
        Xva, yva = _cached("val")
        feats[name] = (Xtr, ytr, Xva, yva)

        nan = int(torch.isnan(Xtr).sum() + torch.isnan(Xva).sum() + torch.isinf(Xtr).sum() + torch.isinf(Xva).sum())
        norms = Xva.float().norm(dim=1)
        dup_val = len(Xva) - len({hashlib.md5(r.numpy().tobytes()).hexdigest() for r in Xva})
        log(f"- cache shapes train {tuple(Xtr.shape)} val {tuple(Xva.shape)} dtype {Xtr.dtype}; NaN/Inf: {nan}; "
            f"val norm mean {norms.mean():.2f} std {norms.std():.2f} min {norms.min():.2f}; duplicated val rows: {dup_val}; "
            f"RankMe(val): {rankme(Xva):.1f}", out)
        log(f"- labels: cached val y == ImageFolder targets: {bool(torch.equal(yva, val_targets))}; "
            f"cached train y == ImageFolder targets: {bool(torch.equal(ytr, train_targets))}; "
            f"val per-class count min/max: {int(torch.bincount(yva, minlength=C).min())}/{int(torch.bincount(yva, minlength=C).max())}", out)
        tr_hashes = {hashlib.md5(r.numpy().tobytes()).hexdigest() for r in Xtr}
        overlap = sum(hashlib.md5(r.numpy().tobytes()).hexdigest() in tr_hashes for r in Xva)
        log(f"- val rows found verbatim in the train bank: {overlap} / {len(Xva)}", out)

        k = settings["report_k"]
        acc = knn_independent(Xtr, ytr, Xva, yva, k=k, C=C, device=device)
        rep = reported.get(name, {}).get("knn", {}).get(str(k), {}).get("top1")
        sh = knn_independent(Xtr, ytr, Xva, yva, k=k, C=C, device=device, shuffle_labels=True)
        log(f"- independent kNN@{k} top-1: **{acc:.2f}%** (reported by src/eval_indomain.py: {rep}); "
            f"label-shuffle control: {sh:.2f}% (chance = {100.0 / C:.2f}%)", out)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        g = torch.Generator().manual_seed(0)
        idx = []
        for c in range(C):
            ii = (ytr == c).nonzero().flatten()
            idx.append(ii[torch.randperm(len(ii), generator=g)[:args.per_class_logreg]])
        idx = torch.cat(idx)
        sc = StandardScaler().fit(Xtr[idx].float().numpy())
        clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=8).fit(sc.transform(Xtr[idx].float().numpy()), ytr[idx].numpy())
        lin = 100.0 * clf.score(sc.transform(Xva.float().numpy()), yva.numpy())
        rep_lin = reported.get(name, {}).get("linear", {}).get("top1")
        log(f"- independent linear probe (sklearn LBFGS logistic regression, {args.per_class_logreg} img/class = {len(idx)} train): "
            f"**{lin:.2f}%** val top-1 (full-train probe reported: {rep_lin})", out)

    log("\n## Cache <-> checkpoint linkage (fresh re-extraction of the first "
        f"{args.n_reextract} val images, loader order; every row against every row)", out)
    sub = torch.utils.data.Subset(val_ds, list(range(args.n_reextract)))
    fresh = {}
    for name, (exp, ckpt, feature) in rows.items():
        fn, dim = load_backbone(exp, ckpt, device, crop_size=settings["crop_size"], feature=feature)
        loader = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=False, num_workers=4)
        fresh[name] = torch.cat([fn(x) for x, _ in loader])
    for name in rows:
        cached = feats[name][2][: args.n_reextract].float()
        row = []
        for other in rows:
            if fresh[other].shape[1] != cached.shape[1]:
                row.append(f"vs fresh({other}): different width")
                continue
            cos = F.cosine_similarity(fresh[other], cached, dim=1)
            row.append(f"vs fresh({other}): mean cos {cos.mean():.4f} (min {cos.min():.3f})")
        log(f"- cached {name} " + "; ".join(row), out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()

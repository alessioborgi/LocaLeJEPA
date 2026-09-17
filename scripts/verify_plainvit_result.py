# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Independent audit of the plain ViT-S/8 IN1K result (report_plainViT_LastExp.md).

Re-derives the k-NN / linear numbers for vit_sigreg and vit_localballreg_l01_k4
with code paths that do NOT share anything with eval_indomain.py, plus controls
that would catch the usual ways such a result can be a bug:

  1. cache integrity: shapes, NaN/Inf, duplicated rows, feature norms, RankMe;
  2. labels: cached y == ImageFolder targets in loader order; 50 val img/class;
  3. train/val disjointness: no val feature vector present in the train bank;
  4. independent weighted k-NN (argsort + bincount, fp32) vs the reported numbers;
  5. label-shuffle control: permuted bank labels must give ~0.1% top-1;
  6. independent linear probe: sklearn multinomial logistic regression on a
     stratified 50 img/class subset of train (different optimiser, no GPU);
  7. cache <-> checkpoint linkage: re-extract the first N val images from the
     checkpoint with a fresh loader and compare to the cached rows (cos ~ 1),
     and to the OTHER backbone's cached rows (cos << 1) to rule out a swap.

    python scripts/verify_plainvit_result.py --out logs/verify_plainvit/verify.md
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval.extract import load_backbone  # noqa: E402
from src.eval.indomain import get_indomain_splits  # noqa: E402

CACHE = "/n/netscratch/dominici_lab/Lab/aborgi/eval_indomain_in1k_cache"
BACKBONES = {
    "vit_sigreg": ("lejepa/vit_in1k_sigreg", "checkpoints/lejepa_vit_in1k_sigreg/last.ckpt", 99),
    "vit_localballreg_l01_k4": ("lejepa/vit_in1k_localballreg_l01_k4", "checkpoints/lejepa_vit_in1k_localballreg_l01_k4/last.ckpt", 99),
}
REPORTED = json.load(open("logs/eval_indomain_in1k_plainvit/results.json")) if os.path.isfile("logs/eval_indomain_in1k_plainvit/results.json") else {}


def log(msg, out):
    print(msg, flush=True)
    out.append(msg)


def rankme(X: torch.Tensor) -> float:
    s = torch.linalg.svdvals(X.float() - X.float().mean(0, keepdim=True))
    p = s / s.sum() + 1e-7
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def knn_independent(Xb, yb, Xq, yq, k=20, T=0.07, C=1000, device="cuda", chunk=256, shuffle_labels=False):
    """Cosine weighted k-NN written from scratch: fp32 bank, argsort, bincount."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/verify_plainvit/verify.md")
    ap.add_argument("--n-reextract", type=int, default=2048)
    ap.add_argument("--per-class-logreg", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = [f"# Verification of the plain ViT-S/8 IN1K result ({time.strftime('%Y-%m-%d %H:%M')})", ""]

    train_ds, val_ds, C = get_indomain_splits("imagenet1k", "data/imagenet", 128)
    val_targets = torch.tensor(val_ds.targets)
    train_targets = torch.tensor(train_ds.targets)

    feats = {}
    for name, (exp, ckpt, ep) in BACKBONES.items():
        log(f"\n## {name}", out)
        Xtr, ytr = (lambda b: (b["X"], b["y"]))(torch.load(f"{CACHE}/{name}/imagenet1k_train_128_ep{ep}.pt", map_location="cpu"))
        Xva, yva = (lambda b: (b["X"], b["y"]))(torch.load(f"{CACHE}/{name}/imagenet1k_val_128_ep{ep}.pt", map_location="cpu"))
        feats[name] = (Xtr, ytr, Xva, yva)

        # 1. cache integrity
        nan = int(torch.isnan(Xtr).sum() + torch.isnan(Xva).sum() + torch.isinf(Xtr).sum() + torch.isinf(Xva).sum())
        norms = Xva.float().norm(dim=1)
        dup_val = len(Xva) - len({hashlib.md5(r.numpy().tobytes()).hexdigest() for r in Xva})
        log(f"- cache shapes train {tuple(Xtr.shape)} val {tuple(Xva.shape)} dtype {Xtr.dtype}; NaN/Inf: {nan}; "
            f"val norm mean {norms.mean():.2f} std {norms.std():.2f} min {norms.min():.2f}; duplicated val rows: {dup_val}; "
            f"RankMe(val,512-d): {rankme(Xva):.1f}", out)

        # 2. labels
        log(f"- labels: cached val y == ImageFolder targets: {bool(torch.equal(yva, val_targets))}; "
            f"cached train y == ImageFolder targets: {bool(torch.equal(ytr, train_targets))}; "
            f"val per-class count min/max: {int(torch.bincount(yva, minlength=C).min())}/{int(torch.bincount(yva, minlength=C).max())}", out)

        # 3. disjointness (exact fp16 row match)
        tr_hashes = {hashlib.md5(r.numpy().tobytes()).hexdigest() for r in Xtr}
        overlap = sum(hashlib.md5(r.numpy().tobytes()).hexdigest() in tr_hashes for r in Xva)
        log(f"- val rows found verbatim in the train bank: {overlap} / {len(Xva)}", out)

        # 4/5. independent kNN + shuffle control
        acc = knn_independent(Xtr, ytr, Xva, yva, device=device)
        rep = REPORTED.get(name, {}).get("knn", {}).get("20", {}).get("top1")
        sh = knn_independent(Xtr, ytr, Xva, yva, device=device, shuffle_labels=True)
        log(f"- independent kNN@20 top-1: **{acc:.2f}%** (reported by eval_indomain.py: {rep}); "
            f"label-shuffle control: {sh:.2f}% (chance = 0.1%)", out)

        # 6. independent linear probe (sklearn, subset)
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
        rep_lin = REPORTED.get(name, {}).get("linear", {}).get("top1")
        log(f"- independent linear probe (sklearn LBFGS logistic regression, {args.per_class_logreg} img/class = {len(idx)} train): "
            f"**{lin:.2f}%** val top-1 (full-train SGD probe reported: {rep_lin})", out)

    # 7. cache <-> checkpoint linkage via fresh re-extraction
    log("\n## Cache <-> checkpoint linkage (fresh re-extraction of the first "
        f"{args.n_reextract} val images, loader order)", out)
    sub = torch.utils.data.Subset(val_ds, list(range(args.n_reextract)))
    fresh = {}
    for name, (exp, ckpt, ep) in BACKBONES.items():
        fn, dim = load_backbone(exp, ckpt, device, crop_size=128)
        loader = torch.utils.data.DataLoader(sub, batch_size=256, shuffle=False, num_workers=4)
        fresh[name] = torch.cat([fn(x) for x, _ in loader])
    for name in BACKBONES:
        cached = feats[name][2][: args.n_reextract].float()
        row = []
        for other in BACKBONES:
            cos = F.cosine_similarity(fresh[other], cached, dim=1)
            row.append(f"vs fresh({other}): mean cos {cos.mean():.4f} (min {cos.min():.3f})")
        log(f"- cached {name} " + "; ".join(row), out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()

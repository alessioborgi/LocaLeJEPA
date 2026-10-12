# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""What the k-NN classifier actually retrieves: image- and patch-level neighbours.

Compares the global LeJEPA baseline (SIGReg) with LocaLeJEPA (lambda 0.1, K 4)
for one encoder family (plain ViT-S/8 by default, --family hvit for H-ViT-S/8),
using the epoch-99 ImageNet-1K checkpoints and the same 128 px eval view
(resize + centre crop) as the k-NN protocol of src/eval_indomain.py.

  (1) images   Top-k neighbours of a few query images in the full 1.28M-image
               training bank, by cosine similarity of the 512-d embedding (the
               exact k-NN feature bank of Table 3, read from the cache_dir of
               configs/eval/indomain_in1k_ep100.yaml). Frames are green when the
               neighbour shares the query's class. The row label gives the
               weighted k=20 vote (T = 0.07) of the evaluation protocol. The
               query itself and near-duplicates (cos > 0.995) are skipped.
  (2) patches  One query patch (8x8 px token) against the mean-centred patch tokens of a
               10k-image bank (10 validation images per class). Hits are
               deduplicated per image (best patch of each image), shown with
               their surrounding context and the matched patch outlined.
  (3) heatmaps Cosine similarity (tokens centred on the bank mean) of the same query patch to every patch of
               five validation images of the query's class.

Stage (1) runs on CPU from the cached features; (2)-(3) need patch tokens and a
GPU forward pass, cached in logs/knn_retrieval/ so re-styling runs on CPU:

    python visualisations/plot_knn_retrieval.py --stage images            # CPU
    python visualisations/plot_knn_retrieval.py --stage neighbours        # CPU: save the retrieved images
    sbatch scripts/slurm_knn_retrieval.sh                                 # GPU: extract + plot
    python visualisations/plot_knn_retrieval.py --stage patches           # CPU: re-plot
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

IMNET = "/n/netscratch/dominici_lab/Lab/aborgi/imagenet/imagenet_full_size/061417"
FEAT_CACHE = "/n/netscratch/dominici_lab/Lab/aborgi/eval_indomain_in1k_cache"
WORK = os.path.join(ROOT, "logs/knn_retrieval")
OUT = os.path.join(ROOT, "paper/figs/knn_retrieval")
CROP, PATCH = 128, 8
GRID = CROP // PATCH

FAMILIES = {
    "vit": ("Plain ViT-S/8", [("vit_sigreg", "lejepa/vit_in1k_sigreg", "LeJEPA (SIGReg)"),
                              ("vit_localballreg_l01_k4", "lejepa/vit_in1k_localballreg_l01_k4",
                               r"LocaLeJEPA ($\lambda$=0.1, K=4)")]),
    "hvit": ("Hyperbolic ViT-S/8", [("hvit_sigreg", "lejepa/hvit_in1k_sigreg", "LeJEPA (SIGReg)"),
                                    ("hvit_localballreg_l01_k4", "lejepa/hvit_in1k_localballreg_l01_k4",
                                     r"LocaLeJEPA ($\lambda$=0.1, K=4)")]),
}
# Query images (ImageNet-1K train files) and hand-picked patches (row, col) on the 16x16 grid.
QUERIES = [
    ("polar bear", "0296/train_442694.JPEG", [("head", 9, 2), ("ice", 13, 8)]),
    ("Komodo dragon", "0048/train_738714.JPEG", [("eye", 10, 11), ("foot", 14, 11)]),
    ("triceratops", "0051/train_145413.JPEG", [("horn", 7, 7), ("frill", 4, 5)]),
]
MAIN_PATCHES = {"head", "eye", "horn"}
BANK_PER_CLASS, HEATMAP_IMAGES, TOPK = 10, 5, 8
GREEN, RED = "#2e9e44", "#d1343a"

VIEW = T.Compose([T.Resize(CROP), T.CenterCrop(CROP)])
NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def categories():
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def file_list(split):
    """Relative paths in ImageFolder order, i.e. the row order of the cached features."""
    path = os.path.join(WORK, f"{split}_files.json")
    if not os.path.isfile(path):
        from torchvision.datasets.folder import make_dataset, IMG_EXTENSIONS
        base = os.path.join(IMNET, split)
        classes = sorted(e.name for e in os.scandir(base) if e.is_dir())
        samples = make_dataset(base, {c: i for i, c in enumerate(classes)}, extensions=IMG_EXTENSIONS)
        os.makedirs(WORK, exist_ok=True)
        json.dump([os.path.relpath(p, base) for p, _ in samples], open(path, "w"))
    return json.load(open(path))


def view(split, rel):
    return VIEW(Image.open(os.path.join(IMNET, split, rel)).convert("RGB"))


def label_of(rel):
    return int(rel.split("/")[0])


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{png,pdf}"))


def frame(ax, color, lw=3.0):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(color); s.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])


def short(name, n=16):
    name = name.split(",")[0]
    return name if len(name) <= n else name[:n - 1] + "."


# --------------------------------------------------------------------------- (1) image-level k-NN
def retrieve_images(family):
    files, cats = file_list("train"), categories()
    index = {f: i for i, f in enumerate(files)}
    out = {}
    for name, _, _ in FAMILIES[family][1]:
        blob = torch.load(os.path.join(FEAT_CACHE, name, "imagenet1k_train_128_ep99.pt"), map_location="cpu")
        X, y = blob["X"], blob["y"].long()                    # fp16 (1.28M, 512): keep it that way
        qidx = [index[rel] for _, rel, _ in QUERIES]
        Q = torch.nn.functional.normalize(X[qidx].float(), dim=1)
        # cosine similarity of every bank row to the queries, in chunks to bound memory
        S = torch.cat([torch.nn.functional.normalize(c.float(), dim=1) @ Q.T for c in X.split(65536)])
        del X, blob
        for (qname, rel, _), qi, sims in zip(QUERIES, qidx, S.T):
            top = sims.topk(TOPK + 64)
            keep = [(int(i), float(s)) for s, i in zip(top.values, top.indices) if i != qi and s < 0.995][:TOPK]
            # weighted k=20 vote of the evaluation protocol (cosine, exp(sim / 0.07)), query excluded
            k20 = sims.topk(21)
            mask = k20.indices != qi
            w = torch.exp(k20.values[mask][:20] / 0.07)
            votes = torch.zeros(1000).index_add_(0, y[k20.indices[mask][:20]], w)
            knn = [(int(i), float(v)) for i, v in zip(k20.indices[mask][:20], k20.values[mask][:20])]
            out[(name, qname)] = dict(hits=keep, knn=knn, pred=int(votes.argmax()), label=int(y[qi]))
            print(f"{name:26s} {qname:14s} k20-vote -> {cats[int(votes.argmax())]:24s} "
                  f"same-class in top-{TOPK}: {sum(int(y[i]) == int(y[qi]) for i, _ in keep)}/{TOPK}")
    return out


def plot_images(family, res, title=False):
    files, cats = file_list("train"), categories()
    disp, models = FAMILIES[family]
    rows = [(q, m) for q in QUERIES for m in models]
    fig, axes = plt.subplots(len(rows), TOPK + 1, figsize=(1.25 * (TOPK + 1), 1.5 * len(rows)),
                             gridspec_kw=dict(wspace=0.06, hspace=0.42, top=0.93 if title else 0.98))
    for r, ((qname, rel, _), (name, _, label)) in enumerate(rows):
        e = res[(name, qname)]
        ax = axes[r, 0]
        ax.imshow(view("train", rel)); frame(ax, "black", 1.5)
        ok = e["pred"] == e["label"]
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
        if r % len(models) == 0:
            ax.set_title(f"query: {qname}", fontsize=7, pad=2)
        ax.set_xlabel(f"k-NN: {short(cats[e['pred']], 14)} " + ("✓" if ok else "✗"),
                      fontsize=6.5, color=GREEN if ok else RED, labelpad=2)
        for c, (i, s) in enumerate(e["hits"], start=1):
            ax = axes[r, c]
            ax.imshow(view("train", files[i]))
            same = label_of(files[i]) == e["label"]
            frame(ax, GREEN if same else RED)
            ax.set_xlabel(f"{short(cats[label_of(files[i])])}\n{s:.2f}", fontsize=5.5, labelpad=1.5)
    if title:
        fig.suptitle(f"{disp}: nearest training images (cosine, 512-d embedding, 1.28M-image bank)",
                     fontsize=9, y=0.965)
    save(fig, f"{family}_images")


def save_neighbours(family, res, k=20, temperature=0.07):
    """Write the k nearest neighbours the k-NN vote uses (query excluded, no de-duplication) to
    <OUT>/neighbours/<family>/<query>/<model>/: the original ImageNet JPEG and the 128 px eval view,
    named by rank, a neighbours.csv with each neighbour's vote weight, and votes.csv per class."""
    import csv, shutil
    files, cats = file_list("train"), categories()
    for qname, rel, _ in QUERIES:
        qdir = os.path.join(OUT, "neighbours", family, qname.replace(" ", "_").lower())
        os.makedirs(qdir, exist_ok=True)
        shutil.copy(os.path.join(IMNET, "train", rel), os.path.join(qdir, "query_" + os.path.basename(rel)))
        view("train", rel).save(os.path.join(qdir, "query_view128.png"))
        for name, _, _ in FAMILIES[family][1]:
            d = os.path.join(qdir, f"{name}_knn{k}")
            os.makedirs(d, exist_ok=True)
            e = res[(name, qname)]
            hits = e["knn"][:k]
            w = np.exp(np.array([s for _, s in hits]) / temperature)
            w = w / w.sum()
            votes = {}
            with open(os.path.join(d, "neighbours.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["rank", "cosine", "vote_weight", "class_index", "class", "same_class", "imagenet_file"])
                for r, ((i, sim), wt) in enumerate(zip(hits, w), start=1):
                    c = label_of(files[i])
                    votes[c] = votes.get(c, 0.0) + wt
                    tag = f"{r:02d}_{cats[c].replace(' ', '_').replace('/', '-')}_{sim:.3f}"
                    shutil.copy(os.path.join(IMNET, "train", files[i]), os.path.join(d, tag + ".JPEG"))
                    view("train", files[i]).save(os.path.join(d, tag + "_view128.png"))
                    wr.writerow([r, f"{sim:.4f}", f"{wt:.4f}", c, cats[c], int(c == e["label"]), "train/" + files[i]])
            with open(os.path.join(d, "votes.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["class_index", "class", "n_neighbours", "vote_share", "is_query_class", "predicted"])
                for c, v in sorted(votes.items(), key=lambda t: -t[1]):
                    n = sum(label_of(files[i]) == c for i, _ in hits)
                    wr.writerow([c, cats[c], n, f"{v:.4f}", int(c == e["label"]), int(c == e["pred"])])
            same = sum(label_of(files[i]) == e["label"] for i, _ in hits)
            print(f"{qname:14s} {name:26s} k={k}: {same}/{k} same class, "
                  f"vote -> {cats[e['pred']]} ({votes.get(e['pred'], 0):.2f} of the weight)")


# --------------------------------------------------------------------------- (2)-(3) patch-level
@torch.no_grad()
def extract_patches(family, device):
    from src.eval.extract import load_backbone
    val = file_list("val")
    by_class = {}
    for i, rel in enumerate(val):
        by_class.setdefault(label_of(rel), []).append(i)
    bank = sorted(i for idx in by_class.values() for i in idx[:: len(idx) // BANK_PER_CLASS][:BANK_PER_CLASS])
    queries = [(q, rel, p) for q, rel, ps in QUERIES for p in ps]
    result = {"bank": np.array(bank)}
    for name, exp, _ in FAMILIES[family][1]:
        extract, dim = load_backbone(exp, os.path.join(ROOT, "checkpoints", exp.replace("/", "_"), "last.ckpt"),
                                     device, crop_size=CROP, feature="patch")
        mu = None

        def tokens(split, rels, bs=256):
            outs = []
            for s in range(0, len(rels), bs):
                x = torch.stack([NORM(view(split, r)) for r in rels[s:s + bs]])
                t = extract(x).to(device)
                outs.append((t if mu is None else torch.nn.functional.normalize(t - mu, dim=-1)).half())
            return torch.cat(outs)
        # Post-norm tokens share a large common direction (every patch has cosine > 0.6 to every
        # other), so centre on the bank's mean token before the cosine, as in DINO patch analyses.
        B = tokens("val", [val[i] for i in bank])                     # (N, 256, D), raw
        mu = torch.stack([b.float().mean((0, 1)) for b in B.split(1024)]).mean(0)
        for b in B.split(1024):
            b.copy_(torch.nn.functional.normalize(b.float() - mu, dim=-1).half())
        print(name, "patch bank", tuple(B.shape))
        Q = tokens("train", [rel for _, rel, _ in QUERIES])            # (3, 256, D)
        for qi, (qname, rel, (pname, r, c)) in enumerate(queries):
            q = Q[[x[1] for x in QUERIES].index(rel), r * GRID + c]     # (D,)
            sims = (B.float() @ q.float()) if B.numel() < 2 ** 31 else torch.cat(
                [(b.float() @ q.float()) for b in B.split(1024)])
            best, arg = sims.max(dim=1)                                  # best patch per bank image
            top = best.topk(40)
            same = by_class[label_of(rel)]
            heat_rel = [val[i] for i in same[1:: len(same) // HEATMAP_IMAGES][:HEATMAP_IMAGES]]
            H = tokens("val", heat_rel)
            result[f"{name}|{qname}|{pname}"] = dict(
                img=np.array([bank[int(i)] for i in top.indices.cpu()]),
                patch=arg[top.indices].cpu().numpy(), sim=top.values.float().cpu().numpy(),
                self=(Q[[x[1] for x in QUERIES].index(rel)].float() @ q.float()).cpu().numpy(),
                heat_rel=heat_rel, heat=(H.float() @ q.float()).cpu().numpy())
        del B, Q
        torch.cuda.empty_cache()
    os.makedirs(WORK, exist_ok=True)
    torch.save(result, os.path.join(WORK, f"{family}_patches.pt"))


def box(ax, r, c, color="red", lw=1.4, off=(0, 0)):
    ax.add_patch(Rectangle((c * PATCH - 0.5 - off[1], r * PATCH - 0.5 - off[0]), PATCH, PATCH,
                           fill=False, ec=color, lw=lw))


def context(img, r, c, size=40):
    """A size x size crop of the 128 px view centred on patch (r, c), clamped to the image."""
    y0 = int(np.clip(r * PATCH + PATCH // 2 - size // 2, 0, CROP - size))
    x0 = int(np.clip(c * PATCH + PATCH // 2 - size // 2, 0, CROP - size))
    return np.asarray(img)[y0:y0 + size, x0:x0 + size], (y0, x0)


def plot_patches(family, which, tag):
    res = torch.load(os.path.join(WORK, f"{family}_patches.pt"), weights_only=False)
    val, cats = file_list("val"), categories()
    disp, models = FAMILIES[family]
    qs = [(q, rel, p) for q, rel, ps in QUERIES for p in ps if p[0] in which]
    rows = [(q, m) for q in qs for m in models]
    K = 6
    fig, axes = plt.subplots(len(rows), K + 1, figsize=(1.3 * (K + 1), 1.45 * len(rows)),
                             gridspec_kw=dict(wspace=0.06, hspace=0.38, top=0.93))
    for i, ((qname, rel, (pname, r, c)), (name, _, label)) in enumerate(rows):
        e = res[f"{name}|{qname}|{pname}"]
        ax = axes[i, 0]
        ax.imshow(view("train", rel)); box(ax, r, c, lw=1.6); frame(ax, "black", 1.2)
        ax.set_title(f"{qname}: {pname}", fontsize=7, pad=2)
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
        for k in range(K):
            j, p, s = int(e["img"][k]), int(e["patch"][k]), float(e["sim"][k])
            pr, pc = divmod(p, GRID)
            crop, off = context(view("val", val[j]), pr, pc)
            ax = axes[i, k + 1]
            ax.imshow(crop, interpolation="nearest"); box(ax, pr, pc, color="yellow", lw=1.4, off=off)
            same = label_of(val[j]) == label_of(rel)
            frame(ax, GREEN if same else RED)
            ax.set_xlabel(f"{short(cats[label_of(val[j])])}\n{s:.2f}", fontsize=5.5, labelpad=1.5)
    fig.suptitle(f"{disp}: nearest patches (one per image) in a 10k-image validation bank (mean-centred tokens)", fontsize=9, y=0.965)
    save(fig, f"{family}_patches{tag}")


def plot_heatmaps(family, which, tag):
    res = torch.load(os.path.join(WORK, f"{family}_patches.pt"), weights_only=False)
    disp, models = FAMILIES[family]
    qs = [(q, rel, p) for q, rel, ps in QUERIES for p in ps if p[0] in which]
    rows = [(q, m) for q in qs for m in models]
    fig, axes = plt.subplots(len(rows), HEATMAP_IMAGES + 2, figsize=(1.3 * (HEATMAP_IMAGES + 2), 1.4 * len(rows)),
                             gridspec_kw=dict(wspace=0.06, hspace=0.3, top=0.93))
    for i, ((qname, rel, (pname, r, c)), (name, _, label)) in enumerate(rows):
        e = res[f"{name}|{qname}|{pname}"]
        ax = axes[i, 0]
        ax.imshow(view("train", rel)); box(ax, r, c, lw=1.6); frame(ax, "black", 1.2)
        ax.set_title(f"{qname}: {pname}", fontsize=7, pad=2)
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
        maps = [e["self"].reshape(GRID, GRID)] + [h.reshape(GRID, GRID) for h in e["heat"]]
        imgs = [view("train", rel)] + [view("val", x) for x in e["heat_rel"]]
        for k, (img, m) in enumerate(zip(imgs, maps)):
            ax = axes[i, k + 1]
            ax.imshow(img)
            ax.imshow(np.kron(m, np.ones((PATCH, PATCH))), cmap="inferno", vmin=0, vmax=1, alpha=0.62,
                      interpolation="nearest")
            if k == 0:
                box(ax, r, c, color="cyan", lw=1.2)
            frame(ax, "black", 0.6)
            if i == 0:
                ax.set_title("same image" if k == 0 else f"{qname} (val)", fontsize=6.5, pad=2)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("cosine similarity to the query patch (clipped at 0)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{disp}: patch-to-patch similarity (mean-centred tokens)", fontsize=9, y=0.965)
    save(fig, f"{family}_heatmaps{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="vit")
    ap.add_argument("--stage", choices=["images", "neighbours", "extract", "patches", "all"], default="all")
    ap.add_argument("--title", action="store_true", help="add a title to the image figure (off for the paper)")
    args = ap.parse_args()
    if args.stage in ("images", "neighbours", "all"):
        res = retrieve_images(args.family)
        if args.stage != "neighbours":
            plot_images(args.family, res, title=args.title)
        if args.stage != "images":
            save_neighbours(args.family, res)
    if args.stage in ("extract", "all"):
        extract_patches(args.family, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if args.stage in ("patches", "extract", "all"):
        allp = {p for _, _, ps in QUERIES for p, _, _ in ps}
        for which, tag in ((MAIN_PATCHES, ""), (allp, "_all")):
            plot_patches(args.family, which, tag)
            plot_heatmaps(args.family, which, tag)


if __name__ == "__main__":
    main()

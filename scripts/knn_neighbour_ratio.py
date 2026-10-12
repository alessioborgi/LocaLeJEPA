# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Same-class share of the k-NN neighborhood, per class and overall.

For every ImageNet-1K validation image x (class C) take its K nearest training
images N (cosine, the 512-d embedding and 1.28M-image bank of the Table 3 k-NN
eval, read from the feature cache of src/eval_indomain.py) and compute

    ratio(x)  = sum_{y in N, y in C} d(x, y) / sum_{z in N} d(x, z),   d = 1 - cos
    purity(x) = |N ∩ C| / K

Both are 1 when the whole neighborhood shares x's class and 0 when none of it
does. Per-class numbers average the 50 validation images of a class; "all" is
the mean over the 50k validation images. Writes logs/knn_retrieval/
neighbour_ratio_<family>.json (all 1000 classes) and a LaTeX table for the
selected classes in paper/tables/knn_neighbour_ratio_<family>.tex.

    sbatch --export=ALL,FAMILY=vit scripts/slurm_knn_neighbour_ratio.sh
    python scripts/knn_neighbour_ratio.py --family vit --table-only      # re-typeset
    python scripts/knn_neighbour_ratio.py --wide     # 36-class and all-class group tables, every family
"""
import argparse
import json
import os

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = "/n/netscratch/dominici_lab/Lab/aborgi/eval_indomain_in1k_cache"
FAMILIES = {
    "vit": ("Plain ViT-S/8", ["vit_sigreg", "vit_localballreg_l01_k4"]),
    "hvit": ("Hyperbolic ViT-S/8", ["hvit_sigreg", "hvit_localballreg_l01_k4"]),
    "d4rn50": ("$D_4$-equivariant ResNet-50", ["d4rn50_sigreg", "d4rn50_localballreg_l01_k4"]),
}
LABEL = {"sigreg": "LeJEPA (SIGReg)", "localballreg_l01_k4": r"LocaLeJEPA ($\lambda{=}0.1$, $K{=}4$)"}
# Ten animal classes (ImageNet-1K index -> short name)
CLASSES = {296: "Polar bear", 294: "Brown bear", 48: "Komodo", 51: "Triceratops", 50: "Alligator",
           63: "Cobra", 147: "Grey whale", 148: "Orca", 149: "Dugong", 150: "Sea lion"}
KS = [10, 20, 100]


def load(name, split):
    blob = torch.load(os.path.join(CACHE, name, f"imagenet1k_{split}_128_ep99.pt"), map_location="cpu")
    return blob["X"], blob["y"].long()


@torch.no_grad()
def neighbour_stats(name, device):
    X, y = load(name, "train")
    V, yv = load(name, "val")
    X = torch.cat([torch.nn.functional.normalize(c.to(device).float(), dim=1).half() for c in X.split(65536)])
    y = y.to(device)
    out = {f"{m}@{k}": [] for m in ("ratio", "purity") for k in KS}
    for q, lq in zip(V.split(512), yv.split(512)):
        q = torch.nn.functional.normalize(q.to(device).float(), dim=1).half()
        s, idx = (q @ X.T).topk(max(KS), dim=1)
        d = (1 - s.float()).clamp_min(0)
        same = (y[idx] == lq.to(device)[:, None]).float()
        for k in KS:
            out[f"ratio@{k}"].append(((d[:, :k] * same[:, :k]).sum(1) / d[:, :k].sum(1).clamp_min(1e-12)).cpu())
            out[f"purity@{k}"].append(same[:, :k].mean(1).cpu())
    per_img = {m: torch.cat(v) for m, v in out.items()}
    res = {m: {"all": float(v.mean()),
               "per_class": [float(v[yv == c].mean()) for c in range(1000)]} for m, v in per_img.items()}
    del X
    torch.cuda.empty_cache()
    return res


def table(family, res, metric="ratio@20"):
    disp, names = FAMILIES[family]
    cols = list(CLASSES)
    k = metric.split("@")[1]
    what = ("distance-weighted same-class share" if metric.startswith("ratio") else "same-class fraction")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Class consistency of the $k$-NN neighborhood (" + disp + r").} For each validation image "
             r"$x$ of class $C$ with $K{=}" + k + r"$ nearest training images $N$ (cosine, the ImageNet-1K $k$-NN bank), "
             + ((r"$\sum_{y\in N\cap C} d(x,y) / \sum_{z\in N} d(x,z)$ with $d = 1-\cos$" if metric.startswith("ratio")
                 else r"$|N\cap C|/K$")) +
             r", averaged over the 50 validation images of each class; \emph{All} averages the 50k validation images. "
             r"1 means the neighborhood lies entirely in the query's class. Bold marks the higher value.}",
             r"\label{tab:knn-neighbor-" + metric.replace("@", "-") + "-" + family + "}",
             r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{l " + "c" * len(cols) + r" | c}", r"\toprule",
             "Method & " + " & ".join(CLASSES.values()) + r" & All \\", r"\midrule"]
    vals = {n: [res[n][metric]["per_class"][c] for c in cols] + [res[n][metric]["all"]] for n in names}
    for n in names:
        cells = []
        for j, v in enumerate(vals[n]):
            best = max(vals[m][j] for m in names)
            cells.append(r"\textbf{%.2f}" % v if abs(v - best) < 1e-12 else "%.2f" % v)
        lines.append(LABEL[n.split("_", 1)[1]] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), default="vit")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    path = os.path.join(ROOT, f"logs/knn_retrieval/neighbour_ratio_{args.family}.json")
    if args.table_only:
        res = json.load(open(path))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res = {n: neighbour_stats(n, device) for n in FAMILIES[args.family][1]}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(res, open(path, "w"))
    for metric in ("ratio@20", "purity@20"):
        out = os.path.join(ROOT, f"paper/tables/knn_neighbour_{metric.replace('@', '')}_{args.family}.tex")
        open(out, "w").write(table(args.family, res, metric))
        print("wrote", out)
    for n in FAMILIES[args.family][1]:
        print(n, "  ".join(f"{m}={res[n][m]['all']:.3f}" for m in res[n]))
        print("   ", " ".join(f"{CLASSES[c]}={res[n]['ratio@20']['per_class'][c]:.2f}" for c in CLASSES))




# --------------------------------------------------------------------------- wider tables (all families)
MORE = {
    "Bears, reptiles, marine mammals": CLASSES,
    "Other animals": {1: "Goldfish", 2: "Great white shark", 22: "Bald eagle", 130: "Flamingo", 145: "King penguin",
                      207: "Golden retriever", 281: "Tabby cat", 292: "Tiger", 340: "Zebra", 386: "African elephant",
                      388: "Giant panda", 301: "Ladybug", 323: "Monarch butterfly"},
    "Objects, food, scenes": {402: "Acoustic guitar", 404: "Airliner", 407: "Ambulance", 483: "Castle",
                              579: "Grand piano", 620: "Laptop", 779: "School bus", 817: "Sports car", 849: "Teapot",
                              954: "Banana", 963: "Pizza", 980: "Volcano", 985: "Daisy"},
}
# WordNet groups of ImageNet-1K by class index (all 1000 classes)
GROUPS = [("Fish", [range(0, 7), range(389, 398)]), ("Birds", [range(7, 25), range(127, 147)]),
          ("Amphibians", [range(25, 33)]), ("Reptiles", [range(33, 69)]), ("Invertebrates", [range(69, 127),
          range(300, 328)]), ("Marine mammals", [range(147, 151)]), ("Dogs", [range(151, 269)]),
          ("Wild canids", [range(269, 281)]), ("Cats", [range(281, 294)]), ("Bears", [range(294, 298)]),
          ("Other mammals", [range(298, 300), range(328, 389)]), ("Objects, food, scenes", [range(398, 1000)])]


def wide_tables(metric="ratio@20"):
    fams = [f for f in FAMILIES if os.path.isfile(os.path.join(ROOT, f"logs/knn_retrieval/neighbour_ratio_{f}.json"))]
    res = {f: json.load(open(os.path.join(ROOT, f"logs/knn_retrieval/neighbour_ratio_{f}.json"))) for f in fams}
    short = {"vit": "ViT-S/8", "hvit": "H-ViT-S/8", "d4rn50": "$D_4$-RN50"}
    head = [r"\toprule",
            " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % short[f] for f in fams) + r" \\",
            "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(fams))),
            " & " + " & ".join(["SIGReg & LocaLeJEPA"] * len(fams)) + r" \\"]

    def row(label, get):
        cells = []
        for f in fams:
            a, b = (get(res[f][n][metric]) for n in FAMILIES[f][1])
            cells += [r"\textbf{%.2f}" % a if a > b else "%.2f" % a, r"\textbf{%.2f}" % b if b > a else "%.2f" % b]
        return label + " & " + " & ".join(cells) + r" \\"

    cap = (r"For each validation image $x$ of class $C$ with $K{=}20$ nearest training images $N$ (cosine, the "
           r"ImageNet-1K $k$-NN bank): $\sum_{y\in N\cap C} d(x,y) / \sum_{z\in N} d(x,z)$, $d = 1-\cos$ "
           r"(1 = the whole neighborhood shares the query's class). LocaLeJEPA is $\lambda{=}0.1$, $K{=}4$. "
           r"Bold marks the higher value per encoder.")
    colspec = "l " + " ".join(["cc"] * len(fams))
    # (a) 36 named classes
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Class consistency of the $k$-NN neighborhood, per class.} " + cap +
             r" Each entry averages the 50 validation images of the class.}",
             r"\label{tab:knn-neighbor-classes}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{" + colspec + "}"] + head
    for g, cl in MORE.items():
        lines += [r"\midrule", r"\multicolumn{%d}{l}{\emph{%s}} \\" % (1 + 2 * len(fams), g)]
        lines += [row(n, lambda m, c=c: m["per_class"][c]) for c, n in cl.items()]
    lines += [r"\midrule", row(r"All 1000 classes", lambda m: m["all"])]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(ROOT, "paper/tables/knn_neighbour_classes.tex"), "w").write("\n".join(lines) + "\n")
    # (b) every class, grouped
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Class consistency of the $k$-NN neighborhood, by class group.} " + cap +
             r" Each entry averages the classes of the group (number in brackets); the last row counts the classes "
             r"on which LocaLeJEPA is higher.}",
             r"\label{tab:knn-neighbor-groups}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{" + colspec + "}"] + head + [r"\midrule"]
    for g, rs in GROUPS:
        idx = [c for r in rs for c in r]
        lines.append(row(f"{g} ({len(idx)})", lambda m, idx=idx: sum(m["per_class"][c] for c in idx) / len(idx)))
    lines += [r"\midrule", row(r"All 1000 classes", lambda m: m["all"])]
    wins = []
    for f in fams:
        a, b = (res[f][n][metric]["per_class"] for n in FAMILIES[f][1])
        wins += [r"\multicolumn{2}{c}{%d / 1000}" % sum(y > x for x, y in zip(a, b))]
    lines += [r"LocaLeJEPA higher & " + " & ".join(wins) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(ROOT, "paper/tables/knn_neighbour_groups.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote paper/tables/knn_neighbour_{classes,groups}.tex")


if __name__ == "__main__":
    import sys
    wide_tables() if "--wide" in sys.argv else main()

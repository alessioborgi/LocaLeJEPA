# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Jonckheere-Terpstra trend test over the BallJEPA locality sweep.

The k-ablation asks an ORDERED question -- "does the metric rise with k?" -- not
a pairwise one, so the statistic has to be a trend test. Pairwise comparisons
across five k values would also need a multiplicity correction that a trend test
does not, and at three seeds per arm they have almost no power: a one-sided
3-vs-3 exact permutation test has only C(6,3) = 20 arrangements, so its smallest
attainable p-value is exactly 0.05.

Jonckheere-Terpstra sums, over every ordered pair of k groups, the number of
cross-group pairs that respect the hypothesised order (ties counted as a half).
Large J means the values climb with k.

    python scripts/ksweep_trend_test.py
    python scripts/ksweep_trend_test.py --metric linear_top1
    python scripts/ksweep_trend_test.py --max-k 8      # reproduce the pre-extension result

TWO-SIDED BY DEFAULT once k passes the number of classes. Through k=8 the
alternative was directional ("accuracy increases with k") and one-sided was
right. The k=16/32 arms were added precisely because the curve may TURN OVER
near #classes=10, and a one-sided test cannot see a peak -- it reports a weak
increase for data that actually rises then falls. Pass --alternative increasing
to recover the original one-sided test, and read --max-k 8 alongside it.

A trend test is also the wrong tool for a peak: if the profile is non-monotone,
J is not small because "nothing happens", it is small because the up and down
halves cancel. The per-group means printed above the statistic are what tell
those apart, so read them first.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

DEFAULT_RESULTS = "logs/eval_indomain_imagenette_ksweep/results.csv"
_K_RE = re.compile(r"(?:^|_)k(\d+)(?:_|$)")


def jt_statistic(groups: list[np.ndarray]) -> float:
    """Sum of Mann-Whitney U over all ordered group pairs; ties count 0.5."""
    total = 0.0
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            a, b = groups[i][:, None], groups[j][None, :]
            total += float((b > a).sum() + 0.5 * (b == a).sum())
    return total


def jt_normal_p(groups: list[np.ndarray], J: float, alternative: str) -> tuple[float, float, float]:
    """Normal approximation: returns (mean, sd, p). Exact under no ties."""
    from math import erfc, sqrt

    sizes = np.array([len(g) for g in groups], dtype=float)
    N = sizes.sum()
    mean = (N**2 - (sizes**2).sum()) / 4.0
    var = (N**2 * (2 * N + 3) - (sizes**2 * (2 * sizes + 3)).sum()) / 72.0
    sd = sqrt(var)
    z = (J - mean) / sd
    upper = 0.5 * erfc(z / sqrt(2.0))
    p = upper if alternative == "increasing" else (
        1.0 - upper if alternative == "decreasing" else 2.0 * min(upper, 1.0 - upper))
    return mean, sd, min(p, 1.0)


def _arrangement_count(sizes: list[int]) -> int:
    """Multinomial N! / prod(n_i!) -- how many distinct group assignments exist."""
    from math import comb

    total, remaining = 1, sum(sizes)
    for n in sizes[:-1]:
        total *= comb(remaining, n)
        remaining -= n
    return total


def _more_extreme(Jp: float, J: float, alternative: str, centre: float) -> bool:
    if alternative == "increasing":
        return Jp >= J
    if alternative == "decreasing":
        return Jp <= J
    return abs(Jp - centre) >= abs(J - centre)


def jt_exact_p(groups: list[np.ndarray], J: float, alternative: str) -> float:
    """Exact permutation p by enumerating every distinct group assignment.

    Only called when the arrangement count is small (three groups of three give
    1680, which is the case the pre-extension k in {2,4,8} result lives in).
    Five groups of three give 168,168,000 and fall back to Monte-Carlo.
    """
    from itertools import combinations

    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    centre = (len(pooled) ** 2 - sum(s**2 for s in sizes)) / 4.0
    extreme = count = 0

    def recurse(remaining: tuple[int, ...], chosen: list[np.ndarray]) -> None:
        nonlocal extreme, count
        if len(chosen) == len(sizes) - 1:
            allocation = chosen + [pooled[list(remaining)]]
            Jp = jt_statistic(allocation)
            count += 1
            extreme += _more_extreme(Jp, J, alternative, centre)
            return
        n = sizes[len(chosen)]
        for pick in combinations(remaining, n):
            rest = tuple(i for i in remaining if i not in pick)
            recurse(rest, chosen + [pooled[list(pick)]])

    recurse(tuple(range(len(pooled))), [])
    return extreme / count


def jt_permutation_p(groups: list[np.ndarray], J: float, alternative: str,
                     n_perm: int, seed: int) -> float:
    """Monte-Carlo permutation p-value, the +1 correction included.

    Exact enumeration is out of reach here (5 groups of 3 give 15!/(3!)^5 =
    168,168,000 arrangements), and the normal approximation is only asymptotic
    at n=3 per group, so the permutation null is the honest reference.
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    cuts = np.cumsum(sizes)[:-1]
    centre = (len(pooled) ** 2 - sum(s**2 for s in sizes)) / 4.0
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        extreme += _more_extreme(jt_statistic(np.split(perm, cuts)), J, alternative, centre)
    return (extreme + 1) / (n_perm + 1)


def load(path: str, metric: str, prefix: str) -> dict[int, list[float]]:
    if not os.path.exists(path):
        sys.exit(f"no results at {path} -- run eval_indomain.py on the cluster first")
    by_k: dict[int, list[float]] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["backbone"]
            if not name.startswith(prefix):
                continue
            m = _K_RE.search(name)
            if not m or not row.get(metric):
                continue
            by_k.setdefault(int(m.group(1)), []).append(float(row[metric]))
    return by_k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--metric", default="knn20_top1")
    ap.add_argument("--prefix", default="ball_k", help="backbone-name prefix selecting the sweep")
    ap.add_argument("--alternative", default="two-sided",
                    choices=["two-sided", "increasing", "decreasing"])
    ap.add_argument("--max-k", type=int, default=None, help="drop k above this")
    ap.add_argument("--n-perm", type=int, default=200_000)
    ap.add_argument("--exact-limit", type=int, default=2_000_000,
                    help="enumerate every arrangement when there are no more than this "
                         "many; otherwise fall back to Monte-Carlo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_k = load(args.results, args.metric, args.prefix)
    if args.max_k:
        by_k = {k: v for k, v in by_k.items() if k <= args.max_k}
    ks = sorted(by_k)
    if len(ks) < 3:
        sys.exit(f"need at least 3 k groups, found {ks or 'none'} in {args.results}")

    groups = [np.array(by_k[k], dtype=float) for k in ks]
    sizes = {len(g) for g in groups}

    print(f"\n{args.metric}  (prefix {args.prefix!r}, {args.results})")
    print("-" * 62)
    print(f'{"k":>5} {"n":>3} {"mean":>8} {"sd":>7}   values')
    for k, g in zip(ks, groups):
        vals = " ".join(f"{v:.2f}" for v in g)
        sd = f"{g.std(ddof=1):.3f}" if len(g) > 1 else "   -  "
        print(f"{k:>5} {len(g):>3} {g.mean():>8.2f} {sd:>7}   {vals}")

    peak = ks[int(np.argmax([g.mean() for g in groups]))]
    print(f"\nhighest group mean at k={peak}" +
          ("  <-- interior peak: the profile is NOT monotone, and a trend test "
           "understates\n    a rise-then-fall shape. Read the means above, not just J."
           if ks[0] != peak != ks[-1] else ""))

    J = jt_statistic(groups)
    mean, sd, p_norm = jt_normal_p(groups, J, args.alternative)

    n_arr = _arrangement_count([len(g) for g in groups])
    if n_arr <= args.exact_limit:
        p_ref, label = jt_exact_p(groups, J, args.alternative), f"exact, {n_arr:,} arrangements"
    else:
        p_ref = jt_permutation_p(groups, J, args.alternative, args.n_perm, args.seed)
        label = (f"Monte-Carlo, {args.n_perm:,} of {n_arr:,} arrangements "
                 f"-- exact enumeration is out of reach")

    print(f"\nJonckheere-Terpstra ({args.alternative})")
    print(f"  J = {J:.1f}   null mean {mean:.1f}, sd {sd:.2f}   z = {(J - mean) / sd:+.3f}")
    print(f"  p (normal approx)      = {p_norm:.4f}")
    print(f"  p ({label}) = {p_ref:.4f}")
    if len(sizes) > 1:
        print(f"  WARNING: unequal group sizes {sorted(len(g) for g in groups)} -- an arm with "
              f"fewer seeds\n           contributes less and the comparison is not balanced.")
    if min(len(g) for g in groups) < 3:
        print("  WARNING: a group has fewer than 3 seeds; the p-value floor is coarse.")
    print()


if __name__ == "__main__":
    main()

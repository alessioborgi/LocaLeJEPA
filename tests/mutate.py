#!/usr/bin/env python
"""Mutation harness: plant a defect, run the suite, record whether it died.

``research_rules/04``: a test you have never seen fail is not evidence. Coverage says a line ran;
this says an assertion would have caught it being wrong.

Two disciplines the rules call out, both implemented here:

* **Difference against a measured baseline, never against zero.** A pre-existing failure would
  otherwise read as a kill for every mutant. The baseline is taken first and a mutant only counts
  as killed if it fails *more* tests than the baseline did.
* **Confirm the revert.** After each mutant the tree is restored from git and the suite re-run,
  so a harness that silently left a defect behind cannot masquerade as a clean sweep.

Usage:
    python tests/mutate.py              # run every mutant
    python tests/mutate.py --report     # tally only
    python tests/mutate.py -k radius    # a subset, matched on the label
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / ".mutation-in-progress"


@dataclass(frozen=True)
class Mutant:
    label: str
    path: str
    old: str
    new: str
    why: str


# Defects of the shape that actually occurs: a sign, a boundary, a dropped call, a swapped
# constant. Each names the property it is meant to destroy.
MUTANTS: list[Mutant] = [
    Mutant("ball-radius-sqrt-d", "src/losses.py",
           "R = float((d + 2) ** 0.5 if radius is None else radius)",
           "R = float((d) ** 0.5 if radius is None else radius)",
           "the default radius no longer whitens the target"),
    Mutant("trapezoid-endpoints", "src/losses.py",
           "weights[[0, -1]] = dt", "weights[[0, -1]] = 2.0 * dt",
           "the quadrature stops being the trapezoid rule, rescaling every regularizer"),
    Mutant("ecf-imaginary-sign", "src/losses.py",
           "+ x_t.sin().mean(-3).square()", "- x_t.sin().mean(-3).square()",
           "the imaginary part of the ECF error is subtracted rather than added"),
    Mutant("drop-n-scaling", "src/losses.py",
           "((err @ weights) * z.size(-2)).mean()", "((err @ weights)).mean()",
           "the statistic no longer scales with the sample count"),
    Mutant("gaussian-cf", "src/losses.py",
           "return torch.exp(-t.square() / 2.0)", "return torch.exp(-t.square())",
           "the Gaussian target CF has the wrong variance"),
    Mutant("ball-density-exponent", "src/losses.py",
           ".pow((d - 1) / 2.0) * w", ".pow((d + 1) / 2.0) * w",
           "the uniform-ball projection density has the wrong exponent"),
    Mutant("min-cluster-comparison", "src/losses.py",
           "if sizes[cluster_idx] < min_size:", "if sizes[cluster_idx] > min_size:",
           "the cluster-size filter keeps exactly the clusters it should drop"),
    # The detach appears on two lines, one per branch of `num_global`. Both are planted: the
    # first version of the guarding test covered only one, and the other mutant survived.
    Mutant("drop-cluster-detach-global", "src/losses.py",
           "cluster_points = proj[:num_global].mean(dim=0).detach()",
           "cluster_points = proj[:num_global].mean(dim=0)",
           "gradients flow through the k-means assignment (num_global branch)"),
    Mutant("drop-cluster-detach-all", "src/losses.py",
           "cluster_points = proj.mean(dim=0).detach()",
           "cluster_points = proj.mean(dim=0)",
           "gradients flow through the k-means assignment (default branch)"),
    Mutant("num-global-off-by-one", "src/losses.py",
           "if num_global < 1 or num_global > V:", "if num_global < 0 or num_global > V:",
           "num_global=0 is accepted and silently selects no reference views"),
    Mutant("weighted-average", "src/losses.py",
           "return (local_losses * (w / w.sum())).sum()", "return (local_losses * w).sum()",
           "the size-weighted average is no longer normalised"),
    Mutant("invariance-reference", "src/losses.py",
           "ref = proj[:num_global].mean(0) if num_global is not None else proj.mean(0)",
           "ref = proj[num_global:].mean(0) if num_global is not None else proj.mean(0)",
           "the invariance loss anchors on the local views instead of the global ones"),
    Mutant("sigma-gn-ratio", "src/losses.py",
           "return math.gamma(1.0 / p) ** 0.5 / (p ** (1.0 / p) * math.gamma(3.0 / p) ** 0.5)",
           "return math.gamma(1.0 / p) ** 0.5 / (p ** (1.0 / p) * math.gamma(2.0 / p) ** 0.5)",
           "the generalized-Gaussian scale no longer gives unit variance"),
    Mutant("l1-sparsity-dim", "src/losses.py",
           "return ((1.0 / D) * (l1 / (l2 + eps)).square()).mean()",
           "return ((1.0 / D) * (l1 / (l2 + eps))).mean()",
           "the l1 sparsity metric drops its square and no longer attains 1/D"),
    Mutant("ntxent-self-similarity", "src/losses.py",
           'sim.fill_diagonal_(float("-inf"))', "pass",
           "self-similarity is left in the softmax denominator"),
    Mutant("vnd-transpose", "src/models/lejepa_module.py",
           "proj = proj.reshape(batch_size, num_views, self.projection_dim).transpose(0, 1)",
           "proj = proj.reshape(batch_size, num_views, self.projection_dim)",
           "projections are (N, V, D) instead of (V, N, D): every statistic is over the wrong axis"),
    Mutant("encode-group-order", "src/models/lejepa_module.py",
           "return torch.cat(group_embs, dim=1).flatten(0, 1)",
           "return torch.cat(group_embs[::-1], dim=1).flatten(0, 1)",
           "local views precede global ones, so num_global slices the wrong views"),
    Mutant("probe-view-index", "src/models/lejepa_module.py",
           "emb_first = emb.reshape(batch_size, num_views, self.embedding_dim)[:, 0]",
           "emb_first = emb.reshape(batch_size, num_views, self.embedding_dim)[:, 1]",
           "the online probe reads the second view rather than the first"),
    Mutant("probe-not-detached", "src/models/lejepa_module.py",
           "self.online_probe(emb_first.detach())", "self.online_probe(emb_first)",
           "the probe's gradient reaches the encoder it is supposed to measure"),
    Mutant("eta-min-group", "src/models/lejepa_module.py",
           'eta_min = self.lr_eta_min_ratio * optimizer.param_groups[0]["lr"]',
           'eta_min = self.lr_eta_min_ratio * optimizer.param_groups[-1]["lr"]',
           "the cosine floor is taken from the probe's learning rate"),
    Mutant("warmup-milestone", "src/models/lejepa_module.py",
           "warmup_steps = min(total_steps, steps_per_epoch * self.warmup_epochs)",
           "warmup_steps = min(total_steps, steps_per_epoch * self.warmup_epochs * 2)",
           "warm-up lasts twice as long as configured"),
    # --- crash-safe writes and the self-healing cache (T.suite-P1-6, C0-3, C0-4) ---
    Mutant("atomic-write-leaves-tmp", "src/eval/io.py",
           "            os.remove(tmp)\n        except OSError:\n            pass\n        raise\n\n\ndef atomic_torch_save",
           "            pass\n        except OSError:\n            pass\n        raise\n\n\ndef atomic_torch_save",
           "an interrupted write leaves a .tmp behind for the next reader to find"),
    Mutant("cache-reraises-instead-of-missing", "src/eval/io.py",
           'logger.warning("unreadable feature cache %s (%s: %s); re-extracting",\n                       path, type(error).__name__, error)\n        return None',
           'logger.warning("unreadable feature cache %s (%s: %s); re-extracting",\n                       path, type(error).__name__, error)\n        raise',
           "a damaged cache wedges the backbone instead of being re-extracted"),
    Mutant("cache-skips-the-shape-check", "src/eval/io.py",
           'if not isinstance(blob, dict) or "X" not in blob or "y" not in blob:',
           "if False:",
           "a readable .pt that is not a feature cache is handed back as one"),
    Mutant("digest-ignores-mtime", "src/eval/io.py",
           'key = f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}".encode()',
           'key = f"{resolved}:{stat.st_size}".encode()',
           "a rewritten checkpoint of identical size reuses the previous run's features"),
    Mutant("non-atomic-indomain-save", "src/eval_indomain.py",
           'with atomic_write(os.path.join(_per_backbone_dir(out_dir), f"{name}.json"), "w", encoding="utf-8") as file:',
           'with open(os.path.join(_per_backbone_dir(out_dir), f"{name}.json"), "w", encoding="utf-8") as file:',
           "the headline table's source file is written non-atomically again"),
    Mutant("cache-existence-only", "src/eval/indomain.py",
           "    blob = load_tensor_cache(cache_path)\n    if blob is not None:",
           '    blob = torch.load(cache_path, map_location="cpu") if (cache_path and os.path.isfile(cache_path)) else None\n    if blob is not None:',
           "the cache is trusted on existence alone, so a truncated one wedges the job"),
    # --- the silent-k substitution, all four copies (T.suite-P2-7) ---
    Mutant("silent-k-indomain", "src/eval_indomain.py",
           'knn = entry["knn"].get(str(report_k))\n',
           'knn = entry["knn"].get(str(report_k)) or next(iter(entry["knn"].values()))\n',
           "the printed table reports the smallest available k under the requested k's heading"),
    Mutant("silent-k-in1k-report", "scripts/make_imagenet1k_results.py",
           'knn = entry["knn"].get(str(k))\n',
           'knn = entry["knn"].get(str(k)) or next(iter(entry["knn"].values()))\n',
           "the ImageNet-1K table substitutes another k"),
    Mutant("silent-k-plainvit", "scripts/make_plainvit_report.py",
           '    return entry["knn"].get(k)',
           '    return entry["knn"].get(k) or next(iter(entry["knn"].values()))',
           "the plain-ViT report substitutes another k"),
    Mutant("silent-k-rankme", "scripts/make_rankme_results.py",
           '        knn = hit["top1"] if hit else None',
           '        knn = (hit or next(iter(ind["knn"].values())))["top1"]',
           "a substituted k enters the RankMe-vs-accuracy correlation"),
    # --- cache identity (C0-4), the coupled writers and readers ---
    Mutant("cache-key-ignores-checkpoint", "src/eval/io.py",
           '    stem += f"_{checkpoint_digest(checkpoint)}"',
           "    pass",
           "two checkpoints sharing a name and an epoch collide again -- C0-4 restored"),
    Mutant("cache-key-drops-crop", "src/eval/io.py",
           'stem = f"{dataset}_{split}_{crop_size}"',
           'stem = f"{dataset}_{split}"',
           "128px and 224px evaluations share one cache entry"),
    Mutant("indomain-hand-built-key", "src/eval_indomain.py",
           "            return feature_cache_path(cache_dir, _name, dataset_name, split, crop_size,\n                                      _epoch, _ckpt)",
           '            return os.path.join(cache_dir, _name, f"{dataset_name}_{split}_{crop_size}_ep{_epoch}.pt")',
           "the writer stops agreeing with the readers that reconstruct its filenames"),
    Mutant("rankme-reader-diverges", "src/eval_rankme.py",
           '        full_cache = feature_cache_path(indomain_cache, name, dataset_name, "train",\n                                        crop_size, epoch, ckpt)',
           '        full_cache = os.path.join(indomain_cache, name, f"{dataset_name}_train_{crop_size}_ep{epoch}.pt") if indomain_cache else None',
           "RankMe silently stops finding the in-domain caches and re-extracts on GPU"),
    Mutant("knn-topk-direction", "src/eval/knn_probe.py",
           "similarity.topk(max_k, dim=1, largest=True)",
           "similarity.topk(max_k, dim=1, largest=False)",
           "the k-NN vote uses the FURTHEST neighbours"),
    Mutant("knn-normalisation", "src/eval/knn_probe.py",
           "query = F.normalize(query, dim=1).to(device=device, dtype=bank_dtype)",
           "query = query.to(device=device, dtype=bank_dtype)",
           "queries are not L2-normalised, so the vote is not cosine"),
    Mutant("knn-percent", "src/eval/knn_probe.py",
           '{"top1": 100.0 * top1[k] / total, "top5": 100.0 * top5[k] / total}',
           '{"top1": top1[k] / total, "top5": top5[k] / total}',
           "accuracies are fractions, dividing every published number by 100"),
    # --- the seeding family (ledger T.suite-P1-5 and the train.py truthiness guard) ---
    Mutant("unseeded-probe-init", "src/eval/linear_probe.py",
           "torch.nn.init.trunc_normal_(head.weight, std=0.01, generator=generator)",
           "torch.nn.init.trunc_normal_(head.weight, std=0.01)",
           "the probe head is initialised from the global RNG, so its seed does not reproduce it"),
    Mutant("probe-init-perturbs-global-rng", "src/eval/linear_probe.py",
           "head = torch.nn.utils.skip_init(torch.nn.Linear, dim, num_classes)",
           "head = torch.nn.Linear(dim, num_classes)",
           "the discarded constructor init advances the caller's global RNG stream"),
    Mutant("seed-truthiness", "src/train.py",
           'if cfg.get("seed") is not None:',
           'if cfg.get("seed"):',
           "seed: 0 silently does not seed"),
    Mutant("standardize-eps", "src/eval/linear_probe.py",
           "std = features.float().std(dim=0).clamp_min(eps)",
           "std = features.float().std(dim=0)",
           "a constant feature dimension divides by zero and NaNs the probe"),
    # --- hygiene batch (T.suite-P3-10, P3-11, P2-9) ---
    # anchored on the following line too: ImageNet.__init__ and ImageNet1KDataModule.__init__
    # now both read `copy_data: bool = False`, so the bare line is ambiguous
    Mutant("copy-data-default-on", "src/data/imagenet1k_datamodule.py",
           "        copy_data: bool = False,\n        index_targets: bool = False,",
           "        copy_data: bool = True,\n        index_targets: bool = False,",
           "constructing ImageNet directly raises NotImplementedError for no stated reason"),
    Mutant("ortho-norm-unvalidated", "src/eval/conformal_autoencoder.py",
           "    if norm not in ORTHO_NORMS:\n        raise ValueError(f\"ortho_norm must be one of {ORTHO_NORMS}, got {norm!r}\")",
           "    if False:\n        raise ValueError(f\"ortho_norm must be one of {ORTHO_NORMS}, got {norm!r}\")",
           "a typo'd ortho_norm silently runs Eq. (6) and changes the inferred dimension"),
    Mutant("hvit-saturation-unwarned", "src/models/components/hyperbolic_vit.py",
           "            if reach > 0.95:", "            if reach > 2.0:",
           "a saturating hyp_c/clip_r pair collapses the representation with no warning"),
    Mutant("dynamic-k-clamp", "src/callbacks/dynamic_num_clusters.py",
           "k = min(self.end, self.start + moved) if self.end >= self.start else max(self.end, self.start - moved)",
           "k = self.start + moved if self.end >= self.start else self.start - moved",
           "the k schedule is never clamped at its endpoint"),
    # --- docs that must keep matching the code (D0.cold-P2-3, P3-4, P3-5) ---
    Mutant("readme-stale-project-name", "README.md",
           "git clone https://github.com/alessioborgi/LocaLeJEPA.git",
           "git clone https://github.com/alessioborgi/BundLeJEPA.git",
           "the quickstart clones a repository that does not exist under that name"),
    Mutant("eval-docstring-incomplete", "src/eval/__init__.py",
           "  - :mod:`src.eval.transfer`              the seven out-of-domain transfer datasets\n",
           "",
           "the eval package advertises fewer modules than it has"),
    Mutant("rankme-normalisation", "src/eval/rankme.py",
           "p = sigma / sigma.sum() + eps", "p = sigma + eps",
           "the spectrum is not normalised, so RankMe stops being scale-invariant"),
]


def run_suite() -> int:
    """Number of failing tests, or -1 if pytest could not run at all."""
    result = subprocess.run(
        # --tb=no: a mutant's failures are counted, never read. -p no:cacheprovider: the run is
        # repeated ~30 times and a stale .pytest_cache would reorder it.
        # --no-cov: the coverage gate would fail every run and drown the failure count.
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider", "--no-cov"],
        cwd=REPO, capture_output=True, text=True,
        # the harness is the one holding the lock, so its own runs must be let through
        env={**os.environ, "LOCALEJEPA_MUTATION_RUNNER": "1"},
    )
    match = re.search(r"(\d+) failed", result.stdout)
    if match:
        return int(match.group(1))
    if re.search(r"\d+ passed", result.stdout):
        return 0
    return -1


@contextmanager
def planting_lock():
    """Announce that ``src/`` is being deliberately corrupted.

    Learned the hard way: while a run was in flight I ran ``pytest`` in another shell and got a
    failure from a *planted* defect, then ran ``git add -A && git commit``. That commit happened
    to land between mutants and was clean -- but had it landed a moment earlier it would have
    committed a deliberate defect under a message claiming the opposite, and the suite would have
    been green on the next run because the harness reverts afterwards.

    The lock makes the window visible: ``tests/conftest.py`` refuses to start a session while it
    exists, so a concurrent test run says why instead of reporting nonsense.
    """
    LOCK.write_text(
        "A mutation run is planting defects in src/ right now.\n"
        "Do not run pytest, and do not commit, until it finishes.\n"
    )
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def restore(path: str) -> None:
    subprocess.run(["git", "checkout", "--", path], cwd=REPO, check=True,
                   capture_output=True)


def plant(mutant: Mutant) -> bool:
    target = REPO / mutant.path
    source = target.read_text()
    if source.count(mutant.old) != 1:
        return False                      # anchor missing or ambiguous: not a valid mutation
    target.write_text(source.replace(mutant.old, mutant.new, 1))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", dest="pattern", default=None, help="only mutants matching this")
    parser.add_argument("--report", action="store_true", help="tally only")
    args = parser.parse_args()

    dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("REFUSING: src/ has uncommitted changes; a mutation run would not be revertible.")
        print(dirty)
        return 2

    selected = [m for m in MUTANTS if not args.pattern or args.pattern in m.label]
    print(f"measuring baseline over {len(selected)} mutants ...")
    baseline = run_suite()
    if baseline < 0:
        print("REFUSING: the suite did not run cleanly enough to measure a baseline.")
        return 2
    print(f"baseline: {baseline} failing test(s)\n")

    killed, survived, invalid = [], [], []
    with planting_lock():
        for mutant in selected:
            if not plant(mutant):
                invalid.append(mutant)
                print(f"  {'INVALID':9s} {mutant.label:28s} anchor not found exactly once")
                continue
            failures = run_suite()
            restore(mutant.path)
            verdict = "KILLED" if failures > baseline else "SURVIVED"
            (killed if failures > baseline else survived).append(mutant)
            if not args.report:
                print(f"  {verdict:9s} {mutant.label:28s} {failures - baseline:+d} failures "
                      f"| {mutant.why}")

    after = run_suite()
    print(f"\nrestored tree: {after} failing test(s) (baseline was {baseline})")
    if after != baseline:
        print("WARNING: the tree did not return to baseline; investigate before trusting this run.")

    total = len(killed) + len(survived)
    rate = 100.0 * len(killed) / total if total else 0.0
    print(f"\n{len(killed)} planted, {len(killed)} killed, {len(survived)} survived "
          f"({rate:.0f}% kill rate)" if not survived else
          f"\n{total} planted, {len(killed)} killed, {len(survived)} survived "
          f"({rate:.0f}% kill rate)")
    for mutant in survived:
        print(f"  SURVIVOR: {mutant.label} -- {mutant.why}")
        print(f"            a survivor is a finding about the TEST, not the code")
    for mutant in invalid:
        print(f"  INVALID:  {mutant.label} -- anchor no longer matches; update the harness")
    return 1 if survived or invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())

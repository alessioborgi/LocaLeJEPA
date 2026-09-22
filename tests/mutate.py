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
    # --- provenance and the placeholder guard (X.fix-P2-2 and the self-destruct defect) ---
    Mutant("generator-overwrites-with-placeholders", "src/eval/io.py",
           "    if not os.path.exists(out_path):\n        return\n    raise SystemExit(",
           "    if not os.path.exists(out_path):\n        return\n    return SystemExit(",
           "a generator run without inputs silently replaces a real table with em dashes"),
    Mutant("provenance-omits-the-digest", "src/eval/io.py",
           'lines += [f"| `{name}` | `{digest}` | {size} |" for name, digest, size in found]',
           'lines += [f"| `{name}` | | {size} |" for name, digest, size in found]',
           "a table cannot be checked against the inputs it claims to come from"),
    Mutant("provenance-hides-an-empty-run", "src/eval/io.py",
           '            "**No input files were found.** This table was generated from nothing',
           '            "**Inputs present.** This table was generated from nothing',
           "a report generated from nothing does not say so"),
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
    # --- realised K (campaign plan C1): the column every ablation table's K axis is read from ---
    Mutant("realised-k-reports-configured", "src/losses.py",
           'stats["effective_clusters"] = float(len(losses))',
           'stats["effective_clusters"] = float(K)',
           "the realised-K column silently shows the requested K"),
    Mutant("realised-k-fallback-hidden", "src/losses.py",
           'stats["fallback"] = 0.0 if losses else 1.0',
           'stats["fallback"] = 0.0',
           "a run whose every step fell back to the global test reports itself as local"),
    Mutant("realised-k-kept-counts-dropped", "src/losses.py",
           'stats["kept_fraction"] = float(sum(weights)) / N',
           'stats["kept_fraction"] = float(sum(sizes)) / N',
           "dropped clusters are counted as kept, so the fraction is always 1.0"),
    Mutant("realised-k-module-never-asks", "src/models/lejepa_module.py",
           "reg = self.reg_loss(proj, stats=reg_stats) if self._reg_takes_stats else self.reg_loss(proj)",
           "reg = self.reg_loss(proj)",
           "the module stops asking, so no local run ever logs its realised K"),
    Mutant("realised-k-namespace-dropped", "src/models/lejepa_module.py",
           '**{f"local/{key}": value for key, value in reg_stats.items()},',
           '**{key: value for key, value in reg_stats.items()},',
           "the diagnostics land outside train/local/*, where nothing reads them"),
    # --- the AICR launcher helper (campaign plan C2a): each is a way the FASRC launcher was wrong ---
    Mutant("launcher-resume-by-mtime", "scripts/aicr/campaign.py",
           "best = max(readable, key=lambda c: c.global_step or 0)",
           "best = max(readable, key=lambda c: c.path.stat().st_mtime)",
           "resume trusts NFS mtimes over the step inside the checkpoint"),
    Mutant("launcher-successor-drops-gpus", "scripts/aicr/campaign.py",
           '        f"--gpus-per-node={gpus}",\n',
           "\n",
           "the successor is submitted without a GPU count -- the FASRC chain's 4->1 GPU revert"),
    Mutant("launcher-geometry-global-batch", "scripts/aicr/campaign.py",
           "    if per_gpu != spec.micro_batch:",
           "    if batch_size != spec.micro_batch:",
           "the geometry asserts the global batch, so a 4-GPU run silently becomes a 16-sample experiment"),
    Mutant("launcher-expand-drops-seeds", "scripts/aicr/campaign.py",
           '    seeds = [int(s) for s in manifest.get("seeds", [0])]',
           "    seeds = [0]",
           "every cell runs once whatever the manifest says about seeds"),
    Mutant("launcher-realised-k-head-not-tail", "scripts/aicr/campaign.py",
           "    tail = values[-last_n:]",
           "    tail = values[:last_n]",
           "the 'last-10-epoch' realised K is the first ten epochs"),
    Mutant("launcher-config-check-never-fails", "scripts/aicr/campaign.py",
           "        if not _values_agree(_parse_override_value(text), actual):",
           "        if False:",
           "a run trained with a different lambda than the manifest passes postflight"),
    Mutant("launcher-marker-always-verified", "scripts/aicr/campaign.py",
           '    name = ".verified" if verdict.ok else ".failed"',
           '    name = ".verified"',
           "a failed run gets the marker the eval preflight requires"),
    # --- calibration callbacks and the data/eval gates (campaign plan C2b, C2c) ---
    Mutant("c2bc-wait-not-counted", "src/callbacks/throughput.py",
           "            self.wait_seconds += now - self._last_batch_end",
           "            pass",
           "the data-wait fraction is always zero, so a decode-starved run looks GPU-bound"),
    Mutant("c2bc-interrupt-fires-every-step", "src/callbacks/throughput.py",
           "            self.fired = True\n            print(f\"SelfInterrupt",
           "            print(f\"SelfInterrupt",
           "the rehearsed SIGTERM is sent on every step after the first"),
    Mutant("c2bc-resumed-check-loose", "scripts/aicr/campaign.py",
           'checks["resumed"] = (second[0] == first[-1] + 1,',
           'checks["resumed"] = (second[0] >= 0,',
           "a calibration that restarted from scratch passes as a resume"),
    Mutant("c2bc-marker-ok-not-read", "scripts/aicr/campaign.py",
           'checks["verified"] = (bool(payload.get("ok")),',
           'checks["verified"] = (True,',
           "an eval runs on a checkpoint whose marker says ok=false"),
    Mutant("c2bc-verify-accepts-empty-split", "scripts/download_transfer_datasets.py",
           "        if len(train) == 0 or len(test) == 0:",
           "        if False:",
           "an empty transfer split is recorded as verified"),
    # --- the campaign configs (C3): data, guarded by the manifest/composition invariants ---
    Mutant("configs-local-arm-flat-again", "configs/experiment/lejepa/vit_in1k_localballreg_l01_k4.yaml",
           "  lr_eta_min_ratio: 0.001", "  lr_eta_min_ratio: 1.0",
           "the local arm silently goes back to a flat schedule"),
    Mutant("configs-vitl-pair-wd-drift", "configs/experiment/lejepa/vitl14_in1k_localballreg_paper.yaml",
           "  weight_decay: 5e-2         # theirs (was 1e-1)", "  weight_decay: 1e-1",
           "the ViT-L pair differs in weight decay, not only in the regularizer"),
    Mutant("configs-d4-baseline-reverts", "configs/experiment/lejepa/d4rn50_in1k_sigreg.yaml",
           "  lr: 2e-3   # decision 11 (was 1e-3)", "  lr: 1e-3",
           "the D4-RN50 baseline leaves the published recipe again"),
    # --- the LeJEPA-protocol probe and its feature mode (campaign C4, decision 3) ---
    Mutant("c4-cls2-last-block-twice", "src/eval/extract.py",
           "    return torch.cat([prefix[:, 0] for _, prefix in outs], dim=1)",
           "    return torch.cat([outs[-1][1][:, 0], outs[-1][1][:, 0]], dim=1)",
           "the 'last two blocks' feature is the last block twice"),
    Mutant("c4-cache-feature-suffix-dropped", "src/eval/io.py",
           '        stem += f"_{feature}"',
           "        pass",
           "cls2 features are served from -- and overwrite -- the encoder-mode cache"),
    Mutant("c4-evaluate-skips-layer-norm", "src/eval/linear_probe.py",
           "        if layer_norm:\n            x = F.layer_norm(x, (x.shape[1],))\n        logits = F.linear(x, weight, bias)",
           "        logits = F.linear(x, weight, bias)",
           "the LeJEPA probe is evaluated without the normalisation it was trained with"),
    # --- the campaign's table generators and their audit (C5b) ---
    Mutant("c5b-published-in-best-table", "scripts/make_imagenet1k_results.py",
           """            row.append(f'— / {pub["linear"]:.2f}†' if pub and kind == "last" else "—")""",
           """            row.append(f'— / {pub["linear"]:.2f}†' if pub else "—")""",
           "the published LeJEPA number appears in the best-checkpoint table as if it were a reproduction"),
    Mutant("c5b-declaration-ties-to-larger-lambda", "scripts/make_ablation_tables.py",
           "        if scores and max(scores) > best:",
           "        if scores and max(scores) >= best:",
           "a tie between two lambdas is resolved towards the larger one, against the written rule"),
    Mutant("c5b-audit-skips-the-grep", "scripts/audit_tables.py",
           "        if not any(rendered in row for row in rows):",
           "        if False:",
           "a hand-edited table passes the audit as long as the archive agrees with cells.json"),
    Mutant("c5b-fallback-marker-dropped", "scripts/make_ablation_tables.py",
           '    if agg["fallback_epochs"]:\n        text += f" ⚠fallback×{agg[\'fallback_epochs\']}"',
           "    pass",
           "a cell whose run fell back to the global statistic is printed like a healthy one"),
    # --- the torus report and the audit's row discipline (C5c) ---
    Mutant("c5c-audit-ignores-the-row", "scripts/audit_tables.py",
           '        candidates = [row for row in rows if cell.get("row", "") in row] if cell.get("row") else rows',
           "        candidates = rows",
           "a value present in any other row vouches for this cell"),
    Mutant("c5c-report-reads-the-first-backbone", "scripts/make_equivariance_report.py",
           "        entry = results.get(run.run_name)\n        if entry is None:\n            lines.append(f\"| {encoder} | {reg} | *pending* | *pending* |\")",
           "        entry = next(iter(results.values()), None)\n        if entry is None:\n            lines.append(f\"| {encoder} | {reg} | *pending* | *pending* |\")",
           "every torus row shows the first backbone's numbers -- the old _load did exactly this"),
    # --- the morning of 2026-09-21 (F44): green-on-one-machine, and the lock's two halves ---
    Mutant("f44-verifier-unsorted-draw", "scripts/verify_imagenet_staging.py",
           "    return sorted(f.path for f in os.scandir(class_dir) if f.is_file())",
           "    return [f.path for f in os.scandir(class_dir) if f.is_file()]",
           "the seeded draw follows readdir order again: green on APFS, red on NFS"),
    Mutant("f44-env-lines-keep-conda-half", "scripts/aicr/campaign.py",
           '        if not line or line.startswith("#") or "==" not in line or line.startswith(("http://", "https://", "-e ")):',
           '        if not line or line.startswith("#"):',
           "the conda half of the lock enters the hash and every preflight refuses again"),
    # --- C7, decision 16: the eval-selected column and the online curve ---
    # the defect is re-admitting a container's NON-BatchNorm children through add_module (torch's converter
    # does that for every child); a BatchNorm's own parent inside escnn is a plain module and admits it,
    # so mutating the BatchNorm insertion alone survived (2026-09-21) -- this one rebuilds the way torch does
    Mutant("c15-eval-config-takes-unverified-cells", "scripts/aicr/make_ablation_eval_config.py",
           '        if require_verified and not (run_dir / ".verified").is_file():',
           "        if False:",
           "a cell whose postflight refused is evaluated and its numbers reach a table (G5/V4)"),
    Mutant("c15-eval-names-file-sorted", "scripts/aicr/make_ablation_eval_config.py",
           '    args.names.write_text("".join(f"{e[\'name\']}\\n" for e in entries), encoding="utf-8")',
           '    args.names.write_text("".join(f"{n}\\n" for n in sorted(e["name"] for e in entries)), encoding="utf-8")',
           "the array index and the config order drift: every task evaluates another cell's checkpoint"),
    Mutant("c15-eval-array-trusts-the-exit-code", "scripts/aicr/slurm_eval_array.sh",
           '[ -f "$RESULT" ] || { echo "=== EVAL FAILED $NAME: no $RESULT ===" >&2; exit 1; }',
           "true",
           "a backbone the eval SKIPPED (missing checkpoint, exit 0) counts as evaluated"),
    Mutant("c14-done-from-any-candidate", "scripts/aicr/campaign.py",
           '                if c.path.name == "last.ckpt" and c.epoch is not None and c.epoch >= max_epochs - 1]',
           '                if c.epoch is not None and c.epoch >= max_epochs - 1]',
           "a run interrupted inside its final epoch is called done; the resubmission skips training and the postflight refuses (F54)"),
    Mutant("c13-local-by-experiment-name", "scripts/aicr/campaign.py",
           "    local_run = run_is_local(run_dir, spec)      # from the recorded config, never the name (F53)",
           '    local_run = any(item.startswith("model.reg_loss.num_clusters=") for item in spec.overrides) or "local" in spec.experiment',
           "the SIGReg control fails its postflight for a realised-K column it never logs (F53)"),
    Mutant("c13-done-cell-skips-the-postflight", "scripts/aicr/slurm_train.sh",
           '    $CAMPAIGN postflight "$GRID" --index "$INDEX" --run-dir "$RUN_DIR" --ckpt-dir "$CKPT_DIR" --lock "$ENV_LOCK" --python "$PY" --train-rc 0   # a complete cell is verified and marked (F53)',
           '    true',
           "a complete cell resubmitted exits 0 without a marker and its wrong `.failed` stays (F53)"),
    Mutant("c12-merge-keeps-the-first-launch", "scripts/aicr/campaign.py",
           "        by_epoch.update(launch)      # this launch supersedes every earlier one on the epochs it logged",
           "        by_epoch = {**launch, **by_epoch}",
           "the interrupted launch's stale rows win over the resumed launch's for the re-trained epochs (F52)"),
    Mutant("c12-postflight-reads-the-newest-file-only", "scripts/aicr/campaign.py",
           "        problems = metrics_complete(csv_dir, spec.max_epochs, local_run=local_run)   # every launch, merged (F52)",
           "        problems = metrics_complete(launches[-1], spec.max_epochs, local_run=local_run)",
           "every resumed run fails its postflight again with a complete artifact (F52)"),
    Mutant("c12-ablation-summary-newest-file-only", "scripts/make_ablation_tables.py",
           "    _, rows = C.merged_metrics_rows(csv_dir)",
           "    _, rows = C.merged_metrics_rows(C.metrics_versions(csv_dir)[-1])",
           "a resumed ablation cell is 'pending' forever and never enters its table (F52)"),
    Mutant("c11-expand-summary-sorted-tags", "scripts/aicr/campaign.py",
           '        blocks = " ".join(f"{t}[{a}-{b}]" for t, a, b in block_index_ranges(runs))   # index order, never sorted (F51)',
           '        blocks = " ".join(f"{t}[{a}-{b}]" for t, a, b in sorted(block_index_ranges(runs)))',
           "the expand summary lists the blocks alphabetically again and --indices are read off the wrong order (F51)"),
    Mutant("c11-submit-names-no-cell", "scripts/aicr/campaign.py",
           '        print(f"# cells: {describe_cells(runs, indices)}", file=sys.stderr)   # F51: verify by name, not by position',
           '        pass',
           "a submission no longer names its cells: the wrong --indices go unnoticed until the run announces itself (F51)"),
    Mutant("c10-last-ckpt-follows-the-monitor", "configs/callbacks/default.yaml",
           "  monitor: null\n  save_top_k: 1\n  save_last: False\n  every_n_epochs: 1",
           '  monitor: "val/probe_acc"\n  save_top_k: 1\n  save_last: True\n  every_n_epochs: 1',
           "last.ckpt is again a copy of best.ckpt: the final-epoch weights are never written (F50)"),
    Mutant("c10-successor-after-a-cancel", "scripts/aicr/campaign.py",
           "    if left is not None and left <= WALL_SIGNAL_WINDOW_S:",
           "    if True:",
           "a cancelled wave spawns a successor per task again (F50)"),
    Mutant("c9-download-keeps-hollow-dirs", "scripts/download_transfer_datasets.py",
           "    hollow = remove_hollow_dirs(root)",
           "    hollow = []",
           "a purged, hollow dataset directory is taken for a downloaded dataset and the extraction is skipped (F49)"),
    Mutant("c9-download-never-touches", "scripts/download_transfer_datasets.py",
           '    print(f"=== touched {touch_tree(root)} entries under {root} (scratch purge is by mtime, F49) ===", flush=True)',
           '    pass',
           "the extracted transfer datasets keep their archives' mtimes and vanish at the next purge sweep (F49)"),
    Mutant("c8-sync-bn-through-add-module", "src/utils/sync_bn.py",
           "            convert_sync_batchnorm_inplace(child, process_group)",
           "            module.add_module(name, convert_sync_batchnorm_inplace(child, process_group))",
           "containers are rebuilt through add_module again and every multi-GPU D4-RN50 launch dies (F48)"),
    Mutant("c8-module-never-syncs-bn", "src/models/lejepa_module.py",
           '        if stage == "fit" and self.sync_batchnorm and trainer is not None and trainer.world_size > 1:',
           '        if False:',
           "BatchNorm statistics are per-rank on every DDP run while the config says synchronized"),
    Mutant("c7b-calibrate-overrides-no-per-step-rows", "scripts/aicr/campaign.py",
           '           "trainer.log_every_n_steps=1"]',
           '           ]',
           "calibrate mode logs every 50 steps again and the seam check calls a working resume broken"),
    Mutant("c7-evalsel-final-not-max", "src/eval/linear_probe.py",
           "            epoch, top1, top5 = max(curve, key=lambda c: c[1])     # earliest epoch on a tie",
           "            epoch, top1, top5 = curve[-1]",
           "the eval-selected probe reports the final epoch instead of the best one: no longer an upper bound"),
    Mutant("c7-online-curve-final-not-max", "scripts/aicr/campaign.py",
           "    best_epoch, best = max(points, key=lambda pt: (pt[1], -pt[0]))",
           "    best_epoch, best = points[-1]",
           "the online curve's maximum is its final value: gate G4a reads the wrong number"),
    Mutant("c7-generator-ignores-probe-select", "scripts/make_imagenet1k_results.py",
           '    assert got_select == select, f"{run_name}: a {got_select!r}-selected result cannot fill a {select!r} cell"',
           "    pass",
           "a strict result fills an eval-selected cell (or the reverse) and the table is mislabelled"),
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

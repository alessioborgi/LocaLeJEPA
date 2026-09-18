# LocalSIGReg and LocalBallReg: algorithms, complexity, and empirical evaluation

> Written against the pre-refactor code (`loss_type` / `local_sigreg_*` / `local_ballreg_*` parameter names); the runs and W&B groups it reports are unchanged.

## Scope and implementation references

This document analyzes **LocalSIGReg and LocalBallReg with per-batch k-means**, relative to the global SIGReg objective used by LeJEPA in this repository. It describes the implementation at commit `a337a0dc1cfaf730d134f3bb3d1d460796c5e010`; the complexity results below are derived from the code, not measured results or claims about every possible LeJEPA implementation.

Sources:

- [`src/losses.py`](../src/losses.py): `invariance_loss`, `sigreg_loss`, the two k-means backends, and `local_sigreg_loss`.
- [`src/models/lejepa_module.py`](../src/models/lejepa_module.py): common training objective and optional cluster-count schedule.
- `src/models/jepa_module.py`: encoder/projector, diagnostics, and training scaffold.
- `scripts/benchmarks/benchmark_local_sigreg_kmeans.py`: initial timing harness, with limitations discussed below.
- `dev/gpu_probe.py`: full-step time and peak memory for any experiment config.
- raw CSV output of every sweep reported below: `docs/data/` on the `benchmarking` branch.
- `scripts/benchmarks/loss_bench.py`: regularizer time and memory in isolation, with no encoder in the loop.

The baseline and local methods use the same encoder, projector, augmentations, invariance term, and optimizer. Only the regularizer changes. Labels are not used for these self-supervised objectives. LocalBallReg is covered in the additional section below; other BundLeJEPA variants are outside scope. The original filename is retained for existing links.

## Notation

| Symbol | Meaning |
|---|---|
| $B$ | Number of original images in one device's batch |
| $V$ | Number of augmented views per image |
| $D$ | Projection dimension, after the projector |
| $S$ | Number of random sketch directions; currently hardcoded to 256 |
| $Q$ | Number of integration knots; default 17 |
| $K$ | Effective cluster count, clipped to $[1,B]$ |
| $I$ | Maximum k-means iterations, at least 1; default 10 |
| $m$ | Minimum accepted cluster size, at least 1; default 4 |
| $n_k$ | Number of original images assigned to cluster $k$ |
| $\mathcal A, K_a, B_a$ | Accepted clusters, their count, and $\sum_{k\in\mathcal A}n_k\le B$ |
| $C_{\mathrm{net}}, M_{\mathrm{net}}$ | Time and memory for the shared network training step, including optimizer state |

The formulas assume a single device and dense eager tensor operations, without activation checkpointing. They count arithmetic and tensor storage; allocator caches, kernel launch latency, and synchronization affect actual measurements. Memory is in scalar elements unless stated otherwise. The regularizer casts projections to FP32 even if the network uses mixed precision.

## Algorithm

For each image, the network produces $V$ projections, collected into $Z\in\mathbb R^{V\times B\times D}$. Let $\mathcal G$ contain the reference/global views, or all views when `num_global=None`.

$$
\bar z_i=\frac{1}{|\mathcal G|}\sum_{v\in\mathcal G}Z_{v,i,:},
\qquad L_{\mathrm{inv}}=\frac{1}{VBD}\sum_{v,i,d}(Z_{v,i,d}-\bar z_{i,d})^2.
$$

The reference mean remains differentiable in the invariance loss. A **detached copy** of the mean is used for clustering: there is no gradient through k-means, cluster labels, or assignments. With the default normalization, assignments use cosine-like neighborhoods. SIGReg still operates on the **raw, uncentered, unnormalized** projections inside each accepted cluster.

```text
Algorithm: one LocalSIGReg training step
Input: B images; encoder f; projector g; reference views G;
       lambda; cluster count K; iteration budget I; minimum size m;
       weighting flag; normalization flag; k-means backend

1. Produce V augmented views per image.
2. Z <- reshape(g(f(views))) to shape [V, B, D].
3. reference <- mean(Z[G], over views).
4. L_inv <- mean((Z - reference)^2).
5. points <- stop_gradient(reference).
6. labels <- KMEANS(points, clip(K, 1, B), max(I, 1), normalization, backend).
7. losses <- []; sizes <- [].
8. For k = 1, ..., K:
       indices <- {i : labels[i] = k}
       If |indices| >= max(m, 1):
           losses.append(SIGREG(Z[:, indices, :]))
           sizes.append(|indices|)
9. If losses is empty:
       L_reg <- SIGREG(Z)                         # global fallback
   Else if weighted:
       L_reg <- sum(sizes[k] * losses[k]) / sum(sizes)
   Else:
       L_reg <- mean(losses)                      # current default
10. L <- (1 - lambda) * L_inv + lambda * L_reg.
11. Zero gradients, backpropagate L, and update f and g.
```

Images in rejected clusters still contribute to invariance. If any clusters survive, rejected images do not contribute to regularization. If none survive, global SIGReg regularizes the entire batch. Clusters are recomputed from scratch at every step; no centroid state is carried between batches.

### SIGReg subroutine

For $Y\in\mathbb R^{V\times n\times D}$, draw $S$ independent Gaussian vectors and normalize them to unit length, obtaining $A\in\mathbb R^{D\times S}$. Each local call draws a fresh matrix; sketches are not shared between clusters.

For knots $t_q$ evenly spaced on $[0,3]$, define

$$
\phi_q=e^{-t_q^2/2},\quad
c_{v,s,q}=\frac1n\sum_i\cos(t_q\langle Y_{v,i,:},A_{:,s}\rangle),\quad
s_{v,s,q}=\frac1n\sum_i\sin(t_q\langle Y_{v,i,:},A_{:,s}\rangle).
$$

With $\Delta t=3/(Q-1)$, the weights are $w_q=2\Delta t\,\phi_q$ for interior knots and $w_q=\Delta t\,\phi_q$ at the endpoints. Then

$$
\operatorname{SIGReg}(Y)=\frac{n}{VS}\sum_{v,s,q}w_q[(c_{v,s,q}-\phi_q)^2+s_{v,s,q}^2].
$$

The multiplier $n$ is part of the actual implementation. Thus “unweighted” means an equal average of **cluster SIGReg statistics**, each already multiplied by its cluster size. The weighted mode adds another explicit size weighting. Neither should be described as automatically identical to a global sample-wise average.

### Relationship to LeJEPA

The baseline sets $L_{\mathrm{reg}}=\operatorname{SIGReg}(Z)$ without clustering. Both methods optimize $(1-\lambda)L_{\mathrm{inv}}+\lambda L_{\mathrm{reg}}$, with default $\lambda=0.02$. For $K=1$, the local objective reduces to global SIGReg (up to sketch randomness), but the code still performs clustering. Local neighborhoods change the objective, rather than being just a faster way to compute global SIGReg.

A standalone, compilable LaTeX pseudocode document is in `docs/local_sigreg_pseudocode.tex` on the `benchmarking` branch.

## Time complexity per batch

### Global SIGReg / LeJEPA

Generating and normalizing the sketch matrix costs $O(DS)$. Projection onto sketches costs $O(VBDS)$; evaluating and reducing sine/cosine at the knots costs $O(VBSQ)$. Invariance costs $O(VBD)$. Backpropagation through these differentiable operations has the same asymptotic order as their forward pass.

$$
T_{\mathrm{LeJEPA}}=O\left(C_{\mathrm{net}}+VBD+VBDS+VBSQ+DS\right).
$$

### LocalSIGReg

1. Compute reference means: $O(VBD)$, an upper bound allowing all views as references.
2. Cluster $B$ points in dimension $D$: $O(IBKD)$. The in-repo backend performs a distance assignment and scatter-based centroid update on each iteration. The installed external backend uses similarity calculations and a dense membership-matrix centroid update, also bounded by $O(IBKD)$. It may stop early, while the in-repo backend runs the requested iteration count.
3. Construct per-cluster masks: $O(BK)$; gather accepted projections: total $O(VB_aD)$.
4. Run local SIGReg calls: $O(VB_aDS+VB_aSQ+K_aDS)$. The sample-dependent terms sum over the accepted population, rather than multiplying the full-batch cost by $K$.

For at least one accepted cluster:

$$
T_{\mathrm{local}}=O\left(C_{\mathrm{net}}+VBD+IBKD+BK+VB_aDS+VB_aSQ+K_aDS\right).
$$

If none are accepted, replace the final three terms by $VBDS+VBSQ+DS$; clustering and masking were still paid. A useful worst-case upper bound is

$$
\boxed{T_{\mathrm{local}}=O\left(C_{\mathrm{net}}+VBD+IBKD+BK+VBS(D+Q)+KDS\right).}
$$

Relative to the global baseline, the additional arithmetic is bounded by $O(IBKD+BK+KDS)$, including repeated sketch construction. This is an upper-bound comparison, not an exact time difference: cluster filtering, fallback, backend convergence, and device utilization change constants and effective work.

The code loops over clusters in Python and calls `mask.sum().item()` for each cluster. On accelerators this can introduce host/device synchronization, so small clusters can be slow despite a modest arithmetic count. There is no backward pass through the $I$ k-means iterations because clustering runs under `no_grad`.

## Memory complexity per batch

### Global baseline

The implementation explicitly constructs knot-expanded tensors of shape $[V,B,S,Q]$. During training, autograd retains tensors needed by sine/cosine backward. Including projections and sketches, a conservative order bound is

$$
M_{\mathrm{LeJEPA}}=O\left(M_{\mathrm{net}}+VBD+VBSQ+DS\right).
$$

This is a tensor-storage bound, not a byte-accurate peak-memory prediction. Several tensors of the leading shape can coexist.

### Local method

Clustering requires point storage $O(BD)$, centroids $O(KD)$, and a dense assignment-distance workspace $O(BK)$ in the in-repo implementation. The installed external backend may chunk similarity calculations, but still constructs a dense $K\times B$ membership mask for centroid updates. Therefore do not claim that its total workspace is independent of $BK$.

All accepted cluster losses are collected before a single backward pass. Their saved tensors coexist, so their total knot-expanded storage is $O(VB_aSQ)$, and their independent saved sketch matrices require $O(K_aDS)$. A conservative whole-step bound is

$$
\boxed{M_{\mathrm{local}}=O\left(M_{\mathrm{net}}+VBD+BD+BK+KD+VBSQ+KDS\right).}
$$

The sum is a valid upper bound; clustering workspace and all regularizer activations do not necessarily peak simultaneously. Exact peaks depend on tensor lifetimes and the backend. Iteration count $I$ does not multiply clustering memory: the iterations are not retained by autograd.

**Local splitting does not imply a $K$-fold training-memory reduction.** Measured full-step memory is indeed unchanged, while the regularizer's own footprint does fall with $K$; both are reported in the measured sections. Such a reduction might apply to a streamed, forward-only regularizer with balanced clusters and prompt tensor release, or to a redesigned backward schedule. It does not describe the current training code. Rejected samples can reduce retained local activations, but the fallback can restore full-batch storage.

As a scale reference, at $B=256,V=4,S=256,Q=17$, one FP32 $[V,B,S,Q]$ tensor contains 4,456,448 elements, or **17 MiB**. This is one tensor, not total peak usage.

## Scaling implications

| Variable changed | Global SIGReg | Additional local-method behavior |
|---|---|---|
| $B$, fixed other parameters | Linear in the bound; measured time is flat below $B=512$ because the kernels do not saturate the device, while measured memory is linear throughout | Also linear if $K,I$ stay fixed; measured time is dominated by per-cluster overhead instead |
| $K$, fixed $B$ | No dependence | Assignment work/workspace and mask scans grow with $K$; more sketch generation and small-kernel overhead. Measured time is linear in $K$ and measured peak memory falls as $1/K$ |
| $K\propto B$ | Linear in $B$ | Dense k-means becomes quadratic in $B$ in time and workspace |
| $I$ | No dependence | Linear worst-case clustering time, no asymptotic memory increase. Measured at 0.71 ms per iteration for the in-repo backend; the external one saturates once it converges |
| $D$ | Sketch projection grows linearly | Clustering and centroid/sketch storage also grow linearly. Measured time and memory are flat over a sixteenfold change in $D$ |
| $V$ | Linear regularizer cost | Local SIGReg also linear; only $B$ averaged points are clustered, not $VB$ points |
| $S$ or $Q$ | Linear in either separately | Same sample-dependent scaling; $S$ additionally increases per-cluster sketches |

With fixed $K,I,D,V,S,Q$, both regularizers scale linearly with batch size. The network's contribution remains separate: changing image resolution, token count, or architecture can dominate the measured step. At fixed dataset size and constant parameters, fewer steps per epoch can offset the linear per-step growth in $B$; practical throughput and optimization behavior still change.

The optional scheduled cluster count uses $K_e=\min(K_{\max},K_{\mathrm{start}}+\lfloor e/\Delta e\rfloor)$. Apply the same formulas at the current epoch's $K_e$; report timings by $K$ rather than averaging the entire schedule into one unexplained number.

The full training scaffold also computes covariance diagnostics with $O(VBD^2)$ work and $O(D^2)$ covariance storage, plus optional online-probe work and logging. These are shared costs and should be separated from objective-only measurements. For distributed runs, $B$ here is the local batch size; the analyzed loss has no cross-device embedding gather. Communication and global-neighborhood variants require a separate analysis.

## Empirical analysis plan: one overnight Imagenette study

### Questions and outputs

Measure (1) objective overhead versus global SIGReg, (2) full-step slowdown and images/s, (3) peak memory versus batch size, and (4) whether the observed trends agree with the symbolic terms. Compare `sigreg`, `local_sigreg/torch`, and `local_sigreg/fast_pytorch_kmeans`. Report measured findings separately from the bounds above. A short quality sanity check is useful, but one night on Imagenette will not establish downstream superiority.

Produce raw step-level measurements, run-level summaries, a machine/environment manifest, plots, and a short results report. Full-step measurements are reported in the final section; the isolated regularizer measurements have not been collected yet.

### Required benchmark work before a comparison run

The current script is a useful smoke test, but it is **not yet sufficient** for the requested empirical comparison:

- Add the global SIGReg baseline. Its current backend loop compares two local implementations only.
- Add separate objective-only, training-step, and end-to-end modes. `local_sigreg_ms_*` currently includes the entire local forward regularizer, not isolated k-means; `total_ms_*` excludes data loading and transfer.
- Add memory measurement, independent repeats, seeds, per-step output, and full configuration/version metadata. Save each completed run immediately so a later OOM does not discard earlier results.
- Replace `itertools.cycle(loader)` with an iterator that restarts the loader at exhaustion. `cycle` caches previously yielded image tensors, can consume substantial host memory, and eventually repeats the same augmentations.
- Add positive-argument validation, empty-loader checks, finite loss/gradient checks, and per-case error/OOM reporting.
- Add cluster diagnostics: actual iteration count when available, accepted cluster count, accepted image fraction, cluster-size distribution, and global-fallback frequency. Collect intrusive diagnostics in separate profiling passes so they do not contaminate headline timings.
- For exact $S$-scaling, make the hardcoded 256 configurable while preserving its default; $Q$ is already a loss argument but needs exposure in the harness. These are optional second-stage changes.

### Fairness and measurement protocol

1. Use one idle device and run cases sequentially. Record hardware, OS, package versions, git commit plus dirty diff, dtype, thread/worker count, power conditions, and any CPU fallback. Verify both backends on the selected device first. Do not silently compare GPU execution with CPU fallback.
2. Use seeds 0, 1, and 2. Within a seed, start methods from the same saved model state and the same bounded set of augmented image batches. Keep augmentation, model initialization, and regularizer random streams separate where possible. Randomize case order to reduce thermal/order bias.
3. For objective-only tests, use identical detached projections with gradients enabled, saved from the same Imagenette encoder state. Include both initialization and a short trained checkpoint if available. This avoids conflating implementation speed with diverging training trajectories. Do not force equal cluster labels in the main native-backend comparison; a fixed-label ablation can separately isolate the local SIGReg calculation.
4. Warm up for 20 steps, then measure 100 steps per case, for three independent repeats. Extend to 200 measured steps only if pilot variance or available time justifies it. Report median, mean, p90, and uncertainty across independent run summaries; consecutive steps are not independent experimental replicates.
5. For headline step timing, synchronize immediately before and after the complete step only. Use a separate instrumented pass for encoder/projector, clustering, local regularizer, backward, and optimizer timings. Internal synchronizations perturb scheduling; do not sum instrumented timings and present them as uninstrumented throughput.
6. On CUDA, reset and read peak allocated and peak reserved memory for each case, with a documented baseline after model/optimizer initialization. On MPS, separately report sampled tensor/driver allocation and process RSS using supported APIs, and explicitly label sampling as a lower bound on peak. Unified-memory counters overlap and must not be added together. Use isolated processes for memory cases to limit allocator-history effects.
7. Report objective forward/backward time, full-step time, original images/s ($B/t$), views/s ($VB/t$), end-to-end epoch time, and memory in MiB. Keep timing passes separate from heavyweight profilers. Preserve failed/OOM cases rather than omitting them.

### Bounded experiment matrix

The first matrix is a sweep of **one factor at a time**, not a Cartesian product. Reference settings: $B=128,V=2,D=64,K=8,I=10,S=256,Q=17,m=4$, unweighted and normalized assignments, FP32.

| Stage | Cases | Purpose |
|---|---|---|
| Pilot | Three methods, $B\in\{32,128\}$, one repeat | Compatibility, finite gradients, memory headroom, runtime estimate |
| Batch scaling | Three methods, $B\in\{16,32,64,128,256\}$, three repeats | Primary time/memory comparison; 45 runs |
| Cluster scaling | Two local backends, $B=256$, $K\in\{1,2,4,8,16,32\}$, three repeats | Assignment overhead, small-cluster effects; 36 runs |
| Iteration scaling | Two local backends, $I\in\{1,5,10,20\}$, three repeats | Worst-case budget versus early stopping; 24 runs |
| Dimension scaling | Three methods, $D\in\{16,64,256\}$, three repeats | Projection versus clustering cost; 27 runs |
| View scaling | Three methods, $V\in\{2,4\}$, three repeats | Verify clustering uses $B$ points; 18 runs |

These are 150 run slots before deduplicating repeated reference settings, at 120 steps each. They are intended primarily as inexpensive objective-only tests. Low-$B$, high-$K$ cases must be interpreted with coverage/fallback diagnostics: apparently fast runs may simply skip many clusters.

Next, run the three methods with the existing TinyConv model over the five batch sizes and three seeds (45 slots). Finally, validate the representative full model using the existing ViT-S/8 Imagenette configuration: 128-pixel crops, **four views**, and **projection dimension 16**. Use two memory-safe batch sizes selected by the pilot, all three methods, and ideally three repeats (18 slots). This bridges the simplified benchmark to the actual experiment settings; the current TinyConv script defaults do not match the ViT configuration.

For training-scaffold measurements, keep covariance diagnostics, online probe, validation, checkpointing, and logging settings identical across methods. Measure a bounded training window with validation/checkpointing excluded, then report an additional end-to-end measurement with the intended normal settings.

### Nightly budget and execution order

Treat an eight-hour window as a planning budget, not a runtime promise:

- First 30 minutes: pilot and runtime/memory calibration.
- Next 2 hours: objective-only primary matrix; prioritize batch and cluster sweeps.
- Next 2 hours: TinyConv full-step and memory tests.
- Next 2 hours: representative ViT runs, reducing the matrix to fit measured throughput.
- Final 90 minutes: optional short quality sanity runs, aggregation, plots, and interruption margin.

After the pilot estimate

$$
T_{\mathrm{plan}}\approx\sum_c R_c[(W_c+N_c)\widehat t_c+t_{\mathrm{setup},c}]+T_{\mathrm{IO}}.
$$

Reserve at least 20% of the remaining wall-clock budget. Stop launching new cases before the deadline; persist partial results. If time is tight, retain the global baseline, both local backends, batch/cluster sweeps, and representative ViT checks; defer secondary dimensions and quality experiments. Never reduce repeats for just one method without documenting it.

An optional quality check should compare identical numbers of optimizer updates with the same starting weights and data order, then evaluate the same frozen-feature k-NN or linear probe protocol. Report total time to reach that checkpoint. Equal-time comparisons answer a different question and should be labeled separately. Full 800-epoch training is outside this first profiling study.

### Existing smoke-test command

The following command is supported by the current script. It only checks local backends and does not implement the complete plan:

```bash
uv run python scripts/benchmarks/benchmark_local_sigreg_kmeans.py \
  --device mps --batch-sizes 16 32 64 128 \
  --warmup-steps 20 --steps 100 \
  --csv outputs/local_sigreg_pilot/local_backends.csv
```

Use `--device cuda` on a CUDA host. Imagenette is expected under `data/imagenette/imagenette2-160/train`; override `--data-root` if needed. The dataset directory exists in the current workspace, but its integrity and runtime compatibility still need the pilot.

### Analysis and acceptance criteria

- Plot objective time, full-step time, and memory against $B$; overlay local/global ratios at matched settings.
- Plot time versus $K$ and $I$, with coverage and fallback annotations. Plot actual iterations where available; equal maximum iterations do not imply equal executed work.
- Use log-log slopes to describe observed scaling over the tested range, with uncertainty and the range reported. A finite sweep cannot prove an asymptotic bound.
- Compare measured trends with $IBKD$, $VB_aS(D+Q)$, and $K_aDS$, allowing an intercept for fixed costs. Do not infer coefficients for correlated terms from a single batch-size sweep.
- Report the maximum successful tested batch size, allocator metric definition, and all OOMs. Do not treat MPS sampled allocation as a CUDA-style exact peak.
- Validate $K=1$ against global SIGReg with controlled sketch RNG, and exercise the all-clusters-rejected fallback. Confirm that grouping is detached and projection gradients are finite.
- Conclude separately on asymptotic scaling, measured overhead on this hardware, and quality sanity checks. Do not claim a winning backend or a measured speed/memory benefit before collecting results.

## LocalBallReg: k-means neighborhoods with a uniform-ball target

**LocalBallReg does use k-means**, with exactly the same assignment code and options as LocalSIGReg. `ballreg_loss` is the inner statistic called on each cluster, just as `sigreg_loss` is the inner statistic for LocalSIGReg. Calling the inner function directly on a whole batch gives a global baseline without clustering. See `ballreg_loss`, `_ball_d_cf`, and `local_ballreg_loss` in [`src/losses.py`](../src/losses.py).

| Method | k-means | Target | Centering before the statistic |
|---|---|---|---|
| Global SIGReg / LeJEPA | No | Isotropic Gaussian | None |
| LocalSIGReg | Yes | Isotropic Gaussian per cluster | None |
| Global BallReg | No | Uniform inside a D-ball | Optional; default false |
| LocalBallReg | Yes | Uniform inside a D-ball per cluster | Per-view cluster mean; default true |

A ball includes its interior; it is not the sphere-surface target used by SUSReg. Distribution matching along random directions is a soft objective, not a hard projection or clipping of all points into a ball. Normalization for k-means affects assignments only, not the vectors passed into the local statistic.

### Differences in the objective

Both local methods use detached view means for clustering, the same minimum cluster size, weighting, and fallback rule. With identical inputs and controlled clustering randomness their assignments match. Their later training trajectories need not match.

For each accepted cluster, LocalBallReg defaults to differentiable centering:

$$
Y_{v,i,:}=Z_{v,i,:}-\mu_{v,k},\qquad
\mu_{v,k}=\frac{1}{n_k}\sum_{j\in C_k}Z_{v,j,:},\quad i\in C_k.
$$

The mean is computed separately for each view. It is **not** the detached k-means centroid or one mean shared across views. For fixed assignments, the centered statistic is invariant to translating all points of a cluster within a view together. Invariance still couples views, and translations may change the next clustering assignment. LocalSIGReg currently has no centering option: its local statistic compares raw projections against a zero-centered Gaussian, and therefore also penalizes location mismatch.

The target characteristic function changes from the Gaussian function to

$$
\phi_{D,R}(t)=
\frac{\int_{-1}^{1}\cos(Rtu)(1-u^2)^{(D-1)/2}\,du}
     {\int_{-1}^{1}(1-u^2)^{(D-1)/2}\,du}.
$$

The default radius is $R=\sqrt{D+2}$, giving identity population covariance. It is determined by dimension unless explicitly configured, not estimated from cluster radii. The code approximates this ratio using $H = 64$ (`n_quad`) Gauss–Legendre nodes. It keeps the **Gaussian integration window** $e^{-t^2/2}$; only the target inside the squared characteristic-function error changes. Sketches, knots, sine/cosine sample averages, and the cluster-size multiplier follow SIGReg.

The default comparison therefore changes **both target shape and centering**. A measured improvement cannot be attributed solely to a uniform-ball target without a centering control. The same loss coefficient also need not produce the same regularizer gradient scale.

### LocalBallReg pseudocode

```text
Algorithm: one LocalBallReg training step
1. Compute Z, the differentiable reference mean, and L_inv as for LocalSIGReg.
2. Run the SAME k-means on stop_gradient(reference), with the same options.
3. For each cluster C_k with n_k >= m:
       r_k <- BALLREG(Z[:, C_k, :], radius, center_clusters, H)
4. If no cluster survives:
       L_reg <- BALLREG(Z, radius, center_clusters, H)
   Else:
       L_reg <- mean(r_k), or sum(n_k * r_k) / sum(n_k) if weighted.
5. Backpropagate (1 - lambda) * L_inv + lambda * L_reg and update the network.

Subroutine BALLREG(Y [V, n, D], radius, center, H):
1. Cast Y to FP32.
2. If center: Y <- Y - mean(Y, over samples, separately per view).
3. Build Q knots on [0, 3] and the SAME Gaussian-windowed trapezoid weights.
4. R <- radius if supplied, otherwise sqrt(D + 2).
5. Without gradients, compute H Gauss-Legendre nodes u_h and weights a_h.
   b_h <- a_h * (1 - u_h^2)^((D - 1)/2)
   target[q] <- sum_h b_h * cos(R * t_q * u_h) / sum_h b_h.
6. Draw and normalize a fresh D-by-S Gaussian sketch matrix A.
7. Compute empirical cosine and sine means over samples, as in SIGREG.
8. Return n/(V*S) * sum_{v,s,q} w_q *
       ((cosine_mean[v,s,q] - target[q])^2 + sine_mean[v,s,q]^2).
```

The fallback preserves `center_clusters`: by default it is **centered global BallReg**, not the uncentered default of a direct `ballreg_loss` call. At $K=1$, the local method matches global BallReg with the same centering/radius/quadrature and sketch randomness, while still paying clustering overhead. LaTeX versions of the wrapper and subroutine are included in the accompanying `.tex` file.

### Time and memory complexity

Let $G(H)$ be the CPU cost of `numpy.polynomial.legendre.leggauss(H)` and $W_G(H)$ its workspace. Keeping these explicit avoids assuming a library routine is constant when sweeping $H$. Once nodes exist, target evaluation costs $O(QH)$ time and $O(QH+H)$ temporary space. Centering costs $O(VnD)$ time and at most $O(VnD)$ extra storage; its backward has the same order.

Global BallReg regularizer cost:

$$
T_{\mathrm{ball}}=O(VBDS+VBSQ+DS+VBD+G(H)+QH).
$$

LocalBallReg training-step cost, with at least one accepted cluster:

$$
T_{\mathrm{local\ ball}}=O\left(C_{\mathrm{net}}+VBD+IBKD+BK
+VB_aS(D+Q)+K_aDS+K_a[G(H)+QH]\right).
$$

The $VBD$ term covers centering, reference construction, and invariance. For no accepted clusters, replace the local statistic terms with one global BallReg call on all $B$ samples; clustering/masking costs remain. Replace $B_a,K_a$ with $B,K$ for a conservative worst-case bound.

Currently every accepted cluster call regenerates quadrature nodes on CPU, transfers them to the device, and recomputes the same target. Neither nodes nor target values are cached. This adds repeated setup overhead. For fixed $H,Q$, it is constant per cluster. Caching by dimension, radius, grid, quadrature settings, device, and dtype is a possible optimization, not part of current-code complexity measurements.

A conservative whole-step memory bound is

$$
M_{\mathrm{local\ ball}}=O\left(M_{\mathrm{net}}+VBD+BD+BK+KD
+VBSQ+KDS+QH+W_G(H)\right).
$$

Centering does not add a higher-order activation term. Quadrature runs under `no_grad`, so its full temporary workspace is reused, not retained for all clusters; small saved target/error tensors are dominated by the existing activation bound. This bound includes host and device storage abstractly; actual host and device metrics must be reported separately. All differentiable cluster graphs still coexist before backward, so the earlier warning against a claimed K-fold training-memory reduction applies equally here.

For fixed $H,Q,S,D,V,K,I$, both local methods scale linearly in $B$. Both have the same $O(IBKD)$ clustering term and can become quadratic in $B$ when $K\propto B$. LocalBallReg adds centering and quadrature, not a different neighborhood algorithm.

### Expanded empirical plan (supersedes the three-method run counts above)

Use five primary methods: global SIGReg, LocalSIGReg with each backend, and LocalBallReg with each backend. Keep all shared settings fixed; use LocalBallReg defaults `center_clusters=true`, `radius=None`, `n_quad=64`. Extend the harness to expose these settings and global BallReg as an ablation. The current smoke-test command above still measures LocalSIGReg only.

| Sweep | Revised cases | Run slots before deduplication |
|---|---|---|
| Batch size | Five methods × five batch sizes × three repeats | 75 |
| Cluster count | Two local targets × two backends × six K values × three repeats | 72 |
| Iteration budget | Two local targets × two backends × four I values × three repeats | 48 |
| Dimension | Five methods × three D values × three repeats | 45 |
| Views | Five methods × two V values × three repeats | 30 |

The 270 objective-only slots are a **candidate pool**, not a promise to finish overnight. TinyConv validation would add 75 slots, and representative ViT validation 30, if all proposed combinations fit. Retain the eight-hour budget and pilot-based estimate above: prioritize all five methods in batch scaling, cluster scaling for both local targets, and a representative ViT check. Defer secondary sweeps and quality runs as needed. Do not silently double the deadline or run methods concurrently on the same device.

Additional targeted controls, at one memory-safe reference setting:

- Compare global BallReg centered/uncentered and LocalBallReg centered/uncentered. A centered LocalSIGReg control would isolate target shape at matched centering, but requires an explicit implementation change: it is not an existing option.
- Check $K=1$ against global BallReg with matching centering and controlled sketches. Check that the all-clusters-rejected fallback preserves centering. Confirm finite gradients through mean subtraction.
- Separately profile node generation, host/device transfer, target evaluation, clustering, and empirical-statistic calculation. Use separate instrumented passes, preserving the uninstrumented headline timing protocol.
- If time permits, sweep $H\in\{32,64,128\}$ on one representative case; fix sketches to compare numerical target/loss stability as well as runtime.
- Record coverage, fallback rate, and loss/gradient magnitudes for both targets. Use identical frozen projections for implementation comparisons; use matched training settings for trajectory comparisons. Differences in cluster occupancy during training must not be mistaken for pure backend efficiency.

The final report should distinguish the common cost of k-means, extra BallReg target/centering work, and the effect of changing the objective. Full-step BallReg numbers are in the final section; isolated measurements separating quadrature, host transfer, and clustering have not been collected yet.

## Measured results: full training step

NVIDIA A100-SXM4-40GB, `bf16-mixed`, ViT-S/8, crop 128, $V=4$, $B=128$, $D=16$, $K=8$, $I=10$, $S=256$, $Q=17$, `fast_pytorch_kmeans` backend. `dev/gpu_probe.py` pulls one real batch and replays it, so the dataloader sits outside the timed loop. The three experiment configs differ only in `loss_type` and the local knobs.

Every method was measured twice, at 30 and at 300 steps, before and after the two implementation fixes described below. Seconds per step:

| Method | before, 30 | before, 300 | after, 30 | after, 300 | vs LeJEPA, after | Peak reserved |
|---|---|---|---|---|---|---|
| `sigreg` (LeJEPA) | 0.264 | 0.266 | 0.263 | 0.263 | — | 22.3 GiB |
| `local_sigreg` | 0.280 | 0.279 | 0.277 | 0.277 | +5.3% | 22.3 GiB |
| `local_ballreg` | 0.296 | 0.298 | 0.279 | 0.278 | +5.7% | 22.3 GiB |

Spread between the 30-step and 300-step measurements is 0-2 ms per method, so the gaps are several times the noise floor. Caching the ball quadrature removed the entire BallReg surcharge: it cost +12.0% of a step before and +5.7% after, which is within 1 ms of LocalSIGReg. Locality itself costs about +5% either way.

**Memory: no measurable cost.** All six runs report 22.3 GiB. The $VBSQ$ bound predicts 8.5 MiB for one knot-expanded tensor at this batch size, which is far below what `max_memory_reserved` can resolve next to ViT activations. This confirms the earlier warning in both directions: local splitting neither multiplies training memory by $K$ nor divides it.

**Time: the gap is not arithmetic.** At $B=128$ the regularizer performs roughly 4.3M operations over an 8.5 MiB $[V,B,S,Q]$ tensor, which is tens of microseconds on this device. The measured gaps of 13 ms and, before the fix, 32 ms are therefore fixed per-cluster overhead rather than work. The isolated measurements below identify which part of that overhead is which.

The practical reading is that locality costs about 5% of a training step and no measurable memory at this operating point.

Caveats:

- $B=128$ rather than the configured 256, which does not fit in 40 GiB without activation checkpointing. Checkpointing would change both columns.
- Cases ran in a fixed order, but thermal drift is ruled out: before the fixes the three timings rose monotonically in run order, and after the fixes the third run dropped to match the second while the order was unchanged.
- The probe replays a single batch, so cluster occupancy after many steps does not represent training. Coverage must come from the isolated measurement.
- Peak memory is `max_memory_reserved`, a quantized allocator pool, not `max_memory_allocated`.
- One measurement per method per step count; no seed repeats at this level.

## Measured results: regularizer in isolation

Same A100, `scripts/benchmarks/loss_bench.py`: a synthetic $(V,N,D)$ tensor, forward and backward only, no encoder and no dataloader. Reference point $V=4$, $B=256$, $D=16$, $K=8$, $I=10$, $S=256$, $Q=17$, FP32, three seeds by 100 steps each, `fast_pytorch_kmeans` unless stated. Peak is `max_memory_allocated` above the input tensor and its gradient. Spread between seeds is 0.01-0.4 ms except where noted.

### The full-step gap is the regularizer and nothing else

At $B=128$, matching the full-step configuration:

| Quantity | Isolated | Full step |
|---|---|---|
| `local_sigreg` minus `sigreg` | 14.3 ms | 13 ms |
| `local_ballreg` minus `sigreg` | 31.2 ms | 32 ms |

Two independent instruments agree to 1-2 ms, so the step-level slowdown is fully accounted for.

### Time is linear in $K$ and flat in $B$

Time in ms, $B=256$:

| $K$ | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| `sigreg` | 1.34 | 1.26 | 1.28 | 1.27 | 1.28 | 1.30 |
| `local_sigreg` | 3.11 | 8.50 | 11.59 | 16.77 | 27.02 | 46.82 |
| `local_ballreg` | 5.04 | 12.83 | 19.69 | 33.25 | 60.04 | 109.31 |

Time in ms, $K=8$:

| $B$ | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| `sigreg` | 1.27 | 1.29 | 1.28 | 1.30 | 1.61 |
| `local_sigreg` | 9.51 | 14.08 | 15.55 | 17.29 | 17.58 |
| `local_ballreg` | 20.25 | 30.55 | 32.46 | 34.06 | 34.12 |

Least-squares fits over $K \ge 2$, with maximum residuals of 0.44 ms and 1.44 ms:

$$
t_{\mathrm{local\ sigreg}} \approx 1.27K + 6.4\ \mathrm{ms},
\qquad
t_{\mathrm{local\ ball}} \approx 3.21K + 7.2\ \mathrm{ms}.
$$

The two sweeps cut along different axes and give opposite answers. Increasing $B$ by 16 times moves the global statistic by 0.3 ms and the local one by 3.5 ms; increasing $K$ by 8 times at fixed $B$ adds 30 ms. Since $B$ is the work axis and $K$ is not, the cost of locality is fixed per-cluster overhead and not arithmetic. Which component of that overhead dominates is settled experimentally below.

The global statistic being flat from $B=32$ to $B=256$ also shows that the linear-in-$B$ bound does not describe this range: the kernels are too small to saturate the device, and linearity only begins to appear at $B=512$.

### The BallReg surcharge is the uncached quadrature

| Measurement | ms |
|---|---|
| BallReg surcharge per cluster, from the $K$ slope | 1.95 |
| One `_ball_d_cf` call, from global `ballreg` minus global `sigreg` | 1.97 |

Two independent estimates of the same quantity agree to one percent. The entire extra cost of LocalBallReg is `leggauss(64)` plus its host-to-device transfer, repeated once per accepted cluster, even though the result depends on neither the grid nor the radius and is therefore identical for every cluster.

### Memory falls as $1/K$

Peak regularizer memory in MiB, $B=256$:

| $K$ | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| measured | 107.96 | 66.25 | 44.07 | 33.28 | 27.14 | 26.58 |
| $22.9 + 85.3/K$ | 108.2 | 65.5 | 44.2 | 33.5 | 28.2 | 25.5 |

The model holds to within 1.1 MiB across the range. The constant term is what autograd retains for every cluster until the single backward pass; the $1/K$ term is the transient knot-expanded copies, of which only one cluster's worth is alive at a time instead of the whole batch's. At $K=8$ this is 3.2 times less than the global statistic. This is a real reduction, though not the $K$-fold one warned against above, and it applies to the regularizer alone: full-step memory was unchanged.

At $K=1$ the local peak equals the global peak exactly, 107.955 MiB in both cases, which confirms the $K=1$ equivalence empirically.

Global peak memory scales linearly in $B$ with a clean constant, measured as exactly 6.0 times one FP32 $[V,B,S,Q]$ tensor at every batch size tested:

| $B$ | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| peak, MiB | 12.76 | 25.50 | 50.99 | 104.96 | 203.89 |
| ratio to one $[V,B,S,Q]$ tensor | 6.00 | 6.00 | 6.00 | 6.17 | 6.00 |

### Dimension and iteration budget

Time in ms at $B=256$ and $K=8$:

| $D$ | `sigreg` | `local_sigreg/torch` | `local_sigreg/fast` | `ballreg` | `local_ballreg/torch` | `local_ballreg/fast` |
|---|---|---|---|---|---|---|
| 16 | 1.24 | 18.02 | 16.24 | 1.36 | 19.99 | 18.17 |
| 64 | 1.25 | 18.47 | 15.27 | 1.41 | 20.28 | 16.98 |
| 256 | 1.24 | 18.59 | 13.46 | 1.41 | 20.22 | 15.13 |

Peak memory across the same sweep is 107.95, 107.81 and 107.25 MiB for the global statistic and 33.28, 32.67 and 33.25 MiB for the local one.

A sixteenfold increase in $D$ changes neither the global time nor the memory of either method, so the $O(VBDS)$ term is as invisible here as the $B$ dependence and for the same reason. The local statistic with the external backend instead becomes faster, 16.24 down to 13.46 ms, while the in-repo backend stays flat between 18.0 and 18.6 ms. The convergence test explains both: random points in higher dimension are more nearly orthogonal, k-means settles in fewer iterations, and only a backend that can stop early collects the saving.

| $I$ | `local_sigreg/torch` | `local_sigreg/fast` | `local_ballreg/torch` | `local_ballreg/fast` |
|---|---|---|---|---|
| 1 | 12.05 | 11.88 | 14.01 | 13.69 |
| 5 | 14.87 | 13.88 | 16.81 | 15.69 |
| 10 | 18.44 | 16.46 | 20.23 | 18.48 |
| 20 | 25.58 | 17.31 | 27.19 | 19.38 |

The in-repo backend is linear in $I$ at 0.71 ms per Lloyd iteration and always spends the full budget. The external backend saturates: from $I=10$ to $I=20$ it adds 0.85 ms where the in-repo one adds 7.14. `fast_pytorch_kmeans` therefore converges at roughly ten iterations, which is where the default already sits, so raising the budget is free with that backend and costs 0.71 ms per iteration with the other. Peak memory is flat in $I$, as the bound predicts.

### Backends and coverage

`fast_pytorch_kmeans` is faster at every $K$, but for two different reasons. At $K=1$ it wins by 6.0 ms because it stops after the first iteration once the error hits the tolerance, while the in-repo backend always runs all $I$ iterations. At larger $K$ the advantage is the slope, 1.27 against 1.40 ms per cluster. Either way the backend choice matters far less than the per-cluster overhead common to both.

At $B=32$ with $K=8$ and $m=4$ only 4 to 5 clusters were accepted and 69 to 72 percent of the batch was covered, so that point is not comparable with the rest: part of the work is simply skipped. The two backends also accepted a different number of clusters, 4 against 5, which confirms that with `mode="cosine"` they are not the same algorithm - the in-repo backend averages normalized points, the external one averages raw points.

### Both ratios must be quoted

In isolation the local statistic is 10 to 13 times more expensive than the global one at the reference point. Within a full training step the same change costs +5% and +12%. Both numbers are correct and quoting either alone is misleading.

## What the two fixes changed

Two changes were made to `src/losses.py` and measured with the same sweeps. Both preserve the loss bitwise: old and new implementations were compared under fixed seeds across 24 configurations, including $K=1$, $K>N$, the all-clusters-rejected fallback, $m=1$, both backends, weighted and unweighted, and a scaled integration grid.

**Cache the ball quadrature.** `_ball_quadrature` memoizes the Gauss-Legendre nodes and density weights on `(d, n_quad, device)`. The key deliberately excludes the grid and the radius, which the nodes do not depend on, so the cache is also correct when the grid is scaled.

**Replace $K$ host synchronizations with one.** A single `torch.bincount(labels, minlength=K).tolist()` before the cluster loop supplies every cluster size, and the boolean mask is now built only for accepted clusters.

| Quantity | before | after |
|---|---|---|
| `ballreg` minus `sigreg`, one quadrature call | 1.97 ms | 0.12 ms |
| `local_ballreg` slope | 3.21 ms per cluster | 1.41 ms per cluster |
| `local_ballreg` at $K=8$ | 33.25 ms | 18.19 ms |
| `local_ballreg` at $K=32$ | 109.31 ms | 51.59 ms |
| `local_sigreg` slope | 1.27 ms per cluster | 1.19 ms per cluster |
| `local_sigreg` at $K=8$ | 16.77 ms | 16.63 ms |
| `local_sigreg` at $K=32$ | 46.82 ms | 44.41 ms |

Peak memory was unchanged to the last digit at every $K$, confirming that neither change touches the memory profile.

The batch sweep was repeated after the fixes, and the parity holds across the whole range rather than only at the reference point. Time in ms with the external backend:

| $B$ | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| `local_sigreg` before | 9.51 | 14.08 | 15.55 | 17.29 | 17.58 |
| `local_sigreg` after | 8.89 | 13.14 | 14.57 | 16.25 | 16.24 |
| `local_ballreg` before | 20.25 | 30.55 | 32.46 | 34.06 | 34.12 |
| `local_ballreg` after | 9.97 | 14.86 | 16.39 | 18.02 | 18.12 |

LocalBallReg is roughly halved at every batch size, 47 to 51 percent, and afterwards sits within 1.8 to 2.0 ms of LocalSIGReg everywhere. The global quadrature surcharge is 0.11 to 0.13 ms across the sweep, against 1.97 ms before.

**The caching result was as predicted.** It removes 94% of the quadrature cost and brings LocalBallReg to parity with LocalSIGReg, 1.41 against 1.19 ms per cluster in isolation and 0.278 against 0.277 s per full step.

**The synchronization result was not.** Removing $K-1$ of the $K$ host synchronizations changed `local_sigreg` by under 1% at $K=8$ and about 5% at $K=32$, comparable to the seed-to-seed spread. The hypothesis that the $K$ `.item()` calls dominate the per-cluster cost is therefore rejected. A plausible reason is that `fast_pytorch_kmeans` already drains the pipeline $I$ times through its own convergence test immediately before the loop, so the queue each `.item()` waits on is short.

What remains, roughly 1.2 ms per cluster, is the launch and dispatch cost of the operations themselves: the inner statistic is about 15 tensor operations plus a comparable backward pass, so a cluster issues on the order of 40 kernels whose individual work is measured in microseconds. Removing it requires processing clusters together rather than one at a time - padding to equal size and evaluating all $K$ in one batched set of kernels, or capturing the loop with CUDA graphs or `torch.compile` - which is a redesign of the loop rather than a local fix, and is out of scope here.

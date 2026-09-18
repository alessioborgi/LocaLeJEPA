# Scheduling the local cluster count k

> Written against the pre-refactor code (`loss_type` / `local_sigreg_*` / `local_ballreg_*` parameter names); the runs and W&B groups it reports are unchanged.

ViT-S/8 on Imagenette, 30 epochs, batch 256, $D=16$, $\lambda=0.1$, lr 1e-3, 3-epoch warmup, grad clip 1.0, `local_ballreg`, `fast_pytorch_kmeans`. Ramps double k every 6 epochs and hold at the endpoint from epoch 24. Metric is the online linear probe. Runs are in the W&B group `k-schedule`.

## Results

| arm | k | probe | seeds |
|---|---|---|---|
| `k1` | 1 | 0.434 | 1 |
| `k6` | 6 | **0.518** | 2, spread ±0.002 |
| `k16` | 16 | **0.515** | 1 |
| `k1to16` | 1→2→4→8→16 | 0.467 | 2, spread ±0.006 |
| `k16to1` | 16→8→4→2→1 | 0.394 | 2, spread ±0.026 |

**Locality helps, and by a lot.** Fixed $k=6$ beats fixed $k=1$, which is the global ball statistic, by 8.5 points. The seed spread at $k=6$ is ±0.002, so this is far outside noise.

**Which k, between 6 and 16, does not matter.** 0.518 against 0.515.

**The curriculum does not help.** The forward ramp reaches 0.467, below both fixed arms whose range it spans. Growing k during training is worse than picking a k and staying there.

**Order does matter, but not enough to rescue the ramp.** Forward beats reversed by 0.073, about three times the larger seed spread, so the direction is a real effect. Both still lose to fixed k.

**Each ramp loses to the fixed arm at its own endpoint**: forward 0.467 against `k16` 0.515, reversed 0.394 against `k1` 0.434. Time spent away from the endpoint is a net cost, not an investment.

## Within-run evidence

Probe by epoch, seed 42:

| epoch | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| `k1` probe | 0.213 | 0.261 | 0.263 | 0.321 | 0.339 | 0.364 | 0.354 | 0.396 | 0.341 | 0.339 |
| `k6` probe | 0.179 | 0.361 | 0.355 | 0.384 | 0.333 | 0.447 | 0.418 | 0.456 | 0.487 | 0.492 |
| `k16` probe | 0.209 | 0.312 | 0.305 | 0.424 | 0.426 | 0.457 | 0.420 | 0.478 | 0.470 | 0.527 |
| `k1to16` k | 1 | 1 | 2 | 2 | 4 | 4 | 8 | 8 | 16 | 16 |
| `k1to16` probe | 0.229 | 0.271 | 0.338 | 0.329 | 0.350 | 0.286 | 0.377 | 0.432 | 0.417 | 0.456 |
| `k16to1` k | 16 | 16 | 8 | 8 | 4 | 4 | 2 | 2 | 1 | 1 |
| `k16to1` probe | 0.188 | 0.287 | 0.274 | 0.323 | 0.408 | 0.385 | 0.391 | **0.442** | 0.376 | 0.375 |

The three fixed arms hold their k for the whole run, so only the two ramps carry a schedule row.

The reversed ramp climbs to 0.442 at epoch 21 while $k=2$, then falls to 0.375 once k reaches 1 and stays there. The damage from low k is visible inside a single run, not only across arms.

## Diagnostics

| arm | reg loss | inv loss | mean $\Sigma_{ii}$ | mean $\Sigma_{ij}^2$ | mean $\lVert z \rVert$ |
|---|---|---|---|---|---|
| `k1` | 1.137 | 0.334 | 0.959 | 0.0042 | 4.19 |
| `k6` | 1.058 | 0.256 | **0.999** | 0.0178 | **4.13** |
| `k16` | 1.279 | 0.203 | 0.820 | 0.0213 | 3.74 |
| `k1to16` | 1.357 | 0.208 | 0.780 | 0.0160 | 3.61 |
| `k16to1` | 1.221 | 0.306 | 0.965 | 0.0063 | 4.35 |

Targets are $\Sigma_{ii} = 1$ and $\lVert z \rVert = R\,D/(D+1) = 3.99$. Only $k=6$ hits both.

At $k=16$ the covariance stalls at 0.82. That is consistent with the appendix below, where the ball statistic barely separates a real ball from a Gaussian once a cluster holds only $D$ samples, but it is not evidence for it: the appendix measures a within-cluster quantity, while `cov_diag_mean` is logged over the whole batch, and the two differ by the spread of the cluster centroids. Separating the readings needs within-cluster covariance logged directly, which `_within_cluster_metrics` on the `ballreg-radius` branch already computes but this branch does not carry.

The probe does not seem to care either way: `k16` scores as well as `k6` while whitening visibly worse.

## Caveats

- 30 epochs. Nothing converged: every arm except `k1` was still climbing at the end.
- The forward ramp trains at its endpoint for only 6 of 30 epochs. A longer schedule would spend most of its budget there, so **this run cannot rule out a curriculum that pays off over a longer horizon**; it shows only that the curriculum does not pay within 30 epochs.
- `k1` and `k16` have one seed each.
- The seed spread grows as k falls, ±0.002 at $k=6$ against ±0.026 for the reversed ramp, so the low-k regime is also the noisy one.

## Practical reading

Fix k somewhere between 6 and 16 and leave it. The schedule adds a knob, costs accuracy at this horizon, and the only motivation left for it is the long-run one the caveats describe.

## Appendix: how much signal a cluster of n points carries

The regularizer is a statistical test evaluated on the $n = B/k$ samples of one cluster. Splitting the batch into more clusters buys locality at the cost of sample size, and this appendix measures the exchange rate.

### Method

At $D=16$ and $V=4$ views, two populations are generated:

- **ball** — points genuinely uniform inside the ball of radius $R=\sqrt{D+2}$, drawn as a uniform direction times $U^{1/D}R$. This is what the target asks for, so the loss should be at its floor.
- **gauss** — points from an isotropic Gaussian, which has **the same covariance** as that ball. Only the shape differs, which is exactly what the ball target is meant to detect and what SIGReg's Gaussian target cannot.

One trial draws a fresh sample of each and evaluates `ballreg_loss(..., center=True)` on both, with the sketch directions redrawn per trial and shared between the two evaluations. 200 trials per row. The reported mean and standard deviation are taken across those 200 loss values.

The standard deviation is deliberately the **spread of a single measurement**, not the standard error of the mean. The question is whether one batch can tell the right shape from the wrong one, not whether 200 trials suffice to prove the two means differ. Dividing by $\sqrt{200}$ would shrink it about fourteenfold and answer a different question.

Separation is the standard two-sample effect size at equal group sizes:

$$
\text{separation}=\frac{\bar L_{\text{gauss}}-\bar L_{\text{ball}}}{\sqrt{(s_{\text{ball}}^{2}+s_{\text{gauss}}^{2})/2}}
$$

Sharing the sketch seed between the two evaluations of a trial turned out not to correlate them, so the paired version of this figure is just the above divided by $\sqrt 2$ and carries no extra information.

### Cluster size, at $D=16$

| $n$ | $n/D$ | ball | $s_{\text{ball}}$ | gauss | $s_{\text{gauss}}$ | gap | pooled $s$ | separation |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.2 | 0.533 | 0.020 | 0.551 | 0.026 | 0.019 | 0.023 | 0.81 |
| 8 | 0.5 | 0.543 | 0.022 | 0.570 | 0.027 | 0.027 | 0.024 | 1.10 |
| 16 | 1.0 | 0.551 | 0.022 | 0.580 | 0.025 | 0.029 | 0.024 | 1.23 |
| 32 | 2.0 | 0.551 | 0.024 | 0.589 | 0.026 | 0.038 | 0.025 | 1.52 |
| 64 | 4.0 | 0.553 | 0.023 | 0.605 | 0.028 | 0.052 | 0.026 | 2.04 |
| 128 | 8.0 | 0.554 | 0.022 | 0.629 | 0.034 | 0.075 | 0.028 | 2.64 |
| 256 | 16.0 | 0.555 | 0.023 | 0.678 | 0.043 | 0.123 | 0.035 | 3.54 |

The **ball** column is the false-alarm level: what the loss reads when the data already matches the target. It is not zero, because a finite sample never reproduces the target characteristic function exactly, and it barely moves with $n$ because the loss multiplies the statistic by the sample count. All of the movement is in the **gauss** column: the wrong shape becomes progressively easier to see as the cluster grows.

The gap grows roughly as $\sqrt n$ while the spreads stay near 0.025, so the two mean columns look almost identical in absolute terms while their separation changes fivefold. At $n=16$ the gap is about one standard deviation, so on a single batch the wrong shape can easily score lower than the right one and the gradient step is driven mostly by which way that batch happened to fluctuate.

Two consequences for choosing $k$:

- Cluster size, not the number of semantic classes, is what bounds $k$. At $B=256$ and $D=16$, $k=16$ already puts a cluster at $n=D$.
- Raising $k$ needs $B$ raised in proportion just to hold the signal constant. Reaching $k=100$ at this $D$ would need $B\approx1600$.

### Projection dimension, at $n=256$

| $D$ | ball | $s_{\text{ball}}$ | gauss | $s_{\text{gauss}}$ | gap | pooled $s$ | separation |
|---|---|---|---|---|---|---|---|
| 4 | 0.532 | 0.075 | 1.596 | 0.227 | 1.064 | 0.169 | 6.29 |
| 16 | 0.555 | 0.023 | 0.678 | 0.043 | 0.123 | 0.035 | 3.54 |
| 64 | 0.564 | 0.017 | 0.580 | 0.019 | 0.016 | 0.018 | 0.91 |
| 256 | 0.569 | 0.017 | 0.572 | 0.016 | 0.003 | 0.017 | 0.16 |

Both SIGReg and BallReg test one-dimensional projections, and by concentration of measure those projections converge to a Gaussian as $D$ grows. The ball target therefore stops being distinguishable from the Gaussian one: by $D=64$ the gap is 0.016 against a spread of 0.018, and at $D=256$ it is gone. The same is visible analytically in `_ball_d_cf`, whose target converges to $e^{-t^2/2}$, which is SIGReg's target. The configured $D=16$ sits near the edge where the distinction still exists.

### Sample size and centering

The two sweeps above are framed around clusters, but nothing in them clusters anything: `ballreg_loss` is called directly on a synthetic sample. What is really being measured is a property of the Epps-Pulley estimator against sample size, and clusters enter only through $n = B/k$. Restating it that way makes a finer grid natural, and makes it obvious to also vary the one other switch the estimator has.

`center` is that switch. `local_ballreg_loss` passes `center_clusters=True`, so every local call subtracts the per-view sample mean. A direct `ballreg_loss` call defaults to `center=False`, so the global baseline does not. Same 200 trials, same two populations.

Centered, which is what the local losses do:

| $n$ | $n/D$ | ball | $s_{\text{ball}}$ | gauss | $s_{\text{gauss}}$ | separation |
|---|---|---|---|---|---|---|
| 1 | 0.06 | 0.427 | 0.000 | 0.427 | 0.000 | undefined |
| 2 | 0.12 | 0.505 | 0.023 | 0.515 | 0.028 | 0.37 |
| 3 | 0.19 | 0.525 | 0.020 | 0.539 | 0.029 | 0.56 |
| 4 | 0.25 | 0.533 | 0.020 | 0.551 | 0.026 | 0.81 |
| 6 | 0.38 | 0.541 | 0.021 | 0.562 | 0.029 | 0.83 |
| 8 | 0.50 | 0.543 | 0.022 | 0.570 | 0.027 | 1.10 |
| 12 | 0.75 | 0.548 | 0.021 | 0.577 | 0.027 | 1.20 |
| 16 | 1.00 | 0.551 | 0.022 | 0.580 | 0.025 | 1.23 |
| 24 | 1.50 | 0.551 | 0.023 | 0.584 | 0.025 | 1.36 |
| 32 | 2.00 | 0.551 | 0.024 | 0.589 | 0.026 | 1.52 |
| 48 | 3.00 | 0.552 | 0.025 | 0.598 | 0.027 | 1.75 |
| 64 | 4.00 | 0.553 | 0.023 | 0.605 | 0.028 | 2.04 |
| 96 | 6.00 | 0.552 | 0.021 | 0.620 | 0.034 | 2.40 |
| 128 | 8.00 | 0.554 | 0.022 | 0.629 | 0.034 | 2.64 |
| 192 | 12.00 | 0.553 | 0.024 | 0.654 | 0.038 | 3.17 |
| 256 | 16.00 | 0.555 | 0.023 | 0.678 | 0.043 | 3.54 |

Uncentered, which is what a direct `ballreg_loss` call does:

| $n$ | $n/D$ | ball | $s_{\text{ball}}$ | gauss | $s_{\text{gauss}}$ | separation |
|---|---|---|---|---|---|---|
| 1 | 0.06 | 1.066 | 0.034 | 1.053 | 0.081 | −0.21 |
| 2 | 0.12 | 1.067 | 0.071 | 1.045 | 0.085 | −0.29 |
| 3 | 0.19 | 1.060 | 0.077 | 1.054 | 0.098 | −0.07 |
| 4 | 0.25 | 1.071 | 0.083 | 1.042 | 0.092 | −0.33 |
| 6 | 0.38 | 1.066 | 0.099 | 1.052 | 0.099 | −0.13 |
| 8 | 0.50 | 1.071 | 0.098 | 1.060 | 0.094 | −0.11 |
| 12 | 0.75 | 1.069 | 0.097 | 1.054 | 0.091 | −0.16 |
| 16 | 1.00 | 1.076 | 0.102 | 1.061 | 0.095 | −0.15 |
| 24 | 1.50 | 1.071 | 0.097 | 1.059 | 0.093 | −0.13 |
| 32 | 2.00 | 1.084 | 0.104 | 1.065 | 0.092 | −0.20 |
| 48 | 3.00 | 1.085 | 0.110 | 1.075 | 0.096 | −0.09 |
| 64 | 4.00 | 1.072 | 0.106 | 1.080 | 0.090 | 0.08 |
| 96 | 6.00 | 1.073 | 0.101 | 1.097 | 0.097 | 0.25 |
| 128 | 8.00 | 1.078 | 0.096 | 1.098 | 0.094 | 0.21 |
| 192 | 12.00 | 1.076 | 0.100 | 1.127 | 0.097 | 0.51 |
| 256 | 16.00 | 1.081 | 0.097 | 1.146 | 0.101 | 0.66 |

**Without centering the statistic does not separate the two shapes at any tested $n$.** Separation is at or below zero up to $n=48$ and reaches only 0.66 at $n=256$, against 3.54 centered.

The mechanism is visible in the spreads: uncentered noise is about 0.10, centered about 0.023. The sample mean fluctuates as $n^{-1/2}$, the imaginary part of the empirical characteristic function picks that up, squaring gives $O(1/n)$, and the loss then multiplies by $n$. What is left is an $O(1)$ contribution from location noise that does not shrink with sample size and buries the shape signal. Centering removes it exactly.

Three consequences:

- Centering is a condition for the ball statistic to work at all, not a stylistic default. This is the quantitative version of the earlier observation that `local_ballreg_loss` centers while `local_sigreg_loss` has no such option.
- Global BallReg as configured, with `center=False`, is a weak baseline: it mostly measures location noise. Comparisons of the local method against it confound locality with centering.
- At $n=1$ the centered residual is identically zero, so the loss is a constant 0.427 regardless of the data, with zero spread. `min_cluster_size=4` sits where separation is 0.81, that is, where the estimator is still barely able to tell the shapes apart.

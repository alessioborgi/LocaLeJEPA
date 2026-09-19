# Local BallReg: the radius sets the collapse threshold

> Written against the pre-refactor code (`loss_type` / `local_sigreg_*` / `local_ballreg_*` parameter names); the runs and W&B groups it reports are unchanged.

ViT-S/8 on Imagenette, $d = 16$, $k = 8$ local clusters, $\lambda = 0.02$, centered residuals,
`loss_type: local_ballreg`. Metrics are the last logged epoch, read from W&B on 2026-09-08.

| $R$ | $P_R$ | Run | Epoch | Val probe | $\overline{\Sigma_{ii}}$ | $\sqrt{\overline{\Sigma_{ij}^2}}$ | $\mathbb{E}\lVert x\rVert$ | $\ell_{\mathrm{inv}}$ | $\ell_{\mathrm{reg}}$ | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $\sqrt{18}$ | off | [a3lfnoqr](https://wandb.ai/snopoffp-snopoffp-personal/BundLeJEPA/runs/a3lfnoqr) | 106 | **65.1%** | 0.602 | 0.156 | 3.29 | 0.0909 | 4.01 | 0.169 |
| $\sqrt{18}$ | on | [1mro3aps](https://wandb.ai/snopoffp-snopoffp-personal/BundLeJEPA/runs/1mro3aps) | 100 | **60.9%** | 0.600 | 0.152 | 3.21 | 0.0923 | 4.00 | 0.170 |
| $1$ | off | [o7fu3ism](https://wandb.ai/snopoffp-snopoffp-personal/BundLeJEPA/runs/o7fu3ism) | 100 | 10.4% | $1.0\cdot10^{-5}$ | $1.1\cdot10^{-5}$ | 0.242 | $6.5\cdot10^{-6}$ | 0.733 | 0.0147 |
| $1$ | on | [xsol1vvb](https://wandb.ai/snopoffp-snopoffp-personal/BundLeJEPA/runs/xsol1vvb) | 100 | 9.9% | $1.3\cdot10^{-4}$ | $1.3\cdot10^{-4}$ | 0.596 | $8.5\cdot10^{-5}$ | 0.677 | 0.0136 |

Both $R = 1$ runs collapse, both $R = \sqrt{d+2}$ runs learn. Ball projection changes nothing
qualitatively. Note the total loss: the collapsed runs sit an order of magnitude *lower*, so the
objective value is not comparable across radii.

## Why the radius decides it

For $X$ uniform on a centered ball of radius $R$ in $\mathbb{R}^d$,

$$\operatorname{Cov}[X] = \frac{R^2}{d+2}\,I, \qquad \mathbb{E}\lVert X\rVert = \frac{d}{d+1}R .$$

So $R = \sqrt{d+2} = \sqrt{18}$ is exactly the whitened target, $\operatorname{Cov} = I$, while $R = 1$
asks for variance $1/18 \approx 0.056$ per coordinate.

| $R$ | Target $\Sigma_{ii}$ | Achieved | Target $\mathbb{E}\lVert x\rVert$ | Achieved |
|---|---:|---:|---:|---:|
| $\sqrt{18}$ | 1 | 0.60 | 3.99 | 3.29 |
| $1$ | 0.056 | $1.0\cdot10^{-5}$ | 0.94 | 0.242 |

At $R = \sqrt{18}$ the representation lands within a factor $1.7$ of its target. At $R = 1$ it sits
$5\cdot10^{3}$ *below* its own target, so this is genuine collapse, not a correctly matched small ball.
Achieved values are the global projector covariance, an upper bound on the within-cluster quantity the
target actually constrains; the $R = 1$ gap is far too large for that distinction to matter.

## The integration grid is the confound

`ballreg_loss` is an Epps-Pulley statistic: it compares one-dimensional random sketches of the batch
against the target characteristic function on a **fixed** grid $t \in [0, 3]$, weighted by trapezoid
rule times a Gaussian window,

$$\ell_{\mathrm{reg}} \approx \int_0^3 w(t)\,\bigl|\hat\varphi_{\text{data}}(t) - \varphi_R(t)\bigr|^2 dt,
\qquad \varphi_R(t) = 1 - \frac{R^2t^2}{2(d+2)} + O(t^4).$$

Collapse means $\hat\varphi \equiv 1$, so the penalty it incurs is however far $\varphi_R$ departs from
$1$ *inside the grid*. But $t$ has units of inverse length, and $\varphi_R$ turns over only at
$t \sim \sqrt{d+2}/R = 1/\sigma_{\text{target}}$:

| $R$ | $\sigma_{\text{target}}$ | $\varphi_R(3)$ | Collapse penalty on $[0,3]$ |
|---|---:|---:|---:|
| $\sqrt{18}$ | 1 | $-0.001$ | 0.427 |
| $1$ | 0.236 | 0.776 | 0.00466 |

At $R = \sqrt{18}$ the target CF decays to zero exactly across the grid, so the test sees its whole
shape. At $R = 1$ the target is still nearly flat at the right edge, i.e. barely distinguishable from
collapse, and the anti-collapse signal is **92x weaker** (computed from `_ball_d_cf`, $d = 16$,
`knots = 17`; the low-frequency $R^4$ scaling would say $18^2 = 324$, the integrated statistic on this
grid says 92).

So the radius changes two things at once: the geometric target, and the statistical power of the test
enforcing it. At $R = 1$ the regularizer is simply too weak to hold variance against
$\ell_{\mathrm{inv}}$, whose own minimum is the collapsed solution. Which of the two causes the
collapse is not yet separated.

## What projection does

The `-norm` runs apply, per centered residual $r$ inside the regularizer only,

$$P_R(r) = r\,\frac{R}{\max(R, \lVert r\rVert)} .$$

This clips from above and does nothing else — no unit-norm rescaling, no whitening, no lower bound on
spread. A collapsed residual is deep inside either ball, so $P_R$ leaves it untouched, which is why
projection cannot rescue $R = 1$. Empirically the projected and unprojected arms agree to within a
few percent on every diagnostic at both radii.

## Does whitening actually work?

Partly. At $R = \sqrt{18}$ variance is preserved at the right order of magnitude, but the covariance
is not identity: off-diagonal RMS $0.15$ against diagonal $0.60$ is a mean correlation of $\approx
0.25$.

It may also be unfinished rather than wrong. The diagonal climbs monotonically for the whole run and
is still climbing at the end -- 0.138, 0.202, 0.376, 0.500, 0.600 at epochs 0, 5, 20, 50, 99 -- and
a3lfnoqr was configured for 300 epochs before it crashed at 106. The gap to $\Sigma_{ii} = 1$ may be a
schedule artifact. Collapse at $R = 1$ needs no such patience: the diagonal is below its own target by
epoch 0 and 60x below by epoch 2, so 20 epochs suffice to reproduce that arm.

The logged diagnostic cannot settle the question either, and that is the main gap. It pools raw
projector outputs over all samples and views, *before* the per-cluster centering the regularizer
uses, so

$$\Sigma_{\text{logged}} = \underbrace{\Sigma_{\text{within}}}_{\text{what BallReg targets}} + \Sigma_{\text{between centroids}} .$$

With $k = 8$ clusters the centroid term alone produces off-diagonal mass. So the measured
correlation is an upper bound on within-cluster correlation, and the supported claim is
**no projection-space collapse**, not **whitening achieved**.

## Caveats and next steps

The four arms are not a matched factorial: a3lfnoqr was configured for 300 epochs and crashed at 106
(65.7% at epoch 99, so the endpoint gap to the projected run is not a projection effect), xsol1vvb
records no seed, and the two older runs do not log `local_ballreg_project_to_ball` at all — their
setting comes from the configs, not the run record.

The radius question itself is settled and needs no further runs. What is unsettled is *why* it is
settled, which takes one short experiment:

1. Rerun $R = 1$ with the grid rescaled by the radius, $t \mapsto t\sqrt{d+2}/R$, which makes the
   statistic dimensionless and equalizes its power across radii — `local_ballreg_scale_grid: true`,
   config `vit_imagenette_local_ballreg_r1_scaled_grid`. If $R = 1$ still collapses, the cause is the
   geometric target; if it stops, the earlier collapse was the grid. Collapse shows up by epoch 2, so
   30 epochs answer it. Sweeping $\lambda$ probes the same axis more crudely.

   The option is a no-op at $R = \sqrt{d+2}$ by construction. Measured on collapsed input, the
   statistic scores 109.2 at $R = \sqrt{18}$, 1.19 at $R = 1$, and 109.2 at $R = 1$ with the scaled
   grid — the 92x gap predicted above, closed exactly.

A matched $R = 1$ baseline of the same length is worth running alongside it, because the diagnostics
are new since these four runs: `train/within_cov_diag_mean`, `train/within_cov_offdiag_sq_mean`
(pooled over clusters, on the centered residuals BallReg sees) and `train/ball_clipped_frac` (share
with $\lVert r\rVert > R$, logged in both arms, so it also says how much $P_R$ *would* clip when it is
off). Claiming whitening rather than absence of collapse would need the $\sqrt{18}$ arm run to
convergence, which is a separate question from the radius comparison.

Use $R = \sqrt{d+2}$ as the default meanwhile.

Code: [`ballreg_loss`, `_project_onto_ball`](src/losses.py), [loss composition](src/models/lejepa_module.py), [covariance diagnostics](src/models/jepa_module.py), [projected config](configs/experiment/lejepa/vit_imagenette_local_ballreg_sqrt_projected.yaml).

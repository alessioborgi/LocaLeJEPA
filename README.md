# LocaLeJEPA: Maximum Entropy Representations under Geometric Priors

A step-by-step walkthrough with a standalone implementation is in [MINIMAL.md](MINIMAL.md).

## Overview

LocaLeJEPA is a Lean Joint Embedding Predictive Architecture that learns embeddings and drives them toward a **locally uniform** distribution. Within each neighbourhood, the target is a uniform distribution on a Euclidean ball — the maximum-entropy distribution for that compact support. Its regularizer, **LocaLeReg**, forms $K$ neighbourhoods per batch using $K$-means on detached per-image mean projections, centres the projections within each neighbourhood, and matches them to the uniform-ball target with a sketched Epps–Pulley test.

For $V$ views of $N$ images with projections $z_i^{(v)}\in\mathbb R^d$ and $\bar z_i=\frac1V\sum_v z_i^{(v)}$,

$$\mathcal L=(1-\lambda)\,\mathcal L_{\rm pred}+\lambda\,\mathcal L_{\rm LocaLeReg},~\text{with}~~\mathcal L_{\rm pred}=\frac1{NVd}\sum_{i,v}\big\|z_i^{(v)}-\bar z_i\big\|^2,~~\text{and}~~\mathcal L_{\rm LocaLeReg}=\frac1{|Q|\,V}\sum_{q\in Q}\sum_{v=1}^V D_{q,v},$$

where $D_{q,v}$ is the sketched Epps–Pulley statistic between the centred projections of neighbourhood $q$ and $\mathrm{Unif}(\mathbb B^d_R)$. The target uniform distribution is on a $d$-dimensional ball with radius $R=\sqrt{d+2}$ so that the target distribution has identity covariance.

![LocaLeJEPA pipeline](assets/LocaLeJEPA_Main_Figure.svg)

## Results

ImageNet-1K, 100 epochs of pretraining, frozen encoder. **Bold**: best within an encoder family.

<table>
<thead>
<tr><th rowspan="2">Method</th><th colspan="2">ImageNet-1K linear (%)</th><th>k-NN (%)</th><th>Transfer (%)</th></tr>
<tr><th>top-1 ↑</th><th>top-5 ↑</th><th>k=20 ↑</th><th>7-dataset avg ↑</th></tr>
</thead>
<tbody>
<tr><th colspan="5" align="left"><i>Plain ViT-S/8 (Euclidean, 21.7M)</i></th></tr>
<tr><td>LeJEPA</td><td align="right">40.7 ± 0.23</td><td align="right">66.9 ± 0.21</td><td align="right">24.3 ± 0.19</td><td align="right">56.7 ± 0.29</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=4)</td><td align="right"><b>49.6 ± 0.23</b></td><td align="right"><b>75.0 ± 0.20</b></td><td align="right"><b>34.1 ± 0.21</b></td><td align="right"><b>63.0 ± 0.32</b></td></tr>
<tr><th colspan="5" align="left"><i>Hyperbolic ViT-S/8 (Poincaré ball, 21.7M)</i></th></tr>
<tr><td>LeJEPA</td><td align="right">41.1 ± 0.22</td><td align="right">67.1 ± 0.21</td><td align="right">25.1 ± 0.19</td><td align="right">57.6 ± 0.60</td></tr>
<tr><td>Rectified LpJEPA</td><td align="right">29.2 ± 0.21</td><td align="right">53.4 ± 0.23</td><td align="right">15.1 ± 0.16</td><td align="right">47.5 ± 0.28</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=4)</td><td align="right"><b>50.5 ± 0.23</b></td><td align="right"><b>75.8 ± 0.19</b></td><td align="right"><b>34.9 ± 0.21</b></td><td align="right"><b>63.9 ± 0.48</b></td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=1→10)</td><td align="right">47.4 ± 0.22</td><td align="right">73.1 ± 0.20</td><td align="right">29.9 ± 0.20</td><td align="right">61.3 ± 0.29</td></tr>
<tr><td>LocaLeJEPA-SIGReg (λ=0.1, K=4)</td><td align="right">40.7 ± 0.22</td><td align="right">66.5 ± 0.21</td><td align="right">26.0 ± 0.20</td><td align="right">57.2 ± 0.27</td></tr>
<tr><th colspan="5" align="left"><i>D<sub>4</sub>-equivariant ResNet-50 (24.4M)</i></th></tr>
<tr><td>LeJEPA</td><td align="right">35.7 ± 0.22</td><td align="right">63.0 ± 0.22</td><td align="right">23.3 ± 0.19</td><td align="right">48.2 ± 0.32</td></tr>
<tr><td>Rectified LpJEPA</td><td align="right">27.9 ± 0.20</td><td align="right">54.1 ± 0.22</td><td align="right">19.7 ± 0.18</td><td align="right">42.6 ± 0.44</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=4)</td><td align="right"><b>42.3 ± 0.22</b></td><td align="right"><b>69.8 ± 0.21</b></td><td align="right"><b>33.0 ± 0.21</b></td><td align="right">54.0 ± 0.31</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=1→10)</td><td align="right">42.2 ± 0.22</td><td align="right">69.5 ± 0.21</td><td align="right">32.5 ± 0.21</td><td align="right"><b>54.3 ± 0.40</b></td></tr>
<tr><td>LocaLeJEPA-SIGReg (λ=0.1, K=4)</td><td align="right">35.9 ± 0.21</td><td align="right">62.7 ± 0.22</td><td align="right">27.0 ± 0.20</td><td align="right">50.3 ± 0.29</td></tr>
</tbody>
</table>

Linear transfer from frozen ImageNet-1K features, top-1 (%), mean over five probe seeds.

<table>
<thead>
<tr><th>Method</th><th>DTD</th><th>Aircraft</th><th>CIFAR-10</th><th>CIFAR-100</th><th>Flowers</th><th>Food</th><th>Pets</th><th>Avg</th></tr>
</thead>
<tbody>
<tr><th colspan="9" align="left"><i>Plain ViT-S/8 (Euclidean, 21.7M)</i></th></tr>
<tr><td>LeJEPA</td><td align="right">56.4 ± 1.5</td><td align="right">26.1 ± 0.8</td><td align="right">84.4 ± 0.4</td><td align="right">61.2 ± 0.6</td><td align="right">68.4 ± 0.6</td><td align="right">48.5 ± 0.3</td><td align="right">52.0 ± 0.8</td><td align="right">56.7 ± 0.3</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=4)</td><td align="right"><b>63.0 ± 1.2</b></td><td align="right"><b>31.1 ± 0.9</b></td><td align="right"><b>86.0 ± 0.4</b></td><td align="right"><b>63.8 ± 0.5</b></td><td align="right"><b>77.4 ± 0.9</b></td><td align="right"><b>55.7 ± 0.3</b></td><td align="right"><b>64.2 ± 1.6</b></td><td align="right"><b>63.0 ± 0.3</b></td></tr>
<tr><th colspan="9" align="left"><i>Hyperbolic ViT-S/8 (Poincaré ball, 21.7M)</i></th></tr>
<tr><td>LeJEPA</td><td align="right">57.4 ± 2.0</td><td align="right">27.3 ± 0.9</td><td align="right">85.3 ± 0.4</td><td align="right">60.3 ± 0.5</td><td align="right">70.9 ± 2.6</td><td align="right">48.6 ± 0.3</td><td align="right">53.4 ± 0.8</td><td align="right">57.6 ± 0.6</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=4)</td><td align="right"><b>62.1 ± 1.2</b></td><td align="right"><b>32.1 ± 0.8</b></td><td align="right"><b>87.5 ± 0.3</b></td><td align="right"><b>65.0 ± 0.5</b></td><td align="right"><b>77.9 ± 3.3</b></td><td align="right"><b>56.4 ± 0.3</b></td><td align="right"><b>65.9 ± 0.9</b></td><td align="right"><b>63.9 ± 0.5</b></td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=1→10)</td><td align="right">59.5 ± 1.4</td><td align="right">28.6 ± 1.0</td><td align="right">86.5 ± 0.3</td><td align="right">64.0 ± 0.6</td><td align="right">74.7 ± 0.6</td><td align="right">54.0 ± 0.3</td><td align="right">62.0 ± 1.0</td><td align="right">61.3 ± 0.3</td></tr>
<tr><td>LocaLeJEPA-SIGReg (λ=0.1, K=4)</td><td align="right">55.7 ± 1.3</td><td align="right">27.0 ± 0.9</td><td align="right">83.1 ± 0.4</td><td align="right">59.0 ± 0.6</td><td align="right">70.5 ± 0.6</td><td align="right">49.0 ± 0.3</td><td align="right">56.5 ± 0.8</td><td align="right">57.2 ± 0.3</td></tr>
<tr><th colspan="9" align="left"><i>D<sub>4</sub>-equivariant ResNet-50 (24.4M)</i></th></tr>
<tr><td>LeJEPA</td><td align="right">51.3 ± 1.4</td><td align="right">16.4 ± 0.7</td><td align="right">83.8 ± 0.4</td><td align="right">55.8 ± 0.5</td><td align="right">52.9 ± 0.7</td><td align="right">34.3 ± 0.3</td><td align="right">42.7 ± 1.1</td><td align="right">48.2 ± 0.3</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=4)</td><td align="right">55.1 ± 1.2</td><td align="right">20.7 ± 1.4</td><td align="right">82.3 ± 0.5</td><td align="right">55.3 ± 0.5</td><td align="right"><b>66.4 ± 0.6</b></td><td align="right"><b>41.8 ± 0.3</b></td><td align="right"><b>56.6 ± 0.8</b></td><td align="right">54.0 ± 0.3</td></tr>
<tr><td>LocaLeJEPA (λ=0.1, K=1→10)</td><td align="right"><b>56.7 ± 1.2</b></td><td align="right"><b>21.7 ± 1.5</b></td><td align="right"><b>84.8 ± 0.4</b></td><td align="right"><b>58.5 ± 0.5</b></td><td align="right">63.5 ± 1.7</td><td align="right">40.5 ± 0.3</td><td align="right">54.6 ± 1.0</td><td align="right"><b>54.3 ± 0.4</b></td></tr>
<tr><td>LocaLeJEPA-SIGReg (λ=0.1, K=4)</td><td align="right">50.0 ± 1.2</td><td align="right">20.4 ± 0.9</td><td align="right">82.8 ± 0.4</td><td align="right">54.9 ± 0.5</td><td align="right">59.0 ± 0.7</td><td align="right">32.9 ± 0.3</td><td align="right">51.8 ± 0.9</td><td align="right">50.3 ± 0.3</td></tr>
</tbody>
</table>

## Installation

Use Python 3.12. Install the base dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Alternatively, without uv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

After activating the environment, replace `uv run python ...` in the commands below with `python ...`. The pip resolves versions from `pyproject.toml`; `uv sync` uses the pinned `uv.lock`.

To log runs to [Weights & Biases](https://wandb.ai), copy `.env.example` to `.env`, set `WANDB_ENTITY` and `WANDB_PROJECT` for your account, and log in once with `uv run wandb login` (or `wandb login` in an activated environment). Keep `WANDB_OFFLINE=false` for online logging, or set it to `true` to save runs locally. You can use `logger=csv` to log without W&B. Optional extras:

| Extra | uv | pip | Needed for |
| --- | --- | --- | --- |
| `equivariant` | `uv sync --extra equivariant` | `python -m pip install -e '.[equivariant]'` | $D_4$-equivariant ResNet-50 (`escnn`; needs `gfortran`) |
| `imagenet` | `uv sync --extra imagenet` | `python -m pip install -e '.[imagenet]'` | staging ImageNet-1K from Hugging Face |
| `experiments` | `uv sync --extra experiments` | `python -m pip install -e '.[experiments]'` | analysis and plotting scripts |

## Quick start

LocaLeReg is a function of the projections in `(V, N, D)` layout. Run from the repository root:

```python
import torch
from src.losses import invariance_loss, local_ballreg_loss

lamb = 0.1
proj = torch.randn(4, 256, 16, requires_grad=True)  # (V, N, D)

loss = (1 - lamb) * invariance_loss(proj) + lamb * local_ballreg_loss(proj, num_clusters=4)
loss.backward()
```

All losses are in [src/losses.py](src/losses.py), the Lightning module in [src/models/lejepa_module.py](src/models/lejepa_module.py).

## Training

ViT-S/8 on Imagenette (downloaded on first run), the recipe of [MINIMAL.md](MINIMAL.md):

```bash
uv run python src/train.py experiment=minimal
```

Every value in [configs/](configs/) can be overridden from the command line:

```bash
uv run python src/train.py \
  experiment=lejepa/vit_in1k_localballreg_l01_k4 \
  model.lamb=0.1 \
  model.reg_loss.num_clusters=4 \
  model.reg_loss.loss.num_slices=256 \
  optimizer.lr=1e-3 \
  optimizer.weight_decay=5e-2 \
  model.probe_lr=1e-3 \
  model.warmup_epochs=10 \
  data.batch_size=64 \
  trainer.accumulate_grad_batches=4 \
  trainer.max_epochs=100 \
  logger.wandb.entity=<ENTITY> \
  logger.wandb.project=<PROJECT>  # logger=csv to log locally
```

| Override | Effect |
| --- | --- |
| `model/reg_loss=sigreg model.lamb=0.02` | LeJEPA baseline |
| `model/reg_loss=locale_sigreg` | LocaLeJEPA-SIGReg (Gaussian target on uncentred neighbourhoods) |
| `callbacks=dynamic_num_clusters callbacks.dynamic_num_clusters.end=10 callbacks.dynamic_num_clusters.step_epochs=10` | $K=1\to10$ schedule |
| `trainer=ddp trainer.devices=4` | multi-GPU |
| `ckpt_path=/path/to/last.ckpt` | resume |
| `paths.data_dir=/your/data/root` | dataset location |

## ImageNet-1K

The datamodule expects `data/imagenet/imagenet_full_size/061417/{train,val}/<class>/*.JPEG`. To build it from the gated [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k):

```bash
uv sync --extra imagenet
uv run python scripts/stage_imagenet1k_hf.py --dest data/imagenet
```

Configs for the rows of the tables above (`experiment=lejepa/<config>`):

| Backbone | LeJEPA | LocaLeJEPA, K=4 | LocaLeJEPA, K=1→10 | LocaLeJEPA-SIGReg |
| --- | --- | --- | --- | --- |
| ViT-S/8 | `vit_in1k_sigreg` | `vit_in1k_localballreg_l01_k4` | | |
| Hyperbolic ViT-S/8 | `hvit_in1k_sigreg` | `hvit_in1k_localballreg_l01_k4` | `hvit_in1k_localballreg_l01_dynk` | `hvit_in1k_localsigreg_l01_k4` |
| $D_4$-equivariant ResNet-50 | `d4rn50_in1k_sigreg` | `d4rn50_in1k_localballreg_l01_k4` | `d4rn50_in1k_localballreg_l01_dynk` | `d4rn50_in1k_localsigreg_l01_k4` |

Rectified LpJEPA: `experiment=rectifiedlpjepa/{hvit,d4rn50}_in1k_rdmreg`. ViT-L/14 multi-crop: `lejepa/vitl14_in1k_{sigreg,localballreg_l01_k4}_paper` with `trainer=ddp trainer.devices=4`.

## Evaluation

Frozen-feature linear probe and k-NN on ImageNet-1K:

```bash
uv run python src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml
```

## License

See [LICENSE](LICENSE) and [NOTICE](src/NOTICE).

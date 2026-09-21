# Campaign manifests

One YAML per wave. `scripts/aicr/campaign.py expand <manifest> --out <grid.json>` turns a manifest
into the runs of a Slurm job array; run names are canonical (block tag + sorted-key aliased
values + seed) and double as the checkpoint directory, which is the whole fix for F35.
`tests/integration/test_campaign_manifests.py` expands every manifest here, composes every
distinct cell, asserts the declared per-GPU micro-batch and the campaign invariants.

| manifest | wave | runs | GPUs/run |
|---|---|---|---|
| `ablation_grid.yaml` | Phase 3a: λ × K × batch factorial + both ramps + scale rows, 3 seeds | 252 | 1 |
| `ablation_batch128.yaml` | the 128×2 batch cell at the declared λ (Phase 4) | 24 | 1 |
| `in1k_baselines.yaml` | Phase 3b: six 128 px baselines | 6 | 4 |
| `in1k_vitl14.yaml` | ViT-L/14 pair at LeJEPA's settings (SIGReg on day 1; local after the declaration) | 2 | 8 |
| `in1k_local.yaml` | Phase 4: three 128 px local arms + the control, at the DECLARED (λ, K) | 4 | 4 |
| `torus.yaml` | Phase 3c: the torus lineup, 800 epochs | 11 (+1 declared) | 1 |
| `calibration_{1,4,8}gpu.yaml` | Phase 2: 200 steps per geometry, `MODE=calibrate` | 2 / 3 / 1 | 1 / 4 / 8 |

Values marked PROVISIONAL are replaced at Phase 4 by the ablation's declaration (decision 14)
and committed before any run that uses them is submitted (gate G6).

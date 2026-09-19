# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate polished Yoda learning-curve plots and build a self-contained
HTML page (PNGs embedded as base64) with the MAE and R^2 data tables.

    python build_lowdata_artifact.py <out_html_path>
"""

import base64
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This file lives in <repo>/visualisations/; read source metrics from the repo's
# logs/ and write rendered figures next to this script under figures/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "logs/eval_angle_probe_lowdata")
OUT = os.path.join(_HERE, "figures")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(DATA, "results_lowdata.json")) as f:
    S = json.load(f)

SIZES = S["sizes"]
NAMES = list(S["backbones"].keys())
NSEEDS = S["n_seeds"]

# Curated, high-separation palette. Global backbones in cool/neutral tones; the
# collapse-fixed local backbones in warm/green tones (winner emphasized).
SERIES = {
    "svit_susreg":          "#7b8794",
    "nvit_sigreg":          "#9aa7d0",
    "nvit_subreg":          "#72b7b2",
    "vit_sigreg":           "#4c78a8",
    "vit_subreg":           "#5b8fb0",
    "localballreg_l0.1_k2": "#e45756",
    "localballreg_l0.1_k4": "#f58518",
    "localballreg_l0.3_k2": "#b279a2",
    "localsigreg_l0.1_k4":  "#54a24b",
}
EMPH = "localballreg_l0.1_k2"


def cell(entry, metric, a):
    m = entry["angles"][f"angle{a}"]
    return m[f"{metric}_mean"], m[f"{metric}_std"]


def make_plot(metric, ylabel, path, ylim=None):
    plt.rcParams.update({
        "font.size": 12, "font.family": "DejaVu Sans",
        "axes.edgecolor": "#c7ccd4", "axes.linewidth": 0.8,
        "axes.labelcolor": "#2a2f37", "text.color": "#2a2f37",
        "xtick.color": "#5c6672", "ytick.color": "#5c6672",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for a in range(2):
        ax = axes[a]
        for n in NAMES:
            eff = [S["backbones"][n]["sizes"][str(ts)]["eff_train"] for ts in SIZES]
            vals = [cell(S["backbones"][n]["sizes"][str(ts)], metric, a) for ts in SIZES]
            m = [v[0] for v in vals]; sd = [v[1] for v in vals]
            lw = 2.4 if n == EMPH else 1.6
            ax.errorbar(eff, m, yerr=sd, marker="o", ms=4.5, lw=lw, capsize=2.5,
                        color=SERIES[n], label=n, alpha=0.95)
        ax.set_xscale("log")
        ax.set_title(f"figurine {a}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("effective # training frames (log scale)")
        if a == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.28, lw=0.7)
        ax.grid(True, which="minor", alpha=0.12, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
    axes[1].legend(fontsize=9.5, framealpha=0.95, edgecolor="#dfe3e9",
                   loc="lower right" if metric == "r2" else "upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    # Always also emit a vector PDF alongside the raster PNG.
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


mae_png = os.path.join(OUT, "lowdata_mae.png")
r2_png = os.path.join(OUT, "lowdata_r2.png")
make_plot("mae", "test MAE  (degrees)", mae_png)
make_plot("r2", "test R²", r2_png, ylim=(-0.25, 1.03))
MAE_B64, R2_B64 = b64(mae_png), b64(r2_png)


def fmt(mean, std, dec):
    txt = f"{mean:.{dec}f} ± {std:.{dec}f}"
    return txt.replace("-", "−")


def table(metric, a, dec):
    best_col_name = "vit_sigreg"
    head = "".join(f"<th>{ts}{'*' if ts == 8097 else ''}</th>" for ts in SIZES)
    rows = []
    for n in NAMES:
        cls = ' class="hl"' if n == best_col_name else ""
        tds = []
        for ts in SIZES:
            mean, std = cell(S["backbones"][n]["sizes"][str(ts)], metric, a)
            tds.append(f"<td>{fmt(mean, std, dec)}</td>")
        rows.append(f'<tr{cls}><th scope="row">{n}</th>{"".join(tds)}</tr>')
    return (f'<div class="tbl-wrap"><table><thead><tr><th scope="col">backbone</th>{head}</tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table></div>')


HTML = f"""<title>Yoda rotation-angle probe — learning curves</title>
<style>
  :root {{
    --ground:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5;
    --ink:#141a20; --muted:#59636f; --faint:#8a94a0; --line:#e1e6ec;
    --accent:#0e7c86; --accent-soft:#dcf1f2; --amber:#a75510; --amber-soft:#fbeede;
    --figmount:#ffffff; --shadow:0 1px 2px rgba(20,26,32,.05), 0 8px 24px rgba(20,26,32,.06);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ground:#0c0f13; --surface:#141a21; --surface-2:#1b222b;
      --ink:#e8ecf1; --muted:#9aa4b1; --faint:#6b7482; --line:#242c36;
      --accent:#3fd0dc; --accent-soft:#0e2b2e; --amber:#f0a35a; --amber-soft:#2a1d10;
      --figmount:#f3f4f6; --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
    }}
  }}
  :root[data-theme="light"] {{
    --ground:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5;
    --ink:#141a20; --muted:#59636f; --faint:#8a94a0; --line:#e1e6ec;
    --accent:#0e7c86; --accent-soft:#dcf1f2; --amber:#a75510; --amber-soft:#fbeede;
    --figmount:#ffffff; --shadow:0 1px 2px rgba(20,26,32,.05), 0 8px 24px rgba(20,26,32,.06);
  }}
  :root[data-theme="dark"] {{
    --ground:#0c0f13; --surface:#141a21; --surface-2:#1b222b;
    --ink:#e8ecf1; --muted:#9aa4b1; --faint:#6b7482; --line:#242c36;
    --accent:#3fd0dc; --accent-soft:#0e2b2e; --amber:#f0a35a; --amber-soft:#2a1d10;
    --figmount:#f3f4f6; --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--ground); color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    line-height:1.55; -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:clamp(24px,5vw,64px) clamp(18px,4vw,40px); }}
  .eyebrow {{
    font:600 12px/1.4 ui-monospace,"SFMono-Regular",Menlo,monospace;
    letter-spacing:.14em; text-transform:uppercase; color:var(--accent); margin:0 0 12px;
  }}
  h1 {{ font-size:clamp(26px,4vw,40px); font-weight:700; letter-spacing:-.02em;
        line-height:1.1; margin:0 0 14px; text-wrap:balance; }}
  .lede {{ font-size:clamp(15px,1.6vw,17px); color:var(--muted); max-width:64ch; margin:0 0 26px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 8px; }}
  .chip {{ font:600 12px/1 ui-monospace,Menlo,monospace; color:var(--muted);
    background:var(--surface-2); border:1px solid var(--line); border-radius:999px;
    padding:7px 12px; letter-spacing:.02em; }}
  .chip b {{ color:var(--ink); font-weight:700; }}
  .rule {{ height:1px; background:var(--line); border:0; margin:34px 0; }}
  h2 {{ font-size:20px; font-weight:700; letter-spacing:-.01em; margin:0 0 4px; }}
  h2 .sub {{ font:500 13px/1 ui-monospace,Menlo,monospace; color:var(--faint); margin-left:2px; }}
  .sec-note {{ color:var(--muted); font-size:14px; margin:2px 0 18px; max-width:70ch; }}
  figure {{ margin:0 0 26px; }}
  .mount {{ background:var(--figmount); border:1px solid var(--line); border-radius:14px;
    padding:14px; box-shadow:var(--shadow); overflow-x:auto; }}
  .mount img {{ display:block; width:100%; height:auto; min-width:680px; border-radius:6px; }}
  figcaption {{ font-size:13px; color:var(--muted); margin-top:10px; padding-left:2px; }}
  figcaption b {{ color:var(--ink); }}
  .tabs {{ display:flex; gap:6px; margin:0 0 16px; }}
  .grouplabel {{ font:600 12px/1 ui-monospace,Menlo,monospace; letter-spacing:.08em;
    text-transform:uppercase; color:var(--faint); margin:22px 0 8px; }}
  .tbl-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); margin:0 0 10px; background:var(--surface); }}
  table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums;
    font-size:12.5px; }}
  thead th {{ position:sticky; top:0; background:var(--surface-2); color:var(--muted);
    font:600 12px/1 ui-monospace,Menlo,monospace; text-align:right; padding:11px 12px;
    white-space:nowrap; border-bottom:1px solid var(--line); }}
  thead th:first-child {{ text-align:left; }}
  tbody th {{ text-align:left; font-weight:600; color:var(--ink); padding:9px 12px;
    white-space:nowrap; font-family:ui-monospace,Menlo,monospace; font-size:12px; }}
  tbody td {{ text-align:right; padding:9px 12px; color:var(--muted);
    font-family:ui-monospace,Menlo,monospace; white-space:nowrap; }}
  tbody tr:not(:last-child) th, tbody tr:not(:last-child) td {{ border-bottom:1px solid var(--line); }}
  tbody tr:hover td, tbody tr:hover th {{ background:color-mix(in srgb, var(--accent) 6%, transparent); }}
  tr.hl th[scope="row"] {{ color:var(--amber); }}
  tr.hl {{ background:var(--amber-soft); }}
  tr.hl td:last-child, tr.hl td:nth-last-child(1) {{ color:var(--ink); font-weight:600; }}
  .foot {{ margin-top:34px; color:var(--faint); font-size:13px; max-width:74ch; }}
  .foot code {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--muted);
    background:var(--surface-2); padding:1px 6px; border-radius:5px; }}
  .foot b {{ color:var(--muted); }}
</style>

<div class="wrap">
  <p class="eyebrow">Frozen-backbone evaluation · bobblehead (Yoda + bulldog)</p>
  <h1>Reading rotation angles out of frozen JEPA backbones</h1>
  <p class="lede">A linear (ridge) probe recovers each figurine's rotation angle from
     frozen 512-d features. Here is the full low-data learning curve — how probe
     accuracy scales with the number of training frames — for nine pretrained
     backbones (five global-regularizer + four collapse-fixed local), each point a
     mean over {NSEEDS} random splits.</p>
  <div class="chips">
    <span class="chip"><b>9</b> backbones</span>
    <span class="chip"><b>10</b> train sizes · 5&#8202;&rarr;&#8202;8097</span>
    <span class="chip"><b>{NSEEDS}</b> seeds each</span>
    <span class="chip"><b>20%</b> held-out test</span>
    <span class="chip">8097 frames total</span>
  </div>

  <hr class="rule">

  <h2>Learning curves <span class="sub">mean &plusmn; std over {NSEEDS} seeds</span></h2>
  <p class="sec-note">Lower is better for MAE, higher for R&sup2;. <b>localballreg</b>
     (lamb0.1, k2 — red, bold), a collapse-fixed local backbone, leads at <em>every</em>
     training size on both figurines, beating the best global backbone (vit_sigreg,
     blue). The earlier collapsed local runs are excluded.</p>

  <figure>
    <div class="mount"><img alt="Test mean angular error vs number of training frames, both figurines, seven backbones" src="{MAE_B64}"></div>
    <figcaption><b>Mean angular error (degrees).</b> Test MAE against effective training-frame
      count (log x), for figurine 0 and figurine 1.</figcaption>
  </figure>

  <figure>
    <div class="mount"><img alt="Test R-squared vs number of training frames, both figurines, seven backbones" src="{R2_B64}"></div>
    <figcaption><b>Coefficient of determination (R&sup2;).</b> Same runs; note every
      backbone is near-random (R&sup2; &le; 0) below ~20 training frames.</figcaption>
  </figure>

  <hr class="rule">

  <h2>Data tables <span class="sub">held-out test, per figurine</span></h2>
  <p class="sec-note">Each cell is mean&#8201;&plusmn;&#8201;std over {NSEEDS} seeds. Columns are the
     number of training frames. <code>8097*</code> = full training pool (effective 5182
     after the 20% test / 16% val hold-out); it reproduces the standard full-data table.</p>

  <div class="grouplabel">Mean angular error — MAE (degrees) · figurine 0</div>
  {table("mae", 0, 2)}
  <div class="grouplabel">Mean angular error — MAE (degrees) · figurine 1</div>
  {table("mae", 1, 2)}
  <div class="grouplabel">R&sup2; · figurine 0</div>
  {table("r2", 0, 3)}
  <div class="grouplabel">R&sup2; · figurine 1</div>
  {table("r2", 1, 3)}

  <p class="foot">
    <b>Method.</b> Frozen backbone &rarr; 512-d feature per frame; closed-form ridge regression
    to (cos&#952;, sin&#952;), &#952;&#770; = atan2(&#349;, &#265;). Per (size, seed): fresh random
    20% test / 16% val / rest split; ridge &#955; picked on val by angular error; metric on the
<<<<<<< HEAD
    held-out test. Reproduce with <code>eval_angle_probe_lowdata.py</code> and
=======
    held-out test. Reproduce with <code>src/eval_angle_probe_lowdata.py</code> and
>>>>>>> 504dbb4b1bdcef4de8af30d106b1e20689fb1f81
    <code>plot_lowdata_curve.py</code>; raw numbers in
    <code>logs/eval_angle_probe_lowdata/results_lowdata.{{json,csv}}</code>.
  </p>
</div>
"""

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "yoda_lowdata.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out_path, f"({len(HTML)//1024} KB)")
print("regenerated plots:", mae_png, r2_png)
print("vector PDFs:", os.path.splitext(mae_png)[0] + ".pdf", os.path.splitext(r2_png)[0] + ".pdf")

"""Documentation claims that can be checked against the files they describe.

Every item here is a correction from the review ledger, turned into a guard. They are cheap
string and config checks, and each one would have caught its own defect -- which is the bar for
including a docs test at all. Anything requiring judgement about *framing* is deliberately absent:
`state/02-METHOD-AND-CLAIMS.md` R1 records the headline-claim wording as the user's decision, not
a testable property.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def tracked_text_files():
    import subprocess

    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout
    for name in out.splitlines():
        path = REPO / name
        if path.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".sh", ".example"}:
            yield path


# --------------------------------------------------------------------------------------
# the rename
# --------------------------------------------------------------------------------------

def test_the_old_project_name_is_gone_outside_results():
    """The repo was renamed BundLeJEPA -> LocaLeJEPA. README's install block still cloned the old
    name *and* `cd`-ed into it, so the quickstart failed at step two even if GitHub redirected
    the clone.

    `results/` is exempt: `LocalReg_complexity.md` explains why its filename was retained, and
    `BallRegRadius_results.md` holds live W&B URLs under the old project that would break.

    Two files are exempt for the same structural reason: they have to *name* the string in order
    to forbid or plant it. `tests/mutate.py` stores the "before" text of every defect, and this
    file states what it is checking for. Without the exemption, planting a mutant for a docs
    regression makes the docs guard fail -- which is exactly what happened when
    `readme-stale-project-name` was added, and it went out in a pushed commit because the mutants
    were verified without re-running the full suite afterwards.

    The exemption is by explicit path, not by directory: a *different* test file reintroducing
    the old name is still a finding, because the reason for exemption is specific to these two.
    """
    exempt = {"tests/mutate.py", "tests/test_docs_match_the_code.py"}
    offenders = [
        f"{path.relative_to(REPO)}:{number}"
        for path in tracked_text_files()
        if str(path.relative_to(REPO)) not in exempt
        if "results/" not in str(path.relative_to(REPO))
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
        if "BundLeJEPA" in line
    ]
    assert not offenders, f"stale project name in: {offenders}"


def test_the_readme_clones_the_repository_that_exists():
    readme = (REPO / "README.md").read_text()
    assert "alessioborgi/LocaLeJEPA" in readme
    assert "clone https://github.com/alessioborgi/BundLeJEPA" not in readme


# --------------------------------------------------------------------------------------
# config overrides quoted in the README must resolve
# --------------------------------------------------------------------------------------

def readme_reg_loss_overrides():
    """Every `model.reg_loss.*=value` the README tells a reader to type."""
    text = (REPO / "README.md").read_text()
    return sorted(set(re.findall(r"model\.reg_loss\.([A-Za-z_.]+)=", text)))


def test_the_readme_quotes_at_least_one_override():
    """Otherwise the test below is vacuously green -- the failure shape that has bitten this
    project five times."""
    assert readme_reg_loss_overrides(), "no model.reg_loss.* override found to check"


@pytest.mark.parametrize("key", readme_reg_loss_overrides())
def test_every_reg_loss_override_in_the_readme_is_a_real_parameter(key):
    """The README documented `model.reg_loss.loss.num_slices=1024`, which **no config sets** --
    the real value is 256 -- and `model.reg_loss.num_slices`, which is not a parameter of
    `local_loss` at all, so Hydra would reject it. A reader following the README got either a
    wrong number or a crash.

    Checked against the function signatures rather than against another document.
    """
    import inspect

    from src import losses

    leaf = key.split(".")[-1]
    if key.startswith("loss."):
        # the inner test: whichever regularizer is placed at `loss`
        candidates = [losses.sigreg_loss, losses.ballreg_loss, losses.uniform_reg_loss]
    else:
        candidates = [losses.local_loss]

    accepted = set()
    for fn in candidates:
        accepted |= set(inspect.signature(fn).parameters)

    assert leaf in accepted, (
        f"README documents model.reg_loss.{key}, but {leaf!r} is not a parameter of any of "
        f"{[f.__name__ for f in candidates]}"
    )


def test_the_readme_does_not_claim_a_num_slices_no_config_sets():
    """1024 appears in exactly one config -- `vitl14_in1k_sigreg_paper.yaml`, the *baseline* arm.
    The README used to present it as what the local configs use."""
    readme = (REPO / "README.md").read_text()
    assert "num_slices=1024" not in readme

    setters = [
        str(path.relative_to(REPO))
        for path in (REPO / "configs").rglob("*.yaml")
        if "num_slices: 1024" in path.read_text()
    ]
    assert setters == ["configs/experiment/lejepa/vitl14_in1k_sigreg_paper.yaml"], (
        f"the set of configs using 1024 slices changed: {setters}. If a local arm now sets it, "
        "the README's note about the ViT-L/14 pair needs revisiting."
    )


# --------------------------------------------------------------------------------------
# the eval package's own docstring
# --------------------------------------------------------------------------------------

def test_the_eval_docstring_lists_every_module_in_the_package():
    """It advertised 2 of 9. A reader looking for the transfer or RankMe probes would conclude
    they were not there."""
    docstring = (REPO / "src" / "eval" / "__init__.py").read_text()
    modules = {p.stem for p in (REPO / "src" / "eval").glob("*.py") if p.stem != "__init__"}

    missing = sorted(m for m in modules if f"src.eval.{m}" not in docstring)
    assert not missing, f"src/eval/__init__.py does not mention: {missing}"


# --------------------------------------------------------------------------------------
# the recipe table
# --------------------------------------------------------------------------------------

def test_the_recipe_table_names_every_parameter_that_differs():
    """`configs/experiment/README.md` says "Everything else is identical" and is the document the
    top-level README points readers at. It listed lr, warm-up and grad-clip but omitted
    `lr_eta_min_ratio`, which is a schedule-*shape* difference: the baseline decays 2e-3 -> 1e-3
    while the local arm is flat at 1e-3.
    """
    from omegaconf import OmegaConf

    text = (REPO / "configs" / "experiment" / "README.md").read_text()
    base = OmegaConf.load(REPO / "configs/experiment/lejepa/vit_in1k_sigreg.yaml")
    ours = OmegaConf.load(REPO / "configs/experiment/lejepa/vit_in1k_localballreg_l01_k4.yaml")

    differing = [
        key for key in ("lr_eta_min_ratio", "warmup_epochs", "lamb")
        if base.get("model", {}).get(key) != ours.get("model", {}).get(key)
        and not (base.get("model", {}).get(key) is None and ours.get("model", {}).get(key) is None)
    ]
    assert differing, "expected the two arms to differ; the fixture configs may have changed"

    # The table writes some of these in prose ("warm-up" rather than `warmup_epochs`), which is
    # fine -- the requirement is that the difference is *stated*, not that the key is spelled
    # exactly. Aliases are listed explicitly rather than matching loosely, so a parameter that
    # genuinely goes undocumented still fails.
    aliases = {
        "warmup_epochs": ("warmup_epochs", "warm-up"),
        "lamb": ("lamb", "λ"),
        "lr_eta_min_ratio": ("lr_eta_min_ratio",),
    }
    for key in differing:
        accepted = aliases.get(key, (key,))
        assert any(name in text for name in accepted), (
            f"model.{key} differs between the compared arms but is absent from the recipe table "
            f"that claims 'everything else is identical' (looked for {accepted})"
        )


# --------------------------------------------------------------------------------------
# the smoke test
# --------------------------------------------------------------------------------------

def test_the_smoke_test_is_labelled_as_not_a_metric():
    """`limit_val_batches=2` at `batch_size=8` is 16 images, all of class 0 of 10, because the val
    loader is unshuffled over a class-sorted ImageFolder. The printed `probe_acc` is noise, and
    two smoke runs differing 0.359 vs 0.031 look like a result.

    The command is deliberately not changed -- a smoke test should stay 30 seconds. The caveat is
    what has to be there.
    """
    readme = (REPO / "README.md").read_text()
    assert "limit_val_batches=2" in readme, "the smoke test command changed; revisit this guard"

    after = readme.split("limit_val_batches=2", 1)[1][:900]
    assert "not metrics" in after or "not a metric" in after, (
        "the smoke test's probe_acc needs an explicit 'this is not a metric' note"
    )

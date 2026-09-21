"""The AICR shell scripts: parseable, thin, and free of the shapes that cost real runs.

Shell cannot be unit-tested the way ``campaign.py`` is, so the guard is structural: every
decision must be delegated (the script calls the helper's subcommands), the resources must not
live in a header (F32/F33: a successor that reverted to 1 GPU), the submitting shell must not
leak in (research_rules/01: ``--export=ALL`` carried a module's LD_LIBRARY_PATH for 23 hours),
and no site path may be tracked (``tests/test_no_absolute_paths.py`` covers that separately).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AICR = REPO / "scripts" / "aicr"
SCRIPTS = sorted(AICR.glob("*.sh"))

FORBIDDEN = {
    "--export=ALL": "a job must not inherit the submitting shell (research_rules/01)",
    "#SBATCH --gpus": "resources live on the submit line, not in a header (F32)",
    "#SBATCH --partition": "the partition is measured at submission, never a header default",
    "#SBATCH --time": "the wall is calibrated per wave, never a header default",
    "--partition=gpu": "FASRC's partition; it does not exist on AICR",
    "uv run": "the AICR environment is conda; uv is not installed there",
    "--qos": "rejected on this cluster",
}

REQUIRED_IN_TRAIN = {
    "set -euo pipefail": "fail on the first unhandled error",
    "srun --ntasks-per-node=": "Lightning's SLURMEnvironment requires ntasks-per-node == devices",
    "SLURM_EXPORT_ENV=ALL": "steps must inherit the script's environment despite --export=NONE",
    "campaign.py": "every decision is delegated to the tested helper",
    " preflight ": "the tree/env check runs before staging",
    " resume ": "resume is decided from inside the checkpoints",
    " postflight ": "the artifact decides, not the exit code",
    "successor": "a resumable exit re-submits with this job's resources",
    "trap cleanup EXIT": "node-local /tmp is cleaned on every exit path",
    "verify_imagenet_staging.py": "a staged tree is verified structurally before training",
    "callbacks.on_exception.dirpath": "the SIGTERM checkpoint lands beside last.ckpt",
    "REPO:?": "BASH_SOURCE is the spool copy under sbatch; the clone must be exported",
}


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
@pytest.mark.parametrize("pattern", sorted(FORBIDDEN))
def test_scripts_avoid_the_known_traps(script, pattern):
    for n, line in enumerate(script.read_text().splitlines(), 1):
        if pattern in line and not line.lstrip().startswith("#"):
            pytest.fail(f"{script.name}:{n} contains {pattern!r}: {FORBIDDEN[pattern]}")


@pytest.mark.parametrize("needle", sorted(REQUIRED_IN_TRAIN))
def test_the_train_script_keeps_its_load_bearing_lines(needle):
    text = (AICR / "slurm_train.sh").read_text()
    assert needle in text, f"slurm_train.sh lost {needle!r}: {REQUIRED_IN_TRAIN[needle]}"


def test_the_train_script_has_no_resource_directives_at_all():
    """Not even a harmless one: the day one is added, the successor path can drift again."""
    directives = [l for l in (AICR / "slurm_train.sh").read_text().splitlines()
                  if re.match(r"^#SBATCH\s+--(gpus|gres|partition|time|mem|cpus|ntasks|nodes|account)", l)]
    assert directives == []


def test_the_roots_template_is_all_placeholders_and_the_real_file_is_ignored():
    template = (AICR / "roots.env.example").read_text()
    assignments = re.findall(r"^export (\w+)=(\S+)", template, flags=re.M)
    assert assignments and all(value == "CHANGE_ME" for _, value in assignments), assignments
    assert {name for name, _ in assignments} == {"PY", "CKPT_ROOT", "LOG_ROOT", "DATA_ROOT", "DATA_TAR", "ENV_LOCK"}
    assert "scripts/aicr/roots.env" in (REPO / ".gitignore").read_text().splitlines()


def test_every_subcommand_the_scripts_call_exists():
    """A script calling a subcommand the helper does not have would fail at 03:00 on a node."""
    import scripts.aicr.campaign as C

    parser_names = set()
    original = C.argparse.ArgumentParser.add_subparsers

    class Spy:
        def __init__(self, sub): self.sub = sub
        def add_parser(self, name, *a, **k):
            parser_names.add(name); return self.sub.add_parser(name, *a, **k)

    C.argparse.ArgumentParser.add_subparsers = lambda self, *a, **k: Spy(original(self, *a, **k))
    try:
        with pytest.raises(SystemExit):
            C.main(["--help"])
    finally:
        C.argparse.ArgumentParser.add_subparsers = original
    called = set()
    for script in SCRIPTS:
        called |= set(re.findall(r"\$CAMPAIGN (\S+)", script.read_text()))
    assert called, "no script calls the helper"
    assert called <= parser_names, f"scripts call subcommands the helper lacks: {sorted(called - parser_names)}"

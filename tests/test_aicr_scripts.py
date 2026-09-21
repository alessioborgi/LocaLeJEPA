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


REQUIRED_PER_SCRIPT = {
    "slurm_train.sh": {
        "MODE:-train": "calibrate mode is opt-in",
        "calibrate-verdict": "a calibration ends with the resumed-run verdict, not an exit code",
        "callbacks.self_interrupt.at_step": "the rehearsed SIGTERM is real, not simulated",
        "callbacks.learning_rate_monitor.logging_interval=step": "per-step LR rows, or the seam check has nothing to read",
        "-calib-${SLURM_JOB_PARTITION": "calibration dirs are keyed by partition and job: two partitions calibrate one geometry at once",
        'CKPT_ROOT="${CKPT_ROOT_OVERRIDE:-$CKPT_ROOT}"': "an ablation wave's checkpoint root comes from --export, never from editing roots.env between submissions: tasks read the file when they START",
        '${CKPT_ROOT_OVERRIDE:+--export "CKPT_ROOT_OVERRIDE=$CKPT_ROOT_OVERRIDE"}': "the successor keeps the checkpoint root of the job it continues, and adds nothing when there is none",
    },
    "slurm_tar_in1k.sh": {
        "tar -tf": "the entry count is compared to the tree, not trusted",
        "sha256sum": "the digest the refresh job re-checks is written beside the tar",
        "MISMATCH": "a short tar is refused loudly",
        "/tmp/*)": "a tar under /tmp is refused (F39)",
        'mkdir "$LOCK"': "one writer at a time: two submissions wrote into one .tmp (F44)",
        "$TAR.tmp\" ] && { echo \"REFUSING": "a leftover .tmp is a running or dead writer, never overwritten",
    },
    "slurm_refresh_tar.sh": {
        "cp \"$TAR\" \"$TAR.refresh\"": "a copy to a NEW inode, fresh under any purge clock",
        "sha256sum -c": "the copy is verified before it replaces the original",
        "--begin=now+25days": "the next refresh is on the scheduler, not in memory",
        "REFUSING": "a purged tar is reported, never silently re-created",
    },
    "slurm_download_transfer.sh": {
        "--verify": "the structural check replaces the existence-only .ready marker",
        "/tmp/*)": "a node-local root is refused (F39)",
    },
    "slurm_eval.sh": {
        "eval-preflight": "the checkpoint must exist and carry .verified before extraction",
        "eval-postflight": "the per-backbone JSON is checked, not the exit code",
        "srun --ntasks-per-node=1": "one task, one GPU",
    },
}


@pytest.mark.parametrize("script, needle", [(s, n) for s, needles in REQUIRED_PER_SCRIPT.items() for n in needles],
                         ids=lambda x: x if isinstance(x, str) and x.endswith(".sh") else None)
def test_each_script_keeps_its_load_bearing_lines(script, needle):
    text = (AICR / script).read_text()
    assert needle in text, f"{script} lost {needle!r}: {REQUIRED_PER_SCRIPT[script][needle]}"


def test_every_script_in_the_directory_is_covered_by_a_requirement_set():
    assert {s.name for s in SCRIPTS} == set(REQUIRED_PER_SCRIPT), "a new script needs its own load-bearing lines"


def test_parameter_expansion_messages_carry_no_quotes():
    """``"${VAR:?word}"`` parses ``word`` with its own quoting rules: an apostrophe in the message
    is an unterminated string and the script dies at parse time -- which ``bash -n`` catches, but
    only after someone has submitted it. Keep the messages plain."""
    for script in SCRIPTS:
        for n, line in enumerate(script.read_text().splitlines(), 1):
            for m in re.finditer(r"\$\{[A-Z_]+:\?([^}]*)\}", line):
                assert "'" not in m.group(1) and '"' not in m.group(1), f"{script.name}:{n}: quote inside ${{VAR:?...}}"


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

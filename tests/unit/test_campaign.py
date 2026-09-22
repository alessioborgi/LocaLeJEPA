"""``scripts/aicr/campaign.py`` -- every decision the AICR launcher makes, and the defect each
one exists to prevent.

The FASRC launcher resumed on the *existence* of ``last.ckpt``, queued its successor before
training started, and let that successor revert to the header's 1-GPU request (F32, F33).
None of that was testable. These tests are the launcher's contract; the shell script only
sequences the functions tested here.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

import scripts.aicr.campaign as C

MANIFEST = {
    "name": "smoke",
    "seeds": [0, 1],
    "gpus": 1,
    "max_epochs": 3,
    "experiment": "ablations/base",
    "short": {"model.lamb": "lam", "model.reg_loss.num_clusters": "k",
              "data.batch_size": "b", "trainer.accumulate_grad_batches": "x"},
    "blocks": [{
        "tag": "fixk",
        "axes": {
            "model.lamb": [0.02, 0.1],
            "model.reg_loss.num_clusters": [1, 8],
            "batch": [
                {"data.batch_size": 64, "trainer.accumulate_grad_batches": 4, "micro_batch": 64},
                {"data.batch_size": 256, "trainer.accumulate_grad_batches": 1, "micro_batch": 256},
            ],
        },
        "overrides": {"logger": "csv"},
    }],
}


def spec(**changes) -> C.RunSpec:
    base = dict(run_name="fixk_b64_lam0.02_k8_x4_s1", experiment="ablations/base",
                overrides=("data.batch_size=64", "logger=csv", "model.lamb=0.02",
                           "model.reg_loss.num_clusters=8", "trainer.accumulate_grad_batches=4"),
                seed=1, gpus=1, micro_batch=64, max_epochs=3, tag="fixk")
    return C.RunSpec(**{**base, **changes})


# --------------------------------------------------------------------------------------
# expand: names are the identity (F35 -- three naming conventions, silent row drop)
# --------------------------------------------------------------------------------------

def test_expand_is_the_cartesian_product_times_seeds():
    runs = C.expand(MANIFEST)
    assert len(runs) == 2 * 2 * 2 * 2
    assert len({r.run_name for r in runs}) == 16, "every run distinguishable by name"


def test_run_names_are_deterministic_and_readable():
    runs = C.expand(MANIFEST)
    assert runs[0].run_name == "fixk_b64_lam0.02_k1_x4_s0"
    assert runs[-1].run_name == "fixk_b256_lam0.1_k8_x1_s1"
    assert C.expand(MANIFEST) == runs, "the same manifest expands identically every time"


def test_run_names_do_not_depend_on_how_the_manifest_was_written():
    """FOUND BY THE CLI TEST: ``yaml.safe_dump`` sorts keys, and the first version named runs in
    axis order, so a re-serialised manifest renamed every run -- F35 through the back door. The
    name is a function of the overrides, canonically ordered, and nothing else."""
    reordered = json.loads(json.dumps(MANIFEST))
    axes = reordered["blocks"][0]["axes"]
    reordered["blocks"][0]["axes"] = {k: axes[k] for k in reversed(list(axes))}
    assert {r.run_name for r in C.expand(reordered)} == {r.run_name for r in C.expand(MANIFEST)}
    assert {r.run_name: r for r in C.expand(reordered)}["fixk_b64_lam0.02_k8_x4_s1"] == spec()


def test_dict_axis_values_set_several_overrides_and_the_micro_batch():
    run = next(r for r in C.expand(MANIFEST) if r.run_name == "fixk_b256_lam0.02_k8_x1_s1")
    assert run.micro_batch == 256
    assert "data.batch_size=256" in run.overrides and "trainer.accumulate_grad_batches=1" in run.overrides
    assert run.hydra_overrides()[0] == "experiment=ablations/base"
    assert run.hydra_overrides()[-1] == "seed=1"
    assert "logger=csv" in run.overrides, "block-level fixed overrides are carried"


def test_two_indistinguishable_cells_are_refused():
    manifest = {**MANIFEST, "blocks": [MANIFEST["blocks"][0], MANIFEST["blocks"][0]]}
    with pytest.raises(ValueError, match="duplicate run name"):
        C.expand(manifest)


@pytest.mark.parametrize("missing, message", [("experiment", "no experiment"), ("micro_batch", "no micro_batch")])
def test_expand_refuses_an_undeclared_experiment_or_micro_batch(missing, message):
    manifest = json.loads(json.dumps(MANIFEST))
    if missing == "experiment":
        del manifest["experiment"]
    else:
        for value in manifest["blocks"][0]["axes"]["batch"]:
            del value["micro_batch"]
    with pytest.raises(ValueError, match=message):
        C.expand(manifest)


@pytest.mark.parametrize("value, text", [(0.1, "0.1"), (1e-3, "0.001"), (2, "2"), (True, "true"), ("fwd", "fwd"), (1.0, "1")])
def test_format_value(value, text):
    assert C.format_value(value) == text


def test_grid_round_trips_through_json(tmp_path):
    runs = C.expand(MANIFEST)
    C.write_grid(runs, tmp_path / "grid.json", manifest_name="smoke", commit="abc123")
    payload, back = C.read_grid(tmp_path / "grid.json")
    assert back == runs and payload["commit"] == "abc123"


# --------------------------------------------------------------------------------------
# geometry: the per-GPU micro-batch is the experiment
# --------------------------------------------------------------------------------------

def test_geometry_prints_what_the_run_computes_on():
    geo = C.geometry(spec(), world_size=1)
    assert geo["micro_batch_per_gpu"] == 64 and geo["accumulate"] == 4 and geo["effective_batch"] == 256
    assert geo["views"] == 4 and geo["crop_size"] == 128 and geo["dataset"] == "imagenette"
    assert geo["steps_per_epoch"] == 148 and geo["precision"] == "bf16-mixed"


def test_geometry_refuses_a_world_size_that_changes_the_experiment():
    """batch_size=64 on 4 GPUs is a 16-sample regularizer batch -- a different experiment."""
    with pytest.raises(ValueError, match="micro-batch is 16 .* declares 64"):
        C.geometry(spec(), world_size=4)


def test_geometry_accepts_the_equivalent_multi_gpu_layout():
    """4 GPUs at batch_size=256, accumulate 1 IS the 64x4 experiment."""
    four = spec(overrides=("data.batch_size=256", "trainer.accumulate_grad_batches=1"), gpus=4)
    geo = C.geometry(four, world_size=4)
    assert geo["micro_batch_per_gpu"] == 64 and geo["effective_batch"] == 256


# --------------------------------------------------------------------------------------
# resume: by the step inside the file, never existence or mtime
# --------------------------------------------------------------------------------------

def ckpt(path: Path, epoch: int, step: int, hparams: dict | None = None) -> Path:
    torch.save({"state_dict": {}, "epoch": epoch, "global_step": step,
                "hyper_parameters": hparams or {"lamb": 0.02, "lr_eta_min_ratio": 0.001, "warmup_epochs": 10}}, path)
    return path


def test_resume_is_fresh_on_an_empty_directory(tmp_path):
    decision = C.resume_decision(tmp_path, max_epochs=100)
    assert decision.kind == "fresh" and decision.path is None


def test_resume_is_done_when_the_last_epoch_is_inside_the_checkpoint(tmp_path):
    ckpt(tmp_path / "last.ckpt", epoch=99, step=14800)
    decision = C.resume_decision(tmp_path, max_epochs=100)
    assert decision.kind == "done" and decision.path.name == "last.ckpt"


def test_resume_picks_the_larger_global_step_not_the_newer_mtime(tmp_path):
    """An on-exception checkpoint from mid-epoch beats an older ``last.ckpt`` even when NFS
    mtimes say otherwise."""
    last = ckpt(tmp_path / "last.ckpt", epoch=40, step=6068)
    on_exc = ckpt(tmp_path / "on_exception.ckpt", epoch=41, step=6200)
    os.utime(on_exc, (1_000_000, 1_000_000))            # make it look OLDER than last.ckpt
    decision = C.resume_decision(tmp_path, max_epochs=100)
    assert decision.kind == "resume" and decision.path == on_exc


def test_resume_survives_a_truncated_checkpoint(tmp_path):
    (tmp_path / "last.ckpt").write_bytes(b"\x80\x02not a checkpoint")
    on_exc = ckpt(tmp_path / "on_exception.ckpt", epoch=12, step=1800)
    decision = C.resume_decision(tmp_path, max_epochs=100)
    assert decision.kind == "resume" and decision.path == on_exc
    assert any(not c.valid for c in decision.candidates), "the unreadable one is reported, not hidden"


def test_resume_is_fresh_and_says_why_when_only_garbage_exists(tmp_path):
    (tmp_path / "last.ckpt").write_bytes(b"\x00" * 64)
    decision = C.resume_decision(tmp_path, max_epochs=100)
    assert decision.kind == "fresh" and "no readable checkpoint" in decision.reason


def test_verify_artifact_needs_the_final_epoch(tmp_path):
    ok, detail = C.verify_artifact(ckpt(tmp_path / "last.ckpt", epoch=98, step=1), max_epochs=100)
    assert not ok and "epoch 98" in detail
    ok, _ = C.verify_artifact(ckpt(tmp_path / "last.ckpt", epoch=99, step=1), max_epochs=100)
    assert ok


# --------------------------------------------------------------------------------------
# postflight: artifact, config, checkpoint, metrics, tree, environment
# --------------------------------------------------------------------------------------

def run_dir(tmp_path: Path, *, lamb: float = 0.02, epochs: int = 3, local: bool = True,
            overrides: list[str] | None = None) -> Path:
    root = tmp_path / "run"
    (root / ".hydra").mkdir(parents=True)
    cfg = {"model": {"lamb": lamb, "lr_eta_min_ratio": 0.001, "warmup_epochs": 10,
                     **({"reg_loss": {"num_clusters": 8}} if local else {"reg_loss": {"num_slices": 256}})},
           "data": {"batch_size": 64}, "trainer": {"accumulate_grad_batches": 4}, "seed": 1,
           "logger": {"csv": {"_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger"}}}
    OmegaConf.save(OmegaConf.create(cfg), root / ".hydra" / "config.yaml")
    given = overrides if overrides is not None else ["experiment=ablations/base", "logger=csv", "seed=1"]
    OmegaConf.save(OmegaConf.create(given), root / ".hydra" / "overrides.yaml")
    csv_dir = root / "csv" / "version_0"
    csv_dir.mkdir(parents=True)
    columns = ["epoch", "step", "train/loss_step", "train/loss_epoch"]
    if local:
        columns += ["train/local/effective_clusters_epoch", "train/local/fallback_epoch",
                    "train/local/configured_clusters_epoch"]
    rows = [",".join(columns)]
    for e in range(epochs):
        rows.append(",".join([str(e), str(e * 10 + 5), "0.5", "", "", "", ""][: len(columns)]))
        epoch_row = [str(e), str(e * 10 + 9), "", "0.4"] + (["7.5", "0.0", "8.0"] if local else [])
        rows.append(",".join(epoch_row))
    (csv_dir / "metrics.csv").write_text("\n".join(rows) + "\n")
    return root


def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return repo, commit


def test_config_matches_spec_on_a_faithful_run(tmp_path):
    assert C.config_matches_spec(run_dir(tmp_path), spec()) == []


def test_config_check_catches_a_run_trained_with_a_different_lambda(tmp_path):
    problems = C.config_matches_spec(run_dir(tmp_path, lamb=0.1), spec())
    assert any("model.lamb" in p and "0.02" in p and "0.1" in p for p in problems)


def test_config_check_verifies_group_overrides_through_the_command_line(tmp_path):
    """``logger=csv`` composes to the whole logger group, so it is checked against
    ``overrides.yaml``: a run launched without it is a mismatch, not a silent pass."""
    problems = C.config_matches_spec(run_dir(tmp_path, overrides=["experiment=ablations/base", "seed=1"]), spec())
    assert any("logger=csv" in p for p in problems)


def test_config_check_catches_the_wrong_experiment(tmp_path):
    problems = C.config_matches_spec(run_dir(tmp_path, overrides=["experiment=lejepa/other", "logger=csv"]), spec())
    assert any("experiment=ablations/base" in p for p in problems)


def test_checkpoint_hparams_must_agree_with_the_config(tmp_path):
    root = run_dir(tmp_path)
    good = ckpt(tmp_path / "good.ckpt", 2, 29)
    assert C.checkpoint_matches_config(good, root) == []
    bad = ckpt(tmp_path / "bad.ckpt", 2, 29, hparams={"lamb": 0.3})
    assert any("lamb" in p for p in C.checkpoint_matches_config(bad, root))


def _csv0(root: Path) -> Path:
    return root / "csv" / "version_0" / "metrics.csv"


def second_launch(root: Path, *, epochs: range, effective: str = "5.0", loss: str = "0.3") -> Path:
    """A resumed launch's ``version_1/metrics.csv`` re-logging ``epochs`` (local columns present)."""
    path = root / "csv" / "version_1" / "metrics.csv"
    path.parent.mkdir(parents=True)
    columns = ["epoch", "step", "train/loss_step", "train/loss_epoch",
               "train/local/effective_clusters_epoch", "train/local/fallback_epoch", "train/local/configured_clusters_epoch"]
    rows = [",".join(columns)]
    for e in epochs:
        rows.append(",".join([str(e), str(e * 10 + 5), "0.5", "", "", "", ""]))
        rows.append(",".join([str(e), str(e * 10 + 9), "", loss, effective, "0.0", "8.0"]))
    path.write_text("\n".join(rows) + "\n")
    return path


def test_metrics_readers_take_each_epoch_from_the_last_launch(tmp_path):
    """F52: a resumed run has one metrics.csv per launch and the interrupted launch's rows for
    the epochs it re-trained are stale. Inverted once on the old readers: the postflight read the
    newest file only ("2 epochs logged, expected 4") and realised_k averaged its two epochs."""
    root = run_dir(tmp_path, epochs=4)                               # version_0: epochs 0-3, K 7.5
    second_launch(root, epochs=range(2, 4))                          # version_1: epochs 2-3 again, K 5.0
    csv_dir = root / "csv"
    columns, rows = C.merged_metrics_rows(csv_dir)
    assert columns[:4] == ["epoch", "step", "train/loss_step", "train/loss_epoch"]
    assert [int(r["epoch"]) for r in rows] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [r["train/loss_epoch"] for r in rows if r["train/loss_epoch"]] == ["0.4", "0.4", "0.3", "0.3"]
    assert C.metrics_complete(csv_dir, 4, local_run=True) == []
    assert any("2 epochs logged, expected 4" in p for p in
               C.metrics_complete(csv_dir / "version_1" / "metrics.csv", 4, local_run=True)), "one file still reads alone"
    out = C.realised_k(csv_dir)
    assert out["epochs"] == 4 and out["mean"] == pytest.approx((2 * 7.5 + 2 * 5.0) / 4)
    assert C.realised_k(csv_dir / "version_0" / "metrics.csv")["mean"] == pytest.approx(7.5)


def test_postflight_verifies_a_resumed_run_across_its_launches(tmp_path):
    """The 05:15 wave's resumed cells ended `.failed` with a complete last.ckpt (F52)."""
    root = run_dir(tmp_path, epochs=3)
    second_launch(root, epochs=range(1, 3))
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    ckpt(ckpt_dir / "last.ckpt", epoch=2, step=29)
    repo, commit = git_repo(tmp_path)
    verdict = C.postflight(spec(), root, ckpt_dir, grid_commit=commit, lock_hash=None, python=None, repo=repo)
    assert verdict.ok, verdict.lines()
    assert verdict.checks["metrics"] == (True, "complete over 2 launch(es)")


def test_metrics_complete_wants_every_epoch_no_nan_and_the_realised_k_column(tmp_path):
    csv_path = _csv0(run_dir(tmp_path))
    assert C.metrics_complete(csv_path, 3, local_run=True) == []
    assert any("epochs logged" in p for p in C.metrics_complete(csv_path, 5, local_run=True))
    csv_no_local = _csv0(run_dir(tmp_path / "b", local=False))
    assert any("effective_clusters" in p for p in C.metrics_complete(csv_no_local, 3, local_run=True))
    assert C.metrics_complete(csv_no_local, 3, local_run=False) == []
    text = csv_path.read_text().replace("0.4", "nan", 1)
    csv_path.write_text(text)
    assert any("nan" in p for p in C.metrics_complete(csv_path, 3, local_run=True))


def test_postflight_verdict_and_marker_on_a_good_run(tmp_path):
    root = run_dir(tmp_path)
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    ckpt(ckpt_dir / "last.ckpt", epoch=2, step=29)
    repo, commit = git_repo(tmp_path)
    verdict = C.postflight(spec(), root, ckpt_dir, grid_commit=commit, lock_hash=None, python=None, repo=repo)
    assert verdict.ok, verdict.lines()
    marker = C.write_marker(ckpt_dir, verdict, spec(), {"train_rc": 0})
    assert marker.name == ".verified" and json.loads(marker.read_text())["ok"] is True


def test_postflight_reads_local_from_the_config_never_from_the_experiment_name(tmp_path):
    """F53: the SIGReg control lives at ``lejepa/vit_in1k_sigreg_localrecipe``; the old name test
    called it local and demanded a realised-K column it never logs. Inverted once on the old
    code: 'local run without train/local/effective_clusters_epoch'."""
    control = spec(experiment="lejepa/vit_in1k_sigreg_localrecipe", run_name="vit_control_lam0.02_s1",
                   overrides=("data.batch_size=64", "logger=csv", "model.lamb=0.02", "trainer.accumulate_grad_batches=4"))
    root = run_dir(tmp_path, local=False,
                   overrides=["experiment=lejepa/vit_in1k_sigreg_localrecipe", "logger=csv", "seed=1"])
    assert not C.run_is_local(root, control)
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    ckpt(ckpt_dir / "last.ckpt", epoch=2, step=29)
    repo, commit = git_repo(tmp_path)
    verdict = C.postflight(control, root, ckpt_dir, grid_commit=commit, lock_hash=None, python=None, repo=repo)
    assert verdict.ok, verdict.lines()
    assert C.run_is_local(run_dir(tmp_path / "local"), spec()), "a config with reg_loss.num_clusters is local"
    (root / ".hydra" / "config.yaml").unlink()
    assert not C.run_is_local(root, control) and C.run_is_local(root, spec()), "without a config: the manifest's overrides"


def test_postflight_fails_on_an_unfinished_artifact_and_writes_failed(tmp_path):
    root = run_dir(tmp_path)
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    ckpt(ckpt_dir / "last.ckpt", epoch=1, step=19)
    repo, commit = git_repo(tmp_path)
    verdict = C.postflight(spec(), root, ckpt_dir, grid_commit=commit, lock_hash=None, python=None, repo=repo)
    assert not verdict.ok and not verdict.checks["artifact"][0]
    marker = C.write_marker(ckpt_dir, verdict, spec())
    assert marker.name == ".failed" and not (ckpt_dir / ".verified").exists()


def test_postflight_refuses_a_dirty_tree_and_a_foreign_commit(tmp_path):
    root = run_dir(tmp_path)
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    ckpt(ckpt_dir / "last.ckpt", epoch=2, step=29)
    repo, commit = git_repo(tmp_path)
    foreign = C.postflight(spec(), root, ckpt_dir, grid_commit="0000000deadbeef", lock_hash=None, python=None, repo=repo)
    assert not foreign.checks["tree"][0] and "expanded at" in foreign.checks["tree"][1]
    (repo / "f.txt").write_text("changed")
    dirty = C.postflight(spec(), root, ckpt_dir, grid_commit=commit, lock_hash=None, python=None, repo=repo)
    assert not dirty.checks["tree"][0] and "uncommitted" in dirty.checks["tree"][1]


def test_preflight_refuses_a_grid_expanded_from_a_dirty_tree(tmp_path):
    repo, commit = git_repo(tmp_path)
    verdict = C.preflight(spec(), grid_commit=commit + "-dirty", lock_hash=None, python=None, repo=repo)
    assert not verdict.ok and "dirty tree" in verdict.checks["tree"][1]
    assert C.preflight(spec(), grid_commit=commit, lock_hash=None, python=None, repo=repo).ok


# --------------------------------------------------------------------------------------
# environment hash, successor, submit, realised K, staging
# --------------------------------------------------------------------------------------

def test_env_hash_ignores_order_and_underscore_spelling_but_not_versions():
    a = "torch==2.14.0\nfast_pytorch_kmeans==0.2.2\n"
    b = "fast-pytorch-kmeans==0.2.2\n# comment\ntorch==2.14.0\n"
    assert C.env_hash(a) == C.env_hash(b)
    assert C.env_hash(a) != C.env_hash(a.replace("2.14.0", "2.13.0"))


FAKE_ENV = {"SLURM_JOB_ID": "12345", "SLURM_JOB_PARTITION": "b200-batch", "SLURM_JOB_ACCOUNT": "acct",
            "SLURM_JOB_NAME": "grid", "SLURM_GPUS_ON_NODE": "4", "SLURM_NTASKS_PER_NODE": "4",
            "SLURM_CPUS_PER_TASK": "16", "SLURM_MEM_PER_NODE": "245760"}


def test_successor_inherits_this_jobs_resources_and_counts_down():
    argv = C.successor_argv(FAKE_ENV, script="s.sh", requeue_left=2, time="06:00:00", exports={"GRID": "g.json"})
    assert "--gpus-per-node=4" in argv and "--ntasks-per-node=4" in argv and "--partition=b200-batch" in argv
    assert "--mem=245760M" in argv and "--dependency=afterany:12345" in argv and "--signal=TERM@120" in argv
    assert argv[-2].startswith("--export=NONE,") and "REQUEUE_LEFT=1" in argv[-2] and "GRID=g.json" in argv[-2]
    assert argv[-1] == "s.sh"


def test_successor_stops_when_the_budget_is_spent():
    assert C.successor_argv(FAKE_ENV, script="s.sh", requeue_left=0, time="01:00:00", exports={}) is None


def test_successor_refuses_to_guess_missing_resources():
    env = {k: v for k, v in FAKE_ENV.items() if k != "SLURM_JOB_PARTITION"}
    with pytest.raises(ValueError, match="SLURM_JOB_PARTITION"):
        C.successor_argv(env, script="s.sh", requeue_left=1, time="01:00:00", exports={})


def test_submit_builds_one_homogeneous_array(tmp_path):
    runs = C.expand(MANIFEST)
    argv = C.submit_command(tmp_path / "grid.json", runs, partition="rtx-batch", account="acct", time="00:45:00",
                            script="scripts/aicr/slurm_train.sh", log_dir="/logs", exports={"REPO": "/r"}, throttle=8)
    assert "--array=0-15%8" in argv and "--gpus-per-node=1" in argv and "--ntasks-per-node=1" in argv
    assert "--cpus-per-task=16" in argv and "--mem=60G" in argv and "--signal=TERM@120" in argv
    assert any(a.startswith("--export=NONE,") and "GRID=" in a and "REPO=/r" in a for a in argv)
    assert "--export=ALL" not in " ".join(argv)


def _manifest_with_blocks_out_of_alphabetical_order() -> dict:
    """Two blocks whose manifest order ('zulu' then 'alpha') is the reverse of their sorted order."""
    block = dict(MANIFEST["blocks"][0])
    return {**MANIFEST, "seeds": [0], "blocks": [{**block, "tag": "zulu"}, {**block, "tag": "alpha"}]}


def test_index_ranges_and_the_expand_summary_follow_the_manifest_order_never_the_alphabet(tmp_path, capsys):
    """F51: the array index counts in manifest order; the summary printed the tags SORTED, and three
    arrays were submitted twice with --indices read off that list. Inverted once on the old summary:
    the order assertion below fails there (it printed ['alpha', 'zulu'])."""
    manifest = _manifest_with_blocks_out_of_alphabetical_order()
    runs = C.expand(manifest)
    assert C.block_index_ranges(runs) == [("zulu", 0, 7), ("alpha", 8, 15)]
    assert [r.tag for r in runs[:8]] == ["zulu"] * 8 and [r.tag for r in runs[8:]] == ["alpha"] * 8
    path = tmp_path / "m.yaml"; path.write_text(yaml.safe_dump(manifest))
    assert C.main(["expand", str(path), "--out", str(tmp_path / "g.json")]) == 0
    out = capsys.readouterr().out
    assert "in index order: zulu[0-7] alpha[8-15]" in out
    assert "['alpha', 'zulu']" not in out


def test_submit_names_every_chosen_cell_by_index_and_run_name(tmp_path, capsys):
    """The printed sbatch line stays the only stdout line (it is copied verbatim); the cells it names
    go to stderr as index=run_name, in manifest order (F51)."""
    manifest = _manifest_with_blocks_out_of_alphabetical_order()
    path = tmp_path / "m.yaml"; path.write_text(yaml.safe_dump(manifest))
    grid = tmp_path / "g.json"
    assert C.main(["expand", str(path), "--out", str(grid)]) == 0
    capsys.readouterr()
    _, runs = C.read_grid(grid)
    assert C.main(["submit", str(grid), "--partition", "rtx-batch", "--indices", "1,9", "--time", "01:00:00",
                   "--log-dir", "/logs", "--export", "REPO=/r"]) == 0
    out, err = capsys.readouterr()
    lines = out.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("sbatch --parsable") and "--array=1,9" in lines[0]
    assert err.strip() == f"# cells: 1={runs[1].run_name} 9={runs[9].run_name}"
    assert runs[1].tag == "zulu" and runs[9].tag == "alpha"
    assert C.main(["submit", str(grid), "--partition", "rtx-batch", "--time", "01:00:00",
                   "--log-dir", "/logs", "--export", "REPO=/r"]) == 0
    out, err = capsys.readouterr()
    assert "--array=0-15" in out and err.strip() == "# cells: 0-15 in index order: zulu[0-7] alpha[8-15]"


def test_submit_names_the_gpu_type_on_preemptable_and_refuses_mixed_gpu_counts(tmp_path):
    runs = C.expand(MANIFEST)
    with pytest.raises(ValueError, match="gpu-type"):
        C.submit_command(tmp_path / "g.json", runs, partition="preemptable", account="a", time="1:00:00",
                         script="s", log_dir="/l", exports={})
    argv = C.submit_command(tmp_path / "g.json", runs, partition="preemptable", account="a", time="1:00:00",
                            script="s", log_dir="/l", exports={}, gpu_type="rtx_pro_6000")
    assert "--gres=gpu:rtx_pro_6000:1" in argv and not any(a.startswith("--gpus-per-node") for a in argv)
    mixed = runs + [spec(run_name="vitl", gpus=8)]
    with pytest.raises(ValueError, match="mixes GPU counts"):
        C.submit_command(tmp_path / "g.json", mixed, partition="rtx-batch", account="a", time="1:00:00",
                         script="s", log_dir="/l", exports={})


def test_realised_k_reads_the_epoch_series_only(tmp_path):
    csv_path = _csv0(run_dir(tmp_path, epochs=12))
    text = csv_path.read_text().replace("7.5,0.0,8.0", "6.0,1.0,8.0", 1)   # one fallback epoch
    csv_path.write_text(text)
    out = C.realised_k(csv_path, last_n=10)
    assert out["epochs"] == 12 and out["configured"] == 8.0 and out["fallback_epochs"] == 1
    assert out["mean"] == pytest.approx((6.0 + 11 * 7.5) / 12) and out["last_mean"] == pytest.approx(7.5)


def test_realised_k_is_explicit_about_a_global_run(tmp_path):
    out = C.realised_k(_csv0(run_dir(tmp_path, local=False)))
    assert out["epochs"] == 0 and out["mean"] is None


def test_stage_check_wants_headroom():
    ok, detail = C.stage_check(free_bytes=100_000_000_000, tar_bytes=150_000_000_000)
    assert not ok and "180 GB" in detail
    assert C.stage_check(free_bytes=200_000_000_000, tar_bytes=150_000_000_000)[0]


# --------------------------------------------------------------------------------------
# the command line the shell script calls
# --------------------------------------------------------------------------------------

def test_cli_expand_show_args_and_stage_check(tmp_path, capsys):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST))
    grid = tmp_path / "grid.json"
    assert C.main(["expand", str(manifest), "--out", str(grid)]) == 0
    assert "16 runs" in capsys.readouterr().out
    _, runs = C.read_grid(grid)
    index = next(i for i, r in enumerate(runs) if r.run_name == "fixk_b64_lam0.02_k8_x4_s1")
    assert C.main(["show", str(grid), "--index", str(index)]) == 0
    assert json.loads(capsys.readouterr().out)["run_name"] == "fixk_b64_lam0.02_k8_x4_s1"
    assert C.main(["args", str(grid), "--index", str(index)]) == 0
    out = capsys.readouterr().out
    assert "export RUN_NAME=fixk_b64_lam0.02_k8_x4_s1" in out and "seed=1" in out
    assert C.main(["stage-check", "--free", "1", "--tar-bytes", "10"]) == 1
    assert C.main(["stage-check", "--free", "100", "--tar-bytes", "10"]) == 0


def test_the_on_exception_checkpoint_is_part_of_the_default_callbacks(tmp_path):
    """The launcher passes ``callbacks.on_exception.dirpath``; the config group must define it
    and it must instantiate to Lightning's OnExceptionCheckpoint."""
    from hydra.utils import instantiate
    from lightning.pytorch.callbacks import OnExceptionCheckpoint

    cfg = C.compose_config(spec().hydra_overrides())
    node = OmegaConf.to_container(cfg.callbacks.on_exception, resolve=False)
    assert node["_target_"].endswith("OnExceptionCheckpoint") and node["filename"] == "on_exception"
    callback = instantiate(cfg.callbacks.on_exception, dirpath=str(tmp_path))
    assert isinstance(callback, OnExceptionCheckpoint)


# --------------------------------------------------------------------------------------
# evaluations never run on an unverified checkpoint; calibration must really resume
# --------------------------------------------------------------------------------------

def eval_config(tmp_path: Path, checkpoint: str = "checkpoints/run_a/last.ckpt") -> Path:
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(yaml.safe_dump({"output_folder": "logs/eval_x",
                                   "backbones": [{"name": "run_a", "experiment": "lejepa/x", "checkpoint": checkpoint}]}))
    return cfg


def test_eval_preflight_wants_an_existing_readable_verified_checkpoint(tmp_path):
    root = tmp_path / "root"
    ckpt_dir = root / "checkpoints" / "run_a"; ckpt_dir.mkdir(parents=True)
    cfg = eval_config(tmp_path)

    missing = C.eval_preflight(cfg, "run_a", root)
    assert not missing.ok and not missing.checks["checkpoint"][0]

    ckpt(ckpt_dir / "last.ckpt", epoch=99, step=14800)
    unverified = C.eval_preflight(cfg, "run_a", root)
    assert not unverified.ok and "missing" in unverified.checks["verified"][1]

    (ckpt_dir / ".verified").write_text(json.dumps({"ok": True, "epoch": 99}))
    assert C.eval_preflight(cfg, "run_a", root).ok

    (ckpt_dir / ".verified").write_text(json.dumps({"ok": False}))
    assert not C.eval_preflight(cfg, "run_a", root).ok, "a .failed-style marker must not pass"


def test_eval_preflight_names_an_unknown_backbone(tmp_path):
    with pytest.raises(ValueError, match="no backbone named 'other'"):
        C.eval_backbone_entry(eval_config(tmp_path), "other")


def test_eval_postflight_wants_the_per_backbone_json(tmp_path):
    root = tmp_path / "root"
    cfg = eval_config(tmp_path)
    assert not C.eval_postflight(cfg, "run_a", root).ok
    out = root / "logs" / "eval_x" / "per_backbone"; out.mkdir(parents=True)
    (out / "run_a.json").write_text("{not json")
    assert not C.eval_postflight(cfg, "run_a", root).checks["parses"][0]
    (out / "run_a.json").write_text(json.dumps({"knn": {"20": {"top1": 50.0}}}))
    assert C.eval_postflight(cfg, "run_a", root).ok


def calibration_run(tmp_path: Path, *, first: range, second: range, lr_jump: float = 0.0) -> Path:
    root = tmp_path / "calib"
    for version, steps in (("version_0", first), ("version_1", second)):
        d = root / "csv" / version; d.mkdir(parents=True)
        rows = ["epoch,step,train/loss_step,lr-AdamW"]
        for step in steps:
            lr = 1e-3 * (1 - step / 400) + (lr_jump if version == "version_1" else 0.0)
            rows.append(f"0,{step},0.5,{lr}")
        (d / "metrics.csv").write_text("\n".join(rows) + "\n")
    return root


def test_launch_and_calibrate_override_sets_are_exactly_these(capsys):
    """F47: the shell composes nothing itself. The tokens are pinned so a change here is a
    change someone meant; the composition against every cell is in the manifests test."""
    assert C.launch_overrides(1, 16) == ["trainer=gpu", "logger=csv", "data.num_workers=16"]
    assert C.launch_overrides(4, 16) == ["trainer=ddp", "trainer.devices=4", "trainer.num_nodes=1", "logger=csv", "data.num_workers=16"]
    calib = C.calibrate_overrides(200)
    assert calib == ["callbacks=calibrate", "+trainer.max_steps=200", "+trainer.limit_val_batches=0",
                     "+trainer.num_sanity_val_steps=0", "callbacks.learning_rate_monitor.logging_interval=step",
                     "trainer.log_every_n_steps=1"]
    assert C.calibrate_overrides(200, 150) == calib + ["callbacks.self_interrupt.at_step=150"]
    assert all(" " not in token for token in calib), "the shell splits the printed line on spaces"
    C.main(["launch-overrides", "--gpus", "8", "--cpus", "16"])
    C.main(["calibrate-overrides", "--max-steps", "200", "--interrupt", "150"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "trainer=ddp trainer.devices=8 trainer.num_nodes=1 logger=csv data.num_workers=16"
    assert out[1].endswith("trainer.log_every_n_steps=1 callbacks.self_interrupt.at_step=150")


def test_resume_shell_output_names_unreadable_candidates_on_a_fresh_start(tmp_path, capsys):
    """F50: Lightning auto-loads any last.ckpt / on_exception.ckpt when an OnExceptionCheckpoint is
    among the callbacks and no path is passed; a fresh start with an unreadable file in place would
    crash at load. The shell output carries the files to move aside."""
    ckpt = tmp_path / "ckpt"; ckpt.mkdir()
    (ckpt / "last.ckpt").write_bytes(b"truncated")
    assert C.main(["resume", "--ckpt-dir", str(ckpt), "--max-epochs", "3", "--shell"]) == 0
    out = capsys.readouterr().out
    assert "export DECISION=fresh" in out and f"export STALE={ckpt / 'last.ckpt'}" in out
    (ckpt / "last.ckpt").unlink()
    C.main(["resume", "--ckpt-dir", str(ckpt), "--max-epochs", "3", "--shell"])
    assert "export STALE=''" in capsys.readouterr().out, "nothing to move on a clean fresh start"


def test_a_successor_writes_its_stdout_into_the_log_dir_never_the_clone():
    """F50: twenty slurm-<id>.out files from cancelled successors made the clone dirty and every
    later preflight would have refused. The successor names its output in the log dir."""
    env = {"SLURM_JOB_ID": "7", "SLURM_JOB_PARTITION": "rtx-batch", "SLURM_JOB_ACCOUNT": "a", "SLURM_JOB_NAME": "ablation_grid",
           "SLURM_GPUS_ON_NODE": "1", "SLURM_NTASKS_PER_NODE": "1", "SLURM_CPUS_PER_TASK": "16"}
    argv = C.successor_argv(env, script="s.sh", requeue_left=1, time="01:00:00", exports={"GRID": "g"}, log_dir="/logs/slurm")
    assert "--output=/logs/slurm/ablation_grid-successor-%j.out" in argv
    assert not any(a == "--output=." for a in argv)


def test_slurm_time_left_parses_every_squeue_form():
    assert C.parse_slurm_time_left("1-02:03:04") == 93784.0
    assert C.parse_slurm_time_left("02:03:04") == 7384.0
    assert C.parse_slurm_time_left("1:59") == 119.0
    assert C.parse_slurm_time_left("45") == 45.0
    for bad in ("UNLIMITED", "INVALID", "NOT_SET", "", None, "x-1:2", "1:2:3:4"):
        assert C.parse_slurm_time_left(bad) is None, bad


def test_a_successor_follows_the_wall_and_never_a_cancel(capsys):
    """F50: `scancel` of the first wave spawned eighteen successors, because every cancelled task's
    postflight found a resumable checkpoint. A SIGTERM that reaches the batch script with more than
    the signal window left is a cancel (or a preemption, which Slurm requeues itself)."""
    assert C.successor_allowed(signalled=False, time_left=None)[0], "a crash or an OOM: resumable"
    assert C.successor_allowed(signalled=True, time_left="1:58")[0], "the wall's --signal=TERM@120"
    ok, why = C.successor_allowed(signalled=True, time_left="23:41:07")
    assert not ok and "cancel" in why
    ok, why = C.successor_allowed(signalled=True, time_left="unknown")
    assert not ok, "an unparsable time left is treated as a cancel, never as a wall"
    env = {"SLURM_JOB_ID": "7", "SLURM_JOB_PARTITION": "rtx-batch", "SLURM_JOB_ACCOUNT": "a", "SLURM_JOB_NAME": "n",
           "SLURM_GPUS_ON_NODE": "1", "SLURM_NTASKS_PER_NODE": "1", "SLURM_CPUS_PER_TASK": "16", "SLURM_MEM_PER_NODE": "61440"}
    import os
    old = dict(os.environ); os.environ.update(env)
    try:
        assert C.main(["successor", "--requeue-left", "1", "--time", "01:00:00", "--script", "s.sh", "--signalled",
                       "--time-left", "23:41:07", "--export", "GRID=g"]) == 1
        assert "cancel" in capsys.readouterr().err
        assert C.main(["successor", "--requeue-left", "1", "--time", "01:00:00", "--script", "s.sh", "--signalled",
                       "--time-left", "1:30", "--export", "GRID=g"]) == 0
        assert "sbatch" in capsys.readouterr().out
    finally:
        os.environ.clear(); os.environ.update(old)


def online_csv(path: Path, values: list[float], *, epoch0: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["epoch,step,train/loss_step,val/probe_acc"]
    for i, v in enumerate(values):
        rows.append(f"{epoch0 + i},{(epoch0 + i) * 10 + 3},0.7,")          # a per-step row: no probe value
        rows.append(f"{epoch0 + i},{(epoch0 + i) * 10 + 9},,{v}")
    path.write_text("\n".join(rows) + "\n")


def test_online_probe_curve_reports_final_and_the_first_epoch_reaching_the_maximum(tmp_path):
    """Decision 16 / gate G4a: LeJEPA reads the best epoch, we read both. Per-step rows (no
    value) are skipped; values are fractions in the CSV and percent out; a tie goes to the
    EARLIER epoch."""
    csv_path = tmp_path / "csv" / "version_0" / "metrics.csv"
    online_csv(csv_path, [0.30, 0.55, 0.60, 0.58, 0.60])
    curve = C.online_probe_curve(csv_path)
    assert curve == {"epochs": 5, "final": 60.0, "max": 60.0, "argmax_epoch": 2}
    assert C.online_probe_curve(tmp_path / "csv") == curve, "a csv directory reads the same single file"


def test_online_probe_curve_spans_the_launches_of_a_resumed_run(tmp_path):
    online_csv(tmp_path / "csv" / "version_0" / "metrics.csv", [0.30, 0.55])
    online_csv(tmp_path / "csv" / "version_1" / "metrics.csv", [0.65, 0.62], epoch0=2)
    online_csv(tmp_path / "csv" / "version_10" / "metrics.csv", [0.61], epoch0=4)     # numeric order, not lexical
    curve = C.online_probe_curve(tmp_path / "csv")
    assert curve == {"epochs": 5, "final": 61.0, "max": 65.0, "argmax_epoch": 2}


def test_online_probe_curve_drops_the_stale_tail_of_an_interrupted_launch(tmp_path):
    """F52: the first launch of task 0 had logged every epoch (last.ckpt was stale at 97); the
    resume re-trained 98-99. Its old rows for those epochs must not enter the curve -- a stale
    0.99 at epoch 3 here would otherwise become the maximum."""
    online_csv(tmp_path / "csv" / "version_0" / "metrics.csv", [0.30, 0.55, 0.60, 0.99])
    online_csv(tmp_path / "csv" / "version_1" / "metrics.csv", [0.58, 0.62], epoch0=2)
    assert C.online_probe_curve(tmp_path / "csv") == {"epochs": 4, "final": 62.0, "max": 62.0, "argmax_epoch": 3}


def test_metrics_readers_accept_lightning_crlf_line_endings(tmp_path):
    """Lightning's CSVLogger writes with csv.DictWriter's default line terminator, CRLF; an awk on
    the cluster missed the last column for that reason (2026-09-22). The Python readers must not."""
    rows = ["epoch,step,train/loss_epoch,val/probe_acc", "0,9,0.5,0.30", "1,19,0.4,0.55", "100,20,,"]
    path = tmp_path / "csv" / "version_0" / "metrics.csv"; path.parent.mkdir(parents=True)
    path.write_bytes(("\r\n".join(rows) + "\r\n").encode())
    assert C.metrics_complete(tmp_path / "csv", 2, local_run=False) == []
    assert C.online_probe_curve(tmp_path / "csv") == {"epochs": 2, "final": 55.0, "max": 55.0, "argmax_epoch": 1}


def test_online_probe_curve_without_the_column_is_empty_not_wrong(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,step,train/loss_step\n0,1,0.5\n")
    assert C.online_probe_curve(path) == {"epochs": 0, "final": None, "max": None, "argmax_epoch": None}


def test_calibrate_verdict_passes_a_true_resume(tmp_path):
    root = calibration_run(tmp_path, first=range(0, 151), second=range(151, 200))
    verdict = C.calibrate_verdict(root, interrupt_step=150, max_steps=200)
    assert verdict.ok, verdict.lines()


def test_calibrate_verdict_catches_a_restart_from_scratch(tmp_path):
    root = calibration_run(tmp_path, first=range(0, 151), second=range(0, 200))
    verdict = C.calibrate_verdict(root, interrupt_step=150, max_steps=200)
    assert not verdict.checks["resumed"][0]


def test_calibrate_verdict_accepts_a_piecewise_constant_lr_across_the_seam(tmp_path):
    """F48: the B=64x4 Imagenette calibration (accumulate_grad_batches=4) was called broken by
    the seam check -- `lr jump 6.7e-06 vs local slope 0.0` -- because the LR only moves every
    fourth row and the two rows before the seam happened to be equal. One scheduler step at the
    seam is continuity; a restart from scratch is still orders of magnitude bigger."""
    root = tmp_path / "calib"
    step_lr = 1e-3 / 400                                     # one scheduler step per 4 rows
    for version, steps in (("version_0", range(0, 151)), ("version_1", range(151, 200))):
        d = root / "csv" / version; d.mkdir(parents=True)
        rows = ["epoch,step,train/loss_step,lr-AdamW"]
        for step in steps:
            rows.append(f"0,{step},0.5,{1e-3 - (step // 4) * step_lr}")
        (d / "metrics.csv").write_text("\n".join(rows) + "\n")
    verdict = C.calibrate_verdict(root, interrupt_step=150, max_steps=200)
    assert verdict.checks["lr_continuous"][0], verdict.lines()
    # the same file with a restart at the seam (LR back to its warm-up value) still fails
    second = root / "csv" / "version_1" / "metrics.csv"
    second.write_text("\n".join(["epoch,step,train/loss_step,lr-AdamW"] + [f"0,{s},0.5,{1e-3 - ((s - 151) // 4) * step_lr}" for s in range(151, 200)]) + "\n")
    assert not C.calibrate_verdict(root, interrupt_step=150, max_steps=200).checks["lr_continuous"][0]


def test_calibrate_verdict_catches_a_learning_rate_jump_at_the_seam(tmp_path):
    root = calibration_run(tmp_path, first=range(0, 151), second=range(151, 200), lr_jump=5e-4)
    verdict = C.calibrate_verdict(root, interrupt_step=150, max_steps=200)
    assert not verdict.checks["lr_continuous"][0]


def test_calibrate_verdict_names_the_row_spacing_when_rows_are_not_per_step(tmp_path):
    """F46: Lightning's default log_every_n_steps=50 wrote rows at 49/99/149 and 199; the seam
    check must fail AND say why, or a working resume looks like a broken one again."""
    root = calibration_run(tmp_path, first=range(49, 150, 50), second=range(199, 200, 50))
    verdict = C.calibrate_verdict(root, interrupt_step=150, max_steps=200)
    assert not verdict.ok
    ok, message = verdict.checks["resumed"]
    assert not ok and "log_every_n_steps=1" in message and "50 steps apart" in message
    # per-step rows carry no hint: the message stays exactly the seam statement
    root2 = calibration_run(tmp_path / "b", first=range(0, 151), second=range(160, 200))
    _, message2 = C.calibrate_verdict(root2, interrupt_step=150, max_steps=200).checks["resumed"]
    assert "log_every_n_steps" not in message2


def test_calibrate_verdict_needs_exactly_two_launches(tmp_path):
    root = tmp_path / "one"; (root / "csv" / "version_0").mkdir(parents=True)
    (root / "csv" / "version_0" / "metrics.csv").write_text("epoch,step,train/loss_step\n0,1,0.5\n")
    assert not C.calibrate_verdict(root, interrupt_step=150, max_steps=200).ok


def test_checkpoint_digest_changes_with_the_file(tmp_path):
    path = ckpt(tmp_path / "last.ckpt", epoch=1, step=2)
    before = C.checkpoint_digest(path)
    assert before == C.checkpoint_digest(path)
    ckpt(path, epoch=1, step=3)
    os.utime(path, (2_000_000_000, 2_000_000_000))
    assert C.checkpoint_digest(path) != before


# --------------------------------------------------------------------------------------
# the environment check reads the lock's pip half (F44)
# --------------------------------------------------------------------------------------

CONDA_HALF = """# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
@EXPLICIT
https://conda.anaconda.org/conda-forge/linux-64/python-3.12.14-h1234.conda#abc
https://conda.anaconda.org/conda-forge/noarch/pip-25.0-pyh.conda#def
"""
PIP_HALF = "torch==2.14.0+cu130\nfast_pytorch_kmeans==0.2.2\nlightning==2.6.6\n"


def test_env_hash_ignores_the_conda_half_of_the_lock():
    """The cluster's lock is two halves (F24/F38); the job hashes a freeze. F44: hashing the whole
    file refused every job at preflight. Only ``name==version`` lines count, on both sides."""
    assert C.env_hash(CONDA_HALF + PIP_HALF) == C.env_hash(PIP_HALF)
    assert C.env_lines(CONDA_HALF + PIP_HALF) == {"torch==2.14.0+cu130", "fast-pytorch-kmeans==0.2.2", "lightning==2.6.6"}
    assert C.env_hash(PIP_HALF) != C.env_hash(PIP_HALF.replace("2.6.6", "2.6.5"))


def test_env_diff_names_the_pins_that_differ():
    live = PIP_HALF.replace("lightning==2.6.6", "lightning==2.6.5") + "pytest==9.1.1\n"
    text = C.env_diff(CONDA_HALF + PIP_HALF, live)
    assert "only in lock: lightning==2.6.6" in text and "only in env: lightning==2.6.5, pytest==9.1.1" in text
    assert C.env_diff(PIP_HALF, PIP_HALF) == "no pin differs"


def test_preflight_reports_the_differing_pins_not_just_a_hash(tmp_path, monkeypatch):
    repo, commit = git_repo(tmp_path)
    monkeypatch.setattr(C, "current_env_text", lambda python: PIP_HALF.replace("2.6.6", "2.6.5"))
    verdict = C.preflight(spec(), grid_commit=commit, lock_hash=C.env_hash(CONDA_HALF + PIP_HALF),
                          python="python", repo=repo, lock_text=CONDA_HALF + PIP_HALF)
    ok, detail = verdict.checks["environment"]
    assert not ok and "only in lock: lightning==2.6.6" in detail and "only in env: lightning==2.6.5" in detail
    monkeypatch.setattr(C, "current_env_text", lambda python: PIP_HALF)
    assert C.preflight(spec(), grid_commit=commit, lock_hash=C.env_hash(CONDA_HALF + PIP_HALF),
                       python="python", repo=repo, lock_text=CONDA_HALF + PIP_HALF).checks["environment"][0]


def test_the_live_environment_is_read_with_pip_list_freeze_format(monkeypatch):
    """The lock was captured with ``pip list --format=freeze``; plain ``pip freeze`` writes
    ``name @ url`` for URL-installed wheels, which would never match."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        class R: stdout = PIP_HALF
        return R()
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    assert C.current_env_hash("python") == C.env_hash(PIP_HALF)
    assert seen["argv"][1:] == ["-m", "pip", "list", "--format=freeze"]

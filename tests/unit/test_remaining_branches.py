"""The branches nothing else reaches.

Not a grab-bag for its own sake: every test here covers a path that only runs under a condition
the rest of the suite never creates -- a download that has not happened, an optional dependency,
a multi-rank logger, a compiled model, a config with the ``extras`` block missing. Those are
exactly the conditions that first occur *on the cluster, in a real job*, which is the worst place
to discover them.

``research_rules/00``: branch coverage is required because line coverage is satisfied by never
taking the ``if``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


# ======================================================================================
# download guards -- the branch that only runs on a machine without the data
# ======================================================================================

def test_imagenette_downloads_only_when_the_directory_is_absent(tmp_path, monkeypatch):
    """Every rank runs ``prepare_data`` on some launchers. Re-downloading a 1.5 GB archive
    because the existence check was missing or inverted would, on AICR, mean four concurrent
    writes into one shared directory."""
    import src.data.imagenette_datamodule as module

    calls = []
    monkeypatch.setattr(module, "download_and_extract_archive",
                        lambda *a, **k: calls.append(k.get("filename")))

    datamodule = module.ImagenetteDataModule(train_transform=lambda x: x,
                                             data_dir=str(tmp_path), num_workers=0)
    datamodule.prepare_data()
    assert calls == ["imagenette2-160.tgz"], "absent: downloaded"

    (tmp_path / "imagenette2-160").mkdir(parents=True)
    datamodule.prepare_data()
    assert len(calls) == 1, "present: not downloaded again"


def test_stl10_fetches_a_distinct_test_split(monkeypatch):
    """The ``test_split != val_split`` guard, from the other side: when they genuinely differ,
    both must be fetched. The existing test only covers the case where they coincide."""
    import src.data.stl10_datamodule as module

    splits = []
    monkeypatch.setattr(module, "STL10",
                        lambda root, split, download, transform=None: splits.append(split))

    module.STL10DataModule(train_transform=lambda x: x, eval_transform=lambda x: x,
                           train_split="unlabeled", val_split="train",
                           test_split="test").prepare_data()
    assert splits == ["unlabeled", "train", "test"]


# ======================================================================================
# bobblehead label discovery -- the scan branches
# ======================================================================================

def bobblehead_dir(tmp_path, n_frames=5):
    """An already-extracted directory, so the tests below exercise label discovery alone."""
    target = tmp_path / "extracted"
    target.mkdir()
    np.save(target / "color_image_export.npy",
            np.zeros((n_frames, 8, 8, 3), dtype=np.uint8))
    return target


def test_a_named_label_file_is_used_directly(tmp_path):
    """``labels_name`` short-circuits the scan. Needed when the directory holds several plausible
    arrays and the scan would pick whichever sorts first."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    np.save(target / "wanted.npy", np.arange(5, dtype=np.float32))
    np.save(target / "aaa_decoy.npy", np.zeros(5, dtype=np.float32) + 99)

    found = _find_labels(target, n_frames=5, labels_name="wanted.npy")
    np.testing.assert_allclose(found[:, 0], np.arange(5))


def test_a_named_label_file_that_is_absent_yields_no_labels(tmp_path):
    """Explicitly asking for a file that is not there must not silently fall back to scanning --
    that would substitute a different label set under the name the caller asked for."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    np.save(target / "other.npy", np.arange(5, dtype=np.float32))
    assert _find_labels(target, n_frames=5, labels_name="absent.npy") is None


def test_a_one_dimensional_label_array_becomes_a_column(tmp_path):
    """Downstream code indexes ``labels[idx]`` and expects a vector per frame. A bare (N,) array
    would give a scalar and change the label's shape for every consumer."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    np.save(target / "angles.npy", np.linspace(0, 1, 5))
    assert _find_labels(target, n_frames=5, labels_name=None).shape == (5, 1)


@pytest.mark.parametrize("suffix, content", [(".csv", "1,2\n3,4\n5,6\n7,8\n9,10\n"),
                                             (".txt", "1 2\n3 4\n5 6\n7 8\n9 10\n")])
def test_text_label_files_are_parsed_with_the_right_delimiter(tmp_path, suffix, content):
    """``,`` for csv and whitespace otherwise. Using the wrong one yields a single column of NaN,
    which would then fail the frame-count check and be silently discarded as 'no labels'."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    (target / f"angles{suffix}").write_text(content)

    found = _find_labels(target, n_frames=5, labels_name=None)
    assert found.shape == (5, 2)
    np.testing.assert_allclose(found[0], [1.0, 2.0])


def test_an_unparseable_candidate_is_skipped_rather_than_fatal(tmp_path):
    """The scan reads every ``.npy``/``.csv``/``.txt`` it finds, including a README or a pickled
    object array. One bad file must not prevent the good one from being found."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    (target / "aaa_notes.txt").write_text("these are notes, not numbers")
    np.save(target / "zzz_angles.npy", np.linspace(0, 1, 5))

    assert _find_labels(target, n_frames=5, labels_name=None).shape == (5, 1)


def test_an_array_of_the_wrong_length_is_not_mistaken_for_labels(tmp_path):
    """The frame-count match is the only thing identifying a labels file during a scan. Without
    it, any stray array would be adopted as the rotation angles."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    np.save(target / "unrelated.npy", np.zeros(17, dtype=np.float32))
    np.save(target / "also_unrelated.npy", np.zeros((3, 7), dtype=np.float32))

    assert _find_labels(target, n_frames=5, labels_name=None) is None


def test_a_directory_is_not_treated_as_a_candidate(tmp_path):
    """``rglob("*")`` returns directories too; ``is_file()`` filters them. A directory named
    ``foo.npy`` is unusual but ``np.load`` on one raises rather than returning None."""
    from src.data.bobblehead_datamodule import _find_labels

    target = bobblehead_dir(tmp_path)
    (target / "confusing.npy").mkdir()
    np.save(target / "real.npy", np.linspace(0, 1, 5))

    assert _find_labels(target, n_frames=5, labels_name=None).shape == (5, 1)


# ======================================================================================
# optional model features
# ======================================================================================

def test_activation_checkpointing_is_requested_only_when_asked():
    """It trades compute for memory and is off by default. Turning it on silently would slow
    every run by ~30% with no configuration change to point at."""
    from src.models.components.timm_encoder import timm_encoder

    plain = timm_encoder(model_name="vit_tiny_patch16_224", num_classes=8, img_size=64,
                         pretrained=False)
    assert plain.grad_checkpointing is False

    checkpointed = timm_encoder(model_name="vit_tiny_patch16_224", num_classes=8, img_size=64,
                                pretrained=False, grad_checkpointing=True)
    assert checkpointed.grad_checkpointing is True


def test_compilation_wraps_every_trainable_submodule_at_fit():
    """``setup(stage="fit")``. All three of encoder, projector and probe must be wrapped -- a
    partially compiled model works, and the uncompiled part simply runs slower, which is
    invisible except as a number in a throughput table."""
    from conftest import TagEncoder
    from src.models.lejepa_module import LeJEPAModule

    module = LeJEPAModule(
        encoder=TagEncoder(4), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=nn.Linear(4, 10), embedding_dim=4, projection_dim=4,
        use_online_probe=True, compile=True,
    )
    module.setup(stage="fit")

    for name in ("encoder", "projector", "online_probe"):
        assert type(getattr(module, name)).__name__ == "OptimizedModule", name


def test_compilation_does_not_happen_outside_fit():
    """``setup`` is called for validate, test and predict too. Compiling on those costs a
    recompilation of a model that is about to be discarded."""
    from conftest import TagEncoder
    from src.models.lejepa_module import LeJEPAModule

    module = LeJEPAModule(
        encoder=TagEncoder(4), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=None, embedding_dim=4, projection_dim=4, use_online_probe=False,
        compile=True,
    )
    module.setup(stage="validate")
    assert type(module.encoder).__name__ != "OptimizedModule"


def test_compilation_is_off_by_default():
    from conftest import TagEncoder
    from src.models.lejepa_module import LeJEPAModule

    module = LeJEPAModule(
        encoder=TagEncoder(4), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=None, embedding_dim=4, projection_dim=4, use_online_probe=False,
    )
    module.setup(stage="fit")
    assert type(module.encoder).__name__ != "OptimizedModule"


# ======================================================================================
# SubReg as a module
# ======================================================================================

def test_the_subreg_module_matches_the_function_it_wraps():
    """The module exists so the random projection matrices are frozen and travel with the
    checkpoint. It must not otherwise change the statistic."""
    from src.losses import SubReg, subreg_loss

    module = SubReg(projection_dim=16, num_subspaces=4, knots=9)
    proj = torch.randn(4, 64, 16)

    torch.manual_seed(0)
    from_module = module(proj)
    torch.manual_seed(0)
    from_function = subreg_loss(proj, module.proj_matrices, knots=9)
    torch.testing.assert_close(from_module, from_function)


def test_the_projection_matrices_are_a_buffer_not_a_parameter():
    """A buffer moves with ``.to(device)`` and is saved in the checkpoint, but is not optimised.
    As a parameter they would be *learned*, and the subspaces would stop being random -- which is
    the entire premise of the estimator."""
    from src.losses import SubReg

    module = SubReg(projection_dim=16, num_subspaces=4)
    assert "proj_matrices" in dict(module.named_buffers())
    assert list(module.parameters()) == []


def test_the_projection_matrices_are_row_orthonormal():
    """From the QR decomposition. Non-orthonormal rows would make the subspace projection a
    general linear map, and the projected coordinates would no longer be standard normal under
    the null -- so the Gaussian target CF would be the wrong target."""
    from src.losses import SubReg

    matrices = SubReg(projection_dim=16, num_subspaces=4).proj_matrices
    assert matrices.shape == (4, 4, 16)
    for matrix in matrices:
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(4), rtol=1e-5, atol=1e-6)


def test_too_many_subspaces_for_the_projection_width_is_refused():
    """``projection_dim // num_subspaces < 1``. The campaign uses projection_dim=16, so
    num_subspaces=32 is a plausible sweep value and would otherwise build empty matrices."""
    from src.losses import SubReg

    with pytest.raises(ValueError, match="num_subspaces=32 is too large"):
        SubReg(projection_dim=16, num_subspaces=32)


def test_the_metric_key_is_the_function_name():
    """``__name__`` is read by the module to build the logged metric key, so a SubReg run's
    curves line up with the function regularizers' in the same dashboard."""
    from src.losses import SubReg

    assert SubReg(projection_dim=16).__name__ == "subreg_loss"


# ======================================================================================
# the angle probe without standardisation
# ======================================================================================

def test_the_low_data_curve_can_skip_standardisation():
    """``standardize=False`` leaves features on their native scale. It is a real configuration --
    the ablation checking that the learning curve is not an artefact of z-scoring -- and the
    ``else`` branch is reachable no other way.

    Both settings must recover a clean circle, so the assertion is on the conclusion rather than
    on the arithmetic: standardising a feature matrix that is already unit-scale should change
    nothing that matters.
    """
    from src.eval.angle_probe import low_data_curve

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200)
    features = np.stack([np.cos(theta), np.sin(theta), rng.normal(size=200) * 0.01], axis=1)

    plain = low_data_curve(features, theta, train_sizes=(25, 50), n_seeds=2, standardize=False)
    scaled = low_data_curve(features, theta, train_sizes=(25, 50), n_seeds=2, standardize=True)

    assert plain["train_sizes"] == [25, 50]
    assert plain["angle0"]["mae_mean"][-1] < 0.2, "a clean circle is recovered without z-scoring"
    assert scaled["angle0"]["mae_mean"][-1] < 0.2


# ======================================================================================
# utils: the extras block and the wandb teardown
# ======================================================================================

def test_a_config_without_an_extras_block_is_a_no_op():
    """Every ``src/eval_*.py`` composes a config that may not carry ``extras``. Reaching into a
    missing key would take down a job at its first line."""
    from src.utils.utils import extras

    extras(OmegaConf.create({"model": {}}))          # must not raise


def test_each_extra_is_applied_independently(monkeypatch):
    """Three independent flags. Chaining them -- so that ``print_config`` only ran when
    ``enforce_tags`` was set -- would be invisible until someone wondered why their config tree
    was not printing."""
    import src.utils.utils as module

    called = []
    monkeypatch.setattr(module.rich_utils, "enforce_tags",
                        lambda cfg, save_to_file: called.append("tags"))
    monkeypatch.setattr(module.rich_utils, "print_config_tree",
                        lambda cfg, resolve, save_to_file: called.append("print"))

    module.extras(OmegaConf.create({"extras": {"enforce_tags": True, "print_config": False}}))
    assert called == ["tags"]

    called.clear()
    module.extras(OmegaConf.create({"extras": {"enforce_tags": False, "print_config": True}}))
    assert called == ["print"]


def test_ignoring_warnings_is_opt_in(monkeypatch):
    """A global ``filterwarnings("ignore")`` suppresses torch's own deprecation and correctness
    warnings for the rest of the process. It must happen only when the config asks."""
    import warnings as warnings_module

    import src.utils.utils as module

    filtered = []
    monkeypatch.setattr(warnings_module, "filterwarnings", lambda action: filtered.append(action))

    module.extras(OmegaConf.create({"extras": {"ignore_warnings": False}}))
    assert filtered == []

    module.extras(OmegaConf.create({"extras": {"ignore_warnings": True}}))
    assert filtered == ["ignore"]


def test_the_task_wrapper_closes_wandb_even_when_the_task_raises(monkeypatch):
    """A multirun whose first job crashes must not leave a wandb run open -- the remaining jobs
    would attach to it and log into the wrong run. The ``finally`` is the point."""
    import importlib.machinery
    import sys
    import types

    import src.utils.utils as module

    finished = []
    fake_wandb = types.ModuleType("wandb")
    # find_spec() raises ValueError on a module whose __spec__ is None, so the stand-in needs one
    fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    fake_wandb.run = object()
    fake_wandb.finish = lambda: finished.append(True)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = OmegaConf.create({"paths": {"output_dir": "/tmp/out"}, "extras": {}})

    @module.task_wrapper
    def failing_task(cfg):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_task(cfg=cfg)
    assert finished == [True]


def test_the_task_wrapper_returns_the_tasks_result_on_success(monkeypatch):
    import importlib.machinery
    import sys
    import types

    import src.utils.utils as module

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    fake_wandb.run = None
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = OmegaConf.create({"paths": {"output_dir": "/tmp/out"}, "extras": {}})

    @module.task_wrapper
    def ok_task(cfg):
        return {"loss": 1.0}, {"model": "m"}

    assert ok_task(cfg=cfg) == ({"loss": 1.0}, {"model": "m"})


# ======================================================================================
# the rank-aware logger
# ======================================================================================

def test_a_rank_zero_only_logger_stays_silent_on_other_ranks(monkeypatch, caplog):
    """DDP runs four processes. Without this, every log line appears four times and a progress
    log becomes unreadable -- and, worse, a 'saved checkpoint' line appears four times for one
    checkpoint."""
    from lightning.pytorch.utilities import rank_zero_only

    from src.utils.pylogger import RankedLogger

    logger = RankedLogger(__name__, rank_zero_only=True)

    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)
    with caplog.at_level(logging.INFO):
        logger.info("from rank zero")
    assert "from rank zero" in caplog.text

    caplog.clear()
    monkeypatch.setattr(rank_zero_only, "rank", 2, raising=False)
    with caplog.at_level(logging.INFO):
        logger.info("from rank two")
    assert caplog.text == ""


def test_a_rank_aware_logger_can_target_one_rank(monkeypatch, caplog):
    """``rank=`` selects a single rank -- used to dump per-rank diagnostics without the other
    three interleaving into them."""
    from lightning.pytorch.utilities import rank_zero_only

    from src.utils.pylogger import RankedLogger

    logger = RankedLogger(__name__, rank_zero_only=False)
    monkeypatch.setattr(rank_zero_only, "rank", 2, raising=False)

    with caplog.at_level(logging.INFO):
        logger.info("only on two", rank=2)
        logger.info("only on three", rank=3)

    assert "only on two" in caplog.text and "only on three" not in caplog.text


def test_every_rank_logs_when_no_rank_is_specified(monkeypatch, caplog):
    from lightning.pytorch.utilities import rank_zero_only

    from src.utils.pylogger import RankedLogger

    logger = RankedLogger(__name__, rank_zero_only=False)
    monkeypatch.setattr(rank_zero_only, "rank", 3, raising=False)

    with caplog.at_level(logging.INFO):
        logger.info("from everywhere")
    assert "from everywhere" in caplog.text


def test_the_message_is_prefixed_with_the_rank(monkeypatch, caplog):
    """So a line from an interleaved multi-rank log can be attributed. Without the prefix a
    four-way interleaving is unreadable exactly when it is most needed."""
    from lightning.pytorch.utilities import rank_zero_only

    from src.utils.pylogger import RankedLogger

    logger = RankedLogger(__name__, rank_zero_only=False)
    monkeypatch.setattr(rank_zero_only, "rank", 1, raising=False)

    with caplog.at_level(logging.INFO):
        logger.info("hello")
    assert "[rank: 1]" in caplog.text


def test_logging_before_the_rank_is_known_is_an_error(monkeypatch):
    """Rather than assuming rank 0. An unset rank means the distributed environment has not been
    initialised, and silently treating that as rank 0 would let all four ranks log as rank 0 --
    the exact confusion the class exists to prevent."""
    from lightning.pytorch.utilities import rank_zero_only

    from src.utils.pylogger import RankedLogger

    monkeypatch.setattr(rank_zero_only, "rank", None, raising=False)
    # ...at an enabled level: the guard sits inside `if self.isEnabledFor(level)`, so a call
    # below the logger's threshold returns silently whatever the rank is.
    with pytest.raises(RuntimeError, match="rank_zero_only.rank` needs to be set"):
        RankedLogger(__name__).error("anything")


# ======================================================================================
# hyperparameter logging without a logger
# ======================================================================================

def test_hyperparameters_are_skipped_when_no_logger_is_attached(caplog):
    """``trainer=cpu`` and the CPU integration tests run with ``logger=[]``. Reaching for
    ``trainer.logger.log_hyperparams`` there would crash a run that is otherwise fine."""
    from src.utils.logging_utils import log_hyperparameters

    trainer = type("T", (), {"logger": None})()
    object_dict = {"cfg": OmegaConf.create({"model": {}}), "model": nn.Linear(1, 1),
                   "trainer": trainer}

    with caplog.at_level(logging.WARNING):
        log_hyperparameters(object_dict)
    assert "Logger not found" in caplog.text

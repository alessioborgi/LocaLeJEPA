"""Project utilities inherited from lightning-hydra-template but used by every run.

``task_wrapper`` wraps the whole of training and ``get_metric_value`` feeds checkpoint selection,
so their failure behaviour matters more than their happy paths.
"""

from __future__ import annotations

import logging

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import extras, get_metric_value, task_wrapper


# --------------------------------------------------------------------------------------
# get_metric_value -- feeds checkpoint selection
# --------------------------------------------------------------------------------------

def test_returns_the_named_metric_as_a_float():
    assert get_metric_value({"val/acc": torch.tensor(0.75)}, "val/acc") == pytest.approx(0.75)


def test_returns_none_when_no_metric_is_requested():
    """``optimized_metric`` is null in every shipped config, so this is the common path."""
    assert get_metric_value({"val/acc": torch.tensor(1.0)}, None) is None
    assert get_metric_value({}, "") is None


def test_a_missing_metric_raises_rather_than_returning_none():
    """Silently returning None would make a hyper-parameter sweep optimise nothing while
    appearing to work."""
    with pytest.raises(Exception, match="Metric value not found"):
        get_metric_value({"val/acc": torch.tensor(1.0)}, "val/does_not_exist")


# --------------------------------------------------------------------------------------
# task_wrapper -- wraps the whole run
# --------------------------------------------------------------------------------------

def test_wrapper_passes_through_the_return_value():
    @task_wrapper
    def task(cfg):
        return {"loss": 1.0}, {"model": "x"}

    metrics, objects = task(DictConfig({"paths": {"output_dir": "/tmp"}}))
    assert metrics == {"loss": 1.0} and objects == {"model": "x"}


def test_wrapper_re_raises_the_original_exception():
    """Swallowing it would let a crashed run look like a completed one -- and with the artifact
    absent, a downstream stage would then fail somewhere unrelated."""
    @task_wrapper
    def task(cfg):
        raise ValueError("training blew up")

    with pytest.raises(ValueError, match="training blew up"):
        task(DictConfig({"paths": {"output_dir": "/tmp"}}))


def test_wrapper_logs_the_output_dir_even_when_the_task_fails(caplog):
    """The ``finally`` block runs on both paths, so the output directory is always reported --
    which is how a failed run is located afterwards."""
    @task_wrapper
    def task(cfg):
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            task(DictConfig({"paths": {"output_dir": "/tmp/somewhere"}}))
    assert any("/tmp/somewhere" in record.message for record in caplog.records)


# --------------------------------------------------------------------------------------
# extras
# --------------------------------------------------------------------------------------

def test_extras_returns_early_without_an_extras_block(caplog):
    with caplog.at_level(logging.WARNING):
        extras(DictConfig({}))
    assert any("Extras config not found" in r.message for r in caplog.records)


def test_extras_can_suppress_warnings():
    import warnings

    extras(DictConfig({"extras": {"ignore_warnings": True, "enforce_tags": False,
                                  "print_config": False}}))
    assert warnings.filters, "a filter must have been installed"


def test_extras_skips_every_optional_step_when_all_are_false():
    extras(DictConfig({"extras": {"ignore_warnings": False, "enforce_tags": False,
                                  "print_config": False}}))


# --------------------------------------------------------------------------------------
# instantiators
# --------------------------------------------------------------------------------------

def test_no_callbacks_gives_an_empty_list(caplog):
    with caplog.at_level(logging.WARNING):
        assert instantiate_callbacks(None) == []
    assert any("No callback configs" in r.message for r in caplog.records)


def test_a_non_dictconfig_callbacks_block_raises():
    with pytest.raises(TypeError, match="Callbacks config must be a DictConfig"):
        instantiate_callbacks(["not", "a", "config"])


def test_entries_without_a_target_are_skipped():
    """``configs/callbacks/default.yaml`` mixes real callbacks with plain scalars, so the filter
    is load-bearing rather than defensive."""
    cfg = OmegaConf.create({
        "model_summary": {"_target_": "lightning.pytorch.callbacks.RichModelSummary"},
        "some_scalar": 3,
    })
    callbacks = instantiate_callbacks(cfg)
    assert len(callbacks) == 1


def test_no_loggers_gives_an_empty_list():
    assert instantiate_loggers(None) == []


def test_a_non_dictconfig_logger_block_raises():
    with pytest.raises(TypeError, match="Logger config must be a DictConfig"):
        instantiate_loggers(["csv"])


def test_a_logger_is_instantiated_from_its_target(tmp_path):
    cfg = OmegaConf.create({
        "csv": {"_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                "save_dir": str(tmp_path)}
    })
    loggers = instantiate_loggers(cfg)
    assert len(loggers) == 1


# --------------------------------------------------------------------------------------
# RankedLogger
# --------------------------------------------------------------------------------------

def test_ranked_logger_prefixes_messages_with_the_rank(caplog):
    logger = RankedLogger(__name__, rank_zero_only=False)
    with caplog.at_level(logging.INFO):
        logger.info("hello")
    assert any("rank" in r.message for r in caplog.records)


def test_ranked_logger_can_target_a_single_rank(caplog):
    """Used to keep multi-rank DDP logs readable; on a single process rank 0 is the only one
    that should emit."""
    logger = RankedLogger(__name__, rank_zero_only=False)
    with caplog.at_level(logging.INFO):
        logger.info("only for rank 1", rank=1)
    assert not [r for r in caplog.records if "only for rank 1" in r.message]

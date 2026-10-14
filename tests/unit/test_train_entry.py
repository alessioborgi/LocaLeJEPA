"""``src/train.py``'s ``train``, with every Hydra-instantiated object replaced by a recorder:
fit from the given checkpoint, test the best checkpoint, return the merged metrics."""

from omegaconf import OmegaConf

import src.train as module


class FakeTrainer:
    def __init__(self, **kwargs):
        self.fit_calls = []
        self.test_calls = []
        self.checkpoint_callback = type("C", (), {"best_model_path": "best.ckpt"})()
        self.callback_metrics = {}

    def fit(self, model, datamodule, ckpt_path):
        self.fit_calls.append(ckpt_path)
        self.callback_metrics = {"train/loss": 1.0}

    def test(self, model, datamodule, ckpt_path):
        self.test_calls.append(ckpt_path)
        self.callback_metrics = {**self.callback_metrics, "test/acc": 0.5}


def test_a_full_run_fits_then_tests_the_best_checkpoint_and_merges_metrics(monkeypatch):
    built = {}

    def instantiate(cfg, **kwargs):
        if cfg._target_ == "trainer":
            built["trainer"] = FakeTrainer(**kwargs)
            return built["trainer"]
        return cfg._target_

    monkeypatch.setattr(module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(module, "instantiate_callbacks", lambda cfg: [])
    monkeypatch.setattr(module, "instantiate_loggers", lambda cfg: [])
    cfg = OmegaConf.create(
        {
            "seed": None,
            "data": {"_target_": "data"},
            "model": {"_target_": "model"},
            "trainer": {"_target_": "trainer"},
            "callbacks": None,
            "logger": None,
            "train": True,
            "test": True,
            "ckpt_path": "resume.ckpt",
            "paths": {"output_dir": "/tmp/out"},
        }
    )

    metrics, objects = module.train(cfg)

    assert built["trainer"].fit_calls == ["resume.ckpt"]
    assert built["trainer"].test_calls == ["best.ckpt"]
    assert metrics == {"train/loss": 1.0, "test/acc": 0.5}
    assert objects["model"] == "model" and objects["datamodule"] == "data"

"""``load_backbone`` against a checkpoint it actually has to load.

The unit tests in ``tests/unit/test_eval_extract.py`` check the prefix-stripping rule by
re-deriving it; useful, but they never execute ``load_backbone``, so they cannot catch a defect in
how it composes the config, resolves the encoder, or applies the strict-load check. These do: they
build a real encoder from a real experiment config, save a Lightning-shaped checkpoint of its own
weights, and load it back.

Everything runs on CPU with ``vit_tiny_patch16_224`` at crop 64 -- a 5.5M-parameter model over 16
patches, which keeps the whole file under a few seconds. The architecture is overridden rather
than using the campaign's ViT-S/8 at 128 because nothing here depends on the size; what is under
test is the loading path.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.eval.extract import extract_features, load_backbone, vit_spatial_patch_tokens


def closed_over(fn, name):
    """Reach a closed-over local of ``extract_fn`` by name.

    ``load_backbone`` returns only the callable, so the encoder it built is otherwise
    unreachable. Looked up through ``co_freevars`` rather than by closure index, so that adding
    an unrelated local to ``extract_fn`` does not silently point these tests at the wrong object.
    """
    inner = getattr(fn, "__wrapped__", fn)   # extract_fn is wrapped in @torch.no_grad()
    index = inner.__code__.co_freevars.index(name)
    return inner.__closure__[index].cell_contents

EXPERIMENT = "lejepa/vit_imagenette_sigreg"
CROP = 64
TINY = ["model.encoder.model_name=vit_tiny_patch16_224", "model.embedding_dim=32"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A checkpoint written the way ``src/train.py`` writes one: every encoder parameter under
    an ``encoder.`` prefix, alongside projector and probe weights that must be ignored."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("PROJECT_ROOT", root)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=os.path.join(root, "configs")):
        cfg = compose(config_name="train",
                      overrides=[f"experiment={EXPERIMENT}", f"data.crop_size={CROP}", *TINY])
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    GlobalHydra.instance().clear()

    state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    state["projector.0.weight"] = torch.zeros(4, 32)      # must be ignored, not rejected
    state["online_probe.weight"] = torch.zeros(10, 32)

    path = tmp_path_factory.mktemp("ckpt") / "last.ckpt"
    torch.save({"state_dict": state, "epoch": 99, "global_step": 1234}, path)
    return str(path)


# --------------------------------------------------------------------------------------
# the encoder feature mode
# --------------------------------------------------------------------------------------

def test_a_saved_encoder_loads_back_and_extracts_embeddings(checkpoint):
    """The round trip. ``embedding_dim`` must come from the config, and the extractor must
    produce exactly that width -- the number every probe downstream sizes itself against."""
    extract_fn, embedding_dim = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                              crop_size=CROP, overrides=TINY)
    assert embedding_dim == 32

    features = extract_fn(torch.randn(3, 3, CROP, CROP))
    assert features.shape == (3, 32)
    assert features.dtype == torch.float32, "features are cast to float32 regardless of AMP"
    assert torch.isfinite(features).all()


def test_the_loaded_encoder_is_frozen_and_in_eval_mode(checkpoint):
    """Three separate things, each of which silently corrupts an evaluation on its own.

    ``requires_grad_(False)`` and ``eval()`` are asserted through their observable consequences:
    the output carries no grad_fn, and two calls on the same input agree exactly. The second is
    the one that matters for ViT specifically -- ``drop_path_rate`` is 0.1 in this config, so a
    backbone left in train mode would return a *different* feature vector each time it saw the
    same image, and every probe would be fitted on noise.
    """
    extract_fn, _ = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                  crop_size=CROP, overrides=TINY)
    images = torch.randn(2, 3, CROP, CROP)

    first = extract_fn(images)
    assert first.grad_fn is None and not first.requires_grad

    torch.testing.assert_close(first, extract_fn(images), rtol=0, atol=0)


def test_the_weights_that_come_back_are_the_weights_that_went_in(checkpoint):
    """Not merely 'a model of the right shape loaded'. A silently-randomised encoder would pass
    every shape assertion above and produce plausible-looking, meaningless features.

    Compared against a second encoder built from the same checkpoint by an independent path.
    """
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    extract_fn, _ = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                  crop_size=CROP, overrides=TINY)

    # extract_fn closes over the encoder; reach it through the closure rather than re-building
    encoder = closed_over(extract_fn, "encoder")
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, loaded[f"encoder.{key}"], rtol=0, atol=0)


def test_a_checkpoint_from_a_different_architecture_is_refused(checkpoint, tmp_path):
    """Strict loading, stated as a consequence: the alternative is a partially-initialised
    encoder that runs and produces numbers. The error must name the counts so the failure is
    diagnosable from a log."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dropped = next(k for k in state["state_dict"] if k.startswith("encoder.blocks.0"))
    del state["state_dict"][dropped]
    broken = tmp_path / "broken.ckpt"
    torch.save(state, broken)

    with pytest.raises(RuntimeError, match="state_dict mismatch"):
        load_backbone(EXPERIMENT, str(broken), torch.device("cpu"), crop_size=CROP, overrides=TINY)


def test_a_torch_compiled_checkpoint_loads_unchanged(checkpoint, tmp_path):
    """``torch.compile`` wraps the module, so every key gains ``_orig_mod.``. A run launched with
    compilation on must stay loadable by an evaluation launched without it."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["state_dict"] = {
        (f"encoder._orig_mod.{k[len('encoder.'):]}" if k.startswith("encoder.") else k): v
        for k, v in state["state_dict"].items()
    }
    compiled = tmp_path / "compiled.ckpt"
    torch.save(state, compiled)

    extract_fn, _ = load_backbone(EXPERIMENT, str(compiled), torch.device("cpu"),
                                  crop_size=CROP, overrides=TINY)
    assert extract_fn(torch.randn(1, 3, CROP, CROP)).shape == (1, 32)


def test_an_unknown_feature_mode_raises_before_any_weight_is_read(checkpoint):
    """The check sits after ``compose`` but before ``torch.load``, which is the right order: a
    typo costs a config composition, not a checkpoint read from a network filesystem."""
    with pytest.raises(ValueError, match="Unknown feature mode"):
        load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=CROP,
                      overrides=TINY, feature="embeddings")


# --------------------------------------------------------------------------------------
# the patch feature mode
# --------------------------------------------------------------------------------------

def test_patch_mode_returns_spatial_tokens_at_the_grid_resolution(checkpoint):
    """16x16 patches at crop 64 with patch size 16 -> 4x4 = 16 tokens, and ``embedding_dim``
    becomes the ViT width rather than the projection head's output."""
    extract_fn, embedding_dim = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                              crop_size=CROP, overrides=TINY, feature="patch")
    assert embedding_dim == 192, "ViT-Tiny's width, not the configured embedding_dim of 32"

    tokens = extract_fn(torch.randn(2, 3, CROP, CROP))
    assert tokens.shape == (2, (CROP // 16) ** 2, 192)


def test_patch_mode_excludes_the_prefix_tokens(checkpoint):
    """The CLS token is not a spatial location. Including it would shift every patch index by
    one and put a global summary vector into a map that is read as a grid."""
    extract_fn, _ = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                  crop_size=CROP, overrides=TINY, feature="patch")
    body = closed_over(extract_fn, "patch_body")

    with torch.no_grad():
        raw = body.forward_features(torch.randn(1, 3, CROP, CROP))
    spatial = extract_fn(torch.randn(1, 3, CROP, CROP))

    assert raw.shape[1] - spatial.shape[1] == body.num_prefix_tokens
    assert body.num_prefix_tokens >= 1, "this ViT has a CLS token, so the test has something to do"


def test_patch_extraction_refuses_a_backbone_without_a_token_grid(checkpoint):
    """ResNets have no ``patch_embed``; asking for patch tokens from one must fail at load time
    rather than returning a pooled vector that is silently treated as a 1x1 map."""
    with pytest.raises(ValueError, match="Patch extraction requires"):
        load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=CROP,
                      overrides=["model.embedding_dim=32", "model/encoder=resnet18",
                                 "~model.encoder.model_name"],
                      feature="patch")


class NotAViT(torch.nn.Module):
    """Minimal stand-ins for the two shapes ``vit_spatial_patch_tokens`` must reject."""

    def __init__(self, tokens, grid=(4, 4), num_prefix_tokens=1):
        super().__init__()
        self._tokens = tokens
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_embed = type("PE", (), {"grid_size": grid})()

    def forward_features(self, images):
        return self._tokens


def test_token_extraction_rejects_a_pooled_output():
    """(B, D) instead of (B, N, D): a backbone that pools internally. Caught explicitly, because
    the slice that follows would happily produce garbage from a 2-D tensor."""
    with pytest.raises(ValueError, match=r"\(B,N,D\) tokens"):
        vit_spatial_patch_tokens(NotAViT(torch.randn(2, 192)), torch.randn(2, 3, CROP, CROP))


def test_token_extraction_rejects_a_token_count_that_is_not_the_grid():
    """The count check is what ties the tokens to a spatial layout. Without it a resolution
    mismatch would reshape 17 tokens into a 4x4 map and drop one, silently."""
    with pytest.raises(ValueError, match="Expected 16 spatial tokens, got 8"):
        vit_spatial_patch_tokens(NotAViT(torch.randn(2, 9, 192)), torch.randn(2, 3, CROP, CROP))


# --------------------------------------------------------------------------------------
# extract_features and its cache
# --------------------------------------------------------------------------------------

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=7, dim=4):
        self.images = torch.randn(n, 3, 8, 8)
        self.labels = torch.arange(n) % 3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def flatten_fn(images):
    return images.flatten(1)[:, :4].float()


def test_features_are_extracted_in_dataset_order_across_batches():
    """``shuffle=False`` and ``drop_last=False``, both load-bearing: features and labels are
    paired positionally downstream, and a dropped final partial batch would silently shrink the
    evaluation set. Checked with 7 samples at batch size 3, so the last batch is partial."""
    dataset = TinyDataset(n=7)
    X, y = extract_features(flatten_fn, dataset, batch_size=3, num_workers=0)

    assert X.shape == (7, 4) and y.shape == (7,)
    torch.testing.assert_close(X, flatten_fn(dataset.images), rtol=0, atol=0)
    torch.testing.assert_close(y, dataset.labels.float(), rtol=0, atol=0)


def test_the_cache_is_written_once_and_read_back_thereafter(tmp_path):
    """The cache is what makes a multi-backbone sweep affordable. It must round-trip exactly --
    a lossy cache would make the cached run and the uncached run report different numbers."""
    dataset = TinyDataset(n=5)
    cache = tmp_path / "nested" / "feats.pt"

    X_first, y_first = extract_features(flatten_fn, dataset, batch_size=2, num_workers=0,
                                        cache_path=str(cache))
    assert cache.is_file(), "the parent directory must be created, not assumed"

    def must_not_be_called(images):
        raise AssertionError("the cache was not used")

    X_second, y_second = extract_features(must_not_be_called, dataset, batch_size=2,
                                          num_workers=0, cache_path=str(cache))
    torch.testing.assert_close(X_first, X_second, rtol=0, atol=0)
    torch.testing.assert_close(y_first, y_second, rtol=0, atol=0)


def test_a_non_tensor_target_is_still_collected(tmp_path):
    """Some torchvision datasets yield tuples or lists of targets rather than a tensor; the
    ``else`` branch handles them and is otherwise never taken by the ImageFolder path."""

    class TupleTargets(TinyDataset):
        def __getitem__(self, index):
            return self.images[index], [float(self.labels[index]), 0.5]

    X, y = extract_features(flatten_fn, TupleTargets(n=4), batch_size=2, num_workers=0)
    assert X.shape == (4, 4) and y.shape == (4, 2)


# --------------------------------------------------------------------------------------
# the projector loader (src/eval/rankme.py), which mirrors load_backbone for projector.* keys
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def projector_checkpoint(tmp_path_factory):
    """A checkpoint carrying both encoder and projector weights -- which is what ``train.py``
    actually writes, and what makes the two loaders' prefix filtering non-trivial."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("PROJECT_ROOT", root)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=os.path.join(root, "configs")):
        cfg = compose(config_name="train",
                      overrides=[f"experiment={EXPERIMENT}", f"data.crop_size={CROP}", *TINY])
    projector = hydra.utils.instantiate(cfg.model.projector)
    GlobalHydra.instance().clear()

    state = {f"projector.{k}": v for k, v in projector.state_dict().items()}
    state["encoder.blocks.0.norm1.weight"] = torch.zeros(192)   # must be ignored
    state["online_probe.weight"] = torch.zeros(10, 32)

    path = tmp_path_factory.mktemp("proj") / "last.ckpt"
    torch.save({"state_dict": state, "epoch": 99}, path)
    return str(path)


def test_the_projector_loads_back_from_a_mixed_checkpoint(projector_checkpoint):
    """The filtering is what is under test: encoder and probe keys share the checkpoint, and a
    prefix match that was too loose would hand them to ``load_state_dict`` as unexpected keys and
    fail the strict check -- or, worse, a match that was too tight would load nothing and pass."""
    from src.eval.rankme import load_projector

    projector, rectify = load_projector(EXPERIMENT, projector_checkpoint, crop_size=CROP,
                                        overrides=TINY)
    assert rectify is False, "plain LeJEPA does not rectify"

    output = projector(torch.randn(4, 32))
    assert output.shape[0] == 4 and torch.isfinite(output).all()


def test_the_loaded_projector_is_frozen_and_in_eval_mode(projector_checkpoint):
    from src.eval.rankme import load_projector

    projector, _ = load_projector(EXPERIMENT, projector_checkpoint, crop_size=CROP,
                                  overrides=TINY)
    assert not projector.training
    assert all(not p.requires_grad for p in projector.parameters())


def test_the_projector_weights_are_the_checkpoints(projector_checkpoint):
    from src.eval.rankme import load_projector

    saved = torch.load(projector_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    projector, _ = load_projector(EXPERIMENT, projector_checkpoint, crop_size=CROP,
                                  overrides=TINY)

    for key, value in projector.state_dict().items():
        torch.testing.assert_close(value, saved[f"projector.{key}"], rtol=0, atol=0)


def test_a_projector_checkpoint_from_a_different_width_is_refused(projector_checkpoint, tmp_path):
    """The projector's width is ``model.projection_dim``, which the campaign varies. Loading a
    16-dim projector's weights into a 32-dim one must fail rather than partially succeed."""
    from src.eval.rankme import load_projector

    state = torch.load(projector_checkpoint, map_location="cpu", weights_only=False)
    dropped = next(k for k in state["state_dict"] if k.startswith("projector."))
    del state["state_dict"][dropped]
    broken = tmp_path / "broken_proj.ckpt"
    torch.save(state, broken)

    with pytest.raises(RuntimeError, match="Projector state_dict mismatch"):
        load_projector(EXPERIMENT, str(broken), crop_size=CROP, overrides=TINY)


def test_a_compiled_projector_checkpoint_loads(projector_checkpoint, tmp_path):
    from src.eval.rankme import load_projector

    state = torch.load(projector_checkpoint, map_location="cpu", weights_only=False)
    state["state_dict"] = {
        (f"projector._orig_mod.{k[len('projector.'):]}" if k.startswith("projector.") else k): v
        for k, v in state["state_dict"].items()
    }
    compiled = tmp_path / "compiled_proj.ckpt"
    torch.save(state, compiled)

    projector, _ = load_projector(EXPERIMENT, str(compiled), crop_size=CROP, overrides=TINY)
    assert projector(torch.randn(2, 32)).shape[0] == 2


def test_rectification_is_read_from_the_experiment_config(projector_checkpoint):
    """``rectify`` decides whether RankMe is computed on the projector output or on its ReLU --
    two different matrices with genuinely different ranks. It must come from the config the
    checkpoint was trained under, not from a caller's assumption."""
    from src.eval.rankme import load_projector

    _, rectify = load_projector(EXPERIMENT, projector_checkpoint, crop_size=CROP,
                                overrides=[*TINY, "+model.rectify=true"])
    assert rectify is True


def test_features_are_projected_in_batches_without_changing_the_result():
    """The batching exists only to bound memory on a 1.28M-row feature matrix, so splitting must
    not change the answer -- asserted at a batch size that forces a partial final chunk.

    NOT bit-exact, and the tolerance says which is which. Measured: 8.9e-8 absolute, 3.9e-6
    relative, because a (7, 8) matmul and a (23, 8) matmul take different blocking paths in the
    BLAS kernel. That must be tolerated. What must be caught -- a dropped final chunk, a wrong
    offset, a batch written back out of order -- moves entries by O(1), four orders of magnitude
    above this bound.
    """
    from src.eval.rankme import project_features

    projector = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.GELU(), torch.nn.Linear(4, 4))
    projector.eval()
    X = torch.randn(23, 8)

    whole = project_features(projector, X, rectify=False, batch_size=10_000)
    chunked = project_features(projector, X, rectify=False, batch_size=7)
    torch.testing.assert_close(whole, chunked, rtol=1e-5, atol=1e-6)
    assert whole.shape == (23, 4), "no row dropped by the partial final chunk"


def test_projection_applies_the_relu_when_rectifying():
    """Rectified LpJEPA's losses see ``relu(projector(.))``, so RankMe must be computed on the
    same thing -- otherwise the reported rank is of a representation the model was never trained
    to shape."""
    from src.eval.rankme import project_features

    projector = torch.nn.Linear(8, 4)
    X = torch.randn(16, 8)

    rectified = project_features(projector, X, rectify=True)
    assert float(rectified.min()) >= 0.0
    torch.testing.assert_close(rectified, torch.relu(project_features(projector, X, rectify=False)))


def test_projection_does_not_build_a_graph():
    """``@torch.no_grad()``. Over a full ImageNet feature matrix an accumulated graph is the
    difference between a job that fits in memory and one that does not."""
    from src.eval.rankme import project_features

    out = project_features(torch.nn.Linear(8, 4), torch.randn(16, 8), rectify=False)
    assert out.grad_fn is None and not out.requires_grad


# --------------------------------------------------------------------------------------
# positional-embedding resampling: patch features at a resolution the checkpoint never saw
# --------------------------------------------------------------------------------------

def test_patch_features_can_be_extracted_at_a_different_resolution(checkpoint):
    """THE REASON THIS BRANCH EXISTS.

    The checkpoint was trained at crop 64 (a 4x4 token grid). Patch-PCA visualisations want more
    spatial detail, so they extract at a larger crop -- here 96, an 6x6 grid. The learned
    positional embedding has one row per training-grid position and simply does not fit.

    ``resample_abs_pos_embed`` interpolates it onto the new grid. Without that step the strict
    ``load_state_dict`` check rejects the checkpoint, so the failure is at least loud -- but the
    tempting 'fix' is to drop strictness, which would load a *randomly initialised* positional
    embedding and produce a patch map that looks plausible and encodes nothing.
    """
    extract_fn, embedding_dim = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                              crop_size=96, overrides=TINY, feature="patch")

    tokens = extract_fn(torch.randn(1, 3, 96, 96))
    assert tokens.shape == (1, (96 // 16) ** 2, embedding_dim)
    assert torch.isfinite(tokens).all()


def test_resampling_preserves_the_prefix_token_rows(checkpoint):
    """``num_prefix_tokens`` is passed to the resampler so the CLS row is carried over rather
    than interpolated together with the spatial grid. Interpolating it in would blend a global
    token into the corner patches -- invisible in any shape, visible only as a patch map with a
    strange corner."""
    at_native = closed_over(
        load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=CROP,
                      overrides=TINY, feature="patch")[0], "patch_body")
    at_larger = closed_over(
        load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=96,
                      overrides=TINY, feature="patch")[0], "patch_body")

    assert at_native.num_prefix_tokens == at_larger.num_prefix_tokens == 1
    torch.testing.assert_close(at_native.pos_embed[:, 0], at_larger.pos_embed[:, 0],
                               rtol=1e-4, atol=1e-5)


def test_the_same_resolution_needs_no_resampling(checkpoint):
    """The shapes match, so the branch is skipped and the embedding is loaded verbatim. Worth
    asserting: an unconditional resample would interpolate the grid onto itself and perturb every
    row for no reason, making the native-resolution result depend on the interpolation kernel."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    body = closed_over(
        load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=CROP,
                      overrides=TINY, feature="patch")[0], "patch_body")

    torch.testing.assert_close(body.pos_embed, saved["encoder.pos_embed"], rtol=0, atol=0)


def test_a_non_square_checkpoint_grid_is_refused(checkpoint, tmp_path):
    """The resampler needs the source grid's side length, recovered as ``sqrt(count)``. A
    checkpoint whose token count is not a perfect square cannot be reshaped into a grid at all,
    and guessing would silently misplace every patch."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pos = state["state_dict"]["encoder.pos_embed"]
    state["state_dict"]["encoder.pos_embed"] = pos[:, :-1]      # 17 rows -> 16 spatial, +1 prefix
    state["state_dict"]["encoder.pos_embed"] = torch.cat(
        [pos[:, :1], pos[:, 1:1 + 7]], dim=1)                   # 7 spatial tokens: not a square
    oblong = tmp_path / "oblong.ckpt"
    torch.save(state, oblong)

    with pytest.raises(ValueError, match="square checkpoint positional grid"):
        load_backbone(EXPERIMENT, str(oblong), torch.device("cpu"), crop_size=96,
                      overrides=TINY, feature="patch")


def test_an_encoder_that_does_not_return_embeddings_is_caught_at_call_time(checkpoint,
                                                                          monkeypatch):
    """``num_classes`` must equal ``embedding_dim`` for LeJEPA -- an encoder left in its default
    classification configuration returns logits, and one left headless returns tokens.

    The check is on ``ndim``, so it fires on the (B, N, D) case. Provoked by replacing the loaded
    encoder's forward, since no config in the repo produces a 3-D encoder output -- which is
    exactly why the branch is otherwise unreachable.
    """
    extract_fn, _ = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                  crop_size=CROP, overrides=TINY)
    encoder = closed_over(extract_fn, "encoder")
    monkeypatch.setattr(encoder, "forward", lambda images: torch.randn(images.shape[0], 5, 32))

    with pytest.raises(RuntimeError, match=r"Expected encoder to return \(B, D\) embeddings"):
        extract_fn(torch.randn(2, 3, CROP, CROP))


# --------------------------------------------------------------------------------------
# the cls2 feature mode (campaign C4: LeJEPA's probe input)
# --------------------------------------------------------------------------------------

def test_cls2_mode_concatenates_the_last_two_class_tokens(checkpoint):
    """vit_tiny has 192 features per block; two class tokens make 384. Checked against a manual
    ``forward_intermediates`` call, so the mode is pinned to timm's own definition of "the last
    two blocks, post-norm", not to whatever the extractor happens to return."""
    extract_fn, dim = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"),
                                    crop_size=CROP, overrides=TINY, feature="cls2")
    body = closed_over(extract_fn, "cls_body")
    assert dim == 2 * body.num_features == 384

    images = torch.randn(3, 3, CROP, CROP)
    feats = extract_fn(images)
    assert feats.shape == (3, 384)
    outs = body.forward_intermediates(images, indices=2, return_prefix_tokens=True, norm=True,
                                      output_fmt="NLC", intermediates_only=True)
    manual = torch.cat([outs[0][1][:, 0], outs[1][1][:, 0]], dim=1)
    torch.testing.assert_close(feats, manual, rtol=1e-5, atol=1e-5)


def test_cls2_is_a_different_representation_from_the_encoder_output(checkpoint):
    """It bypasses the trained ``embedding_dim`` head entirely, so it must not be mistaken for a
    superset of the encoder features: widths differ and the values are not a prefix of each other."""
    enc_fn, enc_dim = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=CROP, overrides=TINY)
    cls_fn, cls_dim = load_backbone(EXPERIMENT, checkpoint, torch.device("cpu"), crop_size=CROP,
                                    overrides=TINY, feature="cls2")
    assert (enc_dim, cls_dim) == (32, 384)
    images = torch.randn(2, 3, CROP, CROP)
    assert not torch.allclose(cls_fn(images)[:, :32], enc_fn(images))


def test_cls2_refuses_a_backbone_without_a_class_token(checkpoint):
    with pytest.raises(ValueError, match="cls2 extraction requires"):
        load_backbone("lejepa/rn18_stl10_ep500", checkpoint, torch.device("cpu"), crop_size=CROP,
                      overrides=["model.embedding_dim=32"], feature="cls2")

"""Spatial-feature PCA colouring.

This module is validation-dense by design and has one reproducibility contract worth pinning:
PCA component signs are otherwise arbitrary, so it fixes them deterministically. Without that,
the same features would render with inverted colours from run to run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.patch_pca import load_spatial_feature_archive, pca_colors


def features(images: int = 2, patches: int = 16, channels: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(images, patches, channels))


# --------------------------------------------------------------------------------------
# archive loading -- every raise
# --------------------------------------------------------------------------------------

def test_archive_round_trips(tmp_path):
    path = tmp_path / "f.npz"
    np.savez(path, features=features(2, 12, 5), grid=np.array([3, 4]))
    loaded, grid = load_spatial_feature_archive(path)
    assert loaded.shape == (2, 12, 5) and grid == (3, 4)


def test_archive_missing_arrays_raise(tmp_path):
    path = tmp_path / "f.npz"
    np.savez(path, features=features())
    with pytest.raises(ValueError, match="requires features and grid"):
        load_spatial_feature_archive(path)


@pytest.mark.parametrize("grid", [np.array([4]), np.array([[2, 2]]), np.array([0, 4]),
                                  np.array([2.0, 2.0])])
def test_archive_rejects_a_malformed_grid(tmp_path, grid):
    path = tmp_path / "f.npz"
    np.savez(path, features=features(1, 4, 5), grid=grid)
    with pytest.raises(ValueError, match="two positive integers"):
        load_spatial_feature_archive(path)


def test_archive_rejects_features_that_do_not_match_the_grid(tmp_path):
    """3x4 = 12 patches; supplying 16 means the spatial layout is wrong, and every rendered
    image would be silently scrambled rather than failing."""
    path = tmp_path / "f.npz"
    np.savez(path, features=features(1, 16, 5), grid=np.array([3, 4]))
    with pytest.raises(ValueError, match="matching grid"):
        load_spatial_feature_archive(path)


# --------------------------------------------------------------------------------------
# colouring -- validation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.zeros((4, 8)),                              # not 3-D
    np.zeros((2, 8, 2)),                           # fewer than 3 channels
])
def test_rejects_malformed_features(bad):
    with pytest.raises(ValueError, match="finite \\(images,patches,channels>=3\\)"):
        pca_colors(bad)


def test_rejects_non_finite_features():
    """A NaN would propagate into every colour rather than failing."""
    f = features()
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pca_colors(f)


@pytest.mark.parametrize("scope, foreground", [("bogus", "high"), ("image", "sideways")])
def test_rejects_an_invalid_scope_or_foreground(scope, foreground):
    with pytest.raises(ValueError, match="Invalid PCA scope or foreground"):
        pca_colors(features(), scope=scope, foreground=foreground)


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_rejects_a_threshold_outside_zero_one(threshold):
    """PC1 is normalised to [0, 1] before thresholding, so anything outside that range would
    silently select everything or nothing."""
    with pytest.raises(ValueError, match="threshold must be in"):
        pca_colors(features(), threshold=threshold)


def test_rejects_constant_features():
    """Zero variance means no principal direction exists; refusing is better than returning an
    arbitrary rotation of nothing."""
    with pytest.raises(ValueError, match="nonconstant"):
        pca_colors(np.ones((1, 8, 4)))


def test_rejects_too_few_patches():
    with pytest.raises(ValueError, match="at least 3 patches"):
        pca_colors(features(images=1, patches=2, channels=5))


# --------------------------------------------------------------------------------------
# colouring -- behaviour
# --------------------------------------------------------------------------------------

def test_output_shapes_and_ranges():
    rgb, pc1, mask, records = pca_colors(features(3, 16, 8))
    assert rgb.shape == (3, 16, 3) and pc1.shape == (3, 16) and mask.shape == (3, 16)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0, "colours must be a valid RGB range"
    assert pc1.min() >= 0.0 and pc1.max() <= 1.0, "PC1 is normalised to [0, 1]"


def test_component_signs_are_deterministic():
    """THE reproducibility contract. PCA signs are arbitrary up to a flip, so the module pins
    them by the largest loading. Two runs on identical input must render identically -- without
    this, the same figure would come out colour-inverted at random.
    """
    f = features(seed=1)
    first, _, _, _ = pca_colors(f)
    second, _, _, _ = pca_colors(f)
    np.testing.assert_array_equal(first, second)


def test_image_scope_fits_each_image_independently():
    """Per-image PCA means changing one image cannot alter another's colours. Group scope must
    not have that property -- that is the difference between the two modes."""
    f = features(3, 16, 8, seed=2)
    per_image, _, _, records = pca_colors(f, scope="image")
    assert len(records) == 3, "one record per image"

    f2 = f.copy()
    f2[2] *= 10.0
    per_image_2, _, _, _ = pca_colors(f2, scope="image")
    np.testing.assert_allclose(per_image[0], per_image_2[0])


def test_group_scope_shares_one_fit_across_images():
    f = features(3, 16, 8, seed=2)
    _, _, _, records = pca_colors(f, scope="group")
    assert len(records) == 1, "a single shared fit"
    assert records[0]["image_indices"] == [0, 1, 2]


def test_thresholding_masks_patches_and_blacks_them_out():
    """Masked-out patches are set to zero rather than left at whatever the PCA produced, so a
    reader cannot mistake background for a low-variance foreground."""
    rgb, _, mask, records = pca_colors(features(2, 32, 8, seed=3), threshold=0.5)
    assert not mask.all(), "a mid threshold must exclude something"
    np.testing.assert_array_equal(rgb[~mask], 0.0)
    assert 0.0 < records[0]["foreground_fraction"] < 1.0


def test_foreground_side_selects_the_opposite_patches():
    """high/low must be complementary: whatever one keeps, the other drops."""
    f = features(1, 32, 8, seed=4)
    _, _, high, _ = pca_colors(f, threshold=0.5, foreground="high")
    _, _, low, _ = pca_colors(f, threshold=0.5, foreground="low")
    assert not (high & low).all()
    assert (high | low).all(), "between them they must cover every patch"


def test_records_carry_enough_to_reproduce_the_projection():
    """The record is what a figure caption would have to cite: components, mean, explained
    variance and the colour range actually used."""
    _, _, _, records = pca_colors(features(2, 16, 8))
    record = records[0]
    assert {"explained_variance_ratio", "components", "mean", "rgb_min", "rgb_max",
            "foreground_fraction", "mask_pc1_components"} <= set(record)
    assert len(record["explained_variance_ratio"]) == 3

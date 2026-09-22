"""The topology metrics must respond to the property each one claims to measure.

Every case below is a point cloud whose topology is known by construction, so a metric
that moves on the wrong case, or fails to move on its own, is broken rather than merely
uninformative.
"""

import numpy as np
import pytest

from src.eval.topology import TopologyOptions, analyse

OBJECTS = 12
POSES = 72
DIM = 16
OPTIONS = TopologyOptions(holdout_objects=4)


def synthetic(kind: str, objects: int = OBJECTS, poses: int = POSES, dim: int = DIM, spread: float = 6.0, seed: int = 0):
    """Circles in one shared plane, circles in unrelated planes, blobs, or one Gaussian."""
    rng = np.random.default_rng(seed)
    angles = np.tile(np.linspace(0, 2 * np.pi, poses, endpoint=False), objects)
    ids = np.repeat(np.arange(1, objects + 1), poses)
    centres = rng.normal(scale=spread, size=(objects, dim))

    def plane():
        u, v = rng.normal(size=(2, dim))
        u /= np.linalg.norm(u)
        v -= u * (v @ u)
        return u, v / np.linalg.norm(v)

    shared = plane()
    x = np.empty((objects * poses, dim))
    for o in range(objects):
        block = slice(o * poses, (o + 1) * poses)
        if kind in {"circles", "aligned"}:
            u, v = shared if kind == "aligned" else plane()
            t = angles[block][:, None]
            x[block] = centres[o] + np.cos(t) * u + np.sin(t) * v
        elif kind == "blobs":
            x[block] = centres[o] + rng.normal(scale=0.5, size=(poses, dim))
        else:
            x[block] = rng.normal(size=(poses, dim))
    return x + rng.normal(scale=0.02, size=x.shape), angles, ids


@pytest.fixture(scope="module")
def scored():
    return {kind: analyse(*synthetic(kind), OPTIONS) for kind in ("aligned", "circles", "blobs", "noise")}


def test_pose_decodes_only_on_circles(scored):
    for kind in ("aligned", "circles"):
        assert scored[kind]["pose_r2_median"] > 0.9
    for kind in ("blobs", "noise"):
        assert scored[kind]["pose_r2_median"] < 0.5


def test_h1_separates_a_circle_from_a_blob(scored):
    assert scored["circles"]["h1_median"] > 0.5
    assert scored["blobs"]["h1_median"] < 0.3


def test_transfer_needs_a_shared_pose_plane(scored):
    assert scored["aligned"]["transfer_r2"] > 0.9
    assert scored["circles"]["transfer_r2"] < 0.5


def test_separation_survives_until_the_clusters_merge(scored):
    for kind in ("aligned", "circles", "blobs"):
        assert scored[kind]["nn_purity"] > 0.95
        assert scored[kind]["global_components"] == OBJECTS
    assert scored["noise"]["nn_purity"] < 0.5
    assert scored["noise"]["global_components"] == 1

# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reset the modification times of a tree -- the scratch purge's clock.

F49 (2026-09-21): AICR's scratch purge deletes files whose *mtime* is older than 30 days and sweeps
daily. Extracting a tarball preserves the archive's mtimes, so an extracted dataset looks expired
the moment it lands and vanishes at the next sweep, its directories left standing: Imagenette went
twice, the seven transfer datasets once. Every extraction onto scratch therefore ends with a
``touch_tree`` of what it wrote, and ``scripts/aicr/slurm_touch_scratch.sh`` repeats it every
~25 days for as long as the campaign runs.
"""

from __future__ import annotations

import os


def touch_tree(root: str | os.PathLike) -> int:
    """Set the mtime and atime of every file and directory under ``root`` (and ``root``) to now;
    returns how many entries were touched, 0 when ``root`` does not exist (a download that
    failed before creating it). Symlinks are touched as links, never followed."""
    if not os.path.lexists(root):
        return 0
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            path = os.path.join(dirpath, name)
            try:
                os.utime(path, None, follow_symlinks=False)
            except NotImplementedError:          # a platform without lutimes
                if not os.path.islink(path):
                    os.utime(path, None)
            count += 1
    os.utime(root, None)
    return count + 1

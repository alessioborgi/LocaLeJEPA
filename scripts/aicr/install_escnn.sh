#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# escnn (the D4-RN50 encoder's dependency) into the pinned campaign environment, reproducibly.
#
# F46: escnn was in no environment file and the D4 cells died at instantiation. It cannot be a pip
# requirement: escnn declares `py3nj` (a Fortran sdist that does not build on Python 3.12, imported
# lazily for SO(3) only), and its dependency `lie-learn` ships a PyPI wheel compiled against NumPy 1.x
# that refuses to import under numpy 2 -- installed without --no-deps it silently downgraded numpy
# 2.4.6 -> 1.26.4 and scipy 1.18.1 -> 1.17.1 (2026-09-21). So: everything with --no-deps, lie_learn
# built from source at a pinned commit against THIS numpy, and the pins diffed before/after
# (state/03 "Installing anything into the pinned environment"). Idempotent: exits 0 when the import
# already works. Linux only (a compiler is needed); on macOS the same steps were done by hand.
#
#   PY=$HOME/conda/envs/localejepa-aicr/bin/python bash scripts/aicr/install_escnn.sh
set -euo pipefail
: "${PY:?PY=<the campaign env python> must be exported}"

LIE_LEARN_SHA=edf012f5f60af320175d2e6269db78b984b5bfc3   # AMLab-Amsterdam/lie_learn master, 2025-02-11
LIE_LEARN_SDIST=https://files.pythonhosted.org/packages/14/6b/a0660174724902ecff31adf58c766e698a5a0a2a187ca1666432a0be4c90/lie_learn-0.0.1.post1.tar.gz
PINS='^(numpy|scipy|torch|torchvision|lightning|timm|scikit-learn|fast[-_]pytorch[-_]kmeans)=='
PIP="env -u LD_LIBRARY_PATH $PY -m pip"

pins() { env -u LD_LIBRARY_PATH "$PY" -m pip list --format=freeze 2>/dev/null | grep -E "$PINS" | sort; }
check() {
    env -u LD_LIBRARY_PATH "$PY" - <<'PYEOF'
import numpy
import lie_learn.representations.SO3.irrep_bases  # the compiled extension: the wheel fails HERE under numpy 2
from escnn import gspaces
print("escnn D4 OK under numpy", numpy.__version__, gspaces.flipRot2dOnR2(N=4))
PYEOF
}

BEFORE=$(pins)
if check 2>/dev/null; then echo "escnn already installed and importable; pins:"; pins; exit 0; fi

$PIP install --no-deps escnn==1.0.11 pymanopt==2.2.1 autograd==1.9.1 joblib==1.6.0

# the two data files lie_learn needs at import; a git install does not ship them
SITE=$(env -u LD_LIBRARY_PATH "$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
DATA=$(mktemp -d)
if [ -d "$SITE/lie_learn" ]; then
    (cd "$SITE/lie_learn" && find . -name '*.npy' | while read -r f; do mkdir -p "$DATA/$(dirname "$f")"; cp "$f" "$DATA/$f"; done)
fi
if [ -z "$(find "$DATA" -name '*.npy')" ]; then
    curl -sSL -o "$DATA/sdist.tar.gz" "$LIE_LEARN_SDIST"
    tar -xzf "$DATA/sdist.tar.gz" -C "$DATA" --strip-components=2 --wildcards '*/lie_learn/*.npy'
fi
[ -n "$(find "$DATA" -name '*.npy')" ] || { echo "no lie_learn data files found" >&2; exit 1; }

$PIP install --no-deps cython==3.3.0
$PIP install --no-cache-dir --no-build-isolation --no-deps --force-reinstall "git+https://github.com/AMLab-Amsterdam/lie_learn@$LIE_LEARN_SHA"
$PIP uninstall -y cython
(cd "$DATA" && find . -name '*.npy' | while read -r f; do mkdir -p "$SITE/lie_learn/$(dirname "$f")"; cp "$f" "$SITE/lie_learn/$f"; done)
rm -rf "$DATA"

check
AFTER=$(pins)
if [ "$BEFORE" != "$AFTER" ]; then
    echo "PINS CHANGED -- restore them (pip install <name>==<version>) before anything else runs:" >&2
    diff <(echo "$BEFORE") <(echo "$AFTER") >&2 || true
    exit 1
fi
echo "PINS UNCHANGED"; pins
echo "escnn INSTALLED: regenerate the environment lock now (state/03) and point ENV_LOCK at it"

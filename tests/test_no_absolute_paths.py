"""No machine-specific absolute paths in the tracked tree.

Absolute paths rooted at a site's filesystem are a defect class, not a tidiness issue: a
config carrying one silently points an evaluation at a cache that does not exist on any other
machine, or -- worse -- at a directory that does exist and holds something else. This repository
carried nine of them from a previous cluster, in files that are still live.

Paths are expected to be RELATIVE, resolved against the directory an evaluation is run from.
Machine-specific locations belong in ``configs/local/default.yaml`` (git-ignored) or in an
environment variable with a relative default.

Run standalone (`python tests/test_no_absolute_paths.py`) or under pytest.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Roots that only exist on one site's filesystem. Deliberately narrow: this test is worthless
# if it cries wolf, and a noisy guard gets deleted rather than fixed.
SITE_ROOTS = ("/n/", "/net/", "/netscratch/", "/home/", "/work/", "/scratch/", "/users/")

# Absolute paths that are portable or are not filesystem paths at all.
ALLOWED = re.compile(
    r"""^\#!/            # shebangs
      | https?://        # URLs
      | /dev/null
      | /etc/profile
      | ^\s*\#\s*/       # a path inside a prose comment is caught by SITE_ROOTS below anyway
    """,
    re.VERBOSE,
)

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".md", ".toml", ".cfg", ".txt"}


def tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [ROOT / f for f in out if Path(f).suffix in TEXT_SUFFIXES]


def offending_lines() -> list[str]:
    hits: list[str] = []
    for path in tracked_text_files():
        if path.resolve() == Path(__file__).resolve():
            continue  # this file names the roots on purpose
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(lines, 1):
            if ALLOWED.search(line):
                continue
            for root in SITE_ROOTS:
                if root in line:
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:110]}")
                    break
    return hits


def test_no_machine_specific_absolute_paths() -> None:
    hits = offending_lines()
    assert not hits, (
        f"{len(hits)} machine-specific absolute path(s) in tracked files. Make them relative "
        "(resolved against the run directory), or move them to configs/local/default.yaml "
        "which is git-ignored:\n  " + "\n  ".join(hits)
    )


if __name__ == "__main__":
    found = offending_lines()
    print(f"{len(found)} offending line(s)")
    for h in found:
        print("  ", h)
    raise SystemExit(1 if found else 0)

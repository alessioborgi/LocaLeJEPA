# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlretrieve

DEFAULT_URL = "https://www.dropbox.com/scl/fo/p3xbos6uwrtyb6g33v9bz/ALyh9wWk1hrB21ltibT5ep8?rlkey=rssxxluclfzz9ad5t8izvmgml&e=1&dl=0"
DEFAULT_FILES = (
    "sq-blob-v3-tm.h5",
    "sq-blob-ul-15k-tm.h5",
    "sq-blob-test2k-tm.h5",
    "mf1-sg3-tm.h5",
    "mf1-sg3-25k-ul-s99-tm.h5",
)


def direct_dropbox_url(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def download_phc(output_dir: Path | str = "data/phc", url: str = DEFAULT_URL) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "phc_dropbox_download"

    print(f"Downloading PhC data from {url}")
    urlretrieve(direct_dropbox_url(url), archive_path)

    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError("Expected Dropbox folder download to be a zip archive.")

    extract_dir = output_dir / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive_members = archive.namelist()
        for filename in DEFAULT_FILES:
            matches = [member for member in archive_members if Path(member).name == filename]
            if matches:
                archive.extract(matches[0], extract_dir)
    archive_path.unlink(missing_ok=True)

    missing = []
    for filename in DEFAULT_FILES:
        matches = list(extract_dir.rglob(filename))
        if not matches:
            missing.append(filename)
            continue
        destination = output_dir / filename
        shutil.copy2(matches[0], destination)
        print(f"Wrote {destination}")

    shutil.rmtree(extract_dir)

    if missing:
        raise FileNotFoundError(f"Downloaded archive did not contain expected files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ESSL/SIB-CL PhC HDF5 files.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dropbox folder URL.")
    parser.add_argument("--output-dir", default="data/phc", type=Path, help="Directory where .h5 files should be placed.")
    args = parser.parse_args()

    download_phc(output_dir=args.output_dir, url=args.url)


if __name__ == "__main__":
    main()

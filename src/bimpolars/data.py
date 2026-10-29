"""Test and example data: a local folder, filled from the bimpolars bucket on first use.

The models the tests and examples use live at :data:`DATA_URL`. :func:`fetch` looks for a file
in :func:`data_dir` first and downloads it only when it is missing there. Point the environment
variable :data:`DATA_DIR_ENV` at a folder that already holds the files (a mirror of the bucket
with the same keys) to work without network access.
"""

from __future__ import annotations

import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

DATA_URL = "http://r2.eu.bedrock.engineer/bimpolars/"
"""The bucket the data is downloaded from; a key is appended to it."""

DATA_DIR_ENV = "BIMPOLARS_DATA_DIR"
"""The environment variable naming the local data folder."""

SIXTY5_STRUCTURAL = "sixty5_O-S1-CON-structural-engineering.ifc"
"""Sixty5 (Strijp-S, Eindhoven) structural model: IFC2X3 in millimetres, 2647 elements."""

SIXTY5_PREFAB_FACADE = "sixty5_O-S1-BWK-BIM-prefab-facade.ifc"
"""Sixty5 prefab facade model: IFC2X3 in millimetres, 1055 elements."""

SIXTY5_ARCHITECTURAL = "sixty5_O-S1-BWK-BIM-architectural.ifc"
"""Sixty5 architectural model: IFC2X3, 343 MB; tests that read it are marked ``slow``."""

RVT_ADVANCED_SAMPLE_ARCH = "rvt-advanced-sample-arch.ifc"
"""Revit Advanced Sample Project, architecture: IFC2X3 in millimetres, 5696 elements."""


def data_dir() -> Path:
    """The local data folder.

    Returns:
        The folder named by :data:`DATA_DIR_ENV`, or ``bimpolars`` under the user's cache
        folder when the variable is not set: ``$XDG_CACHE_HOME`` if set, else ``~/.cache``.
    """
    configured = os.environ.get(DATA_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache_home) / "bimpolars"


def fetch(key: str, *, download: bool = True) -> Path:
    """Gives the local path of a data file, downloading it on first use.

    Args:
        key: The file's key in the bucket, such as :data:`SIXTY5_STRUCTURAL` or
            ``"rvt-advanced-sample/speckle-v3-objects.parquet"``.
        download: Whether to download a file that is not in :func:`data_dir`.

    Returns:
        The path of the file in :func:`data_dir`.

    Raises:
        FileNotFoundError: If the file is missing locally and ``download`` is False.
        urllib.error.URLError: If the download fails.
    """
    path = data_dir() / key
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(
            f"{key!r} is not in {data_dir()} (set {DATA_DIR_ENV} or download it)"
        )
    url = DATA_URL + urllib.parse.quote(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return path

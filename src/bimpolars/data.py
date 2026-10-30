"""Test and example data: a local folder, filled from the bimpolars bucket on first use.

The models the tests and examples use live at :data:`DATA_URL`. :func:`fetch` looks for a file
in :func:`data_dir` first and downloads it only when it is missing there. Point the environment
variable :data:`DATA_DIR_ENV` at a folder that already holds the files (a mirror of the bucket
with the same keys) to work without network access.
"""

from __future__ import annotations

import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path, PureWindowsPath
from typing import NamedTuple

DATA_URL = "http://r2.eu.bedrock.engineer/bimpolars/"
"""The bucket the data is downloaded from; a key is appended to it."""

DATA_DIR_ENV = "BIMPOLARS_DATA_DIR"
"""The environment variable naming the local data folder."""

_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
"""A URL, as opposed to a bucket key or a path (a Windows drive is one letter and a colon)."""

SIXTY5_STRUCTURAL = "sixty5_O-S1-CON-structural-engineering.ifc"
"""Sixty5 (Strijp-S, Eindhoven) structural model: IFC2X3 in millimetres, 2647 elements."""

SIXTY5_PREFAB_FACADE = "sixty5_O-S1-BWK-BIM-prefab-facade.ifc"
"""Sixty5 prefab facade model: IFC2X3 in millimetres, 1055 elements."""

SIXTY5_ARCHITECTURAL = "sixty5_O-S1-BWK-BIM-architectural.ifc"
"""Sixty5 architectural model: IFC2X3, 343 MB; tests that read it are marked ``slow``."""

RVT_ADVANCED_SAMPLE_ARCH = "rvt-advanced-sample-arch.ifc"
"""Revit Advanced Sample Project, architecture: IFC2X3 in millimetres, 5696 elements."""

RVT_SPECKLE_V3_OBJECTS = "rvt-advanced-sample/speckle-v3-objects.parquet"
"""The v3 objects of the Revit Advanced Sample versions on app.speckle.systems, one row per
object (``version_id``, ``object_id``, ``is_root``, ``content``), as
:func:`bimpolars.speckle.version_objects` reads them: the arch, str and mep versions below,
projected from their bundles, and ``5cb009bb77`` (model ``e7f5f7f943``), the IFC export of
the arch model received by Speckle's importer: 9730 objects in metres, no bundle."""

BUNDLE_TABLES: tuple[str, ...] = (
    "eav.eav",
    "eav.model",
    "eav.object_type",
    "eav.objects",
    "eav.path_stats",
    "eav.paths",
    "eav.revit_external_links",
    "eav.revit_views",
    "eav.revit_warnings",
    "eav.type_eav",
    "eav.types",
    "envelope.meta",
    "envelope.node_kinds",
    "envelope.nodes",
    "envelope.rel_types",
    "envelope.relations",
    "envelope.scene_views",
    "geometries",
)
"""The tables of a Revit bundle (rvextract schema 1.5), each a ``<name>.<table>.parquet`` file,
which :func:`fetch_bundle` fetches.

A bundle with more than 1.5 GiB of uncompressed geometry has overflow shards,
``<name>.geometries.<N>.parquet``, which this list does not name and :func:`fetch_bundle`
does not fetch: the elements whose geometry is in them would have no mesh. The RVT samples
have one shard each. Read such a bundle from a complete folder instead."""


class SpeckleVersion(NamedTuple):
    """A Speckle version of the test data: its ids and the key prefix of its bundle.

    Attributes:
        model_id: The Speckle model id.
        version_id: The Speckle version id, as :func:`bimpolars.speckle.version_objects` wants it.
        bundle: The key prefix of the bundle's tables: ``<prefix>.<table>.parquet``, for
            :func:`fetch_bundle`.
    """

    model_id: str
    version_id: str
    bundle: str


_RVT_BUNDLES = "rvt-advanced-sample/speckle-bundle"

RVT_SPECKLE_ARCH = SpeckleVersion(
    "363cae173a", "d5e5bacdde", f"{_RVT_BUNDLES}/363cae173a/d5e5bacdde/advanced_sample_project_arch"
)
"""Revit Advanced Sample Project, architecture, on Speckle: 13,194 v3 objects, 5872 bundle
objects on 5 levels, in millimetres."""

RVT_SPECKLE_STR = SpeckleVersion(
    "35e9a2fa3b", "b66058b4d1", f"{_RVT_BUNDLES}/35e9a2fa3b/b66058b4d1/advanced_sample_project_str"
)
"""Revit Advanced Sample Project, structure, on Speckle: 5355 v3 objects, 935 bundle objects."""

RVT_SPECKLE_MEP = SpeckleVersion(
    "ef7cb591fc", "4413abe507", f"{_RVT_BUNDLES}/ef7cb591fc/4413abe507/advanced_sample_project_mep"
)
"""Revit Advanced Sample Project, MEP, on Speckle: 16,173 v3 objects, 8625 bundle objects."""


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


def bucket_key(key_or_url: str) -> str:
    """Gives the bucket key of a data file, from the key itself or from its URL in the bucket.

    Args:
        key_or_url: A key such as :data:`SIXTY5_STRUCTURAL`, or its URL: :data:`DATA_URL`
            (``http`` or ``https``) followed by the key, percent-encoded or not.

    Returns:
        The key, relative to the bucket and to :func:`data_dir`.

    Raises:
        ValueError: If a URL is not in the bucket (fetch such a file yourself), or the key
            would leave the data folder: an absolute path, a drive, or a ``..`` segment.
    """
    key = key_or_url
    if _URL.match(key_or_url):
        for base in (DATA_URL, DATA_URL.replace("http://", "https://", 1)):
            if key_or_url.startswith(base):
                key = urllib.parse.unquote(key_or_url[len(base) :])
                break
        else:
            raise ValueError(f"{key_or_url!r} is not in the bucket {DATA_URL}: fetch it yourself")
    path = PureWindowsPath(key)  # splits on / and \, and knows drives
    if not key or path.anchor or ".." in path.parts:
        raise ValueError(f"{key_or_url!r} does not name a key inside the bucket")
    return key


def fetch(key: str, *, download: bool = True) -> Path:
    """Gives the local path of a data file, downloading it on first use.

    Args:
        key: The file's key in the bucket, such as :data:`SIXTY5_STRUCTURAL` or
            ``"rvt-advanced-sample/speckle-v3-objects.parquet"``, or its URL in the bucket
            (:func:`bucket_key`).
        download: Whether to download a file that is not in :func:`data_dir`.

    Returns:
        The path of the file in :func:`data_dir`.

    Raises:
        ValueError: If ``key`` is a URL outside the bucket.
        FileNotFoundError: If the file is missing locally and ``download`` is False.
        urllib.error.URLError: If the download fails.
    """
    key = bucket_key(key)
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


def fetch_bundle(prefix: str, *, download: bool = True) -> Path:
    """Gives the local folder of a Speckle bundle, downloading its tables on first use.

    Args:
        prefix: The key prefix of the bundle's tables, such as ``RVT_SPECKLE_ARCH.bundle``,
            or its URL in the bucket (:func:`bucket_key`): every table of
            :data:`BUNDLE_TABLES` is fetched as ``<prefix>.<table>.parquet``. Overflow
            geometry shards are not (see :data:`BUNDLE_TABLES`).
        download: Whether to download the tables that are not in :func:`data_dir`.

    Returns:
        The folder holding the tables, for :func:`bimpolars.read_speckle` and
        :func:`bimpolars.bundle.bundle_objects`.

    Raises:
        ValueError: If ``prefix`` is a URL outside the bucket.
        FileNotFoundError: If a table is missing locally and ``download`` is False.
        urllib.error.URLError: If a download fails.
    """
    for table in BUNDLE_TABLES:
        path = fetch(f"{prefix}.{table}.parquet", download=download)
    return path.parent

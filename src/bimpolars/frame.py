"""The BimDataFrame: a polars DataFrame of BIM elements with one model CRS."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import polars as pl

from bimpolars.geoparquet import GEO_METADATA_KEY, geo_metadata

METADATA_KEY = "bimpolars"
"""The Parquet file metadata key under which a BimDataFrame stores its CRS."""

METADATA_VERSION = 1
"""The layout version of the JSON under :data:`METADATA_KEY`."""

STYLE_DTYPE = pl.Struct(
    {"name": pl.String, "rgba": pl.Array(pl.Float64, 4), "face_count": pl.UInt32}
)
"""One surface style of a mesh: its name, colour (0..1) and how many faces it covers.

A null ``name`` means the faces have no style. Styles with one name can differ in colour, so
group by ``(name, rgba)`` for one material per style.
"""

MESH_DTYPE = pl.Struct(
    {
        "vertices": pl.List(pl.Float64),
        "faces": pl.List(pl.UInt32),
        "styles": pl.List(STYLE_DTYPE),
    }
)
"""A triangle mesh in model coordinates.

``vertices`` is flat (x, y, z, x, y, z, ...) and ``faces`` holds three vertex indices per
triangle. The faces are sorted by style: the first ``styles[0].face_count`` triangles have the
first style, the next ``styles[1].face_count`` the second, and so on, one block per style.
Every face has exactly one style.
"""

BBOX_DTYPE = pl.Struct(
    {name: pl.Float64 for name in ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")}
)
"""The axis-aligned bounding box of a mesh, in model coordinates."""

MODEL_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "class",
    "properties",
    "container",
    "elevation",
    "native_geom",
    "mesh",
    "bbox",
)
"""The columns every reader fills, in order."""


class BimDataFrame:
    """One BIM model: a polars DataFrame of elements and the CRS their geometry is in.

    The frame has one row per element, with or without geometry. The columns the readers fill
    are listed in :data:`MODEL_COLUMNS`; the geometry columns ``mesh``, ``bbox`` and
    ``native_geom`` share the model CRS, as do the simple-feature columns
    :func:`bimpolars.simple_features` adds (``footprint``, WKB). The class is thin: work on
    :attr:`frame` with polars and wrap the result again with :meth:`with_frame`.

    Attributes:
        frame: The elements, one row each.
        crs: The model CRS as a PROJJSON dict.
    """

    __slots__ = ("_crs", "_frame")

    def __init__(self, frame: pl.DataFrame, crs: Mapping[str, Any]) -> None:
        """Wraps a frame of elements and its CRS.

        Args:
            frame: The elements, one row each.
            crs: The model CRS as PROJJSON (see :func:`bimpolars.engineering_crs`).

        Raises:
            TypeError: If ``frame`` is not a polars DataFrame or ``crs`` not a mapping.
            ValueError: If ``crs`` has no ``type``.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"frame must be a polars DataFrame, not {type(frame).__name__}")
        if not isinstance(crs, Mapping):
            raise TypeError(f"crs must be a PROJJSON mapping, not {type(crs).__name__}")
        if "type" not in crs:
            raise ValueError("crs is not PROJJSON: it has no 'type'")
        self._frame = frame
        self._crs: dict[str, Any] = json.loads(json.dumps(crs))

    @property
    def frame(self) -> pl.DataFrame:
        """The elements, one row each."""
        return self._frame

    @property
    def crs(self) -> dict[str, Any]:
        """The model CRS as a PROJJSON dict."""
        return self._crs

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"BimDataFrame({len(self)} rows, crs={self._crs.get('name')!r})\n{self._frame!r}"

    def _repr_html_(self) -> str:
        return (
            f"<p><b>BimDataFrame</b>: crs {self._crs.get('name')!r}</p>{self._frame._repr_html_()}"
        )

    def with_frame(self, frame: pl.DataFrame) -> BimDataFrame:
        """Wraps another frame in the same CRS, for example a filtered or joined one.

        Args:
            frame: The new elements.

        Returns:
            A BimDataFrame with ``frame`` and this model's CRS.
        """
        return BimDataFrame(frame, self._crs)

    def equals(self, other: object) -> bool:
        """Whether another BimDataFrame has the same CRS and the same frame.

        Args:
            other: Anything; only a BimDataFrame can be equal.

        Returns:
            True if the CRS dicts are equal and the frames are equal, nulls included.
        """
        if not isinstance(other, BimDataFrame):
            return False
        return self._crs == other._crs and self._frame.equals(other._frame)

    def write_parquet(self, file: str | Path | IO[bytes], **kwargs: Any) -> None:
        """Writes the frame to Parquet with the CRS in the file metadata.

        The file metadata gets a :data:`METADATA_KEY` entry with the JSON
        ``{"version": 1, "crs": <PROJJSON>}``. Every column, nested ones included, reads back
        with its dtype through :func:`read_parquet`.

        A frame with a ``footprint`` column (see :func:`bimpolars.simple_features`) is written
        as GeoParquet: the file metadata also gets the ``geo`` entry of
        :func:`bimpolars.geoparquet.geo_metadata`, with ``footprint`` as the primary geometry
        in the model CRS and ``bbox`` as its covering, so that GeoParquet readers find the
        footprints with their CRS.

        Args:
            file: Where to write: a path or a binary file object.
            **kwargs: Passed on to :meth:`polars.DataFrame.write_parquet`. A ``metadata`` dict
                is written too, next to the :data:`METADATA_KEY` and ``geo`` entries.
        """
        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata[METADATA_KEY] = json.dumps({"version": METADATA_VERSION, "crs": self._crs})
        geo = geo_metadata(self._frame, self._crs)
        if geo is not None:
            metadata[GEO_METADATA_KEY] = json.dumps(geo)
        self._frame.write_parquet(file, metadata=metadata, **kwargs)

    @classmethod
    def read_parquet(cls, source: str | Path | IO[bytes]) -> BimDataFrame:
        """Reads a Parquet file that :meth:`write_parquet` wrote.

        Args:
            source: The path or binary file object to read.

        Returns:
            The BimDataFrame, CRS included.

        Raises:
            ValueError: If the file has no :data:`METADATA_KEY` metadata, or metadata of a
                layout this version does not read.
        """
        raw = pl.read_parquet_metadata(source).get(METADATA_KEY)
        if raw is None:
            raise ValueError(
                f"not a BimDataFrame: the Parquet file has no {METADATA_KEY!r} metadata"
            )
        metadata = json.loads(raw)
        if metadata.get("version") != METADATA_VERSION:
            raise ValueError(
                f"the {METADATA_KEY!r} metadata has layout version {metadata.get('version')!r}; "
                f"this version of bimpolars reads {METADATA_VERSION}"
            )
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pl.read_parquet(source)
        return cls(frame, metadata["crs"])


def read_parquet(source: str | Path | IO[bytes]) -> BimDataFrame:
    """Reads a BimDataFrame from Parquet; see :meth:`BimDataFrame.read_parquet`.

    Args:
        source: The path or binary file object to read.

    Returns:
        The BimDataFrame, CRS included.
    """
    return BimDataFrame.read_parquet(source)

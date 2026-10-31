"""The BimDataFrame: a polars DataFrame of BIM elements with their CRS in a column."""

from __future__ import annotations

import functools
import importlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO, Any, Literal

import polars as pl
import pyproj

from bimpolars.crs import CRSLike, resolve_crs
from bimpolars.geoparquet import GEO_METADATA_KEY, GEOMETRY_COLUMNS, geo_metadata

CRS_COLUMN = "bim_crs"
"""The column that holds the BIM CRS of every row, as PROJJSON text."""

CRS_DTYPE = pl.Categorical
"""The dtype of :data:`CRS_COLUMN`.

A Categorical keeps one copy of each CRS however many rows share it, in memory and in Parquet
(a dictionary page), also after a file is read back, where a String column holds a copy per row.
"""

METADATA_KEY = "bimpolars"
"""The Parquet file metadata key under which a BimDataFrame stores its layout version."""

METADATA_VERSION = 2
"""The layout version of the JSON under :data:`METADATA_KEY`.

Version 2 keeps the CRS in :data:`CRS_COLUMN`; version 1 kept it in the metadata, as
``{"version": 1, "crs": <PROJJSON>}``, and :func:`read_parquet` still reads it.
"""

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

SWEEP_DTYPE = pl.Struct(
    {
        "type": pl.String,
        "placement": pl.Array(pl.Float64, 16),
        "profile": pl.Binary,
        "profile_type": pl.String,
        "profile_name": pl.String,
        "depth": pl.Float64,
        "direction": pl.Array(pl.Float64, 3),
        "clipped": pl.Boolean,
    }
)
"""One swept solid of an element's native geometry, as a reader found it in the source.

``type`` names the solid (``IfcExtrudedAreaSolid``). ``placement`` is the row-major 4x4 matrix
from the sweep's own coordinates to the BIM CRS: the element's placement, the mapped item's
transformation and the solid's position composed. ``profile`` is the swept area in the sweep's
XY plane as a 2D ISO WKB Polygon (the profile's own position applied, voids as holes),
``profile_type`` and ``profile_name`` are the source's names for it, and ``depth`` and
``direction`` (a unit vector in the sweep's coordinates) are the extrusion. ``clipped`` is
True when the source cut the solid afterwards (a difference or intersection reduced to its
first operand), so that the sweep is bigger than the element; a union lists both operands.
The profile and the depth are in the sweep's own coordinates: :func:`bimpolars.geobim.to_crs`
moves a sweep by composing its affine matrix onto ``placement``, scale included.
"""

NATIVE_GEOM_DTYPE = pl.List(SWEEP_DTYPE)
"""The ``native_geom`` column once a reader fills it: the sweeps of an element's body that
it could read, in the source's order; null for a row whose body has none."""

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
    CRS_COLUMN,
)
"""The columns every reader fills, in order."""

ConcatMethod = Literal["vertical", "vertical_relaxed", "diagonal", "diagonal_relaxed"]
"""The ``how`` of :func:`concat`: the polars concat strategies that stack rows."""


class BimDataFrame:
    """BIM elements: a polars DataFrame with the CRS of their geometry in a column.

    The frame has one row per element, with or without geometry. The columns the readers fill
    are listed in :data:`MODEL_COLUMNS`. The geometry columns ``mesh``, ``bbox`` and
    ``native_geom`` are in the row's BIM CRS, as are the simple-feature columns
    :func:`bimpolars.simple_features` adds (``footprint`` and ``lightbim``, WKB); ``geog`` is
    always OGC:CRS84h. The BIM CRS sits in the :data:`CRS_COLUMN` column (``bim_crs``) as
    PROJJSON text, the same value on every row of one model, so it travels with the rows
    through every polars operation: a concat of models in different CRSs keeps each row's.

    Polars methods work on the BimDataFrame itself: ``bdf.filter(...)``, ``bdf["id"]`` and
    ``pl.DataFrame(bdf)`` go to :attr:`frame`, and a result that is a DataFrame with the
    ``bim_crs`` column comes back as a BimDataFrame. :func:`concat` stacks models.

    Attributes:
        frame: The elements, one row each, ``bim_crs`` included.
        crs: The BIM CRS as a ``pyproj.CRS``, like ``GeoDataFrame.crs``.
        projjson: The BIM CRS as a PROJJSON dict.
    """

    __slots__ = ("_crs_cache", "_frame", "_projjson_cache")

    def __init__(self, frame: pl.DataFrame, crs: CRSLike | None = None) -> None:
        """Wraps a frame of elements, reading the BIM CRS from its ``bim_crs`` column.

        Args:
            frame: The elements, one row each. A ``bim_crs`` column of PROJJSON text, String
                or Categorical, becomes :data:`CRS_DTYPE`.
            crs: The BIM CRS of a frame without the ``bim_crs`` column, in any form: PROJJSON
                (see :func:`bimpolars.engineering_crs`), a ``pyproj.CRS``, or user input such
                as ``"EPSG:7415"``. It is stored as PROJJSON text in a new ``bim_crs`` column,
                the last one. A frame that has the column already must hold ``crs`` on every
                row; :func:`bimpolars.set_crs` replaces a CRS.

        Raises:
            TypeError: If ``frame`` is not a polars DataFrame, or its ``bim_crs`` column is
                neither String nor Categorical.
            ValueError: If the frame has no ``bim_crs`` column and ``crs`` is not given, or
                it has one that does not hold ``crs`` on every row; if ``crs`` is PROJJSON
                without a ``type``, or pyproj cannot read it.
        """
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"frame must be a polars DataFrame, not {type(frame).__name__}")
        if CRS_COLUMN in frame.columns:
            dtype = frame.schema[CRS_COLUMN]
            if dtype == pl.String:
                frame = frame.with_columns(pl.col(CRS_COLUMN).cast(CRS_DTYPE))
            elif dtype != CRS_DTYPE:
                raise TypeError(
                    f"the {CRS_COLUMN} column must hold PROJJSON text, String or Categorical, "
                    f"not {dtype}"
                )
        if crs is not None:
            text = crs_text(_as_projjson(crs))
            if CRS_COLUMN not in frame.columns:
                frame = frame.with_columns(pl.lit(text, dtype=CRS_DTYPE).alias(CRS_COLUMN))
            elif not (frame[CRS_COLUMN] == text).all(ignore_nulls=False):
                raise ValueError(
                    f"the frame's {CRS_COLUMN} column does not hold the CRS "
                    f"{json.loads(text).get('name')!r} on every row: it is "
                    f"{_describe(frame[CRS_COLUMN])}; use bimpolars.set_crs to replace it"
                )
        elif CRS_COLUMN not in frame.columns:
            raise ValueError(
                f"the frame has no {CRS_COLUMN} column: pass its CRS, BimDataFrame(frame, crs)"
            )
        self._frame = frame
        self._projjson_cache: dict[str, Any] | None = None
        self._crs_cache: pyproj.CRS | None = None

    @property
    def frame(self) -> pl.DataFrame:
        """The elements, one row each, with the ``bim_crs`` column."""
        return self._frame

    @property
    def crs(self) -> pyproj.CRS:
        """The BIM CRS as a ``pyproj.CRS``, read from the one value of ``bim_crs``.

        Raises:
            ValueError: If ``bim_crs`` holds more than one CRS, has nulls, or the frame has no
                rows.
        """
        if self._crs_cache is None:
            self._crs_cache = pyproj.CRS.from_json_dict(self.projjson)
        return self._crs_cache

    @property
    def projjson(self) -> dict[str, Any]:
        """The BIM CRS as PROJJSON: the one value of ``bim_crs``, parsed.

        Raises:
            ValueError: As :attr:`crs` raises it.
        """
        if self._projjson_cache is None:
            column = self._frame[CRS_COLUMN]
            values = column.drop_nulls().unique()
            if len(values) != 1 or column.null_count():
                raise ValueError(
                    f"a BimDataFrame needs one CRS here, but its {CRS_COLUMN} column holds "
                    f"{_describe(column)}; take the rows of one CRS first, for example with "
                    f"group_by({CRS_COLUMN!r})"
                )
            self._projjson_cache = json.loads(values[0])
        return self._projjson_cache

    def __len__(self) -> int:
        return len(self._frame)

    def __getattr__(self, name: str) -> Any:
        """Hands polars attributes and methods through to :attr:`frame`.

        A result that is a DataFrame with the ``bim_crs`` column comes back as a BimDataFrame;
        anything else (a Series, a scalar, a frame without ``bim_crs``) as it is. BimDataFrame
        arguments go in as their frames, so ``bdf.join(other, on="id")`` works.
        """
        if name.startswith("_"):  # copy and pickle look these up before _frame is set
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        attribute = getattr(self._frame, name)
        if not callable(attribute):
            return _wrap(attribute)

        @functools.wraps(attribute)
        def method(*args: Any, **kwargs: Any) -> Any:
            args = tuple(_unwrap(arg) for arg in args)
            kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
            return _wrap(attribute(*args, **kwargs))

        return method

    def __getitem__(self, key: Any) -> Any:
        """Indexes :attr:`frame`: ``bdf["id"]`` is the Series, ``bdf[:10]`` a BimDataFrame."""
        return _wrap(self._frame[key])

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *dir(self._frame)})

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """The Arrow PyCapsule stream of :attr:`frame`, for ``pl.DataFrame(bdf)`` and DuckDB."""
        return self._frame.__arrow_c_stream__(requested_schema)

    def __repr__(self) -> str:
        return (
            f"BimDataFrame({len(self)} rows, crs={_label(self._frame[CRS_COLUMN])})\n"
            f"{self._frame.drop(CRS_COLUMN)!r}"
        )

    def _repr_html_(self) -> str:
        return (
            f"<p><b>BimDataFrame</b>: crs {_label(self._frame[CRS_COLUMN])}</p>"
            f"{self._frame.drop(CRS_COLUMN)._repr_html_()}"
        )

    def _display_(self) -> object:
        """What marimo shows: the CRS name above an interactive table without ``bim_crs``."""
        mo = importlib.import_module("marimo")  # only marimo calls this, so it is there
        header = mo.md(f"**BimDataFrame**: crs {_label(self._frame[CRS_COLUMN])}")
        return mo.vstack([header, mo.ui.table(self._frame.drop(CRS_COLUMN))])

    def with_frame(self, frame: pl.DataFrame) -> BimDataFrame:
        """Wraps another frame, in this model's CRS when the frame has no ``bim_crs`` column.

        A frame that kept ``bim_crs`` (a filtered or joined one) is in the CRS its rows say;
        one that lost it (a ``select`` without it, or a frame built from scratch) gets this
        model's.

        Args:
            frame: The new elements.

        Returns:
            A BimDataFrame of ``frame``.

        Raises:
            ValueError: If ``frame`` has no ``bim_crs`` column and this model does not have
                exactly one CRS.
        """
        if CRS_COLUMN in frame.columns:
            return BimDataFrame(frame)
        return BimDataFrame(frame, self.projjson)

    def equals(self, other: object) -> bool:
        """Whether another BimDataFrame has the same frame, ``bim_crs`` included.

        Args:
            other: Anything; only a BimDataFrame can be equal.

        Returns:
            True if the frames are equal, nulls included.
        """
        if not isinstance(other, BimDataFrame):
            return False
        return self._frame.equals(other._frame)

    def write_parquet(self, file: str | Path | IO[bytes], **kwargs: Any) -> None:
        """Writes the frame to Parquet, ``bim_crs`` included.

        The file metadata gets a :data:`METADATA_KEY` entry with the layout version,
        ``{"version": 2}``. Every column, nested ones included, reads back with its dtype
        through :func:`read_parquet`; ``bim_crs`` is stored once per row group, as a
        dictionary.

        A frame with a geometry column of :data:`bimpolars.geoparquet.GEOMETRY_COLUMNS` is
        written as GeoParquet: the file metadata also gets the ``geo`` entry of
        :func:`bimpolars.geoparquet.geo_metadata`, with the first such column as the primary
        geometry (``geog`` in OGC:CRS84h, from :func:`bimpolars.geobim.with_geog`, before
        ``footprint`` in the BIM CRS, from :func:`bimpolars.simple_features`), each with its
        CRS, and ``bbox`` named as the covering of the columns in the BIM CRS (the x and y
        fields each footprint lies within), so that GeoParquet readers find the footprints
        with their CRS and skip row groups for a spatial filter. GeoParquet gives a column one
        CRS, so such a frame needs one BIM CRS.

        Args:
            file: Where to write: a path or a binary file object.
            **kwargs: Passed on to :meth:`polars.DataFrame.write_parquet`. A ``metadata`` dict
                is written too, except that its :data:`METADATA_KEY` and ``geo`` entries,
                if any, are replaced by the ones described above.

        Raises:
            ValueError: If the frame has a geometry column and not exactly one BIM CRS (see
                :attr:`crs`).
        """
        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata[METADATA_KEY] = json.dumps({"version": METADATA_VERSION})
        metadata.pop(GEO_METADATA_KEY, None)
        if any(column in self._frame.columns for column in GEOMETRY_COLUMNS):
            geo = geo_metadata(self._frame, self.projjson)
            if geo is not None:
                metadata[GEO_METADATA_KEY] = json.dumps(geo)
        self._frame.write_parquet(file, metadata=metadata, **kwargs)

    @classmethod
    def read_parquet(cls, source: str | Path | IO[bytes]) -> BimDataFrame:
        """Reads a Parquet file that :meth:`write_parquet` wrote.

        A file with the ``bim_crs`` column reads as it is, also without the
        :data:`METADATA_KEY` entry (written by polars from :attr:`frame`, say). A file of
        layout version 1 has its CRS in that entry instead, and gets the ``bim_crs`` column
        from it.

        Args:
            source: The path or binary file object to read.

        Returns:
            The BimDataFrame, ``bim_crs`` included.

        Raises:
            ValueError: If the file has neither the ``bim_crs`` column nor :data:`METADATA_KEY`
                metadata of layout version 1, or metadata of a layout this version does not
                read.
        """
        raw = pl.read_parquet_metadata(source).get(METADATA_KEY)
        metadata = json.loads(raw) if raw is not None else {}
        version = metadata.get("version")
        if raw is not None and version not in (1, METADATA_VERSION):
            raise ValueError(
                f"the {METADATA_KEY!r} metadata has layout version {version!r}; this version "
                f"of bimpolars reads 1 and {METADATA_VERSION}"
            )
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pl.read_parquet(source)
        if version == 1:
            return cls(frame, metadata["crs"])
        if CRS_COLUMN not in frame.columns:
            raise ValueError(
                f"not a BimDataFrame: the Parquet file has no {CRS_COLUMN!r} column, and no "
                f"{METADATA_KEY!r} metadata with the CRS"
            )
        return cls(frame)


def read_parquet(source: str | Path | IO[bytes]) -> BimDataFrame:
    """Reads a BimDataFrame from Parquet; see :meth:`BimDataFrame.read_parquet`.

    Args:
        source: The path or binary file object to read.

    Returns:
        The BimDataFrame, ``bim_crs`` included.
    """
    return BimDataFrame.read_parquet(source)


def concat(bdfs: Iterable[BimDataFrame], *, how: ConcatMethod = "vertical") -> BimDataFrame:
    """Stacks models into one BimDataFrame, every row keeping its own CRS, like ``pl.concat``.

    Models in different CRSs stack too, each row with its own ``bim_crs``: the result then
    has no single :attr:`BimDataFrame.crs`, and :func:`bimpolars.to_crs` refuses it. Move the
    models into one CRS first (:func:`bimpolars.to_crs`) to work on them as one.

    Args:
        bdfs: The models.
        how: How the columns line up, as for ``pl.concat``: ``"vertical"`` needs the same
            columns, ``"diagonal"`` fills the missing ones with nulls, and the ``_relaxed``
            variants cast to a common dtype.

    Returns:
        The rows of every model, in order.

    Raises:
        TypeError: If an item is not a BimDataFrame.
        ValueError: If there are no models.
    """
    frames: list[pl.DataFrame] = []
    for bdf in bdfs:
        if not isinstance(bdf, BimDataFrame):
            raise TypeError(
                f"concat stacks BimDataFrames, not {type(bdf).__name__}: wrap a frame with "
                f"BimDataFrame(frame, crs) first"
            )
        frames.append(bdf.frame)
    if not frames:
        raise ValueError("concat needs at least one BimDataFrame")
    return BimDataFrame(pl.concat(frames, how=how))


def crs_text(projjson: Mapping[str, Any]) -> str:
    """The text a ``bim_crs`` column holds for a CRS: its PROJJSON, compact.

    Args:
        projjson: The CRS as PROJJSON.

    Returns:
        The JSON text, without spaces.
    """
    return json.dumps(projjson, separators=(",", ":"))


def _as_projjson(crs: CRSLike) -> dict[str, Any]:
    """The PROJJSON of a CRS in any form, the caller's own kept as written."""
    if isinstance(crs, Mapping):
        if "type" not in crs:
            raise ValueError("crs is not PROJJSON: it has no 'type'")
        # PROJJSON is kept as the caller wrote it, without a round trip through pyproj
        projjson: dict[str, Any] = json.loads(json.dumps(crs))
        return projjson
    return resolve_crs(crs)[1]


def _wrap(value: Any) -> Any:
    """A DataFrame with the ``bim_crs`` column as a BimDataFrame; anything else as it is."""
    if isinstance(value, pl.DataFrame) and CRS_COLUMN in value.columns:
        return BimDataFrame(value)
    return value


def _unwrap(value: Any) -> Any:
    """The frame of a BimDataFrame, for a polars method that takes a DataFrame."""
    return value.frame if isinstance(value, BimDataFrame) else value


def _names(column: pl.Series) -> list[str]:
    """The names of the CRSs a ``bim_crs`` column holds, in order of first appearance."""
    texts = column.drop_nulls().unique(maintain_order=True).cast(pl.String)
    return [str(json.loads(text).get("name")) for text in texts]


def _describe(column: pl.Series) -> str:
    """What a ``bim_crs`` column holds, for an error message."""
    names = _names(column)
    parts = []
    if len(names) == 1:
        parts.append(f"the CRS {names[0]!r}")
    elif names:
        parts.append(f"{len(names)} CRSs: " + ", ".join(repr(name) for name in names))
    nulls = column.null_count()
    if nulls:
        parts.append(f"{nulls} of {len(column)} rows without a CRS (null)")
    if not len(column):
        parts.append("no rows")
    return " and ".join(parts)


def _label(column: pl.Series) -> str:
    """The CRS of a ``bim_crs`` column as a display shows it: one name, or what it holds."""
    names = _names(column)
    if len(names) == 1 and not column.null_count():
        return repr(names[0])
    return f"({_describe(column)})"

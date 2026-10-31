"""The BimDataFrame: a polars DataFrame of BIM elements with their CRS in a column."""

from __future__ import annotations

import functools
import importlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any, Literal

import polars as pl
import pyproj
import shapely

from bimpolars.crs import CRSLike, resolve_crs
from bimpolars.geoparquet import (
    COLUMN_CRS,
    GEO_METADATA_KEY,
    GEOG_COLUMN,
    GEOMETRY_COLUMNS,
    geo_metadata,
)

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

_WKT_LENGTH = 60
"""How many characters of a geometry's WKT a display shows."""

ConcatMethod = Literal["vertical", "vertical_relaxed", "diagonal", "diagonal_relaxed"]
"""The ``how`` of :func:`concat`: the polars concat strategies that stack rows."""


class BimDataFrame:
    """BIM elements: a polars DataFrame with the CRS of their geometry in a column.

    The frame has one row per element, with or without geometry. The columns the readers fill
    are listed in ``MODEL_COLUMNS``. The geometry columns ``mesh``, ``bbox`` and
    ``native_geom`` are in the row's BIM CRS, as are the simple-feature columns
    ``bimpolars.simple_features`` adds (``footprint`` and ``lightbim``, WKB); ``geog`` is
    always OGC:CRS84h. The BIM CRS sits in the ``CRS_COLUMN`` column (``bim_crs``) as
    PROJJSON text, the same value on every row of one model, so it travels with the rows
    through every polars operation: a concat of models in different CRSs keeps each row's.
    The column is the one place the CRS is kept: ``crs`` reads it on every call.

    Polars methods work on the BimDataFrame itself: ``bdf.filter(...)``, ``bdf["id"]``,
    ``"id" in bdf`` and ``pl.DataFrame(bdf)`` go to ``frame``, and a result that is a
    DataFrame with a Categorical ``bim_crs`` column comes back as a BimDataFrame. A polars
    method that takes a function (``pipe``, a ``group_by`` loop) hands it the plain frame,
    which the bimpolars functions take as well. ``concat`` stacks models.

    Args:
        frame (pl.DataFrame): The elements, one row each. A ``bim_crs`` column of PROJJSON
            text, String or Categorical, becomes ``CRS_DTYPE``; pyproj must read every value.
        crs (CRSLike | None): The BIM CRS of a frame without the ``bim_crs`` column, in any
            form: PROJJSON (see ``bimpolars.engineering_crs``), a ``pyproj.CRS``, or user
            input such as ``"EPSG:7415"``. It is stored as PROJJSON text in a new ``bim_crs``
            column, where ``MODEL_COLUMNS`` puts it (see ``with_crs_column``). A frame that
            has the column already must hold ``crs`` on every row; ``bimpolars.set_crs``
            replaces a CRS.

    Attributes:
        frame (pl.DataFrame): The elements, one row each, ``bim_crs`` included.
        crs (pyproj.CRS): The BIM CRS as a ``pyproj.CRS``, like ``GeoDataFrame.crs``.
        projjson (dict[str, Any]): The BIM CRS as a PROJJSON dict.

    Raises:
        TypeError: If ``frame`` is not a polars DataFrame.
        TypeError: If the ``bim_crs`` column is neither String nor Categorical.
        ValueError: If the frame has no ``bim_crs`` column and ``crs`` is not given.
        ValueError: If ``crs`` is given and the ``bim_crs`` column does not hold it on every row.
        ValueError: If a ``bim_crs`` value or ``crs`` is not a CRS pyproj can read.
        ValueError: If ``crs`` is PROJJSON without a ``type``.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: pl.DataFrame, crs: CRSLike | None = None) -> None:
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
            texts = _texts(frame[CRS_COLUMN])
            for text in texts:
                _crs_of(text)
        if crs is not None:
            projjson = _as_projjson(crs)
            given = _crs_of(crs_text(projjson))
            if CRS_COLUMN not in frame.columns:
                frame = with_crs_column(frame, projjson)
            elif frame[CRS_COLUMN].null_count() or not all(
                _crs_of(text).is_exact_same(given) for text in texts
            ):
                raise ValueError(
                    f"the frame's {CRS_COLUMN} column does not hold the CRS {given.name!r} on "
                    f"every row: it is {_describe(frame[CRS_COLUMN])}; use bimpolars.set_crs "
                    "to replace it"
                )
        elif CRS_COLUMN not in frame.columns:
            raise ValueError(
                f"the frame has no {CRS_COLUMN} column: pass its CRS, BimDataFrame(frame, crs)"
            )
        self._frame = frame

    @property
    def frame(self) -> pl.DataFrame:
        """The elements, one row each, with the ``bim_crs`` column."""
        return self._frame

    @property
    def crs(self) -> pyproj.CRS:
        """The BIM CRS as a ``pyproj.CRS``: the one CRS of ``bim_crs``, read on every call.

        Values that pyproj reads as exactly the same CRS count as one, so the same PROJJSON
        with its keys in another order, or after a round trip through pyproj, is one CRS.
        An in-place polars method (``bdf.extend(...)``) changes the CRS with the rows.

        Raises:
            ValueError: If ``bim_crs`` holds more than one CRS, or nulls.
            ValueError: If the frame has no rows, and so no CRS.
        """
        return _crs_of(self._crs_text())

    @property
    def projjson(self) -> dict[str, Any]:
        """The BIM CRS as PROJJSON: the one value of ``bim_crs``, parsed.

        Raises:
            ValueError: As ``crs`` raises it.
        """
        projjson: dict[str, Any] = json.loads(self._crs_text())
        return projjson

    def _crs_text(self) -> str:
        """The ``bim_crs`` value of the one CRS the rows are in."""
        if CRS_COLUMN not in self._frame.columns:
            raise ValueError(
                f"the BimDataFrame lost its {CRS_COLUMN} column to an in-place polars method, "
                "such as drop_in_place"
            )
        column = self._frame[CRS_COLUMN]
        if not len(column):
            raise ValueError(
                f"a BimDataFrame with no rows has no CRS: the {CRS_COLUMN} of its rows gives it"
            )
        texts = _texts(column)
        first = _crs_of(texts[0]) if texts else None
        if (
            first is None
            or column.null_count()
            or not all(_crs_of(text).is_exact_same(first) for text in texts[1:])
        ):
            raise ValueError(
                f"a BimDataFrame needs one CRS here, but its {CRS_COLUMN} column holds "
                f"{_describe(column)}; take the rows of one CRS first, for example with "
                f"group_by({CRS_COLUMN!r})"
            )
        return texts[0]

    def __len__(self) -> int:
        return len(self._frame)

    def __contains__(self, key: object) -> bool:
        """Whether ``frame`` has a column of that name, as ``key in frame`` says."""
        return key in self._frame.columns

    def __iter__(self) -> Iterator[pl.Series]:
        """The columns of ``frame`` as Series, as a polars DataFrame iterates."""
        return iter(self._frame)

    def __getattr__(self, name: str) -> Any:
        """Hands polars attributes and methods through to ``frame``.

        A result that is a DataFrame with a Categorical ``bim_crs`` column comes back as a
        BimDataFrame; anything else as it is: a Series, a scalar, a frame without ``bim_crs``
        or one in which it became something else (counts from ``null_count``, text from
        ``describe``). BimDataFrame arguments go in as their frames, so
        ``bdf.join(other, on="id")`` works. A method that changes the frame in place changes
        this BimDataFrame's frame.
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
        """Indexes ``frame``: ``bdf["id"]`` is the Series, ``bdf[:10]`` a BimDataFrame."""
        return _wrap(self._frame[key])

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *dir(self._frame)})

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """The Arrow PyCapsule stream of ``frame``, for ``pl.DataFrame(bdf)`` and DuckDB."""
        return self._frame.__arrow_c_stream__(requested_schema)

    def __narwhals_dataframe__(self) -> object:
        """The narwhals frame of ``frame``, so that ``mo.ui.table(bdf)`` takes the model.

        marimo makes a table of anything narwhals reads, as altair and plotly chart it. A
        wrapper has no public way to hand narwhals its native frame, so this returns the
        compliant frame of ``narwhals.from_native(frame)``, a private attribute that
        ``test_narwhals_and_marimo_tables_read_the_frame`` pins.
        """
        nw = importlib.import_module("narwhals")  # only narwhals calls this, so it is there
        return nw.from_native(self._frame)._compliant_frame

    def __repr__(self) -> str:
        return f"BimDataFrame({len(self)} rows, crs={_label(self._frame)})\n{_named(self._frame)!r}"

    def _repr_html_(self) -> str:
        return (
            f"<p><b>BimDataFrame</b>: crs {_label(self._frame)}</p>"
            f"{_named(self._frame)._repr_html_()}"
        )

    def _display_(self) -> object:
        """What marimo shows: the CRS name above an interactive table of the frame.

        The table shows each row's ``bim_crs`` as the name of its CRS, so that a frame of
        several CRSs says which row is in which, and the WKB geometry columns as the start of
        their WKT. Only the display changes: the values stay PROJJSON and WKB.
        """
        mo = importlib.import_module("marimo")  # only marimo calls this, so it is there
        header = mo.md(f"**BimDataFrame**: crs {_label(self._frame)}")
        table = mo.ui.table(self._frame, format_mapping=_display_formats(self._frame))
        return mo.vstack([header, table])

    def with_frame(self, frame: pl.DataFrame) -> BimDataFrame:
        """Wraps another frame, in this model's CRS when the frame has no ``bim_crs`` column.

        A frame that kept ``bim_crs`` (a filtered or joined one) is in the CRS its rows say;
        one that lost it (a ``select`` without it, or a frame built from scratch) gets this
        model's.

        Args:
            frame (pl.DataFrame): The new elements.

        Returns:
            BimDataFrame: A BimDataFrame of ``frame``.

        Raises:
            ValueError: If ``frame`` lacks ``bim_crs`` and this model does not have one CRS.
        """
        if CRS_COLUMN in frame.columns:
            return BimDataFrame(frame)
        return BimDataFrame(frame, self.projjson)

    def equals(self, other: object) -> bool:
        """Whether another BimDataFrame has the same frame, ``bim_crs`` included.

        Args:
            other (object): Anything; only a BimDataFrame can be equal.

        Returns:
            bool: True if the frames are equal, nulls included.
        """
        if not isinstance(other, BimDataFrame):
            return False
        return self._frame.equals(other._frame)

    def write_parquet(self, file: str | Path | IO[bytes], **kwargs: Any) -> None:
        """Writes the frame to Parquet, ``bim_crs`` included.

        The file metadata gets a ``METADATA_KEY`` entry with the layout version,
        ``{"version": 2}``. Every column, nested ones included, reads back with its dtype
        through ``read_parquet``; ``bim_crs`` is stored once per row group, as a
        dictionary.

        A frame with a geometry column of ``bimpolars.geoparquet.GEOMETRY_COLUMNS`` is
        written as GeoParquet: the file metadata also gets the ``geo`` entry of
        ``bimpolars.geoparquet.geo_metadata``, with the first such column as the primary
        geometry (``geog`` in OGC:CRS84h, from ``bimpolars.geobim.with_geog``, before
        ``footprint`` in the BIM CRS, from ``bimpolars.simple_features``), each with its
        CRS, and ``bbox`` named as the covering of the columns in the BIM CRS (the x and y
        fields each footprint lies within), so that GeoParquet readers find the footprints
        with their CRS and skip row groups for a spatial filter. GeoParquet gives a column one
        CRS, so a frame with ``footprint`` or ``lightbim`` needs one BIM CRS; ``geog`` alone
        needs none, and a frame without rows writes those columns with a null CRS, which
        GeoParquet reads as unknown.

        Args:
            file (str | Path | IO[bytes]): Where to write: a path or a binary file object.
            **kwargs (Any): Passed on to ``polars.DataFrame.write_parquet``. A ``metadata`` dict
                is written too, except that its ``METADATA_KEY`` entry is replaced by the
                one described above, and its ``geo`` entry too when the frame has a geometry
                column.

        Raises:
            ValueError: If the frame has rows, ``footprint`` or ``lightbim``, and not one CRS.
        """
        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata[METADATA_KEY] = json.dumps({"version": METADATA_VERSION})
        present = [column for column in GEOMETRY_COLUMNS if column in self._frame.columns]
        if present:
            in_bim_crs = any(column not in COLUMN_CRS for column in present)
            crs = self.projjson if in_bim_crs and len(self) else None
            metadata[GEO_METADATA_KEY] = json.dumps(geo_metadata(self._frame, crs))
        self._frame.write_parquet(file, metadata=metadata, **kwargs)

    @classmethod
    def read_parquet(cls, source: str | Path | IO[bytes]) -> BimDataFrame:
        """Reads a Parquet file that ``write_parquet`` wrote.

        A file with the ``bim_crs`` column reads as it is, also without the
        ``METADATA_KEY`` entry (written by polars from ``frame``, say). A file of
        layout version 1 has its CRS in that entry instead, and gets the ``bim_crs`` column
        from it.

        Args:
            source (str | Path | IO[bytes]): The path or binary file object to read.

        Returns:
            BimDataFrame: The BimDataFrame, ``bim_crs`` included.

        Raises:
            ValueError: If the file has neither ``bim_crs`` nor ``METADATA_KEY`` of layout 1.
            ValueError: If its metadata has a layout this version does not read.
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
    """Reads a BimDataFrame from Parquet; see ``BimDataFrame.read_parquet``.

    Args:
        source (str | Path | IO[bytes]): The path or binary file object to read.

    Returns:
        BimDataFrame: The BimDataFrame, ``bim_crs`` included.
    """
    return BimDataFrame.read_parquet(source)


def concat(
    bdfs: Iterable[BimDataFrame | pl.DataFrame], *, how: ConcatMethod = "vertical"
) -> BimDataFrame:
    """Stacks models into one BimDataFrame, every row keeping its own CRS, like ``pl.concat``.

    Models in different CRSs stack too, each row with its own ``bim_crs``: the result then
    has no single ``BimDataFrame.crs``, and ``bimpolars.to_crs`` refuses it. Move the
    models into one CRS first (``bimpolars.to_crs``) to work on them as one.

    Args:
        bdfs (Iterable[BimDataFrame | pl.DataFrame]): The models: BimDataFrames, or polars
            DataFrames with the ``bim_crs`` column.
        how (ConcatMethod): How the columns line up, as for ``pl.concat``: ``"vertical"`` needs the
            same columns, ``"diagonal"`` fills the missing ones with nulls, and the ``_relaxed``
            variants cast to a common dtype.

    Returns:
        BimDataFrame: The rows of every model, in order.

    Raises:
        TypeError: If an item is neither a BimDataFrame nor a polars DataFrame.
        ValueError: If there are no models, or the result has no valid ``bim_crs`` column.
    """
    frames: list[pl.DataFrame] = []
    for bdf in bdfs:
        if isinstance(bdf, BimDataFrame):
            frames.append(bdf.frame)
        elif isinstance(bdf, pl.DataFrame):
            frames.append(bdf)
        else:
            raise TypeError(f"concat stacks BimDataFrames, not {type(bdf).__name__}")
    if not frames:
        raise ValueError("concat needs at least one BimDataFrame")
    return BimDataFrame(pl.concat(frames, how=how))


def as_bim_dataframe(bdf: BimDataFrame | pl.DataFrame) -> BimDataFrame:
    """A model as a BimDataFrame, for the functions that take either.

    Args:
        bdf (BimDataFrame | pl.DataFrame): A BimDataFrame, or a polars DataFrame with the
            ``bim_crs`` column (one that a polars method such as ``pipe`` or a ``group_by`` loop
            handed out).

    Returns:
        BimDataFrame: ``bdf`` itself, or the frame as a BimDataFrame.

    Raises:
        TypeError: As ``BimDataFrame`` raises it, for anything else.
        ValueError: As ``BimDataFrame`` raises it, for a frame without a valid ``bim_crs``.
    """
    return bdf if isinstance(bdf, BimDataFrame) else BimDataFrame(bdf)


def with_crs_column(frame: pl.DataFrame, projjson: Mapping[str, Any]) -> pl.DataFrame:
    """The frame with ``bim_crs`` set to one CRS on every row.

    A frame that has the column keeps it in its place. A frame without it gets it where
    ``MODEL_COLUMNS`` puts it, right after the frame's last model column (``bbox`` in a
    model a reader built), or last if it has none, so that a model has one column order
    however it got its CRS.

    Args:
        frame (pl.DataFrame): The elements.
        projjson (Mapping[str, Any]): The CRS as PROJJSON.

    Returns:
        pl.DataFrame: The frame with the CRS in ``bim_crs``, as ``CRS_DTYPE``.
    """
    column = pl.lit(crs_text(projjson), dtype=CRS_DTYPE).alias(CRS_COLUMN)
    if CRS_COLUMN in frame.columns:
        return frame.with_columns(column)
    names = frame.columns
    model = [index for index, name in enumerate(names) if name in MODEL_COLUMNS]
    at = model[-1] + 1 if model else len(names)
    return frame.select(*names[:at], column, *names[at:])


def crs_text(projjson: Mapping[str, Any]) -> str:
    """The text a ``bim_crs`` column holds for a CRS: its PROJJSON, compact.

    Args:
        projjson (Mapping[str, Any]): The CRS as PROJJSON.

    Returns:
        str: The JSON text, without spaces.
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
    """A DataFrame with a Categorical ``bim_crs`` as a BimDataFrame; anything else as it is.

    Only Categorical: polars keeps that dtype through the methods that keep the column a CRS
    (``filter``, ``join``, ``with_columns``), and gives another to the ones that turn it into
    something else (``null_count``, ``describe``, ``implode``).
    """
    if isinstance(value, pl.DataFrame) and value.schema.get(CRS_COLUMN) == CRS_DTYPE:
        return BimDataFrame(value)
    return value


def _unwrap(value: Any) -> Any:
    """The frame of a BimDataFrame, for a polars method that takes a DataFrame."""
    return value.frame if isinstance(value, BimDataFrame) else value


def _texts(column: pl.Series) -> list[str]:
    """The distinct values of a ``bim_crs`` column, in order of first appearance."""
    texts: list[str] = column.drop_nulls().unique(maintain_order=True).cast(pl.String).to_list()
    return texts


@functools.lru_cache(maxsize=64)
def _crs_of(text: str) -> pyproj.CRS:
    """The CRS of a ``bim_crs`` value, parsed once for every frame that holds it."""
    try:
        projjson = json.loads(text)
        if not isinstance(projjson, dict) or "type" not in projjson:
            raise ValueError("it is not a JSON object with a 'type'")
        return pyproj.CRS.from_json_dict(projjson)
    except (ValueError, pyproj.exceptions.CRSError) as error:
        start = text if len(text) <= 60 else f"{text[:60]}..."
        raise ValueError(
            f"the {CRS_COLUMN} column must hold CRSs as PROJJSON text, but it holds "
            f"{start!r}: {error}"
        ) from None


def _names(column: pl.Series) -> list[str]:
    """The names of the CRSs a ``bim_crs`` column holds, in order of first appearance."""
    return [_crs_of(text).name for text in _texts(column)]


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


def _named(frame: pl.DataFrame) -> pl.DataFrame:
    """The frame with each ``bim_crs`` value as the name of its CRS, for a text display."""
    if CRS_COLUMN not in frame.columns:
        return frame
    names = {text: _crs_of(text).name for text in _texts(frame[CRS_COLUMN])}
    return frame.with_columns(pl.col(CRS_COLUMN).cast(pl.String).replace(names))


def _display_formats(frame: pl.DataFrame) -> dict[str, Callable[[Any], str | None]]:
    """The ``format_mapping`` of marimo's table: CRS names for ``bim_crs``, WKT for WKB."""
    formats: dict[str, Callable[[Any], str | None]] = {}
    if CRS_COLUMN in frame.columns:
        formats[CRS_COLUMN] = _crs_name
    for name in GEOMETRY_COLUMNS:
        if frame.schema.get(name) == pl.Binary:
            # geog is in degrees, where the seventh decimal is about a centimetre
            precision = 7 if name == GEOG_COLUMN else 3
            formats[name] = functools.partial(_wkt, precision=precision)
    return formats


def _crs_name(text: str | None) -> str | None:
    """The name of the CRS of a ``bim_crs`` value, for a display."""
    return None if text is None else _crs_of(text).name


def _wkt(wkb: bytes | None, *, precision: int) -> str | None:
    """The start of a WKB geometry's WKT, for a display."""
    if wkb is None:
        return None
    geometry = shapely.from_wkb(wkb, on_invalid="ignore")
    if geometry is None:
        return "(not WKB)"
    text = shapely.to_wkt(geometry, rounding_precision=precision)
    return text if len(text) <= _WKT_LENGTH else f"{text[:_WKT_LENGTH]}…"


def _label(frame: pl.DataFrame) -> str:
    """The CRS of a frame's ``bim_crs`` as a display shows it: one name, or what it holds."""
    if CRS_COLUMN not in frame.columns:
        return f"(no {CRS_COLUMN} column)"
    column = frame[CRS_COLUMN]
    names = _names(column)
    if len(names) == 1 and not column.null_count():
        return repr(names[0])
    return f"({_describe(column)})"

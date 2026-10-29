"""GeoParquet metadata for the geometry columns of a BimDataFrame.

A Parquet file is `GeoParquet <https://geoparquet.org>`_ when its file metadata has a ``geo``
key that names the geometry columns, their encoding and their CRS. :func:`geo_metadata` builds
that entry for a frame with a WKB geometry column from :data:`GEOMETRY_COLUMNS`, with the
model CRS as PROJJSON and the ``bbox`` column as the covering of the primary geometry, so that
DuckDB spatial, geopandas and GDAL read the frame with its CRS. The ``mesh`` and
``native_geom`` columns share the model CRS implicitly: they are not GeoParquet geometries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

GEO_METADATA_KEY = "geo"
"""The Parquet file metadata key of the GeoParquet metadata."""

GEOPARQUET_VERSION = "1.1.0"
"""The GeoParquet specification version the metadata follows."""

GEOMETRY_COLUMNS: tuple[str, ...] = ("footprint",)
"""The WKB geometry columns a BimDataFrame can hold; the first one present is the primary."""

BBOX_COLUMN = "bbox"
"""The column that covers the primary geometry: a Struct with xmin, ymin, xmax and ymax."""

_BBOX_FIELDS = ("xmin", "ymin", "xmax", "ymax")

_WKB_TYPES = {
    1: "Point",
    2: "LineString",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
}

_EWKB_Z = 0x80000000
_EWKB_M = 0x40000000
_EWKB_SRID = 0x20000000


def geo_metadata(frame: pl.DataFrame, crs: Mapping[str, Any]) -> dict[str, Any] | None:
    """Builds the GeoParquet ``geo`` metadata of a frame.

    Every column of :data:`GEOMETRY_COLUMNS` the frame has is listed with ``encoding`` WKB,
    the geometry types found in its values, the model CRS as PROJJSON and, when the frame has a
    :data:`BBOX_COLUMN` Struct, the file's bounding box and that column as ``covering``.

    Args:
        frame: The frame about to be written.
        crs: The model CRS as PROJJSON.

    Returns:
        The metadata as a JSON-ready dict, or None when the frame has no geometry column.

    Raises:
        TypeError: If a geometry column is not Binary.
        ValueError: If a geometry column holds WKB that GeoParquet does not allow.
    """
    present = [name for name in GEOMETRY_COLUMNS if name in frame.columns]
    if not present:
        return None
    covering = BBOX_COLUMN if _is_bbox(frame.schema.get(BBOX_COLUMN)) else None
    columns: dict[str, Any] = {}
    for name in present:
        if frame.schema[name] != pl.Binary:
            raise TypeError(
                f"the geometry column {name!r} must be Binary WKB, not {frame.schema[name]}"
            )
        column: dict[str, Any] = {
            "encoding": "WKB",
            "geometry_types": wkb_geometry_types(frame[name]),
            "crs": dict(crs),
        }
        if covering is not None:
            bounds = frame.select(
                pl.col(covering).struct.field("xmin").min(),
                pl.col(covering).struct.field("ymin").min(),
                pl.col(covering).struct.field("xmax").max(),
                pl.col(covering).struct.field("ymax").max(),
            ).row(0)
            if None not in bounds:
                column["bbox"] = [float(value) for value in bounds]
            column["covering"] = {"bbox": {field: [covering, field] for field in _BBOX_FIELDS}}
        columns[name] = column
    return {"version": GEOPARQUET_VERSION, "primary_column": present[0], "columns": columns}


def _is_bbox(dtype: pl.DataType | None) -> bool:
    """Whether a dtype is a bounding box Struct with float xmin, ymin, xmax and ymax."""
    if not isinstance(dtype, pl.Struct):
        return False
    fields = dtype.to_schema()
    return all(fields.get(name) in (pl.Float32, pl.Float64) for name in _BBOX_FIELDS)


def wkb_geometry_types(wkb: pl.Series) -> list[str]:
    """The GeoParquet ``geometry_types`` of a column of WKB values.

    Only the type codes are read, not the coordinates. ISO WKB (``1003`` for a Polygon Z) and
    extended WKB (the ``0x80000000`` flag) are both understood.

    Args:
        wkb: A Binary Series of WKB geometries; nulls are skipped.

    Returns:
        The names of the geometry types present, such as ``["Polygon Z", "MultiPolygon Z"]``,
        in the order of their type codes; empty for a column without values.

    Raises:
        ValueError: If a value is not WKB, or has M coordinates (not allowed in GeoParquet).
    """
    values = wkb.drop_nulls()
    if values.is_empty():
        return []
    header = pl.DataFrame({"wkb": values}).select(
        order=pl.col("wkb").bin.slice(0, 1),
        little=pl.col("wkb").bin.slice(1, 4).bin.reinterpret(dtype=pl.UInt32, endianness="little"),
        big=pl.col("wkb").bin.slice(1, 4).bin.reinterpret(dtype=pl.UInt32, endianness="big"),
    )
    if header["order"].null_count() or not header["order"].is_in([b"\x00", b"\x01"]).all():
        raise ValueError("the geometry column holds values that are not WKB")
    codes = header.select(
        code=pl.when(pl.col("order") == b"\x01").then(pl.col("little")).otherwise(pl.col("big"))
    )["code"]
    names: dict[int, str] = {}
    for code in sorted(int(code) for code in codes.unique()):
        names[code] = _type_name(code)
    return list(dict.fromkeys(names.values()))


def _type_name(code: int) -> str:
    """The GeoParquet name of a WKB geometry type code."""
    has_z = bool(code & _EWKB_Z)
    has_m = bool(code & _EWKB_M)
    base = code & ~(_EWKB_Z | _EWKB_M | _EWKB_SRID)
    dimension, base = divmod(base, 1000)
    has_z = has_z or dimension in (1, 3)
    has_m = has_m or dimension in (2, 3)
    if base not in _WKB_TYPES or dimension > 3:
        raise ValueError(f"unknown WKB geometry type code {code}")
    if has_m:
        raise ValueError("GeoParquet does not allow geometries with M coordinates")
    return _WKB_TYPES[base] + (" Z" if has_z else "")

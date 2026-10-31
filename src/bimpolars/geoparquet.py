"""GeoParquet metadata for the geometry columns of a BimDataFrame.

A Parquet file is `GeoParquet <https://geoparquet.org>`_ when its file metadata has a ``geo``
key that names the geometry columns, their encoding and their CRS. :func:`geo_metadata` builds
that entry for a frame with a WKB geometry column from :data:`GEOMETRY_COLUMNS`, each with its
CRS as PROJJSON (the BIM CRS, except for the columns of :data:`COLUMN_CRS`: ``geog`` is
always OGC:CRS84h), so that DuckDB spatial, geopandas and GDAL read the frame with its CRS.
When the frame has its ``bbox`` column, the entry also names it as the *covering* of each
column in the BIM CRS: a pointer that tells a reader that each row's geometry lies within
``bbox.xmin..xmax`` and ``bbox.ymin..ymax``, so that a spatial filter can skip row groups. A
GeoParquet covering has those four slots and no z; the column itself keeps all six fields.
``geog`` is in longitude and latitude, which ``bbox`` does not cover, so its entry gets the
bounds of its own coordinates and no covering. The ``mesh`` and ``native_geom`` columns share
the BIM CRS of the ``bim_crs`` column: they are not GeoParquet geometries.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
import shapely

from bimpolars.crs import CRS84H

GEO_METADATA_KEY = "geo"
"""The Parquet file metadata key of the GeoParquet metadata."""

GEOPARQUET_VERSION = "1.1.0"
"""The GeoParquet specification version the metadata follows."""

GEOG_COLUMN = "geog"
"""The column of :func:`bimpolars.geobim.with_geog`: the footprints in OGC:CRS84h."""

GEOMETRY_COLUMNS: tuple[str, ...] = (GEOG_COLUMN, "footprint", "lightbim")
"""The WKB geometry columns a BimDataFrame can hold; the first one present is the primary.

``geog`` (the footprints in OGC:CRS84h, from :func:`bimpolars.geobim.with_geog`) comes before
``footprint`` (in the BIM CRS, from :func:`bimpolars.simple_features`): a reader that takes
the primary column then has longitude and latitude, which export to GeoJSON as they are.
``lightbim`` (in the BIM CRS, the structural-analysis reduction of
:func:`bimpolars.simple_features` with ``lightbim=True``) comes last.
"""

COLUMN_CRS: dict[str, dict[str, Any]] = {GEOG_COLUMN: CRS84H}
"""The CRS of the geometry columns that are not in the BIM CRS, as PROJJSON.

``geog`` is always OGC:CRS84h (:data:`bimpolars.crs.CRS84H`), whatever the BIM CRS is, so
:func:`bimpolars.geobim.to_crs` never moves it.
"""

BBOX_COLUMN = "bbox"
"""The column named as the covering of the primary geometry, when it is a Struct with the
float fields xmin, ymin, xmax and ymax (zmin and zmax may sit there too: a covering has no z)."""

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

_EWKB_FLAGS = 0xE0000000
"""The Z, M and SRID bits of extended WKB, which GeoParquet does not allow."""


def geo_metadata(frame: pl.DataFrame, crs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Builds the GeoParquet ``geo`` metadata of a frame.

    Every column of :data:`GEOMETRY_COLUMNS` the frame has is listed with ``encoding`` WKB,
    the geometry types found in its values and its CRS as PROJJSON: the BIM CRS, or the
    column's own from :data:`COLUMN_CRS` (``geog`` is in OGC:CRS84h). When the frame has a
    :data:`BBOX_COLUMN` Struct, each column in the BIM CRS also gets that column as its
    ``covering`` (the x and y fields: a GeoParquet covering has no z slot) and a ``bbox`` of
    the whole file. That file ``bbox`` is x and y only as well, on purpose: a footprint lies
    flat at its storey's elevation, which is often outside the mesh's ``zmin..zmax`` that the
    column holds, so the mesh's z range would not contain it. A column in another CRS, which
    ``bbox`` does not cover, gets the x and y bounds of its own coordinates as the file
    ``bbox`` and no covering.

    Args:
        frame: The frame about to be written.
        crs: The BIM CRS as PROJJSON, or None when it is unknown (a frame without rows has
            none): the columns in it then get a null ``crs``, GeoParquet's unknown CRS.

    Returns:
        The metadata as a JSON-ready dict, or None when the frame has no geometry column.

    Raises:
        TypeError: If a geometry column is not Binary.
        ValueError: If a geometry column holds values that are not ISO WKB, or WKB that
            GeoParquet does not allow (see :func:`_wkb_geometry_types`).
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
        own_crs = COLUMN_CRS.get(name)
        column_crs = own_crs if own_crs is not None else None if crs is None else dict(crs)
        column: dict[str, Any] = {
            "encoding": "WKB",
            "geometry_types": _wkb_geometry_types(frame[name]),
            "crs": copy.deepcopy(column_crs),
        }
        if own_crs is not None:
            bounds = _wkb_bounds(frame[name])
            if bounds is not None:
                column["bbox"] = bounds
        elif covering is not None:
            extent = frame.select(
                pl.col(covering).struct.field("xmin").min(),
                pl.col(covering).struct.field("ymin").min(),
                pl.col(covering).struct.field("xmax").max(),
                pl.col(covering).struct.field("ymax").max(),
            ).row(0)
            if None not in extent:
                column["bbox"] = [float(value) for value in extent]
            column["covering"] = {"bbox": {field: [covering, field] for field in _BBOX_FIELDS}}
        columns[name] = column
    return {"version": GEOPARQUET_VERSION, "primary_column": present[0], "columns": columns}


def _wkb_bounds(wkb: pl.Series) -> list[float] | None:
    """The x and y bounds ``[xmin, ymin, xmax, ymax]`` of a column of WKB geometries.

    None for a column without a non-empty geometry.
    """
    bounds = shapely.bounds(shapely.from_wkb(wkb.drop_nulls().to_numpy()))
    if len(bounds) == 0 or not np.isfinite(bounds[:, 0]).any():
        return None
    low = np.nanmin(bounds[:, :2], axis=0)
    high = np.nanmax(bounds[:, 2:], axis=0)
    return [float(value) for value in (*low, *high)]


def _is_bbox(dtype: pl.DataType | None) -> bool:
    """Whether a dtype is a bounding box Struct with float xmin, ymin, xmax and ymax."""
    if not isinstance(dtype, pl.Struct):
        return False
    fields = dtype.to_schema()
    return all(fields.get(name) in (pl.Float32, pl.Float64) for name in _BBOX_FIELDS)


def _wkb_geometry_types(wkb: pl.Series) -> list[str]:
    """The GeoParquet ``geometry_types`` of a column of ISO WKB values.

    Only the type codes are read, not the coordinates. GeoParquet's ``WKB`` encoding is ISO
    WKB (``1003`` for a Polygon Z), so extended WKB, with the Z, M or SRID flag of PostGIS in
    the type code, is refused rather than described as ISO.

    Args:
        wkb: A Binary Series of WKB geometries; nulls are skipped.

    Returns:
        The names of the geometry types present, such as ``["Polygon Z", "MultiPolygon Z"]``,
        in the order of their type codes; empty for a column without values.

    Raises:
        ValueError: If a value is not WKB, is extended WKB, or has M coordinates (not allowed
            in GeoParquet).
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
    """The GeoParquet name of an ISO WKB geometry type code."""
    if code & _EWKB_FLAGS:
        raise ValueError(
            f"geometry type code {code:#x} is extended WKB; GeoParquet takes ISO WKB only"
        )
    dimension, base = divmod(code, 1000)
    if base not in _WKB_TYPES or dimension > 3:
        raise ValueError(f"unknown WKB geometry type code {code}")
    if dimension >= 2:
        raise ValueError("GeoParquet does not allow geometries with M coordinates")
    return _WKB_TYPES[base] + (" Z" if dimension == 1 else "")

"""Georeferencing: the Engineering CRS of a model, and moving a model to another CRS.

A model fresh from a reader sits in its own Cartesian coordinates, a plain EngineeringCRS.
Georeferencing says where that is on earth: :func:`derived_engineering_crs` turns a site origin
and a rotation on a projected plus vertical CRS such as EPSG:7415 (RD New + NAP) into an
Engineering CRS that PROJ can transform, :func:`set_crs` labels a model with it,
:func:`to_crs` moves the geometry columns to any CRS pyproj knows, and :func:`with_geog` adds
the footprints in OGC:CRS84h next to the model's own, for GeoJSON and GeoParquet readers on
the web.
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from typing import Any, Protocol

import numpy as np
import polars as pl
import pyproj
import pyproj.exceptions
import shapely
from pyproj.crs import CompoundCRS
from pyproj.transformer import TransformerGroup

from bimpolars.crs import (
    CRS84H,
    PROJJSON_SCHEMA,
    CRSLike,
    LinearUnit,
    _linear_unit,
    as_linear_unit,
    resolve_crs,
    unit_projjson,
)
from bimpolars.features import _clamp_xy
from bimpolars.frame import BBOX_DTYPE, MESH_DTYPE, NATIVE_GEOM_DTYPE, SWEEP_DTYPE, BimDataFrame
from bimpolars.geoparquet import COLUMN_CRS, GEOG_COLUMN, GEOMETRY_COLUMNS
from bimpolars.lightbim import QA_DTYPE

WKB_COLUMNS: tuple[str, ...] = tuple(
    column for column in GEOMETRY_COLUMNS if column not in COLUMN_CRS
)
"""The WKB columns in the model CRS that :func:`to_crs` moves, when present.

They are the :data:`bimpolars.geoparquet.GEOMETRY_COLUMNS` without the columns in a CRS of
their own (:data:`bimpolars.geoparquet.COLUMN_CRS`): ``geog``, the footprints in OGC:CRS84h
of :func:`with_geog`, never changes CRS.
"""

AFFINE_TOLERANCE = 1e-6
"""How far, in metres, a transformation may stray from an affine fit and still be affine."""

_PROBE_HALF_SIZE = 500.0
"""The least half size, in metres, of the cube the affine fit of :func:`to_crs` probes."""

_PROBE_HALF_SIZE_DEGREES = 0.01
"""The same, in degrees, for a model stored in a geographic CRS: about a kilometre."""

_BBOX_FIELDS = ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
"""Warnings of this module point at the caller outside bimpolars."""


def derived_engineering_crs(
    base_crs: CRSLike,
    easting: float,
    northing: float,
    orthometric_height: float,
    rotation: float,
    *,
    unit: LinearUnit | CRSLike | BimDataFrame,
    name: str = "Site",
) -> pyproj.CRS:
    """Derives an Engineering CRS from a projected plus vertical CRS by georeferencing parameters.

    The site origin (0, 0, 0) lands on ``(easting, northing)`` in the projected CRS and on
    ``orthometric_height`` in the vertical CRS, and the site's y-axis points ``rotation``
    degrees counterclockwise from grid north. The scale between site and map is not a
    parameter: it follows from ``unit`` and the base CRS's own units.

    The result is a CompoundCRS of a DerivedProjectedCRS with an Affine parametric
    transformation (EPSG:9624) on the projected CRS and a DerivedVerticalCRS with a Vertical
    Offset (EPSG:9616) on the vertical CRS, with all three axes in ``unit``. PROJ applies a
    deriving conversion from the base to the derived CRS, so the affine parameters hold the
    inverse of the georeferencing (map to site), and the factor between the map unit and the
    site unit sits in the coefficients.

    Args:
        base_crs: A compound CRS of a projected and a vertical CRS, such as ``"EPSG:7415"``
            (RD New + NAP height).
        easting: Easting of the site origin, in the projected CRS's unit.
        northing: Northing of the site origin, in the projected CRS's unit.
        orthometric_height: Height of the site origin, in the vertical CRS's unit.
        rotation: The angle in degrees from grid north to the site's y-axis, counterclockwise
            positive; equally, from grid east to the site's x-axis. Grid north, not true north.
            For an IfcMapConversion it is ``degrees(atan2(XAxisOrdinate, XAxisAbscissa))``.
        unit: The site's length unit: the model whose site it is (``unit=bdf``), whose unit
            it copies, or any other unit :func:`bimpolars.crs.as_linear_unit` takes: a name
            from :data:`bimpolars.crs.LINEAR_UNITS`, a :class:`bimpolars.LinearUnit`, or a
            CRS.
        name: The site's name, used in the names of the CRS and its parts.

    Returns:
        The Engineering CRS, for :func:`set_crs`.

    Raises:
        ValueError: If ``base_crs`` is not a compound CRS of a projected and a vertical CRS
            that PROJ can derive from, or ``unit`` is neither a unit nor a CRS with a linear
            unit.
    """
    base, _ = resolve_crs(base_crs)
    components = base.sub_crs_list
    if len(components) != 2 or not components[0].is_projected or not components[1].is_vertical:
        raise ValueError(
            f"base_crs must be a compound CRS of a projected and a vertical CRS, such as "
            f"EPSG:7415; {base.name!r} is a {base.type_name}"
        )
    projected, vertical = components
    site_unit = unit_projjson(as_linear_unit(unit))
    map_unit = projected.axis_info[0].unit_conversion_factor  # metres per map unit
    height_axis = vertical.axis_info[0]
    theta = math.radians(rotation)
    k = map_unit / site_unit["conversion_factor"]  # map units to site units
    a1, a2 = k * math.cos(theta), k * math.sin(theta)
    b1, b2 = -k * math.sin(theta), k * math.cos(theta)
    a0 = -(a1 * easting + a2 * northing)
    b0 = -(b1 * easting + b2 * northing)

    def parameter(key: str, value: float, unit_json: dict[str, Any] | str) -> dict[str, Any]:
        return {"name": key, "value": float(value), "unit": unit_json}

    site_grid = {
        "type": "DerivedProjectedCRS",
        "name": f"{name} site grid",
        "base_crs": projected.to_json_dict(),
        "conversion": {
            "name": f"{name} georeferencing",
            "method": {
                "name": "Affine parametric transformation",
                "id": {"authority": "EPSG", "code": 9624},
            },
            "parameters": [
                parameter("A0", a0, site_unit),
                parameter("A1", a1, "unity"),
                parameter("A2", a2, "unity"),
                parameter("B0", b0, site_unit),
                parameter("B1", b1, "unity"),
                parameter("B2", b2, "unity"),
            ],
        },
        "coordinate_system": {
            "subtype": "Cartesian",
            "axis": [
                {
                    "name": "Site x",
                    "abbreviation": "x",
                    "direction": "unspecified",
                    "unit": site_unit,
                },
                {
                    "name": "Site y",
                    "abbreviation": "y",
                    "direction": "unspecified",
                    "unit": site_unit,
                },
            ],
        },
    }
    site_z = {
        "type": "DerivedVerticalCRS",
        "name": f"{name} site z",
        "base_crs": vertical.to_json_dict(),
        "conversion": {
            "name": f"{name} orthometric height",
            "method": {"name": "Vertical Offset", "id": {"authority": "EPSG", "code": 9616}},
            "parameters": [
                {
                    "name": "Vertical Offset",
                    "value": -float(orthometric_height),
                    "unit": unit_projjson(
                        LinearUnit(height_axis.unit_name, height_axis.unit_conversion_factor)
                    ),
                    "id": {"authority": "EPSG", "code": 8603},
                }
            ],
        },
        "coordinate_system": {
            "subtype": "vertical",
            "axis": [{"name": "Site z", "abbreviation": "z", "direction": "up", "unit": site_unit}],
        },
    }
    crs = {
        "$schema": PROJJSON_SCHEMA,
        "type": "CompoundCRS",
        "name": f"{name} Engineering CRS",
        "components": [site_grid, site_z],
    }
    try:
        return pyproj.CRS.from_json_dict(crs)
    except pyproj.exceptions.CRSError as error:
        raise ValueError(f"PROJ cannot build the Engineering CRS: {error}") from error


def set_crs(bdf: BimDataFrame, crs: CRSLike) -> BimDataFrame:
    """Labels a model with a CRS without moving its geometry, like ``GeoDataFrame.set_crs``.

    The geometry columns keep their values: ``crs`` states what they are in. Use it with the
    CRS of :func:`derived_engineering_crs` to georeference a model whose reader gave it a plain
    EngineeringCRS, with the same length unit.

    Args:
        bdf: The model.
        crs: The CRS: PROJJSON, a ``pyproj.CRS``, or user input such as ``"EPSG:7415"``.

    Returns:
        The same frame with ``crs`` as its CRS, as PROJJSON.

    Raises:
        ValueError: If pyproj cannot read ``crs``, or it does not have three axes.
    """
    parsed, projjson = resolve_crs(crs)
    _check_3d(parsed)
    return BimDataFrame(bdf.frame, projjson)


def to_crs(bdf: BimDataFrame, crs: CRSLike) -> BimDataFrame:
    """Moves a model to another CRS, like ``GeoDataFrame.to_crs``.

    One pyproj transformation from the model CRS to ``crs`` moves every coordinate of the
    geometry columns together: the ``mesh`` vertices, the WKB columns of :data:`WKB_COLUMNS`
    (``footprint`` and ``lightbim``), and ``elevation``, the level of each storey, taken at
    the plan centre of the model. ``bbox`` is recomputed from the
    moved meshes; a frame without a ``mesh`` column gets the axis-aligned bounds of the moved
    corners of each box instead, which still covers the moved geometry (loosely, under a
    rotation). The model's length unit becomes the target's: a model in millimetres moved to
    EPSG:7415 is in metres.

    ``native_geom``, the sweeps a reader placed in the model CRS
    (:data:`bimpolars.NATIVE_GEOM_DTYPE`), survives only an affine transformation (to the base
    CRS of a :func:`derived_engineering_crs`, for example): the matrix of
    :func:`affine_transformation` is composed onto every sweep's ``placement``, scale
    included, so that the profile and the depth stay in the sweep's own coordinates. The
    column is nulled when the transformation is not affine within :data:`AFFINE_TOLERANCE`
    over the model's extent (and at least a kilometre), as it is to a geographic, geocentric
    or differently projected CRS. ``lightbim_qa.deviation``, a length, follows the change of
    unit the same way: it is scaled by the matrix's linear scale (the cube root of the
    determinant of its 3x3 part, exact for the similarity of a derived engineering CRS), and
    nulled when the transformation is not affine.

    The transformation from or to a derived CRS, such as the compound of
    :func:`derived_engineering_crs`, goes through its base CRS, so that PROJ's geodetic steps
    see heights in the base CRS's unit (see :func:`_transformer`). When the best
    transformation PROJ knows needs grids that are not installed (RDNAPTRANS2018 needs the
    NSGI grids), a ``UserWarning`` names them and PROJ falls back to a less accurate
    operation.

    Args:
        bdf: The model, with a CRS pyproj can transform from: an EngineeringCRS of
            :func:`derived_engineering_crs`, or any georeferenced CRS.
        crs: The target CRS: PROJJSON, a ``pyproj.CRS``, or user input such as
            ``"EPSG:7415"`` or ``"OGC:CRS84h"``. It must have three axes, so that the meshes
            keep one unit per axis: ``"EPSG:28992"`` is refused, ``"EPSG:7415"`` works.

    Returns:
        The model in ``crs``, with the geometry columns moved and ``crs`` as PROJJSON.

    Raises:
        ValueError: If pyproj knows no transformation from the model CRS to ``crs`` (a plain
            EngineeringCRS is not georeferenced: :func:`set_crs` it first), if ``crs`` does
            not have three axes, or if a coordinate does not transform.
        TypeError: If a column of :data:`WKB_COLUMNS` is not Binary, ``native_geom`` holds
            values of another layout than :data:`bimpolars.NATIVE_GEOM_DTYPE` under an affine
            transformation, or ``lightbim_qa`` is not :data:`bimpolars.lightbim.QA_DTYPE`.
    """
    transformer, source, target, projjson = _prepare(bdf, crs)
    frame = bdf.frame
    columns: list[pl.Series] = []
    bbox: pl.Series | None = None
    if "mesh" in frame.columns:
        mesh, bbox = _transform_meshes(frame["mesh"], transformer)
        columns.append(mesh)
        if "bbox" in frame.columns:
            columns.append(bbox)
    elif "bbox" in frame.columns:
        bbox = _transform_bboxes(frame["bbox"], transformer)
        columns.append(bbox)
    if "elevation" in frame.columns:
        columns.append(_transform_elevations(frame, transformer))
    for name in WKB_COLUMNS:
        if name in frame.columns:
            moved = _transform_wkb(frame[name], transformer)
            if bbox is not None and "bbox" in frame.columns:
                moved = _clamp_to_bbox(moved, bbox)
            columns.append(moved)
    matrix: np.ndarray[Any, np.dtype[np.float64]] | None = None
    if "native_geom" in frame.columns or "lightbim_qa" in frame.columns:
        matrix = _affine_matrix(transformer, source, target, frame)
    if "native_geom" in frame.columns:
        if matrix is None:
            columns.append(pl.Series("native_geom", [None] * len(frame), dtype=pl.Null))
        elif frame.schema["native_geom"] == NATIVE_GEOM_DTYPE:
            columns.append(_place_sweeps(frame["native_geom"], matrix))
        elif frame.schema["native_geom"] != pl.Null:
            raise TypeError(
                f"native_geom has the dtype {frame.schema['native_geom']}, not "
                "bimpolars.NATIVE_GEOM_DTYPE, so to_crs cannot place it"
            )
    if "lightbim_qa" in frame.columns:
        columns.append(_scale_deviations(frame["lightbim_qa"], matrix))
    return BimDataFrame(frame.with_columns(columns), projjson)


def _scale_deviations(
    qa: pl.Series, matrix: np.ndarray[Any, np.dtype[np.float64]] | None
) -> pl.Series:
    """Moves the ``deviation`` of ``lightbim_qa``, a length, into the target CRS's unit.

    An affine matrix scales a length by the cube root of the determinant of its 3x3 part,
    exactly for a similarity; without one the deviation is nulled, since a length in degrees
    means nothing.
    """
    if qa.dtype != QA_DTYPE:
        raise TypeError(
            f"lightbim_qa has the dtype {qa.dtype}, not bimpolars.lightbim.QA_DTYPE, so to_crs "
            "cannot scale its deviation"
        )
    scale = None if matrix is None else abs(float(np.linalg.det(matrix[:3, :3]))) ** (1 / 3)
    deviation = pl.col("lightbim_qa").struct.field("deviation")
    scaled = pl.lit(None, dtype=pl.Float64) if scale is None else deviation * scale
    return qa.to_frame().select(
        pl.when(pl.col("lightbim_qa").is_null())
        .then(None)
        .otherwise(pl.col("lightbim_qa").struct.with_fields(scaled.alias("deviation")))
        .cast(QA_DTYPE)
        .alias("lightbim_qa")
    )["lightbim_qa"]


def _place_sweeps(
    native_geom: pl.Series, matrix: np.ndarray[Any, np.dtype[np.float64]]
) -> pl.Series:
    """Composes an affine matrix onto the ``placement`` of every sweep of a ``native_geom``.

    The other fields stay: the profile and the depth are in the sweep's own coordinates, and
    the composed placement, scale and all, takes them to the target CRS.
    """
    exploded = pl.DataFrame(
        {"row": np.arange(len(native_geom)), "sweep": native_geom.alias("sweep")}
    ).explode("sweep")
    placements = exploded["sweep"].struct.field("placement")
    present = placements.is_not_null().to_numpy()
    values = np.full((len(exploded), 16), np.nan)
    if present.any():
        old = placements.drop_nulls().to_numpy().reshape(-1, 4, 4)
        values[present] = (matrix[None, :, :] @ old).reshape(-1, 16)
    placed = pl.Series("placement", values, dtype=pl.Array(pl.Float64, 16))
    sweep = pl.col("sweep")
    rebuilt = exploded.with_columns(
        pl.when(sweep.is_null())
        .then(None)
        .otherwise(
            pl.struct(
                [
                    placed if field.name == "placement" else sweep.struct.field(field.name)
                    for field in SWEEP_DTYPE.fields
                ]
            )
        )
        .cast(SWEEP_DTYPE)
        .alias("sweep")
    )
    # Exploding a null or empty list gives one null row, so every row has a group; the rows
    # that were null stay null, instead of becoming empty lists
    grouped = rebuilt.group_by("row", maintain_order=True).agg(pl.col("sweep").drop_nulls())
    return pl.DataFrame(
        {"was": native_geom.alias("was"), "now": grouped["sweep"].alias("now")}
    ).select(
        pl.when(pl.col("was").is_null())
        .then(None)
        .otherwise(pl.col("now"))
        .cast(NATIVE_GEOM_DTYPE)
        .alias("native_geom")
    )["native_geom"]


def with_geog(bdf: BimDataFrame) -> BimDataFrame:
    """Adds ``geog``: the ``footprint`` of every row in OGC:CRS84h, for GeoJSON and the web.

    ``geog`` is the footprint moved to longitude and latitude in degrees and ellipsoidal height
    in metres on WGS 84 (:data:`bimpolars.crs.CRS84H`), as ISO WKB in a Binary column: a
    ``POLYGON Z`` or ``MULTIPOLYGON Z``, null exactly when ``footprint`` is, and ``POLYGON Z
    EMPTY`` where the footprint is. That is the CRS GeoJSON (RFC 7946) means by its
    coordinates, so the column exports to GeoJSON as it is, and
    :func:`bimpolars.geoparquet.geo_metadata` makes it the primary geometry column of the
    written GeoParquet, with :data:`bimpolars.crs.CRS84H` as its CRS. The model keeps its own
    CRS and its ``footprint`` next to it, and :func:`to_crs` leaves ``geog`` as it is;
    :func:`bimpolars.simple_features` drops it when it computes the footprints again, so run
    ``with_geog`` after it.

    The transformation is the one :func:`to_crs` applies for ``"OGC:CRS84h"``, through the base
    CRS of a derived Engineering CRS (see :func:`_transformer`), and it warns in the same way
    when PROJ's best transformation needs grids that are not installed: without the NSGI grids
    a Dutch site moves by a Helmert transformation that is good to about a metre, and its NAP
    heights pass off as ellipsoidal ones.

    Args:
        bdf: The model with its ``footprint`` column (see :func:`bimpolars.simple_features`),
            in a georeferenced CRS: the Engineering CRS of :func:`derived_engineering_crs`,
            set with :func:`set_crs`, or any CRS pyproj can transform to WGS 84.

    Returns:
        The model with ``geog`` added (or replaced), in the same CRS.

    Raises:
        ValueError: If the frame has no ``footprint`` column, or pyproj knows no
            transformation from the model CRS to WGS 84 (a plain EngineeringCRS is not
            georeferenced: :func:`set_crs` it first), or a coordinate does not transform.
        TypeError: If ``footprint`` is not Binary.
    """
    frame = bdf.frame
    if "footprint" not in frame.columns:
        raise ValueError(
            "with_geog needs the footprint column: add it with bimpolars.simple_features first"
        )
    transformer, _, _, _ = _prepare(bdf, CRS84H)
    geog = _transform_wkb(frame["footprint"], transformer).alias(GEOG_COLUMN)
    return bdf.with_frame(frame.with_columns(geog))


def affine_transformation(
    bdf: BimDataFrame, crs: CRSLike
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """The 4x4 matrix that :func:`to_crs` applies to a placement, when there is one.

    The transformation from the model CRS to ``crs`` is probed on the corners of a cube around
    the model (its extent, at least a kilometre across) and fitted by least squares. It is
    affine when no corner strays more than :data:`AFFINE_TOLERANCE` metres from the fit; a
    geographic or geocentric target never is. The matrix carries the change of unit: from a
    model in millimetres to EPSG:7415 its linear part has a scale of 0.001, which is why
    :func:`to_crs` composes it onto a sweep's ``placement`` rather than onto its profile.

    Args:
        bdf: The model.
        crs: The target CRS, as for :func:`to_crs`.

    Returns:
        The matrix from model coordinates to ``crs`` coordinates (homogeneous, row-major),
        or None when the transformation is not affine over the model.

    Raises:
        ValueError: As :func:`to_crs` raises it.
    """
    transformer, source, target, _ = _prepare(bdf, crs)
    return _affine_matrix(transformer, source, target, bdf.frame)


def linearized_site_to_ecef(crs: CRSLike) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Linearizes the transformation from a site's CRS to ECEF at the site origin.

    Viewers of the earth (CesiumJS, 3D Tiles) place a model with one 4x4 matrix into the
    geocentric frame EPSG:4978. This is that matrix for a model in metres: the origin is where
    the site origin lands, and each column is where one metre along a site axis points.

    Args:
        crs: The site's CRS, as :func:`derived_engineering_crs` gives it: PROJJSON, a
            ``pyproj.CRS``, or user input. Its axes must be in a linear unit.

    Returns:
        A 4x4 matrix from site metres to ECEF metres.

    Raises:
        ValueError: If the CRS's axes are not in a linear unit, or pyproj knows no
            transformation from it to ECEF.
    """
    source, projjson = resolve_crs(crs)
    unit = _linear_unit(projjson)
    to_ecef = _transformer(source, pyproj.CRS("EPSG:4978"))
    # One metre along each site axis, in site units
    origin_and_axes = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) / unit.conversion_factor
    )
    origin, x, y, z = _transform_points(to_ecef, origin_and_axes)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x - origin, y - origin, z - origin])
    matrix[:3, 3] = origin
    return matrix


def _prepare(
    bdf: BimDataFrame, crs: CRSLike
) -> tuple[_Transformation, pyproj.CRS, pyproj.CRS, dict[str, Any]]:
    """Reads the target CRS and builds the transformation from the model CRS to it.

    Returns:
        The transformation, the model CRS, the target CRS and the target's PROJJSON.
    """
    target, projjson = resolve_crs(crs)
    _check_3d(target)
    source = bdf.crs
    return _transformer(source, target), source, target, projjson


class _Transformation(Protocol):
    """What :func:`_transform_points` needs: a ``pyproj.Transformer`` or a chain of them."""

    @property
    def description(self) -> str: ...

    def transform(self, xx: Any, yy: Any, zz: Any) -> Any: ...


class _Chain:
    """Transformers applied one after the other."""

    def __init__(self, steps: list[pyproj.Transformer]) -> None:
        self.steps = steps

    @property
    def description(self) -> str:
        return " | ".join(step.description for step in self.steps)

    def transform(self, xx: Any, yy: Any, zz: Any) -> Any:
        for step in self.steps:
            xx, yy, zz = step.transform(xx, yy, zz)
        return xx, yy, zz


def _transformer(source: pyproj.CRS, target: pyproj.CRS) -> _Transformation:
    """The transformation from the model CRS to a target, with x, y, z in that order.

    A derived CRS (the compound of :func:`derived_engineering_crs`) is passed through its base
    CRS: PROJ otherwise orders the deriving conversion of the vertical CRS after a geodetic
    step, so that a 3D Helmert transformation reads a site's millimetres as metres. The
    conversion to the base CRS is exact, so the chain costs nothing.

    Warns:
        UserWarning: When the best transformation of a step needs grids that are not
            installed, so that PROJ falls back to a less accurate operation.
    """
    steps: list[tuple[pyproj.CRS, pyproj.CRS]] = []
    source_base, target_base = _base_crs(source), _base_crs(target)
    if source_base is not None:
        steps.append((source, source_base))
    steps.append((source_base or source, target_base or target))
    if target_base is not None:
        steps.append((target_base, target))
    try:
        transformers = [
            pyproj.Transformer.from_crs(start, end, always_xy=True) for start, end in steps
        ]
    except pyproj.exceptions.ProjError as error:
        raise ValueError(
            f"pyproj knows no transformation from the model CRS {source.name!r} to "
            f"{target.name!r}; a plain EngineeringCRS is not georeferenced: label the model "
            f"with set_crs and derived_engineering_crs first"
        ) from error
    for start, end in steps:
        _warn_about_missing_grids(start, end)
    if len(transformers) == 1:
        return transformers[0]
    return _Chain(transformers)


def _warn_about_missing_grids(start: pyproj.CRS, end: pyproj.CRS) -> None:
    """Warns when PROJ's best transformation between two CRSs needs grids it does not have.

    The warning is attributed to the caller outside bimpolars, so that Python's default filter
    shows it once per call site.
    """
    message = _missing_grids_message(start, end)
    if message is not None:
        warnings.warn(message, UserWarning, skip_file_prefixes=(_PACKAGE_DIR,))


@functools.cache
def _missing_grids_message(start: pyproj.CRS, end: pyproj.CRS) -> str | None:
    """The text of the grid warning for a pair of CRSs, or None when the best operation runs.

    Memoised per pair: enumerating the candidate operations costs about 0.2 s, and only the
    first check should disturb the warning registry (leaving ``catch_warnings`` invalidates
    it, so that a once-per-call-site filter would show the warning on every call).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(  # pyproj warns too, with less to go on
            "ignore", message="Best transformation is not available", module="pyproj.transformer"
        )
        group = TransformerGroup(start, end, always_xy=True)
    if group.best_available:
        return None
    grids = sorted(
        {
            grid.short_name
            for operation in group.unavailable_operations
            for grid in operation.grids
            if not grid.available
        }
    )
    # The lazy transformer runs the first available operation that is not a ballpark one
    fallback = next(
        (t.description for t in group.transformers if "Ballpark" not in t.description),
        "a ballpark operation",
    )
    return (
        f"the best transformation from {start.name!r} to {end.name!r} needs the grids "
        f"{', '.join(grids)}, which PROJ cannot find; it runs {fallback!r} instead, which is "
        f"less accurate (a Helmert transformation is good to about a metre, and a ballpark "
        f"vertical step passes orthometric heights off as ellipsoidal ones). Install the "
        f"grids or enable the PROJ network (pyproj.network.set_network_enabled(True))"
    )


def _base_crs(crs: pyproj.CRS) -> pyproj.CRS | None:
    """The CRS a derived CRS, or a compound with a derived component, derives from."""
    components = crs.sub_crs_list
    if components:
        if not any(component.is_derived for component in components):
            return None
        return CompoundCRS(
            name=f"base of {crs.name}",
            components=[
                component.source_crs if component.is_derived and component.source_crs else component
                for component in components
            ],
        )
    if crs.is_derived and crs.source_crs is not None:
        return crs.source_crs
    return None


def _check_3d(crs: pyproj.CRS) -> None:
    """Refuses a CRS without three axes: a model's geometry is 3D."""
    if len(crs.axis_info) != 3:
        raise ValueError(
            f"the CRS {crs.name!r} has {len(crs.axis_info)} axes; a model's geometry is 3D, so "
            f"pass a CRS with three, such as a compound of a projected and a vertical CRS "
            f"(EPSG:7415 instead of EPSG:28992)"
        )


def _transform_points(
    transformer: _Transformation, points: np.ndarray[Any, Any]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Transforms an ``(n, 3)`` array of points, refusing non-finite results."""
    x, y, z = transformer.transform(points[:, 0], points[:, 1], points[:, 2])
    moved = np.column_stack([x, y, z]).astype(np.float64)
    if not np.isfinite(moved).all():
        failed = int((~np.isfinite(moved).all(axis=1)).sum())
        raise ValueError(
            f"{failed} of {len(moved)} coordinates did not transform ({transformer.description})"
        )
    return moved


def _transform_meshes(mesh: pl.Series, transformer: _Transformation) -> tuple[pl.Series, pl.Series]:
    """Moves every vertex of a ``mesh`` column; gives the new column and its ``bbox``."""
    vertices = mesh.struct.field("vertices")
    counts = vertices.list.len().fill_null(0).cast(pl.Int64).to_numpy()
    flat = vertices.explode().drop_nulls().to_numpy()
    if len(flat) != counts.sum():
        raise ValueError("a mesh has null vertex coordinates")
    moved = _transform_points(transformer, flat.reshape(-1, 3))
    rows = np.repeat(np.arange(len(mesh)), counts // 3)
    index = pl.DataFrame({"row": np.arange(len(mesh))})
    coordinates = pl.DataFrame({"row": np.repeat(rows, 3), "v": moved.ravel()})
    rebuilt = index.join(
        coordinates.group_by("row", maintain_order=True).agg(pl.col("v")),
        on="row",
        how="left",
        maintain_order="left",
    )["v"].fill_null([])
    new_mesh = (
        pl.DataFrame([mesh.alias("mesh"), rebuilt.alias("vertices")])
        .select(
            pl.when(pl.col("mesh").is_null())
            .then(None)
            .otherwise(
                pl.struct(
                    pl.col("vertices"),
                    pl.col("mesh").struct.field("faces"),
                    pl.col("mesh").struct.field("styles"),
                )
            )
            .alias("mesh")
        )["mesh"]
        .cast(MESH_DTYPE)
    )
    points = pl.DataFrame({"row": rows, "x": moved[:, 0], "y": moved[:, 1], "z": moved[:, 2]})
    bounds = index.join(
        points.group_by("row", maintain_order=True).agg(
            xmin=pl.col("x").min(),
            ymin=pl.col("y").min(),
            zmin=pl.col("z").min(),
            xmax=pl.col("x").max(),
            ymax=pl.col("y").max(),
            zmax=pl.col("z").max(),
        ),
        on="row",
        how="left",
        maintain_order="left",
    )
    return new_mesh, _bbox_series(bounds)


def _transform_bboxes(bbox: pl.Series, transformer: _Transformation) -> pl.Series:
    """Moves a ``bbox`` column on its own: the bounds of the eight moved corners of each box.

    Under a rotation the result is looser than the box of the moved geometry, but it still
    covers it, which is what the GeoParquet covering needs.
    """
    boxes = np.column_stack([bbox.struct.field(field).to_numpy() for field in _BBOX_FIELDS])
    present = np.flatnonzero(bbox.is_not_null().to_numpy())
    low, high = boxes[present, :3], boxes[present, 3:]
    signs = np.array([[sx, sy, sz] for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    corners = low[:, None, :] + signs[None, :, :] * (high - low)[:, None, :]
    moved = _transform_points(transformer, corners.reshape(-1, 3)).reshape(-1, 8, 3)
    values = np.full((len(bbox), 6), np.nan)
    values[present, :3] = moved.min(axis=1)
    values[present, 3:] = moved.max(axis=1)
    bounds = pl.DataFrame(
        {field: values[:, column] for column, field in enumerate(_BBOX_FIELDS)}
    ).fill_nan(None)
    return _bbox_series(bounds)


def _bbox_series(bounds: pl.DataFrame) -> pl.Series:
    """A ``bbox`` column from a frame of bounds, null where ``xmin`` is null."""
    return bounds.select(
        pl.when(pl.col("xmin").is_null())
        .then(None)
        .otherwise(pl.struct(*_BBOX_FIELDS))
        .alias("bbox")
    )["bbox"].cast(BBOX_DTYPE)


def _transform_elevations(frame: pl.DataFrame, transformer: _Transformation) -> pl.Series:
    """Moves the ``elevation`` levels as points at the plan centre of the model."""
    elevation = frame["elevation"].cast(pl.Float64)
    values = elevation.to_numpy()
    present = np.flatnonzero(elevation.is_not_null().to_numpy())
    if len(present) == 0:
        return elevation
    centre = _extent(frame, 0.0)[0]
    points = np.column_stack(
        [np.full(len(present), centre[0]), np.full(len(present), centre[1]), values[present]]
    )
    moved = np.full(len(elevation), np.nan)
    moved[present] = _transform_points(transformer, points)[:, 2]
    return pl.Series("elevation", moved, dtype=pl.Float64).fill_nan(None)


def _transform_wkb(wkb: pl.Series, transformer: _Transformation) -> pl.Series:
    """Moves every coordinate of a column of 3D WKB geometries."""
    if wkb.dtype != pl.Binary:
        raise TypeError(f"the geometry column {wkb.name!r} must be Binary WKB, not {wkb.dtype}")
    geometries = shapely.from_wkb(wkb.to_numpy())

    def move(coordinates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return _transform_points(transformer, coordinates)

    moved = shapely.transform(geometries, move, include_z=True)
    return pl.Series(wkb.name, shapely.to_wkb(moved, flavor="iso").tolist(), dtype=pl.Binary)


def _clamp_to_bbox(wkb: pl.Series, bbox: pl.Series) -> pl.Series:
    """Keeps every moved footprint inside its row's moved ``bbox`` (its GeoParquet covering).

    A footprint vertex can land a rounding step outside the box of the mesh vertices, and
    farther when the transformation lets x and y depend on z (a 3D Helmert does, by less than
    a tenth of a millimetre). Only x and y are clipped; z stays.
    """
    geometries = shapely.from_wkb(wkb.to_numpy())
    bounds = shapely.bounds(geometries)
    low = np.column_stack(
        [bbox.struct.field("xmin").to_numpy(), bbox.struct.field("ymin").to_numpy()]
    )
    high = np.column_stack(
        [bbox.struct.field("xmax").to_numpy(), bbox.struct.field("ymax").to_numpy()]
    )
    with np.errstate(invalid="ignore"):
        outside = (bounds[:, :2] < low).any(axis=1) | (bounds[:, 2:] > high).any(axis=1)
    if not outside.any():
        return wkb
    values = wkb.to_list()
    for row in np.flatnonzero(outside):
        clamped = _clamp_xy(geometries[row], low[row], high[row])
        values[row] = shapely.to_wkb(clamped, flavor="iso")
    return pl.Series(wkb.name, values, dtype=pl.Binary)


def _affine_matrix(
    transformer: _Transformation,
    source: pyproj.CRS,
    target: pyproj.CRS,
    frame: pl.DataFrame,
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """The 4x4 matrix of the transformation when it is affine over the model, else None.

    A geographic or geocentric target is never affine. Otherwise the transformation is probed
    on the corners of a cube around the model (its extent, at least :data:`_PROBE_HALF_SIZE`
    metres each way) and fitted by least squares; it is affine when no corner strays more
    than :data:`AFFINE_TOLERANCE` metres from the fit.
    """
    if target.is_geographic or target.is_geocentric:
        return None
    target_factors = np.array([axis.unit_conversion_factor for axis in target.axis_info])
    if source.is_geographic:
        centre, half = _extent(frame, _PROBE_HALF_SIZE_DEGREES)
    else:
        factor = source.axis_info[0].unit_conversion_factor
        centre, half = _extent(frame, _PROBE_HALF_SIZE / factor)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    probes = np.vstack([centre, centre + half * signs])
    moved = _transform_points(transformer, probes)
    design = np.column_stack([probes - centre, np.ones(len(probes))])
    fit, *_ = np.linalg.lstsq(design, moved, rcond=None)
    residual = np.abs(design @ fit - moved).max(axis=0) * target_factors
    if (residual > AFFINE_TOLERANCE).any():
        return None
    linear = fit[:3].T
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = fit[3] - linear @ centre
    return matrix


def _extent(
    frame: pl.DataFrame, least_half: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], float]:
    """The centre of the model's ``bbox`` column and the half size of the cube around it.

    The origin and ``least_half`` without a usable ``bbox`` column.
    """
    if "bbox" not in frame.columns or frame.schema["bbox"] != BBOX_DTYPE:
        return np.zeros(3), least_half
    bbox = pl.col("bbox").struct
    extent = frame.select(
        bbox.field("xmin").min(),
        bbox.field("ymin").min(),
        bbox.field("zmin").min(),
        bbox.field("xmax").max(),
        bbox.field("ymax").max(),
        bbox.field("zmax").max(),
    ).row(0)
    if None in extent:
        return np.zeros(3), least_half
    low, high = np.array(extent[:3], dtype=float), np.array(extent[3:], dtype=float)
    return (low + high) / 2, max(float((high - low).max()) / 2, least_half)

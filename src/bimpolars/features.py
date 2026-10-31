"""Simple features of BIM elements: the plan footprint of every mesh, as WKB.

A ``BimDataFrame`` fresh from a reader has its geometry as meshes. The GIS side
of a project wants each element as a simple feature: here its footprint, the outline of the
mesh in plan, placed flat at the level of the storey the element belongs to. The footprint is a
``POLYGON Z`` or ``MULTIPOLYGON Z`` in the BIM CRS, stored as ISO WKB in a Binary column so
that the frame writes as GeoParquet (see ``bimpolars.geoparquet``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import shapely
import shapely.errors

from bimpolars.frame import BimDataFrame, as_bim_dataframe
from bimpolars.geoparquet import GEOG_COLUMN
from bimpolars.mesh import mesh_arrays

FOOTPRINT_DTYPE = pl.Binary
"""The dtype of the ``footprint`` column: ISO WKB with Z coordinates."""

FEATURE_COLUMNS: tuple[str, ...] = ("footprint_ref", "footprint")
"""The columns ``simple_features`` adds, in order."""

_REQUIRED_COLUMNS = ("id", "container", "elevation", "mesh", "bbox")

_SLIVER = 1e-9
"""A triangle with less plan area than this fraction of the mesh's plan extent squared is
a sliver (a face that is vertical up to rounding) and is left out of the outline."""


def simple_features(
    bdf: BimDataFrame | pl.DataFrame, *, lightbim: bool = False, kind: pl.Expr | None = None
) -> BimDataFrame:
    """Adds the plan footprint of every element with a mesh, and the LightBIM reduction.

    Two columns are added (or replaced):

    - ``footprint``: the mesh's outline in plan as ``POLYGON Z`` or ``MULTIPOLYGON Z`` in the
      BIM CRS, encoded as ISO WKB (``FOOTPRINT_DTYPE``). It is the union of the mesh's
      triangles projected to the XY plane, so holes and separate parts survive, lying flat at
      the elevation of the ``footprint_ref`` row. A mesh whose triangles are all vertical has
      no plan area and gets ``POLYGON Z EMPTY``; a row without a mesh gets null. Every mesh
      is outlined, an ``IfcOpeningElement``'s void included: filter openings out first for a
      plan of the elements, keeping the spatial rows.
    - ``footprint_ref``: the ``id`` of the row whose level the footprint lies at: the row's
      ``container`` when that has an ``elevation`` (the storey, for most elements), otherwise
      the row itself, at the lowest z of its mesh. Null without a mesh. A frame filtered down
      to its elements has lost the storeys that place the footprints, so it is refused rather
      than silently outlined at every element's own level.

    With ``lightbim=True`` the three columns of ``bimpolars.lightbim.LIGHTBIM_COLUMNS``
    follow: ``lightbim``, the structural-analysis reduction of every member (the axis of a
    beam, the midplane of a wall, the bearing plane of a footing), ``lightbim_kind``, what
    the member is, and ``lightbim_qa``, how that was decided and how far the mesh reaches
    from the reduction. See ``bimpolars.lightbim.lightbim_features``.

    A ``geog`` column (the footprints in OGC:CRS84h of ``bimpolars.geobim.with_geog``) is
    dropped, since it was the previous footprint's: run ``with_geog`` again on the result.

    Args:
        bdf (BimDataFrame | pl.DataFrame): A BimDataFrame with the ``id``, ``container``,
            ``elevation``, ``mesh`` and ``bbox`` columns a reader fills; ``class``, ``properties``
            and ``native_geom`` inform the LightBIM reduction when present.
        lightbim (bool): Whether to add the LightBIM columns.
        kind (pl.Expr | None): With ``lightbim``, an expression that decides the kind of a row ahead
            of the class table and the shape test; see ``bimpolars.lightbim.lightbim_features``.

    Returns:
        BimDataFrame: The BimDataFrame with the columns appended (and ``geog`` dropped), in the same
        CRS.

    Raises:
        ValueError: If a required column is missing, or ``id`` is not unique.
        ValueError: If a row's ``container`` has no row in the frame (see above).
        ValueError: If ``kind`` is given without ``lightbim``.
    """
    if kind is not None and not lightbim:
        raise ValueError("kind decides the LightBIM kind: pass lightbim=True with it")
    frame = as_bim_dataframe(bdf).frame
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"simple_features needs the columns {', '.join(missing)}")
    if frame["id"].n_unique() != len(frame):
        raise ValueError("simple_features needs a unique id per row")
    containers = frame["container"].drop_nulls().unique()
    lost = containers.filter(~containers.is_in(frame["id"].implode()))
    if not lost.is_empty():
        raise ValueError(
            f"simple_features needs the row of every container, but {len(lost)} "
            f"containers have none, such as {lost[0]!r}: keep the spatial rows when "
            "filtering the elements"
        )
    ref = _footprint_ref(frame)
    footprints = _footprints(frame["mesh"], ref["z"])
    frame = frame.drop(GEOG_COLUMN, strict=False).with_columns(ref["footprint_ref"], footprints)
    if lightbim:
        # Imported here: bimpolars.lightbim builds on plan_outline below
        from bimpolars.lightbim import lightbim_features

        frame = frame.with_columns(lightbim_features(BimDataFrame(frame), kind=kind))
    return BimDataFrame(frame)


def _footprint_ref(frame: pl.DataFrame) -> pl.DataFrame:
    """Per row, the ``footprint_ref`` id and the ``z`` its footprint lies at."""
    levels = frame.select(container=pl.col("id"), container_elevation=pl.col("elevation"))
    joined = frame.select(
        "id",
        "container",
        has_mesh=pl.col("mesh").is_not_null(),
        zmin=pl.col("bbox").struct.field("zmin"),
    ).join(levels, on="container", how="left", validate="m:1", maintain_order="left")
    at_container = pl.col("container_elevation").is_not_null()
    return joined.select(
        footprint_ref=pl.when(~pl.col("has_mesh"))
        .then(None)
        .when(at_container)
        .then(pl.col("container"))
        .otherwise(pl.col("id")),
        z=pl.when(at_container).then(pl.col("container_elevation")).otherwise(pl.col("zmin")),
    )


def _footprints(mesh: pl.Series, z: pl.Series) -> pl.Series:
    """The ``footprint`` column: the plan outline of every mesh, flat at ``z``, as WKB."""
    arrays = mesh_arrays(mesh)
    outlines = np.full(len(mesh), None, dtype=object)
    for row in arrays.present:
        outlines[row] = plan_outline(*arrays.row(row))
    heights = z.fill_null(0.0).cast(pl.Float64).to_numpy()
    wkb = shapely.to_wkb(shapely.force_3d(outlines, z=heights), flavor="iso")
    return pl.Series("footprint", wkb.tolist(), dtype=FOOTPRINT_DTYPE)


def plan_outline(vertices: np.ndarray, faces: np.ndarray) -> shapely.Geometry:
    """The outline of a triangle mesh in plan: the union of its triangles projected to XY.

    A triangle whose plan area is below ``_SLIVER`` times the square of the mesh's plan
    extent is a sliver, a face that is vertical up to rounding, and is left out. The outline
    never leaves the mesh's bounding box in plan, so the ``bbox`` column covers it: when
    GEOS's robustness fallback rounds a vertex of the union a step outwards, the outline is
    clamped back to the box. When GEOS's floating-point overlay fails altogether on a mesh
    (it can, on many nearly coincident triangles), the union is redone on a grid of
    ``_SLIVER`` times the extent, which is the fixed-precision overlay that cannot fail.

    Args:
        vertices (np.ndarray): The mesh's vertices, shape ``(n, 3)``.
        faces (np.ndarray): The triangles as vertex indices, shape ``(m, 3)``.

    Returns:
        shapely.Geometry: A 2D Polygon or MultiPolygon, empty when every triangle is vertical or a
        sliver.
    """
    if len(faces) == 0:
        return shapely.Polygon()
    low = vertices[:, :2].min(axis=0)
    high = vertices[:, :2].max(axis=0)
    triangles = vertices[faces, :2]
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    doubled_area = np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])
    extent = float((high - low).max())
    flat = triangles[doubled_area > _SLIVER * extent**2]
    if len(flat) == 0:
        return shapely.Polygon()
    polygons = shapely.polygons(flat)
    try:
        outline = shapely.union_all(polygons)
    except shapely.errors.GEOSException:
        # GEOS's floating-point overlay can fail on a mesh of many nearly coincident
        # triangles (a Speckle stair with 720 faces does it); its fixed-precision overlay
        # does not, so the union is retried on a grid as fine as the sliver rule
        outline = shapely.union_all(polygons, grid_size=_SLIVER * extent)
    if outline.geom_type == "GeometryCollection":
        # A union of areas gives areas; drop anything else GEOS might leave in a collection
        parts = shapely.get_parts(outline)
        outline = shapely.union_all(parts[shapely.get_dimensions(parts) == 2])
    return _clamp(outline, low, high)


def _clamp(geometry: shapely.Geometry, low: np.ndarray, high: np.ndarray) -> shapely.Geometry:
    """Moves the coordinates of a 2D geometry that lie outside a box onto its edge.

    Args:
        geometry (shapely.Geometry): A 2D geometry.
        low (np.ndarray): The box's minimum x and y.
        high (np.ndarray): The box's maximum x and y.

    Returns:
        shapely.Geometry: The geometry itself when its bounds are inside the box, else a clamped
        copy.
    """
    if shapely.is_empty(geometry):
        return geometry
    bounds = shapely.bounds(geometry)
    if (bounds[:2] >= low).all() and (bounds[2:] <= high).all():
        return geometry
    return shapely.transform(geometry, lambda xy: np.clip(xy, low, high))


def _clamp_xy(
    geometry: shapely.Geometry, low: np.ndarray[Any, Any], high: np.ndarray[Any, Any]
) -> shapely.Geometry:
    """Moves the x and y of a 3D geometry's coordinates outside a box onto its edge; z stays.

    ``shapely.transform`` drops z unless ``include_z=True``, which is why this is not
    ``_clamp``.

    Args:
        geometry (shapely.Geometry): A geometry with z coordinates.
        low (np.ndarray[Any, Any]): The box's minimum x and y.
        high (np.ndarray[Any, Any]): The box's maximum x and y.

    Returns:
        shapely.Geometry: The geometry itself when its bounds are inside the box, else a clamped
        copy.
    """
    if shapely.is_empty(geometry):
        return geometry
    bounds = shapely.bounds(geometry)
    if (bounds[:2] >= low).all() and (bounds[2:] <= high).all():
        return geometry

    def clip(coordinates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.column_stack([np.clip(coordinates[:, :2], low, high), coordinates[:, 2:]])

    return shapely.transform(geometry, clip, include_z=True)

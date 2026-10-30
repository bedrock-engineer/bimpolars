"""Reads an IFC file into a BimDataFrame with ifcopenshell."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, cast

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.placement
import ifcopenshell.util.unit
import numpy as np
import polars as pl
import shapely
import shapely.affinity

from bimpolars.crs import LinearUnit, engineering_crs
from bimpolars.frame import NATIVE_GEOM_DTYPE, BimDataFrame
from bimpolars.mesh import UNSTYLED_RGBA, Mesh, mesh_frame
from bimpolars.properties import ATTRIBUTES_PSET, properties_series

DEFAULT_RGBA = UNSTYLED_RGBA
"""The colour of faces without a surface style: ifcopenshell's default grey."""

_SPATIAL_CLASSES = ("IfcSpatialStructureElement", "IfcSpatialElement")

_ARC_STEP = math.radians(11.25)
"""The angle between the points that sample an arc of a profile: 32 points on a circle."""


def read_ifc(
    source: str | os.PathLike[str] | ifcopenshell.file, *, threads: int | None = None
) -> BimDataFrame:
    """Reads every product of an IFC into a BimDataFrame, one row each.

    Elements, spatial structure (site, building, storeys, spaces) and positioning elements
    such as grids all get a row; a row without geometry has a null ``mesh`` and ``bbox``.
    ifcopenshell tessellates the geometry in the model's coordinates and length unit, with the
    placements applied (``use-world-coords``), so the meshes sit where the model puts them.
    Openings are products too: an ``IfcOpeningElement`` row carries the solid mesh of the
    void it cuts, so filter them out before outlining or rendering elements.

    The columns:

    - ``id``: the GlobalId.
    - ``name``: the Name attribute.
    - ``class``: the IFC class, Categorical.
    - ``properties``: a Struct with one Struct per property or quantity set, the type's sets
      included; see :func:`bimpolars.properties.properties_series` for the dtypes. The set
      :data:`ATTRIBUTES_PSET` holds the entity's ``PredefinedType`` (resolved through its
      type and ``ObjectType``) when it has one. A row without any set is a Struct of nulls.
    - ``container``: the ``id`` of the spatial element the row sits in: for an element, its
      storey (or the spatial element that contains its aggregate); for a spatial element, its
      parent in the spatial structure. Null for the site, for an element in no container and
      for ports, which hang off their element rather than a spatial element.
    - ``elevation``: on storeys, the level in model coordinates; see :func:`_elevations`.
    - ``native_geom``: :data:`bimpolars.NATIVE_GEOM_DTYPE`, the ``IfcExtrudedAreaSolid``
      items of the Body representation (mapped items resolved; a difference or intersection
      reduced to its first operand, flagged ``clipped``; both operands of a union), each with
      its placement in model coordinates, its profile as a 2D polygon, depth and direction.
      Items of other kinds (breps, revolved or tapered sweeps) and profiles this reader cannot
      outline are left out, so the list can describe part of a body; null when there is
      nothing to list. The LightBIM
      reduction (:func:`bimpolars.simple_features` with ``lightbim=True``) takes the axis or
      midplane of a member from it.
    - ``mesh``: :data:`bimpolars.MESH_DTYPE`, with faces sorted by style. A style is named
      after its IfcSurfaceStyle, or ``IfcSurfaceStyle-<step id>`` when that has no name; faces
      without any surface style get a null name and ifcopenshell's grey. Two surface styles
      with the same name but different colours give two entries with one name, so group by
      ``(name, rgba)`` when one material per style is needed.
    - ``bbox``: :data:`bimpolars.BBOX_DTYPE` of the mesh.

    The CRS is a plain engineering CRS in the model's length unit: the IFC's coordinates as they
    are. Georeferencing (IfcMapConversion) is not read yet.

    Args:
        source: The IFC file's path, or an open ``ifcopenshell.file``.
        threads: Threads for the tessellation; all CPUs by default.

    Returns:
        The model as a BimDataFrame.
    """
    model = (
        source if isinstance(source, ifcopenshell.file) else ifcopenshell.open(os.fspath(source))
    )
    products = cast(list[ifcopenshell.entity_instance], model.by_type("IfcProduct"))
    shapes = _tessellate(model, threads or os.cpu_count() or 1)

    ids = [str(product.GlobalId) for product in products]
    names = [product.Name for product in products]
    classes = [product.is_a() for product in products]
    containers = [_container(product) for product in products]
    elevations = _elevations(products)
    psets = [_psets(product) for product in products]
    native_geom = _native_geoms(products)
    # Rows without a shape get a null mesh and bbox. The shapes are keyed by STEP id, so two
    # products that share a GlobalId (it happens) each keep their own mesh.
    shape_index = {step_id: position for position, step_id in enumerate(shapes["step_id"])}
    positions = pl.Series([shape_index.get(product.id()) for product in products], dtype=pl.UInt32)
    georeferenced = model.schema != "IFC2X3" and bool(model.by_type("IfcMapConversion"))
    crs = engineering_crs(
        _crs_name(model, source), unit=_length_unit(model), datum="IFC project origin"
    )
    # Everything is plain Python from here on: let a model this function opened go before the
    # properties column, the widest part of a large model, is built
    del products
    if model is not source:
        del model
    properties = properties_series(psets)
    del psets

    frame = pl.DataFrame(
        [
            pl.Series("id", ids, dtype=pl.String),
            pl.Series("name", names, dtype=pl.String),
            pl.Series("class", classes, dtype=pl.Categorical),
            properties,
            pl.Series("container", containers, dtype=pl.String),
            pl.Series("elevation", elevations, dtype=pl.Float64),
            native_geom,
            shapes["mesh"].gather(positions),
            shapes["bbox"].gather(positions),
        ]
    )
    if georeferenced:
        warnings.warn(
            "the IFC has an IfcMapConversion, which read_ifc does not use yet: "
            "the CRS is the model's own coordinates",
            stacklevel=2,
        )
    return BimDataFrame(frame, crs)


def _crs_name(model: ifcopenshell.file, source: object) -> str:
    """Names the CRS after the IfcProject, or after the file."""
    projects = model.by_type("IfcProject")
    name = projects[0].Name if projects else None
    if not name and isinstance(source, str | os.PathLike):
        name = os.path.splitext(os.path.basename(os.fspath(source)))[0]
    return f"{name or 'IFC'} model coordinates"


def _length_unit(model: ifcopenshell.file) -> LinearUnit:
    """The project's length unit; metres when it has none."""
    unit = ifcopenshell.util.unit.get_project_unit(model, "LENGTHUNIT")
    if unit is None:
        return LinearUnit("metre", 1.0)
    name = str(ifcopenshell.util.unit.get_full_unit_name(unit)).lower().replace("_", " ")
    return LinearUnit(name, float(ifcopenshell.util.unit.calculate_unit_scale(model)))


def _is_spatial(product: ifcopenshell.entity_instance) -> bool:
    return any(product.is_a(ifc_class) for ifc_class in _SPATIAL_CLASSES)


def _container(product: ifcopenshell.entity_instance) -> str | None:
    """The GlobalId of the spatial element a product sits in."""
    if _is_spatial(product):
        parent = ifcopenshell.util.element.get_aggregate(product)
        if parent is not None and parent.is_a("IfcProduct"):
            return str(parent.GlobalId)
        return None
    container = ifcopenshell.util.element.get_container(product, should_get_direct=False)
    return None if container is None else str(container.GlobalId)


def _elevations(products: Sequence[ifcopenshell.entity_instance]) -> list[float | None]:
    """The elevation of every storey in model coordinates; None for other products.

    A storey's level is the z of its placement, which is where its elements sit. Some
    exporters place every storey at z = 0 and carry the levels in the ``Elevation``
    attribute instead; when that is the case for the whole model, ``Elevation`` is used.
    """
    storeys = [product for product in products if product.is_a("IfcBuildingStorey")]
    placed = {
        storey.id(): float(ifcopenshell.util.placement.get_storey_elevation(storey))
        for storey in storeys
    }
    attributes = {storey.id(): storey.Elevation for storey in storeys}
    if all(z == 0.0 for z in placed.values()) and any(attributes.values()):
        placed = {step_id: float(attributes[step_id] or 0.0) for step_id in placed}
    return [placed.get(product.id()) for product in products]


def _psets(product: ifcopenshell.entity_instance) -> dict[str, dict[str, Any]]:
    """The product's property and quantity sets, its type's included, without the set ids.

    The set :data:`ATTRIBUTES_PSET` carries the predefined type, which is an attribute rather
    than a property but names the structural kind of a slab (``BASESLAB``), a footing
    (``PILE_CAP``) or a member (``BRACE``).
    """
    psets = ifcopenshell.util.element.get_psets(product)
    sets = {
        name: {key: value for key, value in values.items() if key != "id"}
        for name, values in psets.items()
    }
    predefined_type = ifcopenshell.util.element.get_predefined_type(product)
    if predefined_type:
        sets[ATTRIBUTES_PSET] = {"PredefinedType": str(predefined_type)}
    return sets


def _native_geoms(products: Sequence[ifcopenshell.entity_instance]) -> pl.Series:
    """The ``native_geom`` column: the extruded area solids of every product's body."""
    return pl.Series(
        "native_geom", [_sweeps(product) for product in products], dtype=NATIVE_GEOM_DTYPE
    )


def _sweeps(product: ifcopenshell.entity_instance) -> list[dict[str, Any]] | None:
    """The sweeps of one product as :data:`bimpolars.SWEEP_DTYPE` rows; None without any."""
    if not product.Representation:
        return None
    sweeps = []
    for matrix, item, clipped in _body_items(product):
        if not item.is_a("IfcExtrudedAreaSolid"):
            continue
        try:
            outline = _profile_outline(item.SweptArea)
        except NotImplementedError:
            continue
        direction = np.array(item.ExtrudedDirection.DirectionRatios, dtype=float)
        if len(direction) == 2:
            direction = np.append(direction, 0.0)
        length = float(np.linalg.norm(direction))
        if length == 0.0 or outline.is_empty:
            continue
        placement = matrix
        if item.Position is not None:
            placement = matrix @ ifcopenshell.util.placement.get_axis2placement(item.Position)
        sweeps.append(
            {
                "type": item.is_a(),
                "placement": placement.astype(float).ravel().tolist(),
                "profile": shapely.to_wkb(outline, flavor="iso"),
                "profile_type": item.SweptArea.is_a(),
                "profile_name": item.SweptArea.ProfileName,
                "depth": float(item.Depth),
                "direction": (direction / length).tolist(),
                "clipped": clipped,
            }
        )
    return sweeps or None


def _body_items(
    product: ifcopenshell.entity_instance,
) -> Iterator[tuple[np.ndarray[Any, Any], ifcopenshell.entity_instance, bool]]:
    """Yields the items of a product's Body representation with their placement.

    Mapped items are resolved and boolean results are taken apart: a difference (a clipping)
    or an intersection is reduced to its first operand, the solid before anything was cut
    away, and a union gives both of its operands.

    Yields:
        The matrix from the item's coordinates to model coordinates, the item, and whether a
        boolean result was reduced.
    """
    placement = (
        ifcopenshell.util.placement.get_local_placement(product.ObjectPlacement)
        if product.ObjectPlacement is not None
        else np.eye(4)
    )
    for representation in product.Representation.Representations:
        if representation.RepresentationIdentifier != "Body":
            continue
        yield from _items(representation.Items, placement)


def _items(
    items: Sequence[ifcopenshell.entity_instance],
    matrix: np.ndarray[Any, Any],
    clipped: bool = False,
) -> Iterator[tuple[np.ndarray[Any, Any], ifcopenshell.entity_instance, bool]]:
    """Yields the solids among representation items, mapped items and booleans resolved.

    A union yields both operands; a difference or an intersection yields its first operand,
    flagged as clipped, which the operands of a clipped item inherit.
    """
    for item in items:
        if item.is_a("IfcMappedItem"):
            mapped = matrix @ ifcopenshell.util.placement.get_mappeditem_transformation(item)
            yield from _items(item.MappingSource.MappedRepresentation.Items, mapped, clipped)
            continue
        if item.is_a("IfcCsgSolid"):
            yield from _items([item.TreeRootExpression], matrix, clipped)
        elif item.is_a("IfcBooleanResult") and item.Operator == "UNION":
            yield from _items([item.FirstOperand, item.SecondOperand], matrix, clipped)
        elif item.is_a("IfcBooleanResult"):
            # DIFFERENCE (every IfcBooleanClippingResult) or INTERSECTION: the result is part
            # of the first operand
            yield from _items([item.FirstOperand], matrix, True)
        else:
            yield matrix, item, clipped


def _profile_outline(profile: ifcopenshell.entity_instance) -> shapely.Polygon:
    """The swept area of a profile as a polygon in the extrusion's XY plane.

    Args:
        profile: An IfcRectangleProfileDef (hollow included), IfcCircleProfileDef (hollow
            included), IfcIShapeProfileDef, IfcAsymmetricIShapeProfileDef or
            IfcArbitraryClosedProfileDef (with voids).

    Returns:
        The outline with the profile's own Position applied; voids are holes.

    Raises:
        NotImplementedError: For any other kind of profile, or a curve
            :func:`_curve_points` cannot sample.
    """
    if profile.is_a("IfcRectangleProfileDef"):
        half_x, half_y = profile.XDim / 2, profile.YDim / 2
        holes = []
        if profile.is_a("IfcRectangleHollowProfileDef"):
            t = profile.WallThickness
            holes = [shapely.box(-half_x + t, -half_y + t, half_x - t, half_y - t).exterior.coords]
        outline = shapely.Polygon(shapely.box(-half_x, -half_y, half_x, half_y).exterior, holes)
    elif profile.is_a("IfcCircleProfileDef"):
        outline = shapely.Point(0, 0).buffer(profile.Radius, quad_segs=8)
        if profile.is_a("IfcCircleHollowProfileDef"):
            inner = shapely.Point(0, 0).buffer(profile.Radius - profile.WallThickness, quad_segs=8)
            outline = shapely.Polygon(outline.exterior, [inner.exterior.coords])
    elif profile.is_a("IfcAsymmetricIShapeProfileDef"):
        # IFC2X3 derives it from IfcIShapeProfileDef and names the bottom flange as that does
        ifc2x3 = profile.is_a("IfcIShapeProfileDef")
        bottom_width = profile.OverallWidth if ifc2x3 else profile.BottomFlangeWidth
        bottom = profile.FlangeThickness if ifc2x3 else profile.BottomFlangeThickness
        outline = _i_shape(
            bottom_width,
            profile.TopFlangeWidth,
            profile.OverallDepth,
            profile.WebThickness,
            bottom,
            profile.TopFlangeThickness or bottom,
        )
    elif profile.is_a("IfcIShapeProfileDef"):
        width, flange = profile.OverallWidth, profile.FlangeThickness
        outline = _i_shape(width, width, profile.OverallDepth, profile.WebThickness, flange, flange)
    elif profile.is_a("IfcArbitraryClosedProfileDef"):
        voids = profile.InnerCurves if profile.is_a("IfcArbitraryProfileDefWithVoids") else ()
        outline = shapely.Polygon(
            _curve_points(profile.OuterCurve), [_curve_points(curve) for curve in voids]
        )
    else:
        raise NotImplementedError(profile.is_a())
    position = getattr(profile, "Position", None)
    if position is None:
        return outline
    m = ifcopenshell.util.placement.get_axis2placement(position)
    return shapely.affinity.affine_transform(
        outline, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, -1], m[1, -1]]
    )


def _i_shape(
    bottom_width: float,
    top_width: float,
    depth: float,
    web: float,
    bottom_flange: float,
    top_flange: float,
) -> shapely.Polygon:
    """An I-section centred on its bounding box, flanges of their own width and thickness."""
    b, t, h, w = bottom_width / 2, top_width / 2, depth / 2, web / 2
    lower, upper = -h + bottom_flange, h - top_flange
    return shapely.Polygon(
        [
            (-b, -h), (b, -h), (b, lower), (w, lower), (w, upper), (t, upper),
            (t, h), (-t, h), (-t, upper), (-w, upper), (-w, lower), (-b, lower),
        ]
    )  # fmt: skip


def _curve_points(curve: ifcopenshell.entity_instance) -> list[tuple[float, float]]:
    """Samples a 2D bounded curve of a profile.

    Args:
        curve: An IfcPolyline, IfcIndexedPolyCurve, IfcCircle, IfcTrimmedCurve on a circle
            or a line, or an IfcCompositeCurve of those.

    Returns:
        The points along the curve; arcs get a point every :data:`_ARC_STEP`.

    Raises:
        NotImplementedError: For any other kind of curve.
    """
    if curve.is_a("IfcPolyline"):
        return [(float(p.Coordinates[0]), float(p.Coordinates[1])) for p in curve.Points]
    if curve.is_a("IfcIndexedPolyCurve"):
        return _indexed_poly_curve_points(curve)
    if curve.is_a("IfcCompositeCurve"):
        points: list[tuple[float, float]] = []
        for segment in curve.Segments:
            segment_points = _curve_points(segment.ParentCurve)
            for point in segment_points if segment.SameSense else segment_points[::-1]:
                if not points or math.dist(point, points[-1]) > 1e-9:
                    points.append(point)
        return points
    if curve.is_a("IfcCircle"):
        return _arc_points(curve, 0.0, 2 * math.pi)
    if curve.is_a("IfcTrimmedCurve"):
        basis = curve.BasisCurve
        if basis.is_a("IfcCircle"):
            matrix = ifcopenshell.util.placement.get_axis2placement(basis.Position)
            scale = ifcopenshell.util.unit.calculate_unit_scale(curve.file, "PLANEANGLEUNIT")
            start = _trim_angle(curve.Trim1, matrix, basis.Radius, scale)
            end = _trim_angle(curve.Trim2, matrix, basis.Radius, scale)
            if curve.SenseAgreement and end <= start:
                end += 2 * math.pi
            elif not curve.SenseAgreement and end >= start:
                end -= 2 * math.pi
            return _arc_points(basis, start, end)
        if basis.is_a("IfcLine"):
            ends = [_trim_point(trim, basis) for trim in (curve.Trim1, curve.Trim2)]
            return ends if curve.SenseAgreement else ends[::-1]
    raise NotImplementedError(curve.is_a())


def _indexed_poly_curve_points(curve: ifcopenshell.entity_instance) -> list[tuple[float, float]]:
    """Samples an IFC4 IfcIndexedPolyCurve: straight segments and three-point arcs."""
    coordinates = [(float(x), float(y)) for x, y, *_ in curve.Points.CoordList]
    if not curve.Segments:
        return coordinates
    points: list[tuple[float, float]] = []
    for segment in curve.Segments:
        indices = [int(i) - 1 for i in segment.wrappedValue]
        if segment.is_a("IfcArcIndex"):
            segment_points = _three_point_arc(*(coordinates[i] for i in indices))
        else:
            segment_points = [coordinates[i] for i in indices]
        for point in segment_points:
            if not points or math.dist(point, points[-1]) > 1e-9:
                points.append(point)
    return points


def _three_point_arc(
    start: tuple[float, float], middle: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
    """Points along the arc from ``start`` through ``middle`` to ``end``."""
    (ax, ay), (bx, by), (cx, cy) = start, middle, end
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return [start, end]
    a2, b2, c2 = ax**2 + ay**2, bx**2 + by**2, cx**2 + cy**2
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    radius = math.dist((ux, uy), start)
    first, through, last = (math.atan2(y - uy, x - ux) for x, y in (start, middle, end))
    sweep = (last - first) % (2 * math.pi)
    if (through - first) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    angles = np.linspace(first, first + sweep, max(2, math.ceil(abs(sweep) / _ARC_STEP) + 1))
    return [(ux + radius * math.cos(a), uy + radius * math.sin(a)) for a in angles]


def _trim_angle(
    trim: Sequence[ifcopenshell.entity_instance],
    matrix: np.ndarray[Any, Any],
    radius: float,
    scale: float,
) -> float:
    """A trimming point's angle on its circle, in radians: its parameter, else its position."""
    for value in trim:
        if value.is_a("IfcParameterValue"):
            return float(value.wrappedValue) * scale
    local = np.linalg.solve(matrix[:2, :2], np.subtract(trim[0].Coordinates[:2], matrix[:2, 3]))
    return math.atan2(local[1], local[0])


def _trim_point(
    trim: Sequence[ifcopenshell.entity_instance], line: ifcopenshell.entity_instance
) -> tuple[float, float]:
    """A trimming point on a line: its coordinates, else the point at its parameter."""
    for value in trim:
        if value.is_a("IfcCartesianPoint"):
            return (float(value.Coordinates[0]), float(value.Coordinates[1]))
    t = float(trim[0].wrappedValue)
    origin = line.Pnt.Coordinates
    vector = line.Dir
    ratios = np.array(vector.Orientation.DirectionRatios[:2], dtype=float) * float(vector.Magnitude)
    return (float(origin[0] + t * ratios[0]), float(origin[1] + t * ratios[1]))


def _arc_points(
    circle: ifcopenshell.entity_instance, start: float, end: float
) -> list[tuple[float, float]]:
    """Points on an IfcCircle from angle ``start`` to ``end``, one every :data:`_ARC_STEP`."""
    count = max(2, math.ceil(abs(end - start) / _ARC_STEP) + 1)
    angles = np.linspace(start, end, count)
    matrix = ifcopenshell.util.placement.get_axis2placement(circle.Position)
    local = circle.Radius * np.column_stack([np.cos(angles), np.sin(angles)])
    points = local @ matrix[:2, :2].T + matrix[:2, 3]
    return [(float(x), float(y)) for x, y in points]


class _Styles:
    """Resolves ifcopenshell material names to the IfcSurfaceStyle names of a model."""

    def __init__(self, model: ifcopenshell.file) -> None:
        self._model = model
        self._names: dict[str, str | None] = {}

    def name(self, material: Any) -> str | None:
        """The IfcSurfaceStyle name behind an ifcopenshell material; None for the default."""
        material_name = str(material.name)
        if material_name not in self._names:
            self._names[material_name] = self._resolve(material_name)
        return self._names[material_name]

    def _resolve(self, material_name: str) -> str | None:
        if material_name == "DefaultMaterial":
            return None
        # ifcopenshell names a style after its shading entity: "IfcSurfaceStyleRendering-868"
        prefix, _, step_id = material_name.rpartition("-")
        if prefix.startswith("IfcSurfaceStyle") and step_id.isdigit():
            for style in self._model.get_inverse(self._model.by_id(int(step_id))):
                if style.is_a("IfcSurfaceStyle"):
                    return str(style.Name) if style.Name else f"IfcSurfaceStyle-{style.id()}"
        # A name ifcopenshell passes through, or an id it did not take from an IfcSurfaceStyle:
        # keep it, so that the style stays distinct from the unstyled faces
        return material_name


def _rgba(material: Any) -> tuple[float, float, float, float]:
    """A material's diffuse colour and opacity, each 0..1."""
    components = [float(c) for c in material.diffuse.components]
    if len(components) != 3 or any(np.isnan(components)):
        return DEFAULT_RGBA
    alpha = 1.0
    if material.has_transparency():
        transparency = float(material.transparency)
        if not np.isnan(transparency):
            alpha = 1.0 - transparency
    return (components[0], components[1], components[2], alpha)


def _tessellate(model: ifcopenshell.file, threads: int) -> pl.DataFrame:
    """Tessellates every product with a representation, in world coordinates and model units.

    Returns:
        One row per product that has a shape: ``mesh`` (:data:`bimpolars.MESH_DTYPE`),
        ``bbox`` (:data:`bimpolars.BBOX_DTYPE`) and its STEP id as ``step_id``.

    Raises:
        ValueError: If ifcopenshell yields two shapes for one product.
    """
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("convert-back-units", True)
    styles = _Styles(model)
    shapes: dict[int, Mesh] = {}
    for step_id, geometry in _shapes(settings, model, threads):
        shape = _shape(geometry, styles)
        if shape is None:
            continue
        if step_id in shapes:
            raise ValueError(f"ifcopenshell yielded more than one shape for #{step_id}")
        shapes[step_id] = shape
    return mesh_frame(list(shapes.values())).with_columns(
        pl.Series("step_id", list(shapes), dtype=pl.Int64)
    )


def _shapes(settings: Any, model: ifcopenshell.file, threads: int) -> Iterator[tuple[int, Any]]:
    """Yields (STEP id, geometry) for every shape the ifcopenshell iterator produces."""
    iterator = ifcopenshell.geom.iterator(settings, model, threads)
    if not iterator.initialize():
        return
    while True:
        shape = iterator.get()
        yield int(shape.id), shape.geometry
        if not iterator.next():
            break


def _shape(geometry: Any, styles: _Styles) -> Mesh | None:
    """Reads ifcopenshell geometry into arrays, with the faces sorted by style."""
    # The buffers are the C++ vectors' bytes: doubles for the vertices, ints for the faces and
    # material ids. Reading them avoids boxing every number into a Python object.
    vertices = np.frombuffer(geometry.verts_buffer, dtype=np.float64)
    faces = np.frombuffer(geometry.faces_buffer, dtype=np.int32).reshape(-1, 3)
    material_ids = np.frombuffer(geometry.material_ids_buffer, dtype=np.int32)
    if len(faces) == 0 or len(vertices) == 0:
        return None
    if len(material_ids) != len(faces):
        raise ValueError("ifcopenshell gave a material id count that differs from the face count")
    order = np.argsort(material_ids, kind="stable")
    unique_ids, counts = np.unique(material_ids, return_counts=True)
    materials = geometry.materials
    style_rows = []
    for material_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
        material = materials[material_id] if material_id >= 0 else None
        style_rows.append(
            {
                "name": None if material is None else styles.name(material),
                "rgba": list(DEFAULT_RGBA if material is None else _rgba(material)),
                "face_count": int(count),
            }
        )
    return Mesh(vertices.copy(), faces[order].astype(np.uint32).ravel(), style_rows)

"""Reads an IFC file into a BimDataFrame with ifcopenshell."""

from __future__ import annotations

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

from bimpolars.crs import LinearUnit, engineering_crs
from bimpolars.frame import BimDataFrame
from bimpolars.mesh import UNSTYLED, UNSTYLED_RGBA, Mesh, Style, mesh_frame, styled_mesh
from bimpolars.properties import properties_series

_SPATIAL_CLASSES = ("IfcSpatialStructureElement", "IfcSpatialElement")


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
      included; see :func:`bimpolars.properties.properties_series` for the dtypes. A row
      without any set is a Struct of nulls.
    - ``container``: the ``id`` of the spatial element the row sits in: for an element, its
      storey (or the spatial element that contains its aggregate); for a spatial element, its
      parent in the spatial structure. Null for the site, for an element in no container and
      for ports, which hang off their element rather than a spatial element.
    - ``elevation``: on storeys, the level in model coordinates; see :func:`_elevations`.
    - ``native_geom``: null for now (a later version fills it with the IFC sweeps).
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
            pl.Series("native_geom", [None] * len(ids), dtype=pl.Null),
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
    """The product's property and quantity sets, its type's included, without the set ids."""
    psets = ifcopenshell.util.element.get_psets(product)
    return {
        name: {key: value for key, value in values.items() if key != "id"}
        for name, values in psets.items()
    }


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
        return UNSTYLED_RGBA
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
    """Reads ifcopenshell geometry into a Mesh, with the faces sorted by style."""
    # The buffers are the C++ vectors' bytes: doubles for the vertices, ints for the faces and
    # material ids (-1 for a face without a style). Reading them avoids boxing every number
    # into a Python object.
    vertices = np.frombuffer(geometry.verts_buffer, dtype=np.float64)
    faces = np.frombuffer(geometry.faces_buffer, dtype=np.int32).reshape(-1, 3)
    material_ids = np.frombuffer(geometry.material_ids_buffer, dtype=np.int32)
    if len(faces) == 0 or len(vertices) == 0:
        return None
    if len(material_ids) != len(faces):
        raise ValueError("ifcopenshell gave a material id count that differs from the face count")
    # style 0 is the unstyled faces, style i + 1 the i-th material
    materials = [UNSTYLED, *(Style(styles.name(m), _rgba(m)) for m in geometry.materials)]
    return styled_mesh(vertices.copy(), faces, material_ids + 1, materials)

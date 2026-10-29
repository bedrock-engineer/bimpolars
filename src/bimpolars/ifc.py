"""Reads an IFC file into a BimDataFrame with ifcopenshell."""

from __future__ import annotations

import json
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
from polars.datatypes import DataTypeClass

from bimpolars.crs import engineering_crs
from bimpolars.frame import BBOX_DTYPE, MESH_DTYPE, BimDataFrame

DEFAULT_RGBA = (0.7, 0.7, 0.7, 1.0)
"""The colour of faces without a surface style: ifcopenshell's default grey."""

_SPATIAL_CLASSES = ("IfcSpatialStructureElement", "IfcSpatialElement")


def read_ifc(
    source: str | os.PathLike[str] | ifcopenshell.file, *, threads: int | None = None
) -> BimDataFrame:
    """Reads every product of an IFC into a BimDataFrame, one row each.

    Elements, spatial structure (site, building, storeys, spaces) and positioning elements
    such as grids all get a row; a row without geometry has a null ``mesh`` and ``bbox``.
    ifcopenshell tessellates the geometry in the model's coordinates and length unit, with the
    placements applied (``use-world-coords``), so the meshes sit where the model puts them.

    The columns:

    - ``id``: the GlobalId.
    - ``name``: the Name attribute.
    - ``class``: the IFC class, Categorical.
    - ``properties``: a Struct with one Struct per property or quantity set, the type's sets
      included. A property whose type differs between rows is a String.
    - ``container``: the ``id`` of the spatial element the row sits in: for an element, its
      storey (or the spatial element that contains its aggregate); for a spatial element, its
      parent in the spatial structure. Null for the site.
    - ``elevation``: on storeys, the z of the storey's placement in model coordinates.
    - ``native_geom``: null for now (a later version fills it with the IFC sweeps).
    - ``mesh``: :data:`bimpolars.MESH_DTYPE`, with faces sorted by style. A style is named
      after its IfcSurfaceStyle; faces without one get a null name and ifcopenshell's grey.
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
    unit_name, unit_factor = _length_unit(model)
    products = cast(list[ifcopenshell.entity_instance], model.by_type("IfcProduct"))
    meshes = _tessellate(model, threads or os.cpu_count() or 1)

    ids = [product.GlobalId for product in products]
    names = [product.Name for product in products]
    classes = [product.is_a() for product in products]
    containers = [_container(product) for product in products]
    elevations = [_elevation(product) for product in products]
    properties = _properties_series([_psets(product) for product in products])
    mesh_rows = [meshes.get(product.GlobalId) for product in products]
    bbox_rows = [_bbox(mesh) for mesh in mesh_rows]

    frame = pl.DataFrame(
        [
            pl.Series("id", ids, dtype=pl.String),
            pl.Series("name", names, dtype=pl.String),
            pl.Series("class", classes, dtype=pl.Categorical),
            properties,
            pl.Series("container", containers, dtype=pl.String),
            pl.Series("elevation", elevations, dtype=pl.Float64),
            pl.Series("native_geom", [None] * len(ids), dtype=pl.Null),
            pl.Series("mesh", mesh_rows, dtype=MESH_DTYPE),
            pl.Series("bbox", bbox_rows, dtype=BBOX_DTYPE),
        ]
    )
    if model.schema != "IFC2X3" and model.by_type("IfcMapConversion"):
        warnings.warn(
            "the IFC has an IfcMapConversion, which read_ifc does not use yet: "
            "the CRS is the model's own coordinates",
            stacklevel=2,
        )
    crs = engineering_crs(
        _crs_name(model, source),
        unit=unit_name,
        unit_factor=unit_factor,
        datum="IFC project origin",
    )
    return BimDataFrame(frame, crs)


def _crs_name(model: ifcopenshell.file, source: object) -> str:
    """Names the CRS after the IfcProject, or after the file."""
    projects = model.by_type("IfcProject")
    name = projects[0].Name if projects else None
    if not name and isinstance(source, str | os.PathLike):
        name = os.path.splitext(os.path.basename(os.fspath(source)))[0]
    return f"{name or 'IFC'} model coordinates"


def _length_unit(model: ifcopenshell.file) -> tuple[str, float]:
    """The project's length unit as a (name, metres per unit) pair; metres when it has none."""
    unit = ifcopenshell.util.unit.get_project_unit(model, "LENGTHUNIT")
    if unit is None:
        return "metre", 1.0
    name = str(ifcopenshell.util.unit.get_full_unit_name(unit)).lower().replace("_", " ")
    return name, float(ifcopenshell.util.unit.calculate_unit_scale(model))


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


def _elevation(product: ifcopenshell.entity_instance) -> float | None:
    """The z of a storey's placement in model coordinates; None for other products."""
    if not product.is_a("IfcBuildingStorey"):
        return None
    return float(ifcopenshell.util.placement.get_storey_elevation(product))


def _psets(product: ifcopenshell.entity_instance) -> dict[str, dict[str, Any]]:
    """The product's property and quantity sets, its type's included, without the set ids."""
    psets = ifcopenshell.util.element.get_psets(product)
    return {
        name: {key: value for key, value in values.items() if key != "id"}
        for name, values in psets.items()
    }


def _properties_series(rows: Sequence[dict[str, dict[str, Any]]]) -> pl.Series:
    """Builds the ``properties`` Struct column from the property sets of every row.

    Each property set becomes a Struct field holding a Struct of its properties. A property
    gets the dtype of its values when they agree across rows (Boolean, Int64, Float64, String
    or a List of those; integers and floats together make Float64); when they differ, or when
    a value is nested, the property becomes a String with the value's text (JSON for
    containers).
    """
    kinds: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        for pset, values in row.items():
            pset_kinds = kinds.setdefault(pset, {})
            for key, value in values.items():
                if value is not None:
                    pset_kinds.setdefault(key, set()).add(_kind(value))
    dtypes = {
        pset: {key: _dtype(pset_kinds[key]) for key in pset_kinds if pset_kinds[key]}
        for pset, pset_kinds in kinds.items()
    }
    dtypes = {pset: fields for pset, fields in dtypes.items() if fields}
    if not dtypes:
        return pl.Series("properties", [None] * len(rows), dtype=pl.Null)
    dtype = pl.Struct({pset: pl.Struct(fields) for pset, fields in dtypes.items()})
    coerced = [_coerce_row(row, dtypes) if row else None for row in rows]
    return pl.Series("properties", coerced, dtype=dtype)


def _kind(value: object) -> str:
    """Classifies a property value for dtype inference."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list | tuple):
        inner = {_kind(item) for item in value if item is not None}
        if not inner:
            return "list:empty"
        if inner <= {"int", "float"} and len(inner) > 1:
            return "list:float"
        if len(inner) == 1:
            return "list:" + inner.pop()
    return "other"


_SCALAR_DTYPES: dict[str, pl.DataType] = {
    "bool": pl.Boolean(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "str": pl.String(),
}


def _dtype(kinds: set[str]) -> pl.DataType:
    """The polars dtype for the value kinds a property has across rows."""
    if kinds <= {"int", "float"}:
        return pl.Float64() if "float" in kinds else pl.Int64()
    if len(kinds) == 1:
        (kind,) = kinds
        if kind in _SCALAR_DTYPES:
            return _SCALAR_DTYPES[kind]
    list_kinds = {kind.removeprefix("list:") for kind in kinds if kind.startswith("list:")}
    if list_kinds and len(list_kinds) == len(kinds):
        inner = list_kinds - {"empty"}
        if inner <= {"int", "float"}:
            return pl.List(pl.Float64() if "float" in inner or not inner else pl.Int64())
        if len(inner) == 1 and next(iter(inner)) in _SCALAR_DTYPES:
            return pl.List(_SCALAR_DTYPES[next(iter(inner))])
    return pl.String()


def _coerce_row(
    row: dict[str, dict[str, Any]], dtypes: dict[str, dict[str, pl.DataType]]
) -> dict[str, dict[str, Any] | None]:
    """Converts one row's property values to the dtypes chosen for the column."""
    out: dict[str, dict[str, Any] | None] = {}
    for pset, fields in dtypes.items():
        values = row.get(pset)
        if values is None:
            out[pset] = None
            continue
        out[pset] = {key: _coerce(values.get(key), dtype) for key, dtype in fields.items()}
    return out


def _coerce(value: Any, dtype: pl.DataType | DataTypeClass) -> Any:
    """Converts a property value to what polars expects for ``dtype``."""
    if value is None:
        return None
    if dtype == pl.String():
        if isinstance(value, str):
            return value
        if isinstance(value, list | tuple | dict):
            return json.dumps(value, default=str)
        return str(value)
    if dtype == pl.Float64():
        return float(value)
    if isinstance(dtype, pl.List):
        return [_coerce(item, dtype.inner) for item in value]
    return value


class _Styles:
    """Resolves ifcopenshell material names to the IfcSurfaceStyle names of a model."""

    def __init__(self, model: ifcopenshell.file) -> None:
        self._model = model
        self._names: dict[str, str | None] = {}

    def name(self, material: Any) -> str | None:
        """The IfcSurfaceStyle name behind an ifcopenshell material, or None without one."""
        material_name = str(material.name)
        if material_name not in self._names:
            self._names[material_name] = self._resolve(material_name)
        return self._names[material_name]

    def _resolve(self, material_name: str) -> str | None:
        # ifcopenshell names a style after its rendering entity: "IfcSurfaceStyleRendering-868"
        _, _, step_id = material_name.rpartition("-")
        if step_id.isdigit():
            try:
                shading = self._model.by_id(int(step_id))
            except RuntimeError:
                shading = None
            if shading is not None:
                for style in self._model.get_inverse(shading):
                    if style.is_a("IfcSurfaceStyle"):
                        return None if style.Name is None else str(style.Name)
        if material_name == "DefaultMaterial":
            return None
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


def _tessellate(model: ifcopenshell.file, threads: int) -> dict[str, dict[str, Any]]:
    """Tessellates every product with a representation, in world coordinates and model units.

    Returns:
        Per GlobalId, a ``mesh`` row: vertices, faces sorted by style, and the styles.
    """
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("convert-back-units", True)
    styles = _Styles(model)
    meshes: dict[str, dict[str, Any]] = {}
    for guid, geometry in _shapes(settings, model, threads):
        mesh = _mesh(geometry, styles)
        if mesh is None:
            continue
        earlier = meshes.get(guid)
        meshes[guid] = mesh if earlier is None else _merge(earlier, mesh)
    return meshes


def _shapes(settings: Any, model: ifcopenshell.file, threads: int) -> Iterator[tuple[str, Any]]:
    """Yields (GlobalId, geometry) for every shape the ifcopenshell iterator produces."""
    iterator = ifcopenshell.geom.iterator(settings, model, threads)
    if not iterator.initialize():
        return
    while True:
        shape = iterator.get()
        yield str(shape.guid), shape.geometry
        if not iterator.next():
            break


def _mesh(geometry: Any, styles: _Styles) -> dict[str, Any] | None:
    """Builds a ``mesh`` row from ifcopenshell geometry, with the faces sorted by style."""
    vertices = np.asarray(geometry.verts, dtype=np.float64)
    faces = np.asarray(geometry.faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0 or len(vertices) == 0:
        return None
    material_ids = np.asarray(geometry.material_ids, dtype=np.int64)
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
                "faces": int(count),
            }
        )
    return {
        "vertices": vertices.tolist(),
        "faces": faces[order].astype(np.uint32).ravel().tolist(),
        "styles": style_rows,
    }


def _merge(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Appends one mesh row to another, when a product yields more than one shape."""
    offset = len(first["vertices"]) // 3
    return {
        "vertices": first["vertices"] + second["vertices"],
        "faces": first["faces"] + [index + offset for index in second["faces"]],
        "styles": first["styles"] + second["styles"],
    }


def _bbox(mesh: dict[str, Any] | None) -> dict[str, float] | None:
    """The bounding box of a mesh row, or None without a mesh."""
    if mesh is None:
        return None
    vertices = np.asarray(mesh["vertices"], dtype=np.float64).reshape(-1, 3)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return {
        "xmin": float(low[0]),
        "ymin": float(low[1]),
        "zmin": float(low[2]),
        "xmax": float(high[0]),
        "ymax": float(high[1]),
        "zmax": float(high[2]),
    }

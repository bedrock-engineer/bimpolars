"""Reads a Speckle version into a BimDataFrame from its v3 objects, without specklepy.

A Speckle version is a graph of JSON objects: a root ``Collection`` whose ``__closure`` lists
every object underneath it, DataObjects with a ``displayValue`` of Meshes, long
arrays detached into DataChunks, and instances: an ``InstanceProxy`` places the geometry
of an ``InstanceDefinitionProxy`` (listed on the root) with a 4x4 transform. Those objects
come from a Speckle server, or from a bundle (the Parquet tables a Revit or IFC file is
converted to) projected to v3 objects with specklepy. :func:`version_objects` reads one
version's objects from a local copy; :func:`read_speckle` turns them into a
:class:`~bimpolars.BimDataFrame`, expanding the instances, and reads the Revit categories,
parameters and levels from the version's bundle when one is given. Nothing here needs
specklepy; :mod:`bimpolars.bundle` (the ``[speckle]`` extra) does, to project a bundle on disk
to the objects this module reads.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bimpolars.crs import LinearUnit, engineering_crs
from bimpolars.data import fetch_bundle
from bimpolars.frame import BimDataFrame
from bimpolars.mesh import UNSTYLED, Mesh, Style, mesh_frame, styled_mesh
from bimpolars.properties import properties_series

type SpeckleObject = dict[str, Any]
"""One v3 object as parsed JSON."""

type Bundle = str | os.PathLike[str] | Mapping[str, pl.DataFrame]
"""A Speckle bundle: the folder holding its ``<name>.<table>.parquet`` files, the URL of its
key prefix in the bimpolars bucket (:func:`bundle_folder`), or the tables themselves by name
(``"eav.eav"``, ``"envelope.nodes"``, ...)."""

SPECKLE_UNITS: dict[str, LinearUnit] = {
    "mm": LinearUnit("millimetre", 0.001),
    "cm": LinearUnit("centimetre", 0.01),
    "m": LinearUnit("metre", 1.0),
    "km": LinearUnit("kilometre", 1000.0),
    "in": LinearUnit("inch", 0.0254),
    "ft": LinearUnit("foot", 0.3048),
    "yd": LinearUnit("yard", 0.9144),
    "mi": LinearUnit("mile", 1609.344),
}
"""The length units Speckle objects name in their ``units`` field."""

LEVEL_CLASS = "Levels"
"""The ``class`` of the rows :func:`read_speckle` adds for a bundle's levels."""

LEVEL_ID_PREFIX = "level-"
"""Level rows have the ``id`` ``level-<node id>``, the node id from the bundle's envelope."""

OBJECT_SET = "object"
"""The property set that holds an object's own fields, such as a Revit object's ``family``
and ``type``: the fields that sit next to ``properties`` rather than inside it."""

_INSTANCE_SUFFIX = re.compile(r"^(?P<base>.+)-instance-\d+$")
"""A placement of an element with several: the element's applicationId plus ``-instance-N``."""

_DEFINITION_GEOMETRY = re.compile(r"^def-geo-(?P<index>\d+)$")
"""The applicationId specklepy gives a definition's mesh: the bundle's geometry index."""

_INSTANCE_TYPE = "InstanceProxy"
_COLLECTION_TYPE = "Collection"
_CHUNK_TYPE = "DataChunk"

_SKIP_EVERYWHERE = frozenset({"id", "speckle_type", "applicationId", "totalChildrenCount"})
_SKIP_ON_OBJECT = frozenset({"__closure", "displayValue", "elements", "units", "name", "category"})
_GEOMETRY_DATA = frozenset({"vertices", "faces", "colors", "textureCoordinates", "vertexNormals"})
_CONSUMED_PATHS = frozenset({"name", "category", "units", "speckle_type"})

_LEVEL_KIND = 5
"""The envelope's node kind of a level, when ``envelope.node_kinds`` is not there to say (LEVEL)."""
_MATERIAL_KIND = 3
"""The envelope's node kind of a material (MATERIAL)."""
_ON_LEVEL = 7
"""The envelope's relation from an object to its level, when ``envelope.rel_types`` is not there
to say (ON_LEVEL)."""
_HAS_MATERIAL = 5
"""The envelope's relation from a geometry to its material (HAS_MATERIAL)."""
_SQLITE_CHUNK = 500
"""Objects fetched per SQLite query, under the 999 variables an older SQLite allows."""

_PLACEMENT_SOURCES = {
    "projectBasePoint": "Revit project base point",
    "surveyPoint": "Revit survey point",
    "internalOrigin": "Revit internal origin",
}


def version_objects(
    source: str | os.PathLike[str] | pl.DataFrame | sqlite3.Connection, version: str
) -> list[SpeckleObject]:
    """Reads the v3 objects of one Speckle version from a local copy, root first.

    Two local layouts are read:

    - A Parquet file (or a DataFrame) with one row per object and the columns
      ``version_id``, ``object_id``, ``is_root`` and ``content`` (the object's JSON text), as
      ``rvt-advanced-sample/speckle-v3-objects.parquet`` has them. Other columns, such as
      ``model_id`` and ``model_name``, are ignored; a file is scanned, so only the row groups
      that hold the version are read.
    - A SQLite database (``.db``, ``.sqlite`` or ``.sqlite3``, or an open connection) with the
      tables ``versions(version_id, v3_root_id, ...)`` and ``objects(hash, content)``: the
      ``v3_objects.db`` a Speckle download writes. The root's ``__closure`` names the
      version's other objects.

    Args:
        source: The Parquet or SQLite file, the DataFrame, or the SQLite connection.
        version: The Speckle version id.

    Returns:
        The parsed objects, the root first and its closure after it.

    Raises:
        ValueError: If the version is not in the source, has no root, or lacks objects of
            its closure.
    """
    if isinstance(source, pl.DataFrame):
        return _objects_from_frame(source, version)
    if isinstance(source, sqlite3.Connection):
        return _objects_from_sqlite(source, version)
    path = Path(source)
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            return _objects_from_sqlite(connection, version)
        finally:
            connection.close()
    return _objects_from_frame(pl.scan_parquet(path), version)


def _objects_from_frame(frame: pl.DataFrame | pl.LazyFrame, version: str) -> list[SpeckleObject]:
    rows = (
        frame.lazy()
        .filter(pl.col("version_id") == version)
        .sort("is_root", descending=True, maintain_order=True)
        .collect()
    )
    if rows.is_empty():
        raise ValueError(f"version {version!r} is not in the objects table")
    if not rows["is_root"][0] or rows["is_root"].sum() != 1:
        raise ValueError(f"version {version!r} must have exactly one root object")
    return [json.loads(content) for content in rows["content"]]


def _objects_from_sqlite(connection: sqlite3.Connection, version: str) -> list[SpeckleObject]:
    found = connection.execute(
        "SELECT v3_root_id FROM versions WHERE version_id = ?", (version,)
    ).fetchone()
    if found is None:
        raise ValueError(f"version {version!r} is not in the versions table")
    (root_id,) = found
    root_row = connection.execute(
        "SELECT content FROM objects WHERE hash = ?", (root_id,)
    ).fetchone()
    if root_row is None:
        raise ValueError(f"the root object {root_id!r} of version {version!r} is missing")
    root: SpeckleObject = json.loads(root_row[0])
    closure = list(root.get("__closure", {}))
    contents: dict[str, str] = {}
    for start in range(0, len(closure), _SQLITE_CHUNK):
        ids = closure[start : start + _SQLITE_CHUNK]
        marks = ",".join("?" * len(ids))
        contents.update(
            connection.execute(
                f"SELECT hash, content FROM objects WHERE hash IN ({marks})", ids
            ).fetchall()
        )
    missing = [object_id for object_id in closure if object_id not in contents]
    if missing:
        raise ValueError(
            f"{len(missing)} objects of version {version!r} are missing, such as {missing[0]!r}"
        )
    return [root, *(json.loads(contents[object_id]) for object_id in closure)]


def read_speckle(
    objects: Sequence[Mapping[str, Any]],
    bundle: Bundle | None = None,
    *,
    name: str | None = None,
) -> BimDataFrame:
    """Reads the v3 objects of a Speckle version into a BimDataFrame, one row per element.

    The elements are the objects under the root's ``elements`` (through nested collections),
    except the collections themselves and the objects that belong to an instance definition:
    an ``InstanceProxy`` is expanded in place, its definition's meshes (nested instances
    included) transformed by its 4x4 ``transform`` and scaled from their own ``units`` to the
    model's. A placement named ``<applicationId>-instance-N`` folds into the row of the
    element ``<applicationId>``, as does any other object with that applicationId, so an
    element placed several times, or placed and carrying geometry of its own, is one row with
    one merged mesh. With a bundle, the elements the bundle knows but the version has no
    object for (Revit elements the projection leaves out: curtain walls, room separations, MEP
    systems, wires and circuits) get rows too, with the bundle's category, name, properties
    and level and a null ``mesh``, so that the frame has one row per element of the model, as
    :func:`bimpolars.read_ifc` has.

    The columns:

    - ``id``: the object's ``applicationId`` (the Revit UniqueId, for a Revit model), or its
      Speckle ``id`` when it has none.
    - ``name``: the object's ``name``, else the bundle's.
    - ``class``: the object's ``category``, else the Revit category the bundle has for the
      applicationId, else the object's ``speckle_type``. Categorical.
    - ``properties``: a Struct of sets, each a Struct of values, as
      :func:`bimpolars.properties.properties_series` infers the dtypes. A set is a group of
      the property tree named by its dotted path: ``properties`` holds the values directly in
      the object's property bag (``elementId``, ``typeId``), a set such as
      ``properties.Parameters.Instance Parameters.Dimensions`` holds a group's values
      (``Volume``), and :data:`OBJECT_SET` holds the object's own fields (``family``,
      ``type``). The path is the bundle's eav ``path`` column: ``f"{set}.{name}"``. With a
      bundle, the eav values (type parameters included, instance values winning) fill the
      rows the objects leave bare, such as the instances. A value keeps the unit Revit states
      for its parameter (the eav ``unit`` column: lengths in millimetres, areas in square
      metres, volumes in cubic metres); it is not scaled to the model unit.
    - ``container``: with a bundle, the ``id`` of the row's level (``level-<node id>``); the
      levels get rows of their own (class :data:`LEVEL_CLASS`) with their ``elevation``.
    - ``native_geom``: null.
    - ``mesh``: :data:`bimpolars.MESH_DTYPE` in model units, faces sorted by style; ``bbox``
      its bounds. Quads and n-gons are fan-triangulated. A style is the ``RenderMaterial`` of
      the root's ``renderMaterialProxies`` that targets the mesh's applicationId, the
      element's, or the instance's, with its ``diffuse`` ARGB and ``opacity``. The first proxy
      that does wins when several do, which paints a layered wall, floor or ceiling (one proxy
      per layer, all targeting the element) in its first layer's material. A definition's mesh
      without one (specklepy's plain projection gives none, unlike
      :func:`bimpolars.bundle.bundle_objects`) gets, with a bundle, the material the envelope
      attaches to the bundle geometry it came from (``def-geo-<geometry index>``). A mesh
      without either is unstyled. A ``displayValue`` reference to an object the version
      does not include is skipped, so a row whose geometry is only such references, or only
      an instance of an empty definition, has no mesh.

    The CRS is a plain engineering CRS in the root's ``units``, else the bundle's, else the
    unit most of the meshes name, else metres: the model's own coordinates. A Revit model's
    geolocation is never used.

    Args:
        objects: The version's objects, root first, as :func:`version_objects` or
            :func:`bimpolars.bundle.bundle_objects` gives them.
        bundle: The version's Speckle bundle: a folder of ``<name>.<table>.parquet`` files,
            the URL of its key prefix in the bimpolars bucket (:func:`bundle_folder`), or
            the tables by name. Optional; without it the rows get what the objects carry.
        name: The model's name, for the CRS; the bundle's file name or the root's by default.

    Returns:
        The model as a BimDataFrame.

    Raises:
        ValueError: If the objects have no root, name a unit this module does not know, nest
            instance definitions in a cycle, lack a DataChunk an object refers to, or hold a
            face list or an instance transform of the wrong length; or if the bundle folder is
            missing, its URL is outside the bucket, it lacks ``eav.objects``, ``eav.paths`` or
            ``eav.eav``, or holds two tables of one name.
        urllib.error.URLError: If downloading a table of a bundle named by its URL fails.
    """
    if not objects:
        raise ValueError("no objects: the root object must come first")
    root = objects[0]
    if "elements" not in root and "__closure" not in root:
        raise ValueError("the first object is not a root: it has no elements and no __closure")
    tables = _BundleTables(bundle) if bundle is not None else None
    info = _BundleInfo(tables) if tables is not None else None
    unit = _model_unit(root, objects, info)
    reader = _Reader(objects, unit, info)

    rows: dict[str, _Row] = {}
    for obj in reader.elements(root):
        key = _row_key(obj)
        row = rows.setdefault(key, _Row(key))
        reader.add(row, obj)
    model_name = name or _model_name(root, info)
    datum = "Speckle model origin"
    if info is not None and info.placement_source:
        datum = _PLACEMENT_SOURCES.get(info.placement_source, info.placement_source)
    crs = engineering_crs(f"{model_name} model coordinates", unit=unit, datum=datum)
    return BimDataFrame(_frame(list(rows.values()), info, unit), crs)


class _Row:
    """One element as it is gathered from its objects: fields and mesh parts."""

    __slots__ = ("fields", "key", "klass", "mesh_ids", "name", "parts", "speckle_type")

    def __init__(self, key: str) -> None:
        self.key = key
        self.name: str | None = None
        self.klass: str | None = None
        self.speckle_type: str | None = None
        self.fields: dict[str, dict[str, Any]] = {}
        self.parts: list[_Part] = []
        self.mesh_ids: set[str] = set()


type _Part = tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
    Style | None,
]
"""A piece of a row's mesh: ``(n, 3)`` vertices in model units, ``(m, 3)`` faces and its
style, None when the mesh has none of its own."""


class _Reader:
    """Resolves the object graph: references, chunks, meshes, definitions and materials."""

    def __init__(
        self, objects: Sequence[Mapping[str, Any]], unit: LinearUnit, info: _BundleInfo | None
    ) -> None:
        root = objects[0]
        self.unit = unit
        self.geometry_styles = info.geometry_styles if info is not None else {}
        self.by_id: dict[str, Mapping[str, Any]] = {}
        self.by_app: dict[str, list[Mapping[str, Any]]] = {}
        for obj in objects:
            object_id = obj.get("id")
            if isinstance(object_id, str):
                self.by_id.setdefault(object_id, obj)
            app_id = obj.get("applicationId")
            if isinstance(app_id, str):
                self.by_app.setdefault(app_id, []).append(obj)
        self.definitions: dict[str, Mapping[str, Any]] = {
            str(definition["applicationId"]): definition
            for definition in root.get("instanceDefinitionProxies") or []
            if definition.get("applicationId")
        }
        self.definition_content: set[str] = {
            str(member)
            for definition in self.definitions.values()
            for member in definition.get("objects") or []
        }
        self.materials: dict[str, Style] = {}
        for proxy in root.get("renderMaterialProxies") or []:
            style = _material_style(proxy.get("value") or {})
            for target in proxy.get("objects") or []:
                self.materials.setdefault(str(target), style)
        self._definition_parts: dict[str, list[_Part]] = {}
        self._mesh_parts: dict[str, _Part | None] = {}

    def elements(self, root: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        """The element objects under the root, depth first through the collections."""
        seen: set[int] = set()

        def walk(obj: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if not _is_type(obj, _COLLECTION_TYPE):
                app_id = obj.get("applicationId")
                if not (isinstance(app_id, str) and app_id in self.definition_content):
                    yield obj
            for child in self.referenced(obj.get("elements")):
                if not _is_type(child, _CHUNK_TYPE):
                    yield from walk(child)

        for child in self.referenced(root.get("elements")):
            yield from walk(child)

    def referenced(self, items: Any) -> Iterator[Mapping[str, Any]]:
        """The objects behind a list of references (or inline objects)."""
        for item in items or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("speckle_type") == "reference":
                target = self.by_id.get(str(item.get("referencedId")))
                if target is not None:
                    yield target
            else:
                yield item

    def values(self, items: Any) -> list[Any]:
        """A flat list, with the detached DataChunks inlined."""
        if not isinstance(items, list):
            return []
        if not any(isinstance(item, Mapping) for item in items):
            return items
        out: list[Any] = []
        for item in items:
            if isinstance(item, Mapping):
                chunk = next(self.referenced([item]), None)
                if chunk is None:
                    raise ValueError(
                        f"object {item.get('referencedId')!r} is missing: a DataChunk the "
                        "version's objects do not include"
                    )
                out.extend(chunk.get("data") or [])
            else:
                out.append(item)
        return out

    def add(self, row: _Row, obj: Mapping[str, Any]) -> None:
        """Adds an element object's fields and geometry to its row."""
        app_id = obj.get("applicationId")
        speckle_type = obj.get("speckle_type")
        if not isinstance(speckle_type, str):
            speckle_type = None  # a missing type stays missing, so that the class can be null
        if _is_type(obj, _INSTANCE_TYPE):
            row.speckle_type = row.speckle_type or speckle_type
            row.parts.extend(self.instance_parts(obj, set()))
            return
        if row.name is None and isinstance(obj.get("name"), str):
            row.name = obj["name"]
        if row.klass is None and isinstance(obj.get("category"), str):
            row.klass = obj["category"]
        if speckle_type is not None and (row.speckle_type is None or _is_type(row, _INSTANCE_TYPE)):
            row.speckle_type = speckle_type  # an element's own type over its placements'
        for pset, values in _flatten(obj).items():
            target = row.fields.setdefault(pset, {})
            for key, value in values.items():
                target.setdefault(key, value)
        fallback = self.materials.get(str(app_id)) if app_id else None
        for mesh_id, part in self.object_parts(obj):
            # A mesh and the DataObject that displays it can share an applicationId and both
            # be listed: the mesh counts once
            if mesh_id not in row.mesh_ids:
                row.mesh_ids.add(mesh_id)
                row.parts.append(part if part[2] is not None else (part[0], part[1], fallback))

    def object_parts(self, obj: Mapping[str, Any]) -> list[tuple[str, _Part]]:
        """The meshes of an object, by id: itself when it is a mesh, else its displayValue."""
        meshes = [obj] if "vertices" in obj and "faces" in obj else []
        meshes += [
            shown
            for shown in self.referenced(obj.get("displayValue"))
            if "vertices" in shown and "faces" in shown
        ]
        parts = []
        for mesh in meshes:
            part = self.mesh_part(mesh)
            if part is not None:
                parts.append((str(mesh.get("id")), part))
        return parts

    def mesh_part(self, mesh: Mapping[str, Any]) -> _Part | None:
        """One mesh as vertices in model units, triangles and its own style, memoised."""
        key = str(mesh.get("id"))
        if key not in self._mesh_parts:
            self._mesh_parts[key] = self._decode(mesh)
        return self._mesh_parts[key]

    def _decode(self, mesh: Mapping[str, Any]) -> _Part | None:
        vertices = np.asarray(self.values(mesh.get("vertices")), dtype=np.float64).reshape(-1, 3)
        faces = _triangles(self.values(mesh.get("faces")))
        if len(vertices) == 0 or len(faces) == 0:
            return None
        scale = self.scale(mesh.get("units"))
        if scale != 1.0:
            vertices = vertices * scale
        app_id = mesh.get("applicationId")
        style = self.materials.get(str(app_id)) if app_id else None
        if style is None and isinstance(app_id, str):
            # A definition's mesh is named after the bundle geometry it was projected from
            match = _DEFINITION_GEOMETRY.match(app_id)
            if match:
                style = self.geometry_styles.get(int(match.group("index")))
        return (vertices, faces, style)

    def scale(self, units: Any) -> float:
        """The factor from an object's ``units`` to the model unit."""
        if not units or units == "none":
            return 1.0
        return _unit(str(units)).conversion_factor / self.unit.conversion_factor

    def definition_parts(self, definition_id: str, active: set[str]) -> list[_Part]:
        """A definition's parts in its own coordinates, nested instances expanded."""
        if definition_id in self._definition_parts:
            return self._definition_parts[definition_id]
        if definition_id in active:
            raise ValueError(f"instance definition {definition_id!r} contains itself")
        definition = self.definitions.get(definition_id)
        parts: list[_Part] = []
        seen: set[str] = set()
        if definition is not None:
            active.add(definition_id)
            for member in definition.get("objects") or []:
                for obj in self.by_app.get(str(member), []):
                    if _is_type(obj, _INSTANCE_TYPE):
                        parts.extend(self.instance_parts(obj, active))
                        continue
                    for mesh_id, part in self.object_parts(obj):
                        # specklepy lists a definition's mesh both as the mesh and as the
                        # DataObject that displays it, under one applicationId
                        if mesh_id not in seen:
                            seen.add(mesh_id)
                            parts.append(part)
            active.discard(definition_id)
        self._definition_parts[definition_id] = parts
        return parts

    def instance_parts(self, instance: Mapping[str, Any], active: set[str]) -> list[_Part]:
        """An instance's parts: its definition's, moved, the instance's material filling in."""
        definition_id = instance.get("definitionId")
        if not isinstance(definition_id, str):
            return []
        matrix = _transform(instance.get("transform"), self.scale(instance.get("units")))
        app_id = instance.get("applicationId")
        fallback = None
        if isinstance(app_id, str):
            fallback = self.materials.get(app_id) or self.materials.get(_row_key(instance))
        parts = []
        for vertices, faces, style in self.definition_parts(definition_id, active):
            moved = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            parts.append((moved, faces, style if style is not None else fallback))
        return parts


def bundle_folder(bundle: str | os.PathLike[str]) -> Path:
    """Gives the local folder of a Speckle bundle named by its folder or by its bucket URL.

    Args:
        bundle: The folder holding the bundle's ``<name>.<table>.parquet`` files, or the URL
            of the bundle's key prefix in the bimpolars bucket, ``<DATA_URL><prefix>`` (the
            prefix as :func:`bimpolars.data.fetch_bundle` takes it): the tables are then
            fetched into the data folder, downloaded if they are not there yet.

    Returns:
        The folder.

    Raises:
        ValueError: If the folder does not exist, or the URL is not in the bucket.
        urllib.error.URLError: If downloading a table of a bundle named by its URL fails.
    """
    if isinstance(bundle, str) and "://" in bundle:
        return fetch_bundle(bundle)
    folder = Path(bundle)
    if not folder.is_dir():
        raise ValueError(f"the bundle folder {folder} does not exist")
    return folder


class _BundleTables:
    """Reads the tables of a bundle from its folder or from a mapping, by table name."""

    def __init__(self, bundle: Bundle) -> None:
        self._tables: Mapping[str, pl.DataFrame] | None = None
        self._folder: Path | None = None
        if isinstance(bundle, Mapping):
            self._tables = bundle
        else:
            self._folder = bundle_folder(bundle)

    def get(self, table: str) -> pl.DataFrame | None:
        """One table, or None when the bundle does not have it."""
        if self._tables is not None:
            return self._tables.get(table)
        assert self._folder is not None
        paths = sorted(self._folder.glob(f"*.{table}.parquet"))
        if not paths:
            return None
        if len(paths) > 1:
            raise ValueError(f"{self._folder} holds more than one {table} table: {paths}")
        return pl.read_parquet(paths[0])

    def require(self, table: str) -> pl.DataFrame:
        """One table that the bundle must have."""
        frame = self.get(table)
        if frame is None:
            raise ValueError(f"the bundle has no {table} table")
        return frame


class _BundleInfo:
    """What :func:`read_speckle` takes from a bundle, keyed by applicationId."""

    def __init__(self, tables: _BundleTables) -> None:
        objects = tables.require("eav.objects")
        app_ids: dict[int, str] = dict(objects.select("object_index", "application_id").iter_rows())
        self.app_ids: list[str] = list(dict.fromkeys(app_ids.values()))
        paths: dict[int, str] = dict(
            tables.require("eav.paths").select("path_index", "path").iter_rows()
        )
        self.names: dict[str, str] = {}
        self.categories: dict[str, str] = {}
        self.fields: dict[str, dict[str, dict[str, Any]]] = {}
        by_type: dict[int, list[tuple[str, Any]]] = {}
        type_eav = tables.get("eav.type_eav")
        if type_eav is not None:
            for type_index, path_index, value in _eav_values(type_eav, "type_index"):
                by_type.setdefault(type_index, []).append((paths[path_index], value))
        type_of: dict[int, int] = {}
        object_type = tables.get("eav.object_type")
        if object_type is not None:
            type_of = dict(object_type.select("object_index", "type_index").iter_rows())
        values_of: dict[int, list[tuple[str, Any]]] = {}
        for object_index, path_index, value in _eav_values(
            tables.require("eav.eav"), "object_index"
        ):
            values_of.setdefault(object_index, []).append((paths[path_index], value))
        for object_index, app_id in app_ids.items():
            own = values_of.get(object_index, [])
            typed = by_type.get(type_of.get(object_index, -1), [])
            self._take(app_id, own, typed)
        self.levels: list[tuple[str, str | None, float | None]] = []
        self.level_of: dict[str, str] = {}
        self.geometry_styles: dict[int, Style] = {}
        nodes = tables.get("envelope.nodes")
        relations = tables.get("envelope.relations")
        if nodes is not None and relations is not None:
            # the envelope's lookup tables name its kinds and relations; the numbers of schema
            # 1.5 are the fallback for a bundle without them
            kinds = tables.get("envelope.node_kinds")
            rel_types = tables.get("envelope.rel_types")
            material_kind = _code(kinds, "kind", "MATERIAL", _MATERIAL_KIND)
            level_kind = _code(kinds, "kind", "LEVEL", _LEVEL_KIND)
            has_material = _code(rel_types, "rel", "HAS_MATERIAL", _HAS_MATERIAL)
            on_level = _code(rel_types, "rel", "ON_LEVEL", _ON_LEVEL)
            materials = {
                node_id: _material_style({"name": material, "diffuse": argb, "opacity": opacity})
                for node_id, material, argb, opacity in nodes.filter(
                    pl.col("kind") == material_kind
                )
                .select("id", "name", "argb", "opacity")
                .iter_rows()
            }
            for geometry, node_id in (
                relations.filter(pl.col("rel") == has_material).select("src", "dst").iter_rows()
            ):
                if node_id in materials:
                    self.geometry_styles.setdefault(geometry, materials[node_id])
            for node_id, level_name, elevation in (
                nodes.filter(pl.col("kind") == level_kind)
                .select("id", "name", "elevation")
                .iter_rows()
            ):
                self.levels.append((f"{LEVEL_ID_PREFIX}{node_id}", level_name, elevation))
            for object_index, node_id in (
                relations.filter(pl.col("rel") == on_level).select("src", "dst").iter_rows()
            ):
                if object_index in app_ids:
                    self.level_of[app_ids[object_index]] = f"{LEVEL_ID_PREFIX}{node_id}"
        self.model: dict[str, Any] = {}
        model = tables.get("eav.model")
        if model is not None:
            for path, text, number, flag in model.select(
                "path", "value_string", "value_double", "value_boolean"
            ).iter_rows():
                self.model[path] = _first_value(text, number, flag)
        self.units: str | None = self.model.get("modelPlacement.units") or self.model.get(
            "document.lengthUnit"
        )
        self.placement_source: str | None = self.model.get("modelPlacement.source")
        file_name = self.model.get("document.fileName")
        self.name: str | None = Path(str(file_name)).stem if file_name else None

    def _take(self, app_id: str, own: list[tuple[str, Any]], typed: list[tuple[str, Any]]) -> None:
        """Files an object's eav values, its own winning over its type's, into sets.

        The set is the eav path up to its last dot, the key what follows it, so a group or
        parameter named with a dot (``Nr. of risers``) splits there, where the object's own
        bag, whose keys are whole, would not; the RVT samples have no such name.
        """
        fields: dict[str, dict[str, Any]] = {}
        for path, value in [*own, *typed]:
            pset, _, key = path.rpartition(".")
            if not pset:
                if path == "name":
                    self.names.setdefault(app_id, value)
                elif path == "category":
                    self.categories.setdefault(app_id, value)
                if path in _CONSUMED_PATHS:
                    continue
                pset = OBJECT_SET
            fields.setdefault(pset, {}).setdefault(key, value)
        if fields:
            self.fields[app_id] = fields


def _code(table: pl.DataFrame | None, column: str, name: str, default: int) -> int:
    """The number an envelope lookup table gives a kind or relation name, else the default."""
    if table is not None and "name" in table.columns and column in table.columns:
        found = table.filter(pl.col("name") == name)[column]
        if len(found) == 1 and found[0] is not None:
            return int(found[0])
    return default


def _eav_values(eav: pl.DataFrame, index: str) -> Iterator[tuple[int, int, Any]]:
    """Yields ``(index, path_index, value)`` with the value as str, float or bool."""
    for index_value, path_index, text, number, flag in eav.select(
        index, "path_index", "value_string", "value_double", "value_boolean"
    ).iter_rows():
        value = _first_value(text, number, flag)
        if value is not None:
            yield index_value, path_index, value


def _first_value(text: str | None, number: float | None, flag: bool | None) -> Any:
    """The typed value of an eav row: its number, else its boolean, else its string.

    A numeric value comes with its text as well (``elementId`` is ``130846.0`` and
    ``"130846"``); the number is what specklepy's projection puts on the object.
    """
    if number is not None:
        return number
    if flag is not None:
        return flag
    return text


def _frame(rows: list[_Row], info: _BundleInfo | None, unit: LinearUnit) -> pl.DataFrame:
    """Builds the model columns: the levels, the gathered rows, the bundle's other elements."""
    ids: list[str] = []
    names: list[str | None] = []
    classes: list[str | None] = []
    containers: list[str | None] = []
    elevations: list[float | None] = []
    fields: list[dict[str, dict[str, Any]]] = []
    meshes: list[Mesh] = []
    positions: list[int | None] = []
    if info is not None:
        level_scale = 1.0
        if info.units:
            level_scale = _unit(info.units).conversion_factor / unit.conversion_factor
        for level_id, level_name, elevation in info.levels:
            ids.append(level_id)
            names.append(level_name)
            classes.append(LEVEL_CLASS)
            containers.append(None)
            elevations.append(None if elevation is None else float(elevation) * level_scale)
            fields.append({})
            positions.append(None)
    for row in rows:
        ids.append(row.key)
        klass = row.klass
        row_name = row.name
        row_fields = row.fields
        container = None
        if info is not None:
            klass = klass or info.categories.get(row.key)
            row_name = row_name if row_name is not None else info.names.get(row.key)
            container = info.level_of.get(row.key)
            for pset, values in info.fields.get(row.key, {}).items():
                target = row_fields.setdefault(pset, {})
                for key, value in values.items():
                    target.setdefault(key, value)
        names.append(row_name)
        classes.append(klass or row.speckle_type)
        containers.append(container)
        elevations.append(None)
        fields.append(row_fields)
        mesh = _row_mesh(row.parts)
        if mesh is None:
            positions.append(None)
        else:
            positions.append(len(meshes))
            meshes.append(mesh)
    if info is not None:
        # the bundle's elements that the version has no object for: a row without a mesh
        present = set(ids)
        for app_id in info.app_ids:
            if app_id in present:
                continue
            ids.append(app_id)
            names.append(info.names.get(app_id))
            classes.append(info.categories.get(app_id))
            containers.append(info.level_of.get(app_id))
            elevations.append(None)
            fields.append(dict(info.fields.get(app_id, {})))
            positions.append(None)
    geometry = mesh_frame(meshes)
    gather = pl.Series(positions, dtype=pl.UInt32)
    return pl.DataFrame(
        [
            pl.Series("id", ids, dtype=pl.String),
            pl.Series("name", names, dtype=pl.String),
            pl.Series("class", classes, dtype=pl.Categorical),
            properties_series(fields),
            pl.Series("container", containers, dtype=pl.String),
            pl.Series("elevation", elevations, dtype=pl.Float64),
            pl.Series("native_geom", [None] * len(ids), dtype=pl.Null),
            geometry["mesh"].gather(gather),
            geometry["bbox"].gather(gather),
        ]
    )


def _row_mesh(parts: list[_Part]) -> Mesh | None:
    """Merges a row's parts into one mesh with its faces sorted by style."""
    if not parts:
        return None
    styles: list[Style] = []
    style_index: dict[Style, int] = {}
    vertex_blocks = []
    face_blocks = []
    id_blocks = []
    offset = 0
    for vertices, faces, style in parts:
        key = style if style is not None else UNSTYLED
        if key not in style_index:
            style_index[key] = len(styles)
            styles.append(key)
        vertex_blocks.append(vertices)
        face_blocks.append(faces + offset)
        id_blocks.append(np.full(len(faces), style_index[key], dtype=np.int64))
        offset += len(vertices)
    return styled_mesh(
        np.vstack(vertex_blocks), np.vstack(face_blocks), np.concatenate(id_blocks), styles
    )


def _row_key(obj: Mapping[str, Any]) -> str:
    """The row an object belongs to: its applicationId without ``-instance-N``, else its id."""
    app_id = obj.get("applicationId")
    if isinstance(app_id, str) and app_id:
        match = _INSTANCE_SUFFIX.match(app_id)
        return match.group("base") if match else app_id
    return str(obj.get("id"))


def _is_type(obj: Any, suffix: str) -> bool:
    """Whether an object's ``speckle_type`` names the class ``suffix``.

    A type is a dotted name, or several chained with ``:`` from the base class down
    (``Speckle.Core.Models.Collections.Collection:Objects.Other.Layer``); the last segment of
    any of them counts.
    """
    speckle_type = obj.speckle_type if isinstance(obj, _Row) else obj.get("speckle_type")
    if not isinstance(speckle_type, str):
        return False
    return any(name.rsplit(".", 1)[-1] == suffix for name in speckle_type.split(":"))


def _flatten(obj: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Files an object's fields and property bag into sets named by dotted path.

    A value's set is the dotted path of the group it sits in; the object's own fields form
    :data:`OBJECT_SET`. References, the structural fields and, on a mesh that is an element
    itself, the geometry arrays are left out.
    """
    sets: dict[str, dict[str, Any]] = {}
    skip_on_object = _SKIP_ON_OBJECT
    if "vertices" in obj and "faces" in obj:
        skip_on_object = skip_on_object | _GEOMETRY_DATA  # a mesh that is an element itself

    def walk(node: Mapping[str, Any], path: str) -> None:
        for key, value in node.items():
            if key in _SKIP_EVERYWHERE or (not path and key in skip_on_object):
                continue
            if _is_reference(value) or (
                isinstance(value, list) and value and all(_is_reference(item) for item in value)
            ):
                continue
            if isinstance(value, Mapping):
                walk(value, f"{path}.{key}" if path else key)
            else:
                sets.setdefault(path or OBJECT_SET, {})[key] = value

    walk(obj, "")
    return sets


def _is_reference(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("speckle_type") == "reference"


def _triangles(faces: list[Any]) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Speckle's face list (``n, i1..in`` per face; 0 and 1 mean 3 and 4) as triangles."""
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    flat = np.asarray(faces, dtype=np.int64)
    if len(flat) % 4 == 0:
        heads = flat[0::4]
        if np.all((heads == 3) | (heads == 0)):
            return flat.reshape(-1, 4)[:, 1:].copy()
    out: list[tuple[int, int, int]] = []
    position = 0
    while position < len(flat):
        count = int(flat[position])
        if count < 3:
            count += 3
        corners = flat[position + 1 : position + 1 + count].tolist()
        if len(corners) != count:
            raise ValueError("a face of the mesh runs past the end of its face list")
        out.extend((corners[0], corners[i], corners[i + 1]) for i in range(1, count - 1))
        position += count + 1
    return np.asarray(out, dtype=np.int64).reshape(-1, 3)


def _transform(transform: Any, translation_scale: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    """An instance's row-major 4x4, its translation scaled into model units."""
    if transform is None:
        return np.eye(4)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"an instance transform must have 16 values, not {matrix.size}")
    matrix = matrix.reshape(4, 4).copy()
    matrix[:3, 3] *= translation_scale
    return matrix


def _material_style(material: Mapping[str, Any]) -> Style:
    """A RenderMaterial as a style: its name and its diffuse colour with its opacity."""
    argb = material.get("diffuse")
    if isinstance(argb, int | float):
        value = int(argb) & 0xFFFFFFFF
        rgb = ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
    else:
        rgb = (255, 255, 255)
    opacity = material.get("opacity")
    alpha = float(opacity) if isinstance(opacity, int | float) else 1.0
    name = material.get("name")
    return Style(str(name) if name else None, (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha))


def _unit(units: str) -> LinearUnit:
    """The linear unit a Speckle ``units`` string names."""
    try:
        return SPECKLE_UNITS[units]
    except KeyError:
        known = ", ".join(SPECKLE_UNITS)
        raise ValueError(f"unknown Speckle unit {units!r}; expected one of {known}") from None


def _model_unit(
    root: Mapping[str, Any], objects: Sequence[Mapping[str, Any]], info: _BundleInfo | None
) -> LinearUnit:
    """The model unit: the root's ``units``, else the bundle's, else the meshes' most common.

    Without any of those the model is taken to be in metres, as :func:`bimpolars.read_ifc`
    assumes for an IFC without a length unit.
    """
    units = root.get("units")
    if isinstance(units, str) and units and units != "none":
        return _unit(units)
    if info is not None and info.units:
        return _unit(info.units)
    counts: dict[str, int] = {}
    for obj in objects:
        mesh_units = obj.get("units")
        if "vertices" in obj and isinstance(mesh_units, str) and mesh_units != "none":
            counts[mesh_units] = counts.get(mesh_units, 0) + 1
    if counts:
        return _unit(max(counts, key=lambda key: counts[key]))
    return SPECKLE_UNITS["m"]


def _model_name(root: Mapping[str, Any], info: _BundleInfo | None) -> str:
    """The model's name: the bundle's file name, else the root's name, else Speckle."""
    if info is not None and info.name:
        return info.name
    root_name = root.get("name")
    if isinstance(root_name, str) and root_name and root_name != "Received model":
        return root_name
    return "Speckle"


__all__ = [
    "LEVEL_CLASS",
    "LEVEL_ID_PREFIX",
    "OBJECT_SET",
    "SPECKLE_UNITS",
    "Bundle",
    "SpeckleObject",
    "bundle_folder",
    "read_speckle",
    "version_objects",
]

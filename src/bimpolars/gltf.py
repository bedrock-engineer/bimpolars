"""Binary glTF (glb) of a BimDataFrame's meshes, one primitive and material per style.

glTF 2.0 is what CesiumJS, three.js, Blender and most 3D viewers load: triangle meshes in
metres with the y-axis up. :func:`element_meshes_glb` writes the meshes of a
:class:`~bimpolars.BimDataFrame` as one glb, the faces grouped by style into one primitive
each, and :func:`triangles_glb` writes any triangles in metres the same way. Both keep the
model's z-up coordinates in the vertex data and turn the scene y-up with a rotation on the root
node (:data:`Y_UP_ROTATION`), so a reader that applies node transforms shows the model upright
and one that reads the accessors finds the model as it is, in metres.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from bimpolars.crs import linear_unit
from bimpolars.frame import BimDataFrame

GLTF_VERSION = "2.0"
"""The glTF version the files declare."""

Y_UP_ROTATION: tuple[float, float, float, float] = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
"""The rotation (a unit quaternion, x, y, z, w) of the root node of every glb written here.

A quarter turn about the x-axis that takes the model's z-up coordinates to glTF's y-up: model
(x, y, z) lands on scene (x, z, -y), the convention of 3D Tiles. In CesiumJS, load the glb with
``forwardAxis: Cesium.Axis.X`` (and the default up axis) to place the model's own axes with the
model matrix.
"""

EXCLUDED_CLASSES: tuple[str, ...] = ("IfcOpeningElement",)
"""The classes :func:`element_meshes_glb` leaves out: an opening's mesh is the void it cuts."""

_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_UP = (0.0, 0.0, 1.0)

_Floats = npt.NDArray[np.float64]
_Ints = npt.NDArray[np.int64]
_Material = tuple[str | None, tuple[float, float, float, float]]


def element_meshes_glb(bdf: BimDataFrame, *, normals: bool = True) -> bytes:
    """Writes the meshes of a BimDataFrame as binary glTF, one primitive per style.

    Every row with a mesh is written, except the :data:`EXCLUDED_CLASSES` (an
    ``IfcOpeningElement`` carries the mesh of the void it cuts). The vertices go from the
    model's length unit (:func:`bimpolars.linear_unit` of the CRS) to glTF's metres. The faces
    are grouped by style, ``(name, rgba)`` of the mesh's ``styles``, into one primitive each,
    with one material per style of that name and colour (``alphaMode`` BLEND when the colour is
    translucent, double sided, matte), so that a viewer renders the model the way its authoring
    tool coloured it.

    The scene is one root node, named after the CRS and carrying it as ``extras.crs``, that
    turns the model y-up (:data:`Y_UP_ROTATION`), with one child node holding the mesh. The
    child is translated to the centre of the model's bounding box and the vertex data is
    relative to it, so that the float32 vertices of a model far from its origin (georeferenced
    in a projected CRS, for instance) keep their precision.

    Args:
        bdf: A BimDataFrame with the ``mesh`` column (and ``class``, to leave openings out).
        normals: Whether to write normals. With them every triangle gets its own three
            vertices with the face normal, so the glb is flat shaded in every viewer and holds
            three vertices per face of the ``mesh`` column (the meshes share vertices between
            faces of different orientation, so per-vertex normals cannot be flat). Without
            them the vertices and faces are written as the ``mesh`` column holds them, indexed
            and about a quarter of the size: a glTF reader then computes flat normals itself
            (the specification requires it), though CesiumJS renders such a model unlit and
            three.js smooths it.

    Returns:
        The glb file content.

    Raises:
        ValueError: If the frame has no ``mesh`` column, or a mesh's styles do not cover its
            faces exactly.
    """
    frame = bdf.frame
    if "mesh" not in frame.columns:
        raise ValueError("element_meshes_glb needs the mesh column")
    # Select before filtering: a filter copies every column, and properties can be gigabytes
    columns = [name for name in ("class", "mesh") if name in frame.columns]
    keep = pl.col("mesh").is_not_null()
    if "class" in columns:
        keep &= ~pl.col("class").cast(pl.String).is_in(EXCLUDED_CLASSES)
    meshes = frame.select(columns).filter(keep)["mesh"]
    vertices, faces, materials, face_material = _geometry(
        meshes, linear_unit(bdf.crs).conversion_factor
    )
    glb = _Glb(name=bdf.crs.get("name"), extras={"crs": bdf.crs})
    if len(vertices) == 0:
        return glb.to_bytes()
    centre = glb.centre(vertices)
    vertices = vertices - centre
    material_ids = [glb.material(name, rgba) for name, rgba in materials]
    position = None if normals else glb.vec3(vertices, _ARRAY_BUFFER, bounds=True)
    groups = _grouped(faces, face_material, len(materials))
    for material, group in zip(material_ids, groups, strict=True):
        if normals:
            glb.flat_primitive(vertices[group], material)
        elif position is not None:
            glb.indexed_primitive(group, position, len(vertices), material)
    return glb.to_bytes(centre)


def triangles_glb(groups: Mapping[str, tuple[npt.ArrayLike, Sequence[float]]]) -> bytes:
    """Writes groups of triangles in metres, z up, as flat-shaded binary glTF.

    Each group becomes one primitive with one material, for terrain, extruded footprints or
    any geometry that is not a BimDataFrame. The scene is laid out as in
    :func:`element_meshes_glb`: a root node that turns the triangles y-up and a child node at
    the centre of their bounding box.

    Args:
        groups: Per material name, the triangles as an ``(n, 3, 3)`` array of corner
            coordinates in metres and the colour as RGB or RGBA in 0..1.

    Returns:
        The glb file content.

    Raises:
        ValueError: If an array is not ``(n, 3, 3)`` or a colour is not three or four values
            in 0..1.
    """
    corners: dict[str, _Floats] = {}
    for name, (triangles, _) in groups.items():
        array = np.asarray(triangles, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            raise ValueError(
                f"the triangles of {name!r} must have shape (n, 3, 3), not {array.shape}"
            )
        corners[name] = array
    glb = _Glb()
    stacked = [array.reshape(-1, 3) for array in corners.values() if len(array)]
    if not stacked:
        return glb.to_bytes()
    centre = glb.centre(np.concatenate(stacked))
    for name, (_, colour) in groups.items():
        if len(corners[name]):
            glb.flat_primitive(corners[name] - centre, glb.material(name, _rgba(colour)))
    return glb.to_bytes(centre)


def _rgba(colour: Sequence[float]) -> tuple[float, float, float, float]:
    """A colour as RGBA in 0..1, from RGB or RGBA."""
    values = [float(c) for c in colour]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4 or any(not 0.0 <= c <= 1.0 for c in values):
        raise ValueError(f"a colour is three or four values in 0..1, not {list(colour)!r}")
    return (values[0], values[1], values[2], values[3])


def _geometry(meshes: pl.Series, scale: float) -> tuple[_Floats, _Ints, list[_Material], _Ints]:
    """Concatenates the meshes of a ``mesh`` Series, scaled to metres.

    Returns:
        The vertices ``(n, 3)``, the faces ``(m, 3)`` as indices into them, the distinct
        styles as ``(name, rgba)`` sorted by name (unstyled last), and per face the index of
        its style.

    Raises:
        ValueError: If a mesh's styles do not cover its faces exactly, or a face index is out
            of range.
    """
    vertices = meshes.struct.field("vertices")
    faces = meshes.struct.field("faces")
    styles = meshes.struct.field("styles")
    vertex_counts = (vertices.list.len() // 3).cast(pl.Int64)
    face_counts = (faces.list.len() // 3).cast(pl.Int64)
    styled_counts = (
        styles.list.eval(pl.element().struct.field("face_count")).list.sum().fill_null(0)
    )
    if (styled_counts.cast(pl.Int64) != face_counts).any():
        raise ValueError("the styles of a mesh must cover its faces exactly, one block each")

    all_vertices = vertices.explode().drop_nulls().to_numpy().reshape(-1, 3) * scale
    all_vertices = np.ascontiguousarray(all_vertices, dtype=np.float64)
    offsets = np.repeat(
        (vertex_counts.cum_sum() - vertex_counts).to_numpy(), face_counts.to_numpy()
    )
    all_faces = faces.explode().drop_nulls().to_numpy().astype(np.int64).reshape(-1, 3)
    all_faces = all_faces + offsets[:, None]
    if len(all_faces) and (all_faces.min() < 0 or all_faces.max() >= len(all_vertices)):
        raise ValueError("a face of a mesh points outside its vertices")

    style_rows = styles.explode().drop_nulls().struct.unnest()
    style_rows = style_rows.filter(pl.col("face_count") > 0)
    rgbas = style_rows["rgba"].to_numpy().reshape(-1, 4).tolist()
    keys: list[_Material] = [
        (name, (rgba[0], rgba[1], rgba[2], rgba[3]))
        for name, rgba in zip(style_rows["name"].to_list(), rgbas, strict=True)
    ]
    materials = sorted(set(keys), key=lambda key: (key[0] is None, key[0] or "", key[1]))
    index = {key: position for position, key in enumerate(materials)}
    face_material = np.repeat(
        np.array([index[key] for key in keys], dtype=np.int64),
        style_rows["face_count"].cast(pl.Int64).to_numpy(),
    )
    return all_vertices, all_faces, materials, face_material


def _grouped(faces: _Ints, face_material: _Ints, count: int) -> list[_Ints]:
    """The faces of each material, in the order of the faces."""
    order = np.argsort(face_material, kind="stable")
    bounds = np.searchsorted(face_material[order], np.arange(count + 1))
    return [faces[order[bounds[i] : bounds[i + 1]]] for i in range(count)]


class _Glb:
    """Accumulates one glTF mesh of primitives with their materials and writes the glb."""

    def __init__(self, name: str | None = None, extras: Mapping[str, Any] | None = None) -> None:
        self._name = name
        self._extras = extras
        self._buffer = bytearray()
        self._views: list[dict[str, Any]] = []
        self._accessors: list[dict[str, Any]] = []
        self._materials: list[dict[str, Any]] = []
        self._primitives: list[dict[str, Any]] = []

    @staticmethod
    def centre(vertices: _Floats) -> _Floats:
        """The centre of the vertices' bounding box: the mesh node's translation."""
        result: _Floats = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        return result

    def material(self, name: str | None, rgba: tuple[float, float, float, float]) -> int:
        """Adds a matte, double-sided material and gives its index."""
        material: dict[str, Any] = {
            "pbrMetallicRoughness": {
                "baseColorFactor": list(rgba),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "doubleSided": True,
        }
        if name is not None:
            material["name"] = name
        if rgba[3] < 1.0:
            material["alphaMode"] = "BLEND"
        self._materials.append(material)
        return len(self._materials) - 1

    def flat_primitive(self, corners: _Floats, material: int) -> None:
        """Adds a non-indexed primitive: every triangle its own vertices with the face normal.

        Args:
            corners: The triangles as ``(n, 3, 3)`` corner coordinates.
            material: The index of the primitive's material.
        """
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(lengths > 0, normals / np.where(lengths > 0, lengths, 1.0), _UP)
        self._primitives.append(
            {
                "attributes": {
                    "POSITION": self.vec3(corners.reshape(-1, 3), _ARRAY_BUFFER, bounds=True),
                    "NORMAL": self.vec3(np.repeat(normals, 3, axis=0), _ARRAY_BUFFER),
                },
                "material": material,
            }
        )

    def indexed_primitive(
        self, faces: _Ints, position: int, vertex_count: int, material: int
    ) -> None:
        """Adds a primitive that indexes a shared POSITION accessor.

        Args:
            faces: The triangles as ``(n, 3)`` indices into the accessor.
            position: The index of the POSITION accessor.
            vertex_count: How many vertices that accessor holds, for the index type.
            material: The index of the primitive's material.
        """
        # The largest value of an index type is reserved, so 65535 vertices still fit a short
        if vertex_count <= 65535:
            indices = self._accessor(faces.ravel().astype(np.uint16), _UNSIGNED_SHORT, "SCALAR")
        else:
            indices = self._accessor(faces.ravel().astype(np.uint32), _UNSIGNED_INT, "SCALAR")
        self._views[-1]["target"] = _ELEMENT_ARRAY_BUFFER
        self._primitives.append(
            {"attributes": {"POSITION": position}, "indices": indices, "material": material}
        )

    def vec3(self, array: _Floats, target: int, *, bounds: bool = False) -> int:
        """Adds a float32 VEC3 accessor over an ``(n, 3)`` array and gives its index."""
        index = self._accessor(array.astype(np.float32), _FLOAT, "VEC3", bounds=bounds)
        self._views[-1]["target"] = target
        return index

    def _accessor(
        self, data: npt.NDArray[Any], component_type: int, kind: str, *, bounds: bool = False
    ) -> int:
        data = np.ascontiguousarray(data)
        self._views.append(
            {"buffer": 0, "byteOffset": len(self._buffer), "byteLength": data.nbytes}
        )
        self._buffer.extend(data.tobytes())
        self._buffer.extend(b"\0" * (-len(self._buffer) % 4))
        accessor = {
            "bufferView": len(self._views) - 1,
            "componentType": component_type,
            "count": len(data),
            "type": kind,
        }
        if bounds:
            accessor["min"] = data.min(axis=0).tolist()
            accessor["max"] = data.max(axis=0).tolist()
        self._accessors.append(accessor)
        return len(self._accessors) - 1

    def to_bytes(self, translation: _Floats | None = None) -> bytes:
        """Packs the glb: the JSON chunk and, when there is geometry, the binary chunk.

        Args:
            translation: The mesh node's translation in model coordinates (metres).
        """
        root: dict[str, Any] = {"rotation": list(Y_UP_ROTATION)}
        if self._name is not None:
            root["name"] = self._name
        if self._extras is not None:
            root["extras"] = dict(self._extras)
        gltf: dict[str, Any] = {
            "asset": {"version": GLTF_VERSION, "generator": "bimpolars"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [root],
        }
        if self._primitives:
            root["children"] = [1]
            child: dict[str, Any] = {"mesh": 0}
            if translation is not None:
                child["translation"] = translation.tolist()
            gltf["nodes"].append(child)
            gltf["meshes"] = [{"primitives": self._primitives}]
            gltf["materials"] = self._materials
            gltf["buffers"] = [{"byteLength": len(self._buffer)}]
            gltf["bufferViews"] = self._views
            gltf["accessors"] = self._accessors
        json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
        json_chunk += b" " * (-len(json_chunk) % 4)
        chunks = [struct.pack("<I4s", len(json_chunk), b"JSON"), json_chunk]
        if self._primitives:
            chunks += [struct.pack("<I4s", len(self._buffer), b"BIN\0"), bytes(self._buffer)]
        length = 12 + sum(len(chunk) for chunk in chunks)
        return b"".join([struct.pack("<4sII", b"glTF", 2, length), *chunks])

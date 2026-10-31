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

from bimpolars.crs import _linear_unit
from bimpolars.frame import BimDataFrame, as_bim_dataframe

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

UNIT = "metre"
"""The length unit of the vertex data, whatever the BIM CRS's unit: glTF's metre."""

_FLOAT = 5126
"""The accessor ``componentType`` of float32."""

_UNSIGNED_SHORT = 5123
"""The accessor ``componentType`` of uint16, for the indices of up to 65535 vertices."""

_UNSIGNED_INT = 5125
"""The accessor ``componentType`` of uint32, for the indices of more vertices."""

_ARRAY_BUFFER = 34962
"""The bufferView ``target`` of vertex attributes."""

_ELEMENT_ARRAY_BUFFER = 34963
"""The bufferView ``target`` of indices."""

_UP = (0.0, 0.0, 1.0)
"""The normal written for a degenerate triangle, which has none of its own."""

_Floats = npt.NDArray[np.float64]
_Ints = npt.NDArray[np.int64]
_Material = tuple[str | None, tuple[float, float, float, float]]


def element_meshes_glb(bdf: BimDataFrame | pl.DataFrame, *, normals: bool = True) -> bytes:
    """Writes the meshes of a BimDataFrame as binary glTF, one primitive per style.

    Every row with a mesh is written, except the :data:`EXCLUDED_CLASSES` (an
    ``IfcOpeningElement`` carries the mesh of the void it cuts). The vertices go from the
    model's length unit (the unit of the CRS's axes) to glTF's metres. The faces
    are grouped by style, ``(name, rgba)`` of the mesh's ``styles``, into one primitive each,
    with one material per style of that name and colour (``alphaMode`` BLEND when the colour is
    translucent, double sided, matte), so that a viewer renders the model the way its authoring
    tool coloured it.

    The scene is one root node that turns the model y-up (:data:`Y_UP_ROTATION`), with one
    child node holding the mesh. The root node is named after the CRS and carries it as
    ``extras.crs``, next to ``extras.unit``, the :data:`UNIT` of the vertex data: the CRS keeps
    its own unit (millimetres for most IFCs) while the data under the node is in metres. The
    child is translated to the centre of the written vertices' bounding box and the vertex data
    is relative to it, so that the float32 vertices of a model far from its origin
    (georeferenced in a projected CRS, for instance) keep their precision.

    Args:
        bdf: A BimDataFrame with the ``mesh`` column (and ``class``, to leave openings out),
            or a polars DataFrame with those and ``bim_crs``, in one CRS.
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
        ValueError: If the frame has no ``mesh`` column, not exactly one BIM CRS (a frame
            without rows has none), the CRS has no linear unit, a mesh's
            styles do not cover its faces exactly, a style has no colour, or a face points
            outside its mesh's vertices.
    """
    bdf = as_bim_dataframe(bdf)
    frame = bdf.frame
    if "mesh" not in frame.columns:
        raise ValueError("element_meshes_glb needs the mesh column")
    projjson = bdf.projjson
    scale = _linear_unit(projjson).conversion_factor
    # Select before filtering: a filter copies every column, and properties can be gigabytes
    columns = [name for name in ("class", "mesh") if name in frame.columns]
    keep = pl.col("mesh").is_not_null()
    if "class" in columns:
        # fill_null: a null class is not an excluded one, and ~null would drop the row
        keep &= ~pl.col("class").cast(pl.String).is_in(EXCLUDED_CLASSES).fill_null(False)
    meshes = frame.select(columns).filter(keep)["mesh"]
    vertices, faces, materials, face_material = _geometry(meshes, scale)
    glb = _Glb(name=projjson.get("name"), extras={"crs": projjson, "unit": UNIT})
    if len(vertices) == 0:
        return glb.to_bytes()
    vertices = glb.centre(vertices)
    material_ids = [glb.material(name, rgba) for name, rgba in materials]
    groups = _grouped(faces, face_material, len(materials))
    if normals:
        for material, group in zip(material_ids, groups, strict=True):
            glb.flat_primitive(vertices[group], material)
    else:
        position = glb.positions(vertices)
        for material, group in zip(material_ids, groups, strict=True):
            glb.indexed_primitive(group, position, len(vertices), material)
    return glb.to_bytes()


def triangles_glb(groups: Mapping[str, tuple[npt.ArrayLike, Sequence[float]]]) -> bytes:
    """Writes groups of triangles in metres, z up, as flat-shaded binary glTF.

    Each group becomes one primitive with one material, for terrain, extruded footprints or
    any geometry that is not a BimDataFrame. The scene is laid out as in
    :func:`element_meshes_glb`: a root node that turns the triangles y-up and a child node at
    the centre of the written vertices' bounding box.

    Args:
        groups: Per material name, the triangles as an ``(n, 3, 3)`` array of corner
            coordinates in metres and the colour as RGB or RGBA in 0..1.

    Returns:
        The glb file content.

    Raises:
        ValueError: If an array is not ``(n, 3, 3)`` or a colour is not three or four values
            in 0..1.
    """
    parsed: list[tuple[str, _Floats, tuple[float, float, float, float]]] = []
    for name, (triangles, colour) in groups.items():
        array = np.asarray(triangles, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            raise ValueError(
                f"the triangles of {name!r} must have shape (n, 3, 3), not {array.shape}"
            )
        if len(array):
            parsed.append((name, array, _rgba(colour)))
    glb = _Glb()
    if not parsed:
        return glb.to_bytes()
    corners = glb.centre(np.concatenate([array.reshape(-1, 3) for _, array, _ in parsed]))
    splits = np.cumsum([len(array) * 3 for _, array, _ in parsed])[:-1]
    for (name, _, rgba), centred in zip(parsed, np.split(corners, splits), strict=True):
        glb.flat_primitive(centred.reshape(-1, 3, 3), glb.material(name, rgba))
    return glb.to_bytes()


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
        ValueError: If a mesh's styles do not cover its faces exactly, a style has no colour,
            or a face points outside its mesh's vertices.
    """
    vertices = meshes.struct.field("vertices")
    faces = meshes.struct.field("faces")
    styles = meshes.struct.field("styles")
    vertex_counts = (vertices.list.len() // 3).cast(pl.Int64).to_numpy()
    face_counts = (faces.list.len() // 3).cast(pl.Int64).to_numpy()
    styled_counts = (
        styles.list.eval(pl.element().struct.field("face_count")).list.sum().fill_null(0)
    )
    if (styled_counts.cast(pl.Int64).to_numpy() != face_counts).any():
        raise ValueError("the styles of a mesh must cover its faces exactly, one block each")

    all_vertices = vertices.explode().drop_nulls().to_numpy().reshape(-1, 3) * scale
    all_vertices = np.ascontiguousarray(all_vertices, dtype=np.float64)
    local_faces = faces.explode().drop_nulls().to_numpy().astype(np.int64).reshape(-1, 3)
    if len(local_faces) and (
        local_faces.min() < 0
        or (local_faces.max(axis=1) >= np.repeat(vertex_counts, face_counts)).any()
    ):
        raise ValueError("a face of a mesh points outside its vertices")
    offsets = np.repeat(np.cumsum(vertex_counts) - vertex_counts, face_counts)
    all_faces = local_faces + offsets[:, None]

    style_rows = styles.explode().drop_nulls().struct.unnest()
    style_rows = style_rows.filter(pl.col("face_count") > 0)
    if style_rows["rgba"].null_count():
        raise ValueError("every style needs an rgba colour")
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
        self._translation: _Floats | None = None
        self._buffer = bytearray()
        self._views: list[dict[str, Any]] = []
        self._accessors: list[dict[str, Any]] = []
        self._materials: list[dict[str, Any]] = []
        self._primitives: list[dict[str, Any]] = []

    def centre(self, vertices: _Floats) -> _Floats:
        """Makes the centre of the vertices' bounding box the mesh node's translation.

        Args:
            vertices: Every vertex the file will hold, ``(n, 3)``.

        Returns:
            The vertices relative to that centre, as the accessors store them.
        """
        self._translation = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        result: _Floats = vertices - self._translation
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
                    "POSITION": self._vec3(corners.reshape(-1, 3), bounds=True),
                    "NORMAL": self._vec3(np.repeat(normals, 3, axis=0)),
                },
                "material": material,
            }
        )

    def positions(self, vertices: _Floats) -> int:
        """Adds the POSITION accessor that indexed primitives share and gives its index."""
        return self._vec3(vertices, bounds=True)

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
            indices = self._accessor(
                faces.ravel().astype(np.uint16), _UNSIGNED_SHORT, "SCALAR", _ELEMENT_ARRAY_BUFFER
            )
        else:
            indices = self._accessor(
                faces.ravel().astype(np.uint32), _UNSIGNED_INT, "SCALAR", _ELEMENT_ARRAY_BUFFER
            )
        self._primitives.append(
            {"attributes": {"POSITION": position}, "indices": indices, "material": material}
        )

    def _vec3(self, array: _Floats, *, bounds: bool = False) -> int:
        """Adds a float32 VEC3 vertex attribute over an ``(n, 3)`` array and gives its index."""
        return self._accessor(
            array.astype(np.float32), _FLOAT, "VEC3", _ARRAY_BUFFER, bounds=bounds
        )

    def _accessor(
        self,
        data: npt.NDArray[Any],
        component_type: int,
        kind: str,
        target: int,
        *,
        bounds: bool = False,
    ) -> int:
        """Appends the data to the buffer in its own bufferView and gives the accessor's index."""
        data = np.ascontiguousarray(data)
        self._views.append(
            {
                "buffer": 0,
                "byteOffset": len(self._buffer),
                "byteLength": data.nbytes,
                "target": target,
            }
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

    def to_bytes(self) -> bytes:
        """Packs the glb: the JSON chunk and, when there is geometry, the binary chunk.

        Raises:
            ValueError: If a value that reaches the JSON is NaN or infinite (a NaN vertex
                gives NaN bounds), which JSON cannot hold.
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
            if self._translation is not None:
                child["translation"] = self._translation.tolist()
            gltf["nodes"].append(child)
            gltf["meshes"] = [{"primitives": self._primitives}]
            gltf["materials"] = self._materials
            gltf["buffers"] = [{"byteLength": len(self._buffer)}]
            gltf["bufferViews"] = self._views
            gltf["accessors"] = self._accessors
        json_chunk = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode()
        json_chunk += b" " * (-len(json_chunk) % 4)
        chunks = [struct.pack("<I4s", len(json_chunk), b"JSON"), json_chunk]
        if self._primitives:
            chunks += [struct.pack("<I4s", len(self._buffer), b"BIN\0"), bytes(self._buffer)]
        length = 12 + sum(len(chunk) for chunk in chunks)
        return b"".join([struct.pack("<4sII", b"glTF", 2, length), *chunks])

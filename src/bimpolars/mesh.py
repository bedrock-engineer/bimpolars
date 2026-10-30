"""Triangle meshes as the readers build them, and the ``mesh`` and ``bbox`` columns they fill.

Every reader ends with one :class:`Mesh` per row that has geometry: flat vertices, faces sorted
by style and the styles themselves. :func:`mesh_frame` turns those into the ``mesh``
(:data:`bimpolars.MESH_DTYPE`) and ``bbox`` (:data:`bimpolars.BBOX_DTYPE`) columns without
boxing every coordinate into a Python object, and :func:`mesh_arrays` reads a ``mesh`` column
back into flat arrays the same way, for the functions that walk the meshes row by row.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from bimpolars.frame import BBOX_DTYPE, MESH_DTYPE, STYLE_DTYPE

type Rgba = tuple[float, float, float, float]
"""A colour and opacity, each 0..1."""

UNSTYLED_RGBA: Rgba = (0.7, 0.7, 0.7, 1.0)
"""The colour of faces without a style: the grey ifcopenshell gives them."""


class Mesh(NamedTuple):
    """One row's triangle mesh: flat vertices, faces sorted by style, and the styles.

    Attributes:
        vertices: ``(x, y, z, x, y, z, ...)`` as float64.
        faces: Three vertex indices per triangle as uint32, in style order: the first
            ``styles[0]["face_count"]`` triangles have the first style, and so on.
        styles: Per style its ``name`` (None for unstyled faces), ``rgba`` (a list of four
            floats 0..1) and ``face_count``, as :data:`bimpolars.STYLE_DTYPE` holds them.
    """

    vertices: np.ndarray[Any, np.dtype[np.float64]]
    faces: np.ndarray[Any, np.dtype[np.uint32]]
    styles: list[dict[str, Any]]


def styled_mesh(
    vertices: np.ndarray[Any, np.dtype[np.float64]],
    faces: np.ndarray[Any, np.dtype[np.integer[Any]]],
    style_ids: np.ndarray[Any, np.dtype[np.integer[Any]]],
    styles: Sequence[tuple[str | None, Rgba]],
) -> Mesh:
    """Builds a :class:`Mesh` from faces that each carry a style.

    The faces are sorted by style id (a stable sort, so the faces of one style keep their
    order) and only the styles that have faces are listed, in id order.

    Args:
        vertices: ``(n, 3)`` or flat vertex coordinates.
        faces: ``(m, 3)`` vertex indices.
        style_ids: Per face, the index of its style in ``styles``; ``m`` entries.
        styles: The ``(name, rgba)`` styles the ids index.

    Returns:
        The mesh with its faces grouped by style.

    Raises:
        ValueError: If ``style_ids`` has not one entry per face.
    """
    faces = np.asarray(faces).reshape(-1, 3)
    style_ids = np.asarray(style_ids, dtype=np.int64)
    if len(style_ids) != len(faces):
        raise ValueError(f"{len(faces)} faces but {len(style_ids)} style ids")
    order = np.argsort(style_ids, kind="stable")
    unique_ids, counts = np.unique(style_ids, return_counts=True)
    style_rows = [
        {"name": styles[style_id][0], "rgba": list(styles[style_id][1]), "face_count": int(count)}
        for style_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True)
    ]
    return Mesh(
        np.ascontiguousarray(vertices, dtype=np.float64).ravel(),
        faces[order].astype(np.uint32).ravel(),
        style_rows,
    )


class MeshArrays(NamedTuple):
    """The vertices and faces of a ``mesh`` column as flat arrays, with per-row offsets.

    Attributes:
        vertices: All vertices, ``(n, 3)`` float64, row after row.
        faces: All faces, ``(m, 3)`` int64, row after row, with indices local to their row.
        vertex_offsets: ``len(column) + 1`` offsets into ``vertices``: row ``i`` owns
            ``vertices[vertex_offsets[i]:vertex_offsets[i + 1]]``.
        face_offsets: The same for ``faces``.
        present: The indices of the rows that have a mesh.
    """

    vertices: np.ndarray[Any, np.dtype[np.float64]]
    faces: np.ndarray[Any, np.dtype[np.int64]]
    vertex_offsets: np.ndarray[Any, np.dtype[np.int64]]
    face_offsets: np.ndarray[Any, np.dtype[np.int64]]
    present: np.ndarray[Any, np.dtype[np.int64]]

    def row(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """The ``(vertices, faces)`` of one row, as views into the flat arrays."""
        return (
            self.vertices[self.vertex_offsets[index] : self.vertex_offsets[index + 1]],
            self.faces[self.face_offsets[index] : self.face_offsets[index + 1]],
        )


def mesh_arrays(mesh: pl.Series) -> MeshArrays:
    """Reads a ``mesh`` column into flat numpy arrays without boxing a coordinate.

    Args:
        mesh: A column of :data:`bimpolars.MESH_DTYPE`, nulls allowed.

    Returns:
        The arrays and the offsets that slice them per row; see :class:`MeshArrays`.

    Raises:
        ValueError: If a mesh has null coordinates or indices.
    """
    vertices = mesh.struct.field("vertices")
    faces = mesh.struct.field("faces")
    vertex_counts = (vertices.list.len().fill_null(0) // 3).cast(pl.Int64).to_numpy()
    face_counts = (faces.list.len().fill_null(0) // 3).cast(pl.Int64).to_numpy()
    all_vertices = vertices.explode().drop_nulls().to_numpy()
    all_faces = faces.explode().drop_nulls().to_numpy()
    if len(all_vertices) != 3 * vertex_counts.sum() or len(all_faces) != 3 * face_counts.sum():
        raise ValueError("a mesh has null vertex coordinates or face indices")
    return MeshArrays(
        np.ascontiguousarray(all_vertices, dtype=np.float64).reshape(-1, 3),
        all_faces.astype(np.int64).reshape(-1, 3),
        np.concatenate([[0], np.cumsum(vertex_counts)]),
        np.concatenate([[0], np.cumsum(face_counts)]),
        np.flatnonzero(mesh.is_not_null().to_numpy()),
    )


def mesh_frame(meshes: Sequence[Mesh]) -> pl.DataFrame:
    """Builds the ``mesh`` and ``bbox`` columns of rows that have geometry.

    Args:
        meshes: One mesh per row with geometry, each with at least one vertex and one face.

    Returns:
        A frame with the columns ``mesh`` (:data:`bimpolars.MESH_DTYPE`) and ``bbox``
        (:data:`bimpolars.BBOX_DTYPE`), one row per mesh, in order. Gather it into place for
        the rows without geometry.
    """
    if not meshes:
        return pl.DataFrame(
            [
                pl.Series("mesh", [], dtype=MESH_DTYPE),
                pl.Series("bbox", [], dtype=BBOX_DTYPE),
            ]
        )
    # polars reads the numpy arrays without boxing their values; the casts turn the Array
    # dtype it infers when every mesh has the same size into the List dtype of MESH_DTYPE
    vertices = pl.Series("vertices", [mesh.vertices for mesh in meshes])
    faces = pl.Series("faces", [mesh.faces for mesh in meshes])
    low = np.array([mesh.vertices.reshape(-1, 3).min(axis=0) for mesh in meshes])
    high = np.array([mesh.vertices.reshape(-1, 3).max(axis=0) for mesh in meshes])
    return pl.DataFrame(
        [
            vertices.cast(pl.List(pl.Float64)),
            faces.cast(pl.List(pl.UInt32)),
            pl.Series("styles", [mesh.styles for mesh in meshes], dtype=pl.List(STYLE_DTYPE)),
            pl.Series("xmin", low[:, 0]),
            pl.Series("ymin", low[:, 1]),
            pl.Series("zmin", low[:, 2]),
            pl.Series("xmax", high[:, 0]),
            pl.Series("ymax", high[:, 1]),
            pl.Series("zmax", high[:, 2]),
        ]
    ).select(
        pl.struct("vertices", "faces", "styles").alias("mesh"),
        pl.struct("xmin", "ymin", "zmin", "xmax", "ymax", "zmax").alias("bbox"),
    )

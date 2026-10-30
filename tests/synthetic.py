"""Synthetic BimDataFrames for the tests: boxes as meshes in a three-level spatial structure,
and the helpers the test modules share."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import polars as pl

from bimpolars import BBOX_DTYPE, MESH_DTYPE, BimDataFrame, engineering_crs

BOX_FACES = [
    [0, 1, 2],
    [0, 2, 3],
    [4, 6, 5],
    [4, 7, 6],
    [0, 4, 5],
    [0, 5, 1],
    [1, 5, 6],
    [1, 6, 2],
    [2, 6, 7],
    [2, 7, 3],
    [3, 7, 4],
    [3, 4, 0],
]


def box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> dict[str, Any]:
    """A closed box as a mesh row: 8 vertices and 12 triangles in one style."""
    vertices = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    return mesh(vertices, BOX_FACES)


def mesh(vertices: list[list[float]], faces: list[list[int]]) -> dict[str, Any]:
    return {
        "vertices": [float(value) for vertex in vertices for value in vertex],
        "faces": [index for face in faces for index in face],
        "styles": [{"name": None, "rgba": [0.7, 0.7, 0.7, 1.0], "face_count": len(faces)}],
    }


def merged(*meshes: dict[str, Any]) -> dict[str, Any]:
    """Several meshes as one mesh row."""
    out: dict[str, Any] = {"vertices": [], "faces": [], "styles": []}
    for part in meshes:
        offset = len(out["vertices"]) // 3
        out["vertices"] += part["vertices"]
        out["faces"] += [index + offset for index in part["faces"]]
        out["styles"] += part["styles"]
    return out


def bbox(row: dict[str, Any] | None) -> dict[str, float] | None:
    if row is None:
        return None
    vertices = np.asarray(row["vertices"]).reshape(-1, 3)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return dict(zip(("xmin", "ymin", "zmin", "xmax", "ymax", "zmax"), [*low, *high], strict=True))


def model(rows: list[dict[str, Any]]) -> BimDataFrame:
    """A BimDataFrame from rows with id, class, container, elevation and mesh."""
    frame = pl.DataFrame(
        [
            pl.Series("id", [row["id"] for row in rows], dtype=pl.String),
            pl.Series("name", [row.get("name") for row in rows], dtype=pl.String),
            pl.Series("class", [row["class"] for row in rows], dtype=pl.Categorical),
            pl.Series("properties", [None] * len(rows), dtype=pl.Null),
            pl.Series("container", [row.get("container") for row in rows], dtype=pl.String),
            pl.Series("elevation", [row.get("elevation") for row in rows], dtype=pl.Float64),
            pl.Series("native_geom", [None] * len(rows), dtype=pl.Null),
            pl.Series("mesh", [row.get("mesh") for row in rows], dtype=MESH_DTYPE),
            pl.Series("bbox", [bbox(row.get("mesh")) for row in rows], dtype=BBOX_DTYPE),
        ]
    )
    return BimDataFrame(frame, engineering_crs("Synthetic", unit="millimetre"))


SPATIAL = [
    {"id": "site", "class": "IfcSite"},
    {"id": "building", "class": "IfcBuilding", "container": "site"},
    {"id": "storey", "class": "IfcBuildingStorey", "container": "building", "elevation": 3000.0},
]


def style_faces(frame: pl.DataFrame) -> pl.DataFrame:
    """Per row with a mesh: the face count and the sum of the styles' face counts."""
    mesh = pl.col("mesh").struct
    return frame.filter(pl.col("mesh").is_not_null()).select(
        "id",
        n_faces=mesh.field("faces").list.len() // 3,
        n_faces_styled=mesh.field("styles")
        .list.eval(pl.element().struct.field("face_count"))
        .list.sum(),
        n_styles=mesh.field("styles").list.len(),
    )


def run_without(module: str, code: str) -> list[str]:
    """Runs Python code in a subprocess that cannot import ``module``; the lines it prints.

    ``sys.modules[module] = None`` makes every import of the module raise ImportError, which is
    how the tests check that the core does without an optional dependency.
    """
    script = f"import sys; sys.modules[{module!r}] = None\n{code}"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.strip().splitlines()

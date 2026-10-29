"""element_meshes_glb and triangles_glb: synthetic models and the Sixty5 structural model.

The glb files are read back two ways: trimesh, an independent glTF reader that applies the node
transforms, checks that they load and gives the geometry in the scene's y-up frame; a small
reader in this module checks the file layout itself (header, chunks, accessors).
"""

from __future__ import annotations

import io
import json
import struct
from typing import Any

import numpy as np
import polars as pl
import pytest
import trimesh

from bimpolars import BimDataFrame, element_meshes_glb, engineering_crs, triangles_glb
from bimpolars.gltf import Y_UP_ROTATION
from tests.synthetic import SPATIAL, box, merged, model

GREY = [0.7, 0.7, 0.7, 1.0]
BLUE = [0.2, 0.3, 0.8, 1.0]
GLASS = [0.6, 0.8, 0.9, 0.4]


def styled(row: dict[str, Any], name: str | None, rgba: list[float]) -> dict[str, Any]:
    """A synthetic mesh with one named style."""
    return {**row, "styles": [{"name": name, "rgba": rgba, "face_count": len(row["faces"]) // 3}]}


def read_glb(glb: bytes) -> tuple[dict[str, Any], bytes]:
    """Splits a glb into its JSON and its binary chunk, checking the container layout."""
    magic, version, length = struct.unpack_from("<4sII", glb, 0)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(glb)
    assert length % 4 == 0
    json_length, json_type = struct.unpack_from("<I4s", glb, 12)
    assert json_type == b"JSON"
    gltf = json.loads(glb[20 : 20 + json_length])
    binary = b""
    if 20 + json_length < length:
        bin_length, bin_type = struct.unpack_from("<I4s", glb, 20 + json_length)
        assert bin_type == b"BIN\0"
        binary = glb[28 + json_length : 28 + json_length + bin_length]
        assert gltf["buffers"] == [{"byteLength": bin_length}]
    return gltf, binary


def accessor(gltf: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    """Decodes an accessor into a numpy array."""
    item = gltf["accessors"][index]
    view = gltf["bufferViews"][item["bufferView"]]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[item["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3}[item["type"]]
    offset = view["byteOffset"] + item.get("byteOffset", 0)
    assert offset % np.dtype(dtype).itemsize == 0
    assert view["byteLength"] >= item["count"] * width * np.dtype(dtype).itemsize
    data = np.frombuffer(binary, dtype=dtype, count=item["count"] * width, offset=offset)
    return data.reshape(item["count"], width) if width > 1 else data


def primitive_triangles(gltf: dict[str, Any], binary: bytes, primitive: dict[str, Any]) -> Any:
    """A primitive's triangles as an (n, 3, 3) array of corners in the mesh node's frame."""
    positions = accessor(gltf, binary, primitive["attributes"]["POSITION"]).astype(np.float64)
    if "indices" in primitive:
        indices = accessor(gltf, binary, primitive["indices"]).astype(np.int64)
        assert indices.max() < len(positions)
        return positions[indices.reshape(-1, 3)]
    return positions.reshape(-1, 3, 3)


def model_triangles(gltf: dict[str, Any], binary: bytes) -> dict[int, Any]:
    """Per material index, the triangles in model coordinates (metres, z up)."""
    root, child = gltf["nodes"]
    assert root["children"] == [1]
    translation = np.asarray(child["translation"])
    out = {}
    for primitive in gltf["meshes"][child["mesh"]]["primitives"]:
        assert primitive["material"] not in out
        out[primitive["material"]] = primitive_triangles(gltf, binary, primitive) + translation
    return out


def corner_set(triangles: Any) -> set[tuple[float, ...]]:
    return {tuple(np.round(t.ravel(), 6).tolist()) for t in triangles}


def mesh_corners(row: dict[str, Any], scale: float = 0.001) -> Any:
    vertices = np.asarray(row["vertices"]).reshape(-1, 3) * scale
    return vertices[np.asarray(row["faces"]).reshape(-1, 3)]


def load(glb: bytes) -> trimesh.Scene:
    scene = trimesh.load(io.BytesIO(glb), file_type="glb", process=False)
    assert isinstance(scene, trimesh.Scene)
    return scene


def y_up(xyz: Any) -> Any:
    """Model coordinates (z up) in the glTF scene frame (y up)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.stack([xyz[..., 0], xyz[..., 2], -xyz[..., 1]], axis=-1)


@pytest.fixture
def styled_model() -> BimDataFrame:
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = styled(box(0, 0, 0, 500, 500, 3000), "Concrete", GREY)
    beam = {"id": "beam", "class": "IfcBeam", "container": "storey"}
    beam["mesh"] = merged(
        styled(box(500, 0, 2500, 4500, 300, 3000), "Steel", BLUE),
        styled(box(4500, 0, 2500, 5000, 300, 3000), "Concrete", GREY),
    )
    window = {"id": "window", "class": "IfcWindow", "container": "storey"}
    window["mesh"] = styled(box(1000, 0, 0, 2000, 100, 2000), "Glass", GLASS)
    opening = {"id": "opening", "class": "IfcOpeningElement", "container": "storey"}
    opening["mesh"] = styled(box(1000, -50, 0, 2000, 150, 2000), None, GREY)
    grid = {"id": "grid", "class": "IfcGrid", "container": "storey"}
    return model([*SPATIAL, column, beam, window, opening, grid])


def test_one_primitive_and_material_per_style(styled_model: BimDataFrame) -> None:
    gltf, _ = read_glb(element_meshes_glb(styled_model))
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["scene"] == 0
    assert gltf["scenes"] == [{"nodes": [0]}]
    materials = gltf["materials"]
    assert [m["name"] for m in materials] == ["Concrete", "Glass", "Steel"]
    assert [m["pbrMetallicRoughness"]["baseColorFactor"] for m in materials] == [GREY, GLASS, BLUE]
    assert [m.get("alphaMode") for m in materials] == [None, "BLEND", None]
    assert all(m["doubleSided"] for m in materials)
    assert all(m["pbrMetallicRoughness"]["metallicFactor"] == 0.0 for m in materials)
    primitives = gltf["meshes"][0]["primitives"]
    assert [p["material"] for p in primitives] == [0, 1, 2]
    assert len(gltf["meshes"]) == 1


def test_openings_and_rows_without_a_mesh_are_left_out(styled_model: BimDataFrame) -> None:
    gltf, binary = read_glb(element_meshes_glb(styled_model))
    by_material = model_triangles(gltf, binary)
    rows = {row["id"]: row for row in styled_model.frame.iter_rows(named=True)}
    column, beam, window = (mesh_corners(rows[i]["mesh"]) for i in ("column", "beam", "window"))
    assert sum(len(t) for t in by_material.values()) == 48
    assert corner_set(by_material[0]) == corner_set(column) | corner_set(beam[12:])
    assert corner_set(by_material[1]) == corner_set(window)
    assert corner_set(by_material[2]) == corner_set(beam[:12])


def test_flat_normals_give_every_triangle_its_own_vertices(styled_model: BimDataFrame) -> None:
    gltf, binary = read_glb(element_meshes_glb(styled_model))
    for primitive in gltf["meshes"][0]["primitives"]:
        assert "indices" not in primitive
        positions = accessor(gltf, binary, primitive["attributes"]["POSITION"])
        normals = accessor(gltf, binary, primitive["attributes"]["NORMAL"])
        assert positions.shape == normals.shape
        assert len(positions) % 3 == 0
        corners = positions.reshape(-1, 3, 3).astype(np.float64)
        face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
        assert np.repeat(face_normals, 3, axis=0) == pytest.approx(normals, abs=1e-6)
        assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0, abs=1e-6)
    counts = [
        gltf["accessors"][p["attributes"]["POSITION"]]["count"]
        for p in gltf["meshes"][0]["primitives"]
    ]
    assert counts == [72, 36, 36]


def test_without_normals_the_glb_holds_the_mesh_column(styled_model: BimDataFrame) -> None:
    gltf, binary = read_glb(element_meshes_glb(styled_model, normals=False))
    primitives = gltf["meshes"][0]["primitives"]
    assert len(primitives) == 3
    assert {p["attributes"]["POSITION"] for p in primitives} == {
        primitives[0]["attributes"]["POSITION"]
    }
    assert all("NORMAL" not in p["attributes"] for p in primitives)
    positions = accessor(gltf, binary, primitives[0]["attributes"]["POSITION"])
    # column 8 + beam 16 + window 8 vertices; the opening's and the grid's are not written
    assert len(positions) == 32
    assert gltf["accessors"][primitives[0]["indices"]]["componentType"] == 5123
    assert sum(gltf["accessors"][p["indices"]]["count"] for p in primitives) == 48 * 3
    by_material = model_triangles(gltf, binary)
    rows = {row["id"]: row for row in styled_model.frame.iter_rows(named=True)}
    expected = np.concatenate([mesh_corners(rows[i]["mesh"]) for i in ("column", "beam", "window")])
    assert corner_set(np.concatenate(list(by_material.values()))) == corner_set(expected)


def test_scene_is_in_metres_and_y_up(styled_model: BimDataFrame) -> None:
    glb = element_meshes_glb(styled_model)
    gltf, _ = read_glb(glb)
    root, child = gltf["nodes"]
    assert root["rotation"] == pytest.approx(Y_UP_ROTATION)
    assert root["name"] == "Synthetic"
    assert root["extras"]["crs"] == styled_model.crs
    assert "mesh" not in root
    assert child["translation"] == pytest.approx([2.5, 0.25, 1.5])
    position = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert (np.asarray(position["min"]) >= -2.5).all()
    assert (np.asarray(position["max"]) <= 2.5).all()
    scene = load(glb)
    assert len(scene.geometry) == 3
    low, high = y_up([0, 0, 0]), y_up([5, 0.5, 3])
    assert scene.bounds[0] == pytest.approx(np.minimum(low, high), abs=1e-6)
    assert scene.bounds[1] == pytest.approx(np.maximum(low, high), abs=1e-6)
    merged_mesh = scene.to_mesh()
    assert len(merged_mesh.faces) == 48
    assert len(merged_mesh.vertices) == 48 * 3


def test_a_model_in_metres_is_not_scaled() -> None:
    slab = {"id": "slab", "class": "IfcSlab", "mesh": box(0, 0, 0, 10, 20, 0.3)}
    bdf = model([slab])
    bdf = BimDataFrame(bdf.frame, engineering_crs("Metric", unit="metre"))
    scene = load(element_meshes_glb(bdf))
    assert scene.bounds[1] - scene.bounds[0] == pytest.approx([10, 0.3, 20], abs=1e-6)


def test_a_model_far_from_the_origin_keeps_its_precision() -> None:
    far = {"id": "far", "class": "IfcSlab"}
    far["mesh"] = box(160_000_000, 384_000_000, 18_000, 160_000_500, 384_000_500, 18_125.5)
    glb = element_meshes_glb(model([far]))
    gltf, binary = read_glb(glb)
    (triangles,) = model_triangles(gltf, binary).values()
    assert triangles.min(axis=0).min(axis=0) == pytest.approx([160_000, 384_000, 18], abs=1e-6)
    assert triangles.max(axis=0).max(axis=0) == pytest.approx(
        [160_000.5, 384_000.5, 18.1255], abs=1e-6
    )


def test_a_model_without_meshes_gives_an_empty_scene() -> None:
    glb = element_meshes_glb(model(SPATIAL))
    gltf, binary = read_glb(glb)
    assert binary == b""
    assert "meshes" not in gltf
    assert "buffers" not in gltf
    assert "children" not in gltf["nodes"][0]
    scene = load(glb)
    assert scene.is_empty


def test_element_meshes_glb_checks_its_input() -> None:
    bdf = model(SPATIAL)
    with pytest.raises(ValueError, match="mesh"):
        element_meshes_glb(bdf.with_frame(bdf.frame.drop("mesh")))
    broken = {"id": "broken", "class": "IfcSlab", "mesh": box(0, 0, 0, 1, 1, 1)}
    broken["mesh"]["styles"][0]["face_count"] = 11
    with pytest.raises(ValueError, match="styles"):
        element_meshes_glb(model([broken]))


def test_a_frame_without_the_class_column_writes_every_mesh() -> None:
    bdf = model([{"id": "a", "class": "IfcOpeningElement", "mesh": box(0, 0, 0, 1, 1, 1)}])
    assert "meshes" not in read_glb(element_meshes_glb(bdf))[0]
    unclassed = bdf.with_frame(bdf.frame.drop("class"))
    assert len(read_glb(element_meshes_glb(unclassed))[0]["meshes"][0]["primitives"]) == 1


def test_triangles_glb_writes_flat_shaded_groups() -> None:
    floor = np.array([[[0, 0, 0], [10, 0, 0], [10, 10, 0]], [[0, 0, 0], [10, 10, 0], [0, 10, 0]]])
    wall = np.array([[[0, 0, 0], [0, 10, 0], [0, 10, 5]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    glb = triangles_glb({"floor": (floor, (0.5, 0.5, 0.5)), "wall": (wall, [1.0, 0.0, 0.0, 0.5])})
    gltf, binary = read_glb(glb)
    assert [m["name"] for m in gltf["materials"]] == ["floor", "wall"]
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [0.5, 0.5, 0.5, 1.0]
    assert gltf["materials"][1]["alphaMode"] == "BLEND"
    assert "extras" not in gltf["nodes"][0]
    by_material = model_triangles(gltf, binary)
    assert corner_set(by_material[0]) == corner_set(floor)
    assert corner_set(by_material[1]) == corner_set(wall)
    normals = accessor(gltf, binary, gltf["meshes"][0]["primitives"][1]["attributes"]["NORMAL"])
    assert normals[:3] == pytest.approx(np.array([[1.0, 0, 0]] * 3))
    assert normals[3:] == pytest.approx(
        np.array([[0, 0, 1.0]] * 3)
    )  # a degenerate triangle points up
    scene = load(glb)
    assert scene.bounds[1] - scene.bounds[0] == pytest.approx([10, 5, 10])
    with pytest.raises(ValueError, match="colour"):
        triangles_glb({"bad": (floor, (2.0, 0.0, 0.0))})
    with pytest.raises(ValueError, match="shape"):
        triangles_glb({"bad": (floor.reshape(-1, 3), (0.0, 0.0, 0.0))})


def elements(bdf: BimDataFrame) -> pl.DataFrame:
    """The rows element_meshes_glb writes, without the properties column."""
    return bdf.frame.drop("properties", strict=False).filter(
        pl.col("mesh").is_not_null() & (pl.col("class") != "IfcOpeningElement")
    )


def test_sixty5_structural_glb(sixty5_structural: BimDataFrame) -> None:
    rows = elements(sixty5_structural)
    struct_ = pl.col("mesh").struct
    counts = rows.select(
        vertices=(struct_.field("vertices").list.len() // 3).sum(),
        faces=(struct_.field("faces").list.len() // 3).sum(),
    ).row(0, named=True)
    assert counts == {"vertices": 89_740, "faces": 177_130}
    styles = rows.select(struct_.field("styles")).explode("styles").unnest("styles")
    assert len(styles.select("name", "rgba").unique()) == 12

    glb = element_meshes_glb(sixty5_structural)
    gltf, binary = read_glb(glb)
    assert len(gltf["materials"]) == 12
    primitives = gltf["meshes"][0]["primitives"]
    assert len(primitives) == 12
    assert (
        sum(gltf["accessors"][p["attributes"]["POSITION"]]["count"] for p in primitives)
        == 3 * 177_130
    )
    scene = load(glb)
    assert len(scene.geometry) == 12
    loaded = scene.to_mesh()
    assert len(loaded.faces) == 177_130
    assert len(loaded.vertices) == 3 * 177_130
    bbox = rows.select(
        pl.col("bbox").struct.field(f).min()
        if f.endswith("min")
        else pl.col("bbox").struct.field(f).max()
        for f in ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
    ).row(0)
    low, high = y_up(bbox[:3]) / 1000, y_up(bbox[3:]) / 1000
    assert scene.bounds[0] == pytest.approx(np.minimum(low, high), abs=1e-5)
    assert scene.bounds[1] == pytest.approx(np.maximum(low, high), abs=1e-5)

    indexed, binary = read_glb(element_meshes_glb(sixty5_structural, normals=False))
    primitives = indexed["meshes"][0]["primitives"]
    assert len(primitives) == 12
    position = indexed["accessors"][primitives[0]["attributes"]["POSITION"]]
    assert position["count"] == 89_740
    assert indexed["accessors"][primitives[0]["indices"]]["componentType"] == 5125
    assert sum(indexed["accessors"][p["indices"]]["count"] for p in primitives) == 3 * 177_130
    # The same triangles either way
    flat = model_triangles(gltf, binary=read_glb(glb)[1])
    both = model_triangles(indexed, binary)
    for material in range(12):
        assert np.sort(flat[material].reshape(-1, 9), axis=0) == pytest.approx(
            np.sort(both[material].reshape(-1, 9), axis=0), abs=1e-5
        )

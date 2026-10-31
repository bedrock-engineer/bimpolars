"""read_speckle on hand-made v3 objects: instances, nested definitions, units, materials,
the property layout and the bundle join, without the RVT data."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np
import polars as pl
import pytest

from bimpolars import MESH_DTYPE, BimDataFrame, read_speckle, version_objects
from bimpolars.crs import as_linear_unit
from bimpolars.frame import MODEL_COLUMNS
from bimpolars.mesh import UNSTYLED_RGBA
from bimpolars.speckle import LEVEL_CLASS, OBJECT_SET, _material_style, _triangles

BOX_VERTICES = [
    [0, 0, 0],
    [100, 0, 0],
    [100, 100, 0],
    [0, 100, 0],
    [0, 0, 100],
    [100, 0, 100],
    [100, 100, 100],
    [0, 100, 100],
]
BOX_FACES = (
    "3,0,1,2,3,0,2,3,3,4,6,5,3,4,7,6,3,0,4,5,3,0,5,1,3,1,5,6,3,1,6,2,3,2,6,7,3,2,7,3,3,3,7,4,"
    "3,3,4,0"
)

GREY = -4144960  # ARGB 0xFFC0C0C0


def ref(object_id: str) -> dict[str, str]:
    return {"referencedId": object_id, "speckle_type": "reference"}


def mesh(
    object_id: str,
    vertices: list[list[float]],
    faces: list[int],
    *,
    app_id: str | None = None,
    units: str = "mm",
) -> dict[str, Any]:
    return {
        "id": object_id,
        "speckle_type": "Objects.Geometry.Mesh",
        "applicationId": app_id,
        "units": units,
        "vertices": [float(value) for vertex in vertices for value in vertex],
        "faces": faces,
    }


def chunked_mesh(object_id: str, app_id: str) -> list[dict[str, Any]]:
    """A box whose vertices and faces are detached into DataChunks."""
    vertices = [float(value) for vertex in BOX_VERTICES for value in vertex]
    faces = [int(value) for value in BOX_FACES.split(",")]
    return [
        {
            "id": object_id,
            "speckle_type": "Objects.Geometry.Mesh",
            "applicationId": app_id,
            "units": "mm",
            "vertices": [ref(f"{object_id}-v0"), ref(f"{object_id}-v1")],
            "faces": [ref(f"{object_id}-f")],
        },
        {
            "id": f"{object_id}-v0",
            "speckle_type": "Speckle.Core.Models.DataChunk",
            "data": vertices[:12],
        },
        {
            "id": f"{object_id}-v1",
            "speckle_type": "Speckle.Core.Models.DataChunk",
            "data": vertices[12:],
        },
        {"id": f"{object_id}-f", "speckle_type": "Speckle.Core.Models.DataChunk", "data": faces},
    ]


def data_object(object_id: str, app_id: str, meshes: list[str], **fields: Any) -> dict[str, Any]:
    return {
        "id": object_id,
        "speckle_type": "Objects.Data.DataObject",
        "applicationId": app_id,
        "displayValue": [ref(mesh_id) for mesh_id in meshes],
        **fields,
    }


def instance(
    object_id: str,
    app_id: str,
    definition: str,
    translation: tuple[float, float, float],
    **fields: Any,
) -> dict[str, Any]:
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return {
        "id": object_id,
        "speckle_type": "Speckle.Core.Models.Instances.InstanceProxy",
        "applicationId": app_id,
        "definitionId": definition,
        "transform": matrix.ravel().tolist(),
        "units": fields.pop("units", "mm"),
        "maxDepth": 0,
        **fields,
    }


def definition(app_id: str, members: list[str]) -> dict[str, Any]:
    return {
        "id": f"id-{app_id}",
        "speckle_type": "Speckle.Core.Models.Instances.InstanceDefinitionProxy",
        "applicationId": app_id,
        "name": "test",
        "objects": members,
    }


def material_proxy(name: str, argb: int, opacity: float, targets: list[str]) -> dict[str, Any]:
    return {
        "id": f"proxy-{name}",
        "speckle_type": "Objects.Other.RenderMaterialProxy",
        "applicationId": f"mat-{name}",
        "objects": targets,
        "value": {
            "speckle_type": "Objects.Other.RenderMaterial",
            "name": name,
            "diffuse": argb,
            "opacity": opacity,
        },
    }


def model_objects() -> list[dict[str, Any]]:
    """A root with a wall, a door in a collection, placed columns, a nested definition, a bare
    mesh and a placed element with geometry of its own."""
    objects: list[dict[str, Any]] = [
        # the wall: a quad and a triangle in mm, plus a box in metres
        mesh(
            "wall-mesh",
            [[0, 0, 0], [1000, 0, 0], [1000, 0, 3000], [0, 0, 3000], [2000, 0, 0]],
            [1, 0, 1, 2, 3, 3, 1, 4, 2],
            app_id="w1",
        ),
        mesh(
            "wall-mesh-m",
            [[v / 1000 for v in vertex] for vertex in BOX_VERTICES],
            [int(v) for v in BOX_FACES.split(",")],
            app_id="w1",
            units="m",
        ),
        data_object("wall", "w1", ["wall-mesh", "wall-mesh-m"], name="Wall 1"),
        # the door, in a nested collection, with a property bag and fields of its own
        mesh("door-mesh", BOX_VERTICES, [int(v) for v in BOX_FACES.split(",")], app_id="d1"),
        data_object(
            "door",
            "d1",
            ["door-mesh"],
            name="Door 1",
            category="Doors",
            family="Single",
            properties={
                "elementId": 7.0,
                "Parameters": {"Dimensions": {"Width": 900.0}},
                "tags": ["a", "b"],
            },
            location={"speckle_type": "Objects.Geometry.Point", "x": 1.0, "y": 2.0, "units": "mm"},
        ),
        {
            "id": "coll",
            "speckle_type": "Speckle.Core.Models.Collections.Collection",
            "applicationId": "coll-1",
            "name": "Doors",
            "elements": [ref("door")],
        },
        # the column definition: a box held by a def-geo DataObject, placed twice by one element
        *chunked_mesh("def-mesh", "def-geo-0"),
        data_object("def-object", "def-geo-0", ["def-mesh"], name="geometry", properties={}),
        instance("col-a", "col-1", "def-1", (1000, 0, 0)),
        instance("col-b", "col-1-instance-1", "def-1", (5000, 0, 0)),
        # a definition that nests the column definition, placed once, with its transform in metres
        instance("nested", "nested-inst-1", "def-1", (10, 0, 0)),
        instance("i2", "i2", "def-2", (0, 0.02, 0), units="m"),
        # a mesh that is an element of its own, and an instance of an empty definition
        mesh("bare-mesh", BOX_VERTICES, [int(v) for v in BOX_FACES.split(",")], app_id="m1"),
        instance("empty", "e1", "def-empty", (0, 0, 0)),
        # an element with geometry of its own and a placement (bundle_objects' -instance-0)
        mesh("beam-mesh", BOX_VERTICES, [int(v) for v in BOX_FACES.split(",")], app_id="beam"),
        data_object("beam", "beam", ["beam-mesh"], name="Beam"),
        instance("beam-placed", "beam-instance-0", "def-1", (0, 0, 5000)),
    ]
    root = {
        "id": "root",
        "speckle_type": "Speckle.Core.Models.Collections.Collection",
        "applicationId": "artifact-root",
        "name": "Synthetic",
        "units": "mm",
        "elements": [
            ref(obj["id"])
            for obj in objects
            if obj["speckle_type"].split(".")[-1] not in ("Mesh", "DataChunk")
            or obj["id"] == "bare-mesh"
        ],
        "instanceDefinitionProxies": [
            definition("def-1", ["def-geo-0"]),
            definition("def-2", ["nested-inst-1"]),
            definition("def-empty", []),
        ],
        "renderMaterialProxies": [
            material_proxy("Oak", -8355712, 1.0, ["w1"]),
            material_proxy("Steel", GREY, 0.5, ["col-1", "beam"]),
            material_proxy("Second", -1, 1.0, ["w1"]),
        ],
        "__closure": {obj["id"]: 1 for obj in objects},
    }
    return [root, *objects]


def bundle_tables() -> dict[str, pl.DataFrame]:
    """A bundle that knows the wall, the column and the empty instance, one level and one
    material on geometry 0."""
    objects = pl.DataFrame({"object_index": [0, 1, 2], "application_id": ["w1", "col-1", "e1"]})
    paths = pl.DataFrame(
        {
            "path_index": [0, 1, 2, 3, 4, 5],
            "path": [
                "category",
                "name",
                "family",
                "properties.elementId",
                "properties.Parameters.Dimensions.Length",
                "units",
            ],
        }
    )
    eav = pl.DataFrame(
        {
            "object_index": [0, 0, 0, 1, 1, 1, 1, 2, 2],
            "path_index": [0, 3, 5, 0, 1, 2, 4, 0, 1],
            "value_string": [
                "Walls",
                None,
                "mm",
                "Columns",
                "Column 1",
                "Round",
                None,
                "Walls",
                "Empty",
            ],
            "value_double": [None, 42.0, None, None, None, None, 3000.0, None, None],
            "value_boolean": [None] * 9,
        }
    )
    type_eav = pl.DataFrame(
        {
            "type_index": [0, 0],
            "path_index": [4, 3],
            "value_string": [None, None],
            "value_double": [1.0, 99.0],
            "value_boolean": [None, None],
        }
    )
    object_type = pl.DataFrame({"object_index": [1], "type_index": [0]})
    nodes = pl.DataFrame(
        {
            "id": [7, 8],
            "kind": [5, 3],
            "name": ["Level 1", "Concrete"],
            "elevation": [3.0, None],
            "argb": [None, GREY],
            "opacity": [None, 0.75],
        }
    )
    relations = pl.DataFrame(
        {"rel": [7, 7, 5], "src": [0, 1, 0], "dst": [7, 7, 8], "ord": [0, 0, 0]}
    )
    model = pl.DataFrame(
        {
            "path": ["modelPlacement.units", "modelPlacement.source", "document.fileName"],
            "value_string": ["m", "surveyPoint", "synthetic.rvt"],
            "value_double": [None, None, None],
            "value_boolean": [None, None, None],
        }
    )
    return {
        "eav.objects": objects,
        "eav.paths": paths,
        "eav.eav": eav,
        "eav.type_eav": type_eav,
        "eav.object_type": object_type,
        "envelope.nodes": nodes,
        "envelope.relations": relations,
        "eav.model": model,
    }


@pytest.fixture(scope="module")
def bare() -> BimDataFrame:
    return read_speckle(model_objects())


@pytest.fixture(scope="module")
def bundled() -> BimDataFrame:
    return read_speckle(model_objects(), bundle_tables())


def row(bdf: BimDataFrame, row_id: str) -> dict[str, Any]:
    return bdf.frame.filter(pl.col("id") == row_id).row(0, named=True)


def face_count(mesh_row: dict[str, Any]) -> int:
    return len(mesh_row["faces"]) // 3


def test_columns_and_crs(bare: BimDataFrame) -> None:
    assert bare.frame.columns == list(MODEL_COLUMNS)
    assert bare.frame.schema["mesh"] == MESH_DTYPE
    assert bare.frame.schema["class"] == pl.Categorical
    assert bare.projjson["type"] == "EngineeringCRS"
    assert bare.projjson["name"] == "Synthetic model coordinates"
    assert bare.projjson["datum"]["name"] == "Speckle model origin"
    assert as_linear_unit(bare.projjson).name == "millimetre"


def test_one_row_per_element_and_placements_fold(bare: BimDataFrame) -> None:
    # the collection, the definition geometry and the nested instance are not rows; the
    # column's two placements and the beam's own geometry plus placement are one row each
    assert bare.frame["id"].to_list() == ["w1", "d1", "col-1", "i2", "m1", "e1", "beam"]
    assert bare.frame["class"].to_list() == [
        "Objects.Data.DataObject",
        "Doors",
        "Speckle.Core.Models.Instances.InstanceProxy",
        "Speckle.Core.Models.Instances.InstanceProxy",
        "Objects.Geometry.Mesh",
        "Speckle.Core.Models.Instances.InstanceProxy",
        "Objects.Data.DataObject",
    ]
    assert bare.frame["name"].to_list() == ["Wall 1", "Door 1", None, None, None, None, "Beam"]
    assert bare.frame["container"].null_count() == 7
    assert bare.frame["elevation"].null_count() == 7


def test_meshes_are_merged_scaled_and_triangulated(bare: BimDataFrame) -> None:
    wall = row(bare, "w1")
    # the quad fans into two triangles, the triangle stays, the box in metres adds 12 faces
    assert face_count(wall["mesh"]) == 2 + 1 + 12
    assert wall["bbox"] == {
        "xmin": 0.0,
        "ymin": 0.0,
        "zmin": 0.0,
        "xmax": 2000.0,
        "ymax": 100.0,
        "zmax": 3000.0,
    }
    vertices = np.asarray(wall["mesh"]["vertices"]).reshape(-1, 3)
    assert len(vertices) == 5 + 8
    assert vertices[5:].max() == 100.0  # the metre box, now in millimetres
    column = row(bare, "col-1")
    assert face_count(column["mesh"]) == 24
    assert column["bbox"] == {
        "xmin": 1000.0,
        "ymin": 0.0,
        "zmin": 0.0,
        "xmax": 5100.0,
        "ymax": 100.0,
        "zmax": 100.0,
    }
    faces = np.asarray(column["mesh"]["faces"]).reshape(-1, 3)
    assert faces.max() == 15  # the second placement's vertices are offset
    beam = row(bare, "beam")
    assert face_count(beam["mesh"]) == 24
    assert beam["bbox"]["zmax"] == 5100.0


def test_nested_instance_and_metre_transform(bare: BimDataFrame) -> None:
    # def-2 holds def-1 moved by 10 mm in x; i2 places it 0.02 m = 20 mm along y
    nested = row(bare, "i2")
    assert face_count(nested["mesh"]) == 12
    assert nested["bbox"] == {
        "xmin": 10.0,
        "ymin": 20.0,
        "zmin": 0.0,
        "xmax": 110.0,
        "ymax": 120.0,
        "zmax": 100.0,
    }


def test_a_mesh_can_be_an_element_and_an_empty_definition_gives_no_mesh(bare: BimDataFrame) -> None:
    assert face_count(row(bare, "m1")["mesh"]) == 12
    empty = row(bare, "e1")
    assert empty["mesh"] is None and empty["bbox"] is None


def test_styles_from_the_render_material_proxies(bare: BimDataFrame) -> None:
    wall = row(bare, "w1")["mesh"]["styles"]
    # the first proxy that targets the wall wins; its ARGB decodes to the diffuse colour
    assert [style["name"] for style in wall] == ["Oak"]
    assert wall[0]["rgba"] == pytest.approx([128 / 255, 128 / 255, 128 / 255, 1.0])
    assert wall[0]["face_count"] == 15
    column = row(bare, "col-1")["mesh"]["styles"]
    # the material targets the element; both placements, -instance-1 included, inherit it
    assert [(style["name"], style["face_count"]) for style in column] == [("Steel", 24)]
    assert column[0]["rgba"] == pytest.approx([192 / 255, 192 / 255, 192 / 255, 0.5])
    door = row(bare, "d1")["mesh"]["styles"]
    assert [(style["name"], style["rgba"]) for style in door] == [(None, list(UNSTYLED_RGBA))]
    beam = row(bare, "beam")["mesh"]["styles"]
    assert sorted((style["name"], style["face_count"]) for style in beam) == [("Steel", 24)]


def test_properties_layout(bare: BimDataFrame) -> None:
    psets = bare.frame.schema["properties"].to_schema()
    assert set(psets) == {OBJECT_SET, "properties", "properties.Parameters.Dimensions", "location"}
    assert psets["properties"].to_schema() == {"elementId": pl.Float64, "tags": pl.List(pl.String)}
    assert psets["properties.Parameters.Dimensions"].to_schema() == {"Width": pl.Float64}
    assert psets[OBJECT_SET].to_schema() == {"family": pl.String}
    assert psets["location"].to_schema() == {"x": pl.Float64, "y": pl.Float64, "units": pl.String}
    door = row(bare, "d1")["properties"]
    assert door["properties"] == {"elementId": 7.0, "tags": ["a", "b"]}
    assert door[OBJECT_SET] == {"family": "Single"}
    assert door["properties.Parameters.Dimensions"] == {"Width": 900.0}
    wall = row(bare, "w1")["properties"]
    assert all(value is None for value in wall.values())


def test_bundle_fills_class_name_properties_and_levels(bundled: BimDataFrame) -> None:
    frame = bundled.frame
    assert frame["id"].to_list() == ["level-7", "w1", "d1", "col-1", "i2", "m1", "e1", "beam"]
    level = row(bundled, "level-7")
    assert level["class"] == LEVEL_CLASS
    assert level["name"] == "Level 1"
    assert level["elevation"] == 3000.0  # the bundle is in metres, the model in millimetres
    assert level["mesh"] is None
    assert frame["container"].to_list() == [
        None,
        "level-7",
        None,
        "level-7",
        None,
        None,
        None,
        None,
    ]
    # the object's own category and name win; the bundle fills the instance's
    assert row(bundled, "w1")["class"] == "Walls"
    assert row(bundled, "d1")["class"] == "Doors"
    assert row(bundled, "col-1")["class"] == "Columns"
    assert row(bundled, "col-1")["name"] == "Column 1"
    assert row(bundled, "e1")["class"] == "Walls"
    assert row(bundled, "i2")["class"] == "Speckle.Core.Models.Instances.InstanceProxy"
    column = row(bundled, "col-1")["properties"]
    assert column[OBJECT_SET] == {"family": "Round"}
    # the instance value wins over the type's; the type adds what the instance lacks
    assert column["properties.Parameters.Dimensions"] == {"Length": 3000.0, "Width": None}
    assert column["properties"]["elementId"] == 99.0
    assert row(bundled, "w1")["properties"]["properties"]["elementId"] == 42.0
    assert bundled.projjson["name"] == "synthetic model coordinates"
    assert bundled.projjson["datum"]["name"] == "Revit survey point"
    assert (
        as_linear_unit(bundled.projjson).name == "millimetre"
    )  # the root's units win over the bundle's


def test_bundle_materials_for_definition_geometry(bundled: BimDataFrame) -> None:
    # geometry 0 (def-geo-0) carries a material in the envelope: it wins over the instance's
    column = row(bundled, "col-1")["mesh"]["styles"]
    assert [(style["name"], style["face_count"]) for style in column] == [("Concrete", 24)]
    assert column[0]["rgba"] == pytest.approx([192 / 255, 192 / 255, 192 / 255, 0.75])
    beam = row(bundled, "beam")["mesh"]["styles"]
    assert sorted((style["name"], style["face_count"]) for style in beam) == [
        ("Concrete", 12),
        ("Steel", 12),
    ]


def test_bundle_elements_without_an_object_get_rows() -> None:
    # a curtain wall the projection left out: the bundle knows it, the version does not
    tables = bundle_tables()
    tables["eav.objects"] = pl.concat(
        [tables["eav.objects"], pl.DataFrame({"object_index": [3], "application_id": ["cw1"]})]
    )
    tables["eav.eav"] = pl.concat(
        [
            tables["eav.eav"],
            pl.DataFrame(
                {
                    "object_index": [3, 3, 3],
                    "path_index": [0, 1, 2],
                    "value_string": ["Walls", "Curtain Wall 1", "Curtain Wall"],
                    "value_double": [None, None, None],
                    "value_boolean": [None, None, None],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    tables["envelope.relations"] = pl.concat(
        [
            tables["envelope.relations"],
            pl.DataFrame({"rel": [7], "src": [3], "dst": [7], "ord": [0]}),
        ]
    )
    bdf = read_speckle(model_objects(), tables)
    assert bdf.frame["id"].to_list() == [
        "level-7",
        "w1",
        "d1",
        "col-1",
        "i2",
        "m1",
        "e1",
        "beam",
        "cw1",
    ]
    wall = row(bdf, "cw1")
    assert wall["class"] == "Walls"
    assert wall["name"] == "Curtain Wall 1"
    assert wall["container"] == "level-7"
    assert wall["properties"][OBJECT_SET] == {"family": "Curtain Wall"}
    assert wall["mesh"] is None and wall["bbox"] is None
    assert read_speckle(model_objects()).frame["id"].to_list()[-1] == "beam"


def test_the_bundle_types_values_per_path() -> None:
    # a path is a number (or a boolean) only when every value of it has one (#35): Marks are
    # text, also over the number the wall's object carries; Type Mark is text across the
    # objects and their types; a name stays text
    tables = bundle_tables()
    identity = "properties.Identity Data"
    tables["eav.paths"] = pl.concat(
        [
            tables["eav.paths"],
            pl.DataFrame(
                {
                    "path_index": [6, 7, 8],
                    "path": [f"{identity}.Mark", f"{identity}.Fire Rated", f"{identity}.Type Mark"],
                }
            ),
        ]
    )
    named = (pl.col("object_index") == 2) & (pl.col("path_index") == 1)
    tables["eav.eav"] = pl.concat(
        [
            tables["eav.eav"].with_columns(
                value_string=pl.when(named).then(pl.lit("192754")).otherwise("value_string"),
                value_double=pl.when(named).then(192_754.0).otherwise("value_double"),
            ),
            pl.DataFrame(
                {
                    "object_index": [0, 1, 2, 0, 1, 0],
                    "path_index": [6, 6, 6, 7, 7, 8],
                    "value_string": ["129", "104A", "007", "true", "false", "12"],
                    "value_double": [129.0, None, 7.0, None, None, 12.0],
                    "value_boolean": [None, None, None, True, False, None],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    tables["eav.type_eav"] = pl.concat(
        [
            tables["eav.type_eav"],
            pl.DataFrame(
                {
                    "type_index": [0],
                    "path_index": [8],
                    "value_string": ["B"],
                    "value_double": [None],
                    "value_boolean": [None],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    objects = model_objects()
    wall = next(obj for obj in objects if obj["id"] == "wall")
    wall["properties"] = {"Identity Data": {"Mark": 129.0}}  # as specklepy builds it
    bdf = read_speckle(objects, tables)
    schema = bdf.frame.schema["properties"].to_schema()
    assert schema[identity].to_schema() == {
        "Mark": pl.String,
        "Fire Rated": pl.Boolean,
        "Type Mark": pl.String,
    }
    assert schema["properties"].to_schema()["elementId"] == pl.Float64
    sets = {row_id: row(bdf, row_id)["properties"][identity] for row_id in ("w1", "col-1", "e1")}
    assert sets == {
        "w1": {"Mark": "129", "Fire Rated": True, "Type Mark": "12"},
        "col-1": {"Mark": "104A", "Fire Rated": False, "Type Mark": "B"},
        "e1": {"Mark": "007", "Fire Rated": None, "Type Mark": None},
    }
    assert row(bdf, "e1")["name"] == "192754"


def test_envelope_kinds_and_relations_are_looked_up_by_name() -> None:
    # a bundle that numbers LEVEL and ON_LEVEL differently says so in its lookup tables
    tables = bundle_tables()
    tables["envelope.nodes"] = tables["envelope.nodes"].with_columns(kind=pl.Series([9, 3]))
    tables["envelope.relations"] = tables["envelope.relations"].with_columns(
        rel=pl.Series([11, 11, 5])
    )
    tables["envelope.node_kinds"] = pl.DataFrame({"kind": [9, 3], "name": ["LEVEL", "MATERIAL"]})
    tables["envelope.rel_types"] = pl.DataFrame(
        {"rel": [11, 5], "name": ["ON_LEVEL", "HAS_MATERIAL"]}
    )
    renumbered = read_speckle(model_objects(), tables)
    assert renumbered.frame["id"][0] == "level-7"
    assert row(renumbered, "w1")["container"] == "level-7"
    assert [style["name"] for style in row(renumbered, "col-1")["mesh"]["styles"]] == ["Concrete"]
    # without the lookup tables the renumbered bundle has no levels
    del tables["envelope.node_kinds"], tables["envelope.rel_types"]
    assert read_speckle(model_objects(), tables).frame["id"][0] == "w1"


def test_missing_references() -> None:
    # a DataChunk the version lacks is an error; a displayValue it lacks is skipped
    objects = [obj for obj in model_objects() if obj["id"] != "def-mesh-f"]
    with pytest.raises(ValueError, match="'def-mesh-f' is missing"):
        read_speckle(objects)
    objects = [obj for obj in model_objects() if obj["id"] != "door-mesh"]
    assert row(read_speckle(objects), "d1")["mesh"] is None


def test_speckle_type_chains_and_a_missing_type() -> None:
    objects = model_objects()
    collection = next(obj for obj in objects if obj["id"] == "coll")
    collection["speckle_type"] = "Speckle.Core.Models.Collections.Collection:Objects.Other.Layer"
    bare_mesh = next(obj for obj in objects if obj["id"] == "bare-mesh")
    del bare_mesh["speckle_type"]
    frame = read_speckle(objects).frame
    assert "coll-1" not in frame["id"].to_list()  # a chained type is still a collection
    assert frame.filter(pl.col("id") == "m1")["class"][0] is None  # not the string "None"


def test_bundle_folder_and_missing_tables(tmp_path) -> None:
    tables = bundle_tables()
    for table, frame in tables.items():
        frame.write_parquet(tmp_path / f"synthetic.{table}.parquet")
    from_folder = read_speckle(model_objects(), tmp_path)
    assert from_folder.equals(read_speckle(model_objects(), tables))
    minimal = {table: tables[table] for table in ("eav.objects", "eav.paths", "eav.eav")}
    partial = read_speckle(model_objects(), minimal)
    assert partial.frame["id"].to_list()[0] == "w1"  # no levels without the envelope
    assert row(partial, "col-1")["class"] == "Columns"
    with pytest.raises(ValueError, match=r"no eav\.objects table"):
        read_speckle(model_objects(), {"eav.paths": tables["eav.paths"]})
    with pytest.raises(ValueError, match="does not exist"):
        read_speckle(model_objects(), tmp_path / "missing")
    tables["eav.eav"].write_parquet(tmp_path / "other.eav.eav.parquet")
    with pytest.raises(ValueError, match=r"more than one eav\.eav"):
        read_speckle(model_objects(), tmp_path)


def test_input_errors() -> None:
    with pytest.raises(ValueError, match="root object must come first"):
        read_speckle([])
    with pytest.raises(ValueError, match="not a root"):
        read_speckle([{"id": "x", "speckle_type": "Objects.Data.DataObject"}])
    objects = model_objects()
    objects[0]["units"] = "furlong"
    with pytest.raises(ValueError, match="unknown Speckle unit 'furlong'"):
        read_speckle(objects)
    objects = model_objects()
    # def-loop holds an instance of itself; an element placing it would never finish
    objects[0]["instanceDefinitionProxies"].append(definition("def-loop", ["looper"]))
    objects.append(instance("looper", "looper", "def-loop", (0, 0, 0)))
    objects.append(instance("outer", "outer", "def-loop", (0, 0, 0)))
    objects[0]["elements"] += [ref("looper"), ref("outer")]
    with pytest.raises(ValueError, match="contains itself"):
        read_speckle(objects)


def test_unit_fallbacks() -> None:
    objects = model_objects()
    del objects[0]["units"]
    # without units on the root: the meshes' most common unit, or the bundle's
    assert as_linear_unit(read_speckle(objects).projjson).name == "millimetre"
    in_metres = read_speckle(objects, bundle_tables())
    assert as_linear_unit(in_metres.projjson).name == "metre"
    assert row(in_metres, "w1")["bbox"]["xmax"] == 2.0
    for obj in objects:
        obj.pop("units", None)
    assert as_linear_unit(read_speckle(objects).projjson).name == "metre"


def test_name_overrides() -> None:
    assert read_speckle(model_objects(), name="Mine").projjson["name"] == "Mine model coordinates"
    objects = model_objects()
    objects[0]["name"] = "Received model"
    assert read_speckle(objects).projjson["name"] == "Speckle model coordinates"


def test_triangles() -> None:
    assert _triangles([]).shape == (0, 3)
    assert _triangles([3, 0, 1, 2, 0, 3, 4, 5]).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert _triangles([1, 0, 1, 2, 3]).tolist() == [[0, 1, 2], [0, 2, 3]]
    assert _triangles([5, 0, 1, 2, 3, 4]).tolist() == [[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    with pytest.raises(ValueError, match="past the end"):
        _triangles([4, 0, 1, 2])


def test_material_style() -> None:
    assert _material_style({"name": "Grey", "diffuse": GREY, "opacity": 1.0}) == (
        "Grey",
        (192 / 255, 192 / 255, 192 / 255, 1.0),
    )
    assert _material_style({}) == (None, (1.0, 1.0, 1.0, 1.0))
    assert _material_style({"name": "", "diffuse": -16777216, "opacity": None}) == (
        None,
        (0.0, 0.0, 0.0, 1.0),
    )


def test_version_objects_from_frame_and_sqlite(tmp_path) -> None:
    objects = model_objects()
    frame = pl.DataFrame(
        {
            "version_id": ["v1"] * len(objects) + ["v2"],
            "object_id": [obj["id"] for obj in objects] + ["other"],
            "is_root": [index == 0 for index in range(len(objects))] + [True],
            "content": [json.dumps(obj) for obj in objects] + ["{}"],
        }
    )
    assert version_objects(frame, "v1") == objects
    # the root comes first whatever the row order; the rest keep theirs
    shuffled = frame.reverse()
    from_shuffled = version_objects(shuffled, "v1")
    assert from_shuffled == [objects[0], *objects[:0:-1]]
    parquet = tmp_path / "objects.parquet"
    shuffled.write_parquet(parquet)
    assert version_objects(parquet, "v1") == from_shuffled
    with pytest.raises(ValueError, match="not in the objects table"):
        version_objects(frame, "v3")
    with pytest.raises(ValueError, match="exactly one root"):
        version_objects(frame.with_columns(is_root=pl.lit(False)), "v1")

    db = tmp_path / "v3_objects.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE versions (version_id TEXT, v3_root_id TEXT, bundle_dir TEXT)"
        )
        connection.execute("CREATE TABLE objects (hash TEXT PRIMARY KEY, content TEXT)")
        connection.execute("INSERT INTO versions VALUES ('v1', 'root', 'bundles/v1')")
        connection.executemany(
            "INSERT INTO objects VALUES (?, ?)", [(obj["id"], json.dumps(obj)) for obj in objects]
        )
    from_db = version_objects(db, "v1")
    assert from_db[0] == objects[0]
    assert sorted(obj["id"] for obj in from_db) == sorted(obj["id"] for obj in objects)
    with sqlite3.connect(db) as connection:
        assert version_objects(connection, "v1") == from_db
        with pytest.raises(ValueError, match="not in the versions table"):
            version_objects(connection, "v9")
        connection.execute("DELETE FROM objects WHERE hash = 'door'")
    with pytest.raises(ValueError, match="1 objects of version 'v1' are missing"):
        version_objects(db, "v1")

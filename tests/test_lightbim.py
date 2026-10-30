"""LightBIM: the reduction of synthetic meshes and of the Sixty5 structural model."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import polars as pl
import pytest
import shapely

from bimpolars import (
    FEATURE_COLUMNS,
    LIGHTBIM_COLUMNS,
    BimDataFrame,
    data,
    read_parquet,
    simple_features,
)
from bimpolars.frame import MODEL_COLUMNS
from bimpolars.geoparquet import GEO_METADATA_KEY
from bimpolars.lightbim import CLASSES, KIND_DTYPE, LIGHTBIM_DTYPE, QA_DTYPE, Kind
from bimpolars.mesh import mesh_arrays
from tests.synthetic import SPATIAL, box, merged, model, rotated_box, sweep, translation

PROXY = "IfcBuildingElementProxy"


def reduce(rows: list[dict[str, Any]], **kwargs: Any) -> pl.DataFrame:
    """The LightBIM columns of the element rows, unnested, keyed by id."""
    bdf = simple_features(model([*SPATIAL, *rows]), lightbim=True, **kwargs)
    return bdf.frame.tail(len(rows)).select(
        "id",
        "lightbim",
        pl.col("lightbim_kind").struct.unnest(),
        pl.col("lightbim_qa").struct.unnest(),
    )


def geometry(frame: pl.DataFrame, row_id: str) -> Any:
    wkb = frame.filter(pl.col("id") == row_id)["lightbim"][0]
    return None if wkb is None else shapely.from_wkb(wkb)


def row(frame: pl.DataFrame, row_id: str) -> dict[str, Any]:
    return frame.filter(pl.col("id") == row_id).row(0, named=True)


def coordinates(geom: Any) -> np.ndarray:
    return shapely.get_coordinates(geom, include_z=True)


def plane_area(geom: Any) -> float:
    """The area of a flat 3D polygon in its own plane (shapely's ``area`` is the plan area)."""
    xyz = coordinates(geom)
    centre = xyz.mean(axis=0)
    axes = np.linalg.svd(xyz - centre, full_matrices=False)[2][:2]

    def flat(ring: Any) -> np.ndarray:
        return (shapely.get_coordinates(ring, include_z=True) - centre) @ axes.T

    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return sum(
        shapely.Polygon(flat(part.exterior), [flat(ring) for ring in part.interiors]).area
        for part in parts
    )


def test_columns_and_dtypes() -> None:
    column = {"id": "c", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(10, 20, 3200, 510, 820, 6000)
    bdf = simple_features(model([*SPATIAL, column]), lightbim=True)
    frame = bdf.frame
    assert frame.columns == [*MODEL_COLUMNS, *FEATURE_COLUMNS, *LIGHTBIM_COLUMNS]
    assert frame.schema["lightbim"] == LIGHTBIM_DTYPE
    assert frame.schema["lightbim_kind"] == KIND_DTYPE
    assert frame.schema["lightbim_qa"] == QA_DTYPE
    for name in LIGHTBIM_COLUMNS:
        assert frame[name].null_count() == 3  # the spatial rows have no mesh
    line = shapely.from_wkb(frame["lightbim"][3])
    assert line.geom_type == "LineString" and line.has_z
    assert coordinates(line).tolist() == [[260, 420, 3200], [260, 420, 6000]]
    assert frame["lightbim_kind"][3] == {"member": "curve", "role": "column", "load_bearing": None}
    qa = frame["lightbim_qa"][3]
    assert qa["rule"] == "class" and qa["source"] == "mesh"
    assert qa["deviation"] == pytest.approx(math.hypot(250, 400))
    twice = simple_features(bdf, lightbim=True)
    assert twice.frame.columns == frame.columns and twice.equals(bdf)


def test_the_shape_test_tries_the_plate_before_the_line() -> None:
    plate = {"id": "plate", "class": PROXY, "mesh": box(0, 0, 0, 10_000, 10_000, 300)}
    bar = {"id": "bar", "class": PROXY, "mesh": box(0, 0, 0, 300, 300, 3000)}
    block = {"id": "block", "class": PROXY, "mesh": box(0, 0, 0, 1000, 800, 600)}
    wide = {"id": "wide", "class": PROXY, "mesh": box(0, 0, 0, 2000, 300, 6000)}
    frame = reduce([plate, bar, block, wide])
    assert frame["member"].to_list() == ["surface", "curve", "mass", "surface"]
    assert frame["role"].null_count() == 4
    assert frame["rule"].to_list() == ["shape"] * 4
    assert frame["source"].to_list() == ["mesh"] * 4
    surface = geometry(frame, "plate")
    assert surface.geom_type == "Polygon" and surface.area == 1e8
    assert set(coordinates(surface)[:, 2]) == {150.0}
    assert row(frame, "plate")["deviation"] == 150
    line = geometry(frame, "bar")
    assert coordinates(line).tolist() == [[150, 150, 0], [150, 150, 3000]]
    mass = geometry(frame, "block")
    assert mass.area == 1000 * 800 and set(coordinates(mass)[:, 2]) == {300.0}
    assert row(frame, "block")["deviation"] == 300
    # 2000 wide, 300 thick and 6000 long: a plate by the plate test, before the line test
    assert set(coordinates(geometry(frame, "wide"))[:, 1]) == {150.0}


def test_the_principal_axes_find_an_inclined_brace() -> None:
    brace = {"id": "brace", "class": PROXY}
    brace["mesh"] = rotated_box(5000, 300, 300, centre=(0, 0, 5000), about_y=40)
    frame = reduce([brace])
    assert row(frame, "brace")["member"] == "curve"
    line = geometry(frame, "brace")
    ends = coordinates(line)
    direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    along = np.array([math.cos(math.radians(40)), 0, -math.sin(math.radians(40))])
    assert abs(direction @ along) == pytest.approx(1, abs=1e-9)
    assert np.linalg.norm(ends[1] - ends[0]) == pytest.approx(5000)  # .length is 2D
    assert row(frame, "brace")["deviation"] == pytest.approx(math.hypot(150, 150))
    # An upright box around the same brace is 4000 x 300 x 3400: it would pass the plate test
    upright = {"id": "upright", "class": PROXY, "mesh": box(0, 0, 0, 4000, 300, 3400)}
    assert row(reduce([upright]), "upright")["member"] == "surface"


def test_a_wall_with_openings_gets_holes_and_parts() -> None:
    window = {"id": "window", "class": "IfcWallStandardCase", "container": "storey"}
    window["mesh"] = merged(
        box(0, 0, 0, 1000, 300, 3000),
        box(2000, 0, 0, 6000, 300, 3000),
        box(1000, 0, 0, 2000, 300, 1000),
        box(1000, 0, 2000, 2000, 300, 3000),
    )
    door = {"id": "door", "class": "IfcWallStandardCase", "container": "storey"}
    door["mesh"] = merged(box(0, 0, 0, 3000, 300, 3000), box(4000, 0, 0, 6000, 300, 3000))
    frame = reduce([window, door])
    assert frame["member"].to_list() == ["surface", "surface"]
    assert frame["role"].to_list() == ["wall", "wall"]
    holed = geometry(frame, "window")
    assert holed.geom_type == "Polygon" and len(holed.interiors) == 1
    assert plane_area(holed) == pytest.approx(6000 * 3000 - 1000 * 1000)
    xyz = coordinates(holed)
    assert set(xyz[:, 1]) == {150.0}  # the midplane of a 300 thick wall along x
    assert xyz[:, 2].min() == 0 and xyz[:, 2].max() == 3000
    assert row(frame, "window")["deviation"] == 150
    parts = geometry(frame, "door")
    assert parts.geom_type == "MultiPolygon" and len(parts.geoms) == 2
    assert plane_area(parts) == pytest.approx(5000 * 3000)


def test_the_sweep_places_the_axis() -> None:
    profile = shapely.box(-250, -400, 250, 400)
    # The sweep says the profile's centroid is at x = 300, not at the mesh's centre x = 250
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(0, 0, 0, 500, 800, 3000)
    column["native_geom"] = [sweep(translation(300, 400, 0), profile, 3000)]
    longer = sweep(translation(300, 400, 0), profile, 4000, clipped=True)
    clipped = {**column, "id": "clipped", "native_geom": [longer]}
    # A beam extruded across its width: the profile is the side view, the depth the width
    sideways = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    beam = {"id": "beam", "class": "IfcBeam", "container": "storey"}
    beam["mesh"] = box(0, 0, 0, 6000, 300, 400)
    beam["native_geom"] = [sweep(sideways, shapely.box(0, 0, 6000, 400), 300)]
    frame = reduce([column, clipped, beam])
    assert frame["source"].to_list() == ["extrusion", "extrusion, cuts ignored", "extrusion"]
    assert frame["rule"].to_list() == ["class"] * 3
    line = geometry(frame, "column")
    assert coordinates(line).tolist() == [[300, 400, 0], [300, 400, 3000]]
    assert row(frame, "column")["deviation"] == pytest.approx(math.hypot(300, 400))
    assert coordinates(geometry(frame, "clipped")).tolist() == [[300, 400, 0], [300, 400, 3000]]
    assert coordinates(geometry(frame, "beam")).tolist() == [[0, 150, 200], [6000, 150, 200]]
    assert row(frame, "beam")["role"] == "beam"


def test_the_sweep_places_the_midplane() -> None:
    slab = {"id": "slab", "class": "IfcSlab", "container": "storey"}
    slab["mesh"] = box(0, 0, 0, 4000, 3000, 250)
    slab["native_geom"] = [sweep(np.eye(4), shapely.box(0, 0, 4000, 3000), 250)]
    wall = {"id": "wall", "class": "IfcWallStandardCase", "container": "storey"}
    wall["mesh"] = box(0, 0, 0, 6000, 300, 3000)
    wall["native_geom"] = [sweep(np.eye(4), shapely.box(0, 0, 6000, 300), 3000)]
    frame = reduce([slab, wall])
    assert frame["member"].to_list() == ["surface", "surface"]
    assert frame["role"].to_list() == ["slab", "wall"]
    assert frame["source"].to_list() == ["extrusion", "extrusion"]
    flat = geometry(frame, "slab")
    assert set(coordinates(flat)[:, 2]) == {125.0} and flat.area == 4000 * 3000
    upright = geometry(frame, "wall")
    assert set(coordinates(upright)[:, 1]) == {150.0}
    assert plane_area(upright) == pytest.approx(6000 * 3000)
    assert frame["deviation"].to_list() == [125, 150]


def test_supports_and_masses() -> None:
    footing = {"id": "footing", "class": "IfcFooting", "container": "storey"}
    footing["mesh"] = box(0, 0, -1000, 2000, 2000, 0)
    raft = {"id": "raft", "class": "IfcSlab", "container": "storey"}
    raft["mesh"] = box(0, 0, -800, 9000, 9000, 0)
    raft["properties"] = {"attributes": {"PredefinedType": "BASESLAB"}}
    floor = {"id": "floor", "class": "IfcSlab", "container": "storey"}
    floor["mesh"] = box(0, 0, 0, 9000, 9000, 250)
    floor["properties"] = {"attributes": {"PredefinedType": "FLOOR"}}
    block = {"id": "block", "class": PROXY, "mesh": box(0, 0, 0, 1000, 800, 600)}
    pile = {"id": "pile", "class": "IfcFooting", "mesh": box(0, 0, -12_000, 400, 400, 0)}
    frame = reduce([footing, raft, floor, block, pile])
    assert frame["member"].to_list() == ["support", "support", "surface", "mass", "curve"]
    assert frame["role"].to_list() == ["footing", "raft", "slab", None, "pile"]
    assert frame["load_bearing"].to_list() == [True, True, None, None, True]
    assert frame["rule"].to_list() == ["class", "class", "class", "shape", "shape"]
    bearing = geometry(frame, "footing")
    assert set(coordinates(bearing)[:, 2]) == {-1000.0} and bearing.area == 2000 * 2000
    assert row(frame, "footing")["deviation"] == 1000
    assert set(coordinates(geometry(frame, "raft"))[:, 2]) == {-800.0}
    assert set(coordinates(geometry(frame, "floor"))[:, 2]) == {125.0}
    assert set(coordinates(geometry(frame, "block"))[:, 2]) == {300.0}
    assert coordinates(geometry(frame, "pile")).tolist() == [[200, 200, -12_000], [200, 200, 0]]


def test_a_mass_whose_mesh_is_not_closed_sits_at_the_centre_of_its_box() -> None:
    # A box with an inside-out box on top: the signed volume is 1000 x 1000 x (1000 - 800),
    # around a centre far below the mesh (z = -3100), which falls back to the box centre
    upper = box(0, 0, 1000, 1000, 1000, 1800)
    upper["faces"] = np.asarray(upper["faces"]).reshape(-1, 3)[:, ::-1].ravel().tolist()
    planting = {
        "id": "planting",
        "class": PROXY,
        "mesh": merged(box(0, 0, 0, 1000, 1000, 1000), upper),
    }
    closed = {
        "id": "closed",
        "class": PROXY,
        "mesh": merged(box(0, 0, 0, 1000, 1000, 1000), box(0, 0, 1000, 500, 500, 1800)),
    }
    frame = reduce([planting, closed])
    assert frame["member"].to_list() == ["mass", "mass"]
    assert set(coordinates(geometry(frame, "planting"))[:, 2]) == {900.0}
    assert row(frame, "planting")["deviation"] == 900.0
    # A closed mesh keeps its centre of gravity: 1e9 mm3 at z 500 and 0.2e9 at z 1400
    assert coordinates(geometry(frame, "closed"))[:, 2] == pytest.approx(650.0)


def test_the_kind_expression_wins() -> None:
    block = {"id": "block", "class": PROXY, "mesh": box(0, 0, 0, 1000, 800, 600)}
    slab = {"id": "slab", "class": "IfcSlab", "mesh": box(0, 0, 0, 9000, 9000, 250)}
    as_brace = (
        pl.when(pl.col("class") == PROXY)
        .then(pl.struct(member=pl.lit("curve"), role=pl.lit("brace"), load_bearing=pl.lit(True)))
        .otherwise(None)
    )
    frame = reduce([block, slab], kind=as_brace)
    assert row(frame, "block")["member"] == "curve"
    assert row(frame, "block")["role"] == "brace"
    assert row(frame, "block")["load_bearing"] is True
    assert row(frame, "block")["rule"] == "kind"
    assert geometry(frame, "block").geom_type == "LineString"
    assert row(frame, "slab")["rule"] == "class"  # a null kind falls through
    # A String decides the member only; a Struct with a role only leaves the member to the table
    everything_mass = reduce([block, slab], kind=pl.lit("mass"))
    assert everything_mass["member"].to_list() == ["mass", "mass"]
    assert everything_mass["rule"].to_list() == ["kind", "kind"]
    # A member other than the table's drops the table's role: the slab is no "mass slab", and
    # as a curve its role follows the orientation of its axis
    assert everything_mass["role"].to_list() == [None, None]
    as_curve = reduce([slab], kind=pl.lit("curve"))
    assert as_curve.select("member", "role", "rule").row(0) == ("curve", "beam", "kind")
    # A support the kind gives stays one, however slender: a tall pedestal is no pile
    pedestal = {"id": "pedestal", "class": "IfcFooting", "mesh": box(0, 0, 0, 500, 500, 6000)}
    proxy = {"id": "proxy", "class": PROXY, "mesh": box(0, 0, 0, 500, 500, 6000)}
    footing = pl.struct(member=pl.lit("support"), role=pl.lit("footing"))
    pedestals = reduce([pedestal, proxy], kind=footing)
    assert pedestals.select("member", "role", "rule").rows() == [
        ("support", "footing", "kind"),
        ("support", "footing", "kind"),
    ]
    assert set(coordinates(geometry(pedestals, "pedestal"))[:, 2]) == {0.0}
    assert reduce([pedestal]).select("member", "role", "rule").row(0) == ("curve", "pile", "shape")
    plates = reduce([block, slab], kind=pl.struct(role=pl.lit("plate")))
    assert plates["member"].to_list() == ["mass", "surface"]
    assert plates["role"].to_list() == ["plate", "plate"]
    assert plates["rule"].to_list() == ["shape", "class"]
    with pytest.raises(ValueError, match="member 'pillar'"):
        reduce([block], kind=pl.lit("pillar"))
    with pytest.raises(ValueError, match="fields"):
        reduce([block], kind=pl.struct(kind=pl.lit("curve")))
    with pytest.raises(TypeError, match="Struct"):
        reduce([block], kind=pl.lit(1))
    with pytest.raises(ValueError, match="lightbim=True"):
        simple_features(model(SPATIAL), kind=pl.lit("mass"))


def test_classes_that_are_no_member_and_rows_without_a_mesh_are_null() -> None:
    assert CLASSES["IfcOpeningElement"] is None
    opening = {"id": "opening", "class": "IfcOpeningElement", "container": "storey"}
    opening["mesh"] = box(0, 0, 0, 1000, 300, 2000)
    unmeshed = {"id": "ghost", "class": "IfcColumn", "container": "storey"}
    frame = reduce([opening, unmeshed])
    assert frame["lightbim"].null_count() == 2
    assert frame["member"].null_count() == 2 and frame["rule"].null_count() == 2
    # The caller's kind can still make an opening a member
    with_kind = reduce([opening], kind=pl.lit("mass"))
    assert with_kind["member"].to_list() == ["mass"]
    assert simple_features(model(SPATIAL), lightbim=True).frame["lightbim"].null_count() == 3


def test_load_bearing_comes_from_the_properties_else_the_table() -> None:
    partition = {"id": "partition", "class": "IfcWallStandardCase", "container": "storey"}
    partition["mesh"] = box(0, 0, 0, 3000, 100, 2600)
    partition["properties"] = {"Pset_WallCommon": {"LoadBearing": False}}
    revit = {"id": "revit", "class": "Walls", "container": "storey"}
    revit["mesh"] = box(0, 0, 0, 3000, 300, 2600)
    revit["properties"] = {
        "properties.Parameters.Instance Parameters.Structural": {"Structural": "Yes"}
    }
    column = {"id": "column", "class": "Structural Columns", "container": "storey"}
    column["mesh"] = box(0, 0, 0, 400, 400, 3000)
    # An exporter's own set that disagrees: the Pset_ set wins, whatever the order of the sets
    # in the properties Struct (the order in which the rows first give them)
    archicad = {"id": "archicad", "class": "IfcWallStandardCase", "container": "storey"}
    archicad["mesh"] = box(0, 0, 0, 3000, 200, 2600)
    archicad["properties"] = {
        "ArchiCADProperties": {"LoadBearing": "No"},
        "Pset_WallCommon": {"LoadBearing": True},
    }
    frame = reduce([archicad, partition, revit, column])  # ArchiCADProperties comes first
    assert frame["load_bearing"].to_list() == [True, False, True, True]
    assert frame["member"].to_list() == ["surface", "surface", "surface", "curve"]
    assert frame["role"].to_list() == ["wall", "wall", "wall", "column"]


def test_a_role_left_open_by_the_table_follows_the_orientation() -> None:
    assert CLASSES["IfcMember"] == Kind("curve", None, None)
    flat = {"id": "flat", "class": "IfcMember", "mesh": box(0, 0, 0, 4000, 200, 200)}
    upright = {"id": "upright", "class": "IfcMember", "mesh": box(0, 0, 0, 200, 200, 4000)}
    inclined = {"id": "inclined", "class": "Structural Framing"}
    inclined["mesh"] = rotated_box(4000, 200, 200, centre=(0, 0, 0), about_y=45)
    frame = reduce([flat, upright, inclined])
    assert frame["role"].to_list() == ["beam", "column", "brace"]
    assert frame["rule"].to_list() == ["class"] * 3


def test_lightbim_is_a_geoparquet_geometry(tmp_path: Any) -> None:
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(10, 20, 3200, 510, 820, 6000)
    slab = {"id": "slab", "class": "IfcSlab", "container": "storey"}
    slab["mesh"] = box(-1000, -1000, 2800, 1000, 1000, 3200)
    bdf = simple_features(model([*SPATIAL, column, slab]), lightbim=True)
    path = tmp_path / "lightbim.parquet"
    bdf.write_parquet(path)
    geo = json.loads(pl.read_parquet_metadata(path)[GEO_METADATA_KEY])
    assert geo["primary_column"] == "footprint"
    assert list(geo["columns"]) == ["footprint", "lightbim"]
    assert geo["columns"]["lightbim"]["geometry_types"] == ["LineString Z", "Polygon Z"]
    assert geo["columns"]["lightbim"]["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert read_parquet(path).equals(bdf)


# Sixty5


@pytest.fixture(scope="module")
def sixty5_lightbim(sixty5_structural: BimDataFrame) -> BimDataFrame:
    return simple_features(sixty5_structural, lightbim=True)


def unnested(bdf: BimDataFrame) -> pl.DataFrame:
    return bdf.frame.select(
        "id",
        "name",
        "class",
        "bbox",
        "footprint",
        "lightbim",
        pl.col("lightbim_kind").struct.unnest(),
        pl.col("lightbim_qa").struct.unnest(),
    )


def test_sixty5_structural_kinds(sixty5_lightbim: BimDataFrame) -> None:
    frame = unnested(sixty5_lightbim)
    counts = frame.group_by("class", "member", "role").len().sort("class", "member", "role")
    assert {tuple(key): n for *key, n in counts.iter_rows()} == {
        ("IfcBeam", "curve", "beam"): 354,
        ("IfcBuilding", None, None): 1,
        ("IfcBuildingElementProxy", "mass", None): 1,
        ("IfcBuildingStorey", None, None): 19,
        ("IfcColumn", "curve", "column"): 387,
        ("IfcFooting", "support", "footing"): 106,
        ("IfcGrid", None, None): 19,
        ("IfcOpeningElement", None, None): 1284,
        ("IfcPile", "curve", "pile"): 146,
        ("IfcSite", None, None): 1,
        ("IfcSlab", "support", "raft"): 21,  # PredefinedType BASESLAB
        ("IfcSlab", "surface", "slab"): 49,  # PredefinedType FLOOR
        ("IfcWall", "surface", "wall"): 3,
        ("IfcWallStandardCase", "surface", "wall"): 296,
    }
    members = frame.filter(pl.col("member").is_not_null())
    assert len(members) == 2647 - 1284
    assert members["lightbim"].null_count() == 0
    # Pset_*Common.LoadBearing is True on every beam, column, wall and floor; the table says
    # it for the piles, footings and rafts; the proxy (the origin marker) is undecided
    proxy = pl.col("class") == PROXY
    assert members.filter(~proxy)["load_bearing"].all()
    assert members.filter(proxy)["load_bearing"].null_count() == 1
    assert dict(members["rule"].value_counts().iter_rows()) == {"class": 1362, "shape": 1}
    # The axis or midplane comes from the IFC sweep wherever there is one: the breps (4 beams,
    # 8 columns, a slab, 3 walls), the supports and the proxy fall back to the mesh
    sources = members.group_by("source").len()
    assert dict(sources.iter_rows()) == {
        "extrusion": 339 + 379 + 146 + 48 + 276,
        "extrusion, cuts ignored": 11 + 20,
        "mesh": 4 + 8 + 1 + 3 + 106 + 21 + 1,
    }
    geometries = shapely.from_wkb(members["lightbim"].to_numpy())
    assert shapely.has_z(geometries).all()  # is_valid is 2D: a vertical polygon has no plan area
    types = pl.Series([g.geom_type for g in geometries])
    by_member = pl.DataFrame({"member": members["member"], "type": types})
    assert {(m, t): n for m, t, n in by_member.group_by("member", "type").len().iter_rows()} == {
        ("curve", "LineString"): 887,
        ("mass", "Polygon"): 1,
        ("support", "MultiPolygon"): 1,
        ("support", "Polygon"): 126,
        ("surface", "MultiPolygon"): 15,  # walls cut in two by a full-height opening
        ("surface", "Polygon"): 333,
    }


def test_sixty5_structural_deviations_are_bounded(sixty5_lightbim: BimDataFrame) -> None:
    frame = unnested(sixty5_lightbim)
    # A curve member's deviation is the half-diagonal of its section: the deepest beam is
    # 600 x 1200 (671), the piles are round with a radius of 300
    curves = frame.filter(pl.col("member") == "curve")
    assert curves["deviation"].max() < 700
    assert curves.filter(pl.col("class") == "IfcPile")["deviation"].to_numpy() == pytest.approx(300)
    # A surface's deviation is its half-thickness: walls up to 510, floors up to 400
    surfaces = frame.filter(pl.col("member") == "surface")
    assert surfaces["deviation"].max() < 300
    assert surfaces["deviation"].min() >= 100
    # A support's deviation is its height above the bearing level
    supports = frame.filter(pl.col("member") == "support")
    box = supports["bbox"].struct.unnest()
    assert supports["deviation"].to_numpy() == pytest.approx((box["zmax"] - box["zmin"]).to_numpy())


def test_sixty5_structural_curves_lie_on_their_axis(sixty5_lightbim: BimDataFrame) -> None:
    frame = unnested(sixty5_lightbim)
    curves = frame.filter(pl.col("member") == "curve")
    lines = shapely.from_wkb(curves["lightbim"].to_numpy())
    ends = np.array([coordinates(line) for line in lines])
    box = curves["bbox"].struct.unnest().to_numpy()
    low, high = box[:, :3], box[:, 3:]
    assert (ends.min(axis=1) >= low - 1e-6).all() and (ends.max(axis=1) <= high + 1e-6).all()
    # Columns and piles are vertical, from the bottom to the top of their mesh, with the axis
    # at the centroid of the profile. The plan centre of a column's box is within 0.02 mm of
    # it; ifcopenshell tessellates the round piles 599.51 x 599.76 mm, so the centre of their
    # box is 0.24 mm off the circle's centre, where the axis is
    upright = curves["class"].is_in(["IfcColumn", "IfcPile"]).to_numpy()
    assert np.abs(ends[upright, 0, :2] - ends[upright, 1, :2]).max() < 1e-6
    assert ends[upright, :, 2].min(axis=1) == pytest.approx(low[upright, 2])
    assert ends[upright, :, 2].max(axis=1) == pytest.approx(high[upright, 2])
    offset = np.abs(ends.mean(axis=1)[:, :2] - (low[:, :2] + high[:, :2]) / 2).max(axis=1)
    assert offset[(curves["class"] == "IfcColumn").to_numpy()].max() < 0.02
    assert offset[(curves["class"] == "IfcPile").to_numpy()].max() < 0.25
    # Every beam runs along a model axis and spans its mesh along it. 89 of them are steel Z
    # sections 200 to 363 mm long and 424 mm high, extruded along their length: the axis
    # follows the extrusion, not the larger extent of the box
    beams = (curves["class"] == "IfcBeam").to_numpy()
    vectors = ends[beams, 1] - ends[beams, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    directions = np.abs(vectors / lengths[:, None])
    assert (directions.max(axis=1) > 1 - 1e-9).all()
    along = directions.argmax(axis=1)
    extents = (high[beams] - low[beams])[np.arange(len(along)), along]
    assert lengths == pytest.approx(extents)
    assert lengths.min() == pytest.approx(200) and lengths.max() == pytest.approx(21_880)
    z_sections = curves.filter(pl.col("class") == "IfcBeam")["name"].str.contains("Z_profiel")
    assert z_sections.sum() == 89
    assert lengths[z_sections.to_numpy()].max() < 424
    # A beam with a sweep lies on the centroid line of its profile: on the centre of the
    # section for the rectangles, HEAs and hollow sections, off it by the centroid's offset
    # for the angles (43 mm for an L200x200x20), the UNPs and the Z sections
    swept = sixty5_lightbim.frame.filter(
        (pl.col("class") == "IfcBeam")
        & pl.col("lightbim_qa").struct.field("source").str.starts_with("extrusion")
    )
    assert len(swept) == 350
    for wkb, sweeps in zip(swept["lightbim"], swept["native_geom"], strict=True):
        start, end = coordinates(shapely.from_wkb(wkb))
        sweep = max(sweeps, key=lambda s: shapely.from_wkb(s["profile"]).area * s["depth"])
        centroid = shapely.from_wkb(sweep["profile"]).centroid
        point = (np.array(sweep["placement"]).reshape(4, 4) @ [centroid.x, centroid.y, 0, 1])[:3]
        axis = (end - start) / np.linalg.norm(end - start)
        relative = point - start
        assert np.linalg.norm(relative - (relative @ axis) * axis) < 1e-6


def test_sixty5_structural_surfaces_and_supports(sixty5_lightbim: BimDataFrame) -> None:
    frame = unnested(sixty5_lightbim)
    floors = frame.filter(pl.col("role") == "slab")
    box = floors["bbox"].struct.unnest()
    for geom, zmin, zmax, footprint in zip(
        shapely.from_wkb(floors["lightbim"].to_numpy()),
        box["zmin"],
        box["zmax"],
        shapely.from_wkb(floors["footprint"].to_numpy()),
        strict=True,
    ):
        z = coordinates(geom)[:, 2]
        assert z.min() == pytest.approx((zmin + zmax) / 2) == pytest.approx(z.max())
        assert geom.area == pytest.approx(footprint.area, rel=1e-9)
    walls = frame.filter(pl.col("role") == "wall")
    box = walls["bbox"].struct.unnest()
    meshes = mesh_arrays(
        sixty5_lightbim.frame.filter(pl.col("lightbim_kind").struct.field("role") == "wall")["mesh"]
    )
    for i, (geom, zmin, zmax) in enumerate(
        zip(shapely.from_wkb(walls["lightbim"].to_numpy()), box["zmin"], box["zmax"], strict=True)
    ):
        xyz = coordinates(geom)
        # The plan coordinates lie on one line: the wall's midline, halfway through the wall
        centre = xyz[:, :2].mean(axis=0)
        _, singular, axes = np.linalg.svd(xyz[:, :2] - centre, full_matrices=False)
        assert singular[1] < 1e-6 * singular[0]
        depth = (meshes.row(i)[0][:, :2] - centre) @ axes[1]
        assert abs(depth.max() + depth.min()) < 1e-6
        assert xyz[:, 2].min() == pytest.approx(zmin) and xyz[:, 2].max() == pytest.approx(zmax)
    supports = frame.filter(pl.col("member") == "support")
    box = supports["bbox"].struct.unnest()
    for geom, zmin, footprint in zip(
        shapely.from_wkb(supports["lightbim"].to_numpy()),
        box["zmin"],
        shapely.from_wkb(supports["footprint"].to_numpy()),
        strict=True,
    ):
        assert set(coordinates(geom)[:, 2]) == {zmin}
        assert geom.area == pytest.approx(footprint.area, rel=1e-9)


# The Revit Advanced Sample, structure, read from Speckle: the Revit half of the class table


@pytest.fixture(scope="module")
def rvt_structure() -> BimDataFrame:
    import bimpolars

    model = data.RVT_SPECKLE_STR
    objects = bimpolars.version_objects(data.fetch(data.RVT_SPECKLE_V3_OBJECTS), model.version_id)
    bdf = bimpolars.read_speckle(objects, data.fetch_bundle(model.bundle))
    return simple_features(bdf, lightbim=True)


def test_rvt_structure_kinds(rvt_structure: BimDataFrame) -> None:
    frame = unnested(rvt_structure)
    members = frame.filter(pl.col("member").is_not_null())
    kinds = members.group_by("class", "member", "role").len()
    found = {(klass, member, role) for klass, member, role, _ in kinds.iter_rows()}
    # Columns and framing by the table, the framing's role by the angle of its axis (posts
    # and braces among the beams); foundations are footings, or piles by their shape
    assert found == {
        ("Structural Columns", "curve", "column"),
        ("Structural Framing", "curve", "beam"),
        ("Structural Framing", "curve", "brace"),
        ("Structural Framing", "curve", "column"),
        ("Structural Foundations", "support", "footing"),
        ("Structural Foundations", "curve", "pile"),
        ("Walls", "surface", "wall"),
        ("Floors", "surface", "slab"),
    }
    # Revit's Structural parameter says every one of them bears load
    assert members["load_bearing"].all()
    piles = members.filter(pl.col("role") == "pile")
    assert len(piles) > 200 and (piles["rule"] == "shape").all()
    # Reinforcement and stiffeners have a mesh but are no members
    rebar = frame.filter(pl.col("class").is_in(["Structural Rebar", "Structural Stiffeners"]))
    assert len(rebar) > 0 and rebar["lightbim"].null_count() == len(rebar)

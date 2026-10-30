"""simple_features: footprints of synthetic meshes and of the Sixty5 models."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
import shapely

from bimpolars import (
    FEATURE_COLUMNS,
    FOOTPRINT_DTYPE,
    BimDataFrame,
    data,
    engineering_crs,
    read_ifc,
    simple_features,
)
from bimpolars.features import _clamp, plan_outline
from bimpolars.frame import MODEL_COLUMNS
from tests.synthetic import SPATIAL, box, merged, mesh, model


def footprints(bdf: BimDataFrame) -> list[Any]:
    return [None if wkb is None else shapely.from_wkb(wkb) for wkb in bdf.frame["footprint"]]


def z_values(geometry: Any) -> set[float]:
    return {float(z) for z in shapely.get_coordinates(geometry, include_z=True)[:, 2]}


def test_box_footprint_lies_at_its_storey() -> None:
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(10, 20, 3200, 510, 820, 6000)
    bdf = simple_features(model([*SPATIAL, column]))
    frame = bdf.frame
    assert frame.columns == [*MODEL_COLUMNS, *FEATURE_COLUMNS]
    assert frame.schema["footprint"] == FOOTPRINT_DTYPE
    assert frame.schema["footprint_ref"] == pl.String
    assert frame["footprint_ref"].to_list() == [None, None, None, "storey"]
    assert frame["footprint"].null_count() == 3
    footprint = footprints(bdf)[3]
    assert footprint.geom_type == "Polygon"
    assert footprint.has_z
    assert footprint.area == 500 * 800
    assert footprint.bounds == (10, 20, 510, 820)
    assert z_values(footprint) == {3000.0}
    assert bdf.crs == engineering_crs("Synthetic", unit="millimetre")


def test_element_without_a_level_lies_at_its_lowest_z() -> None:
    in_building = {"id": "a", "class": "IfcSlab", "container": "building"}
    in_building["mesh"] = box(0, 0, -600, 1000, 1000, -200)
    unplaced = {"id": "b", "class": "IfcSlab", "mesh": box(0, 0, 100, 10, 10, 200)}
    bdf = simple_features(model([*SPATIAL, in_building, unplaced]))
    assert bdf.frame["footprint_ref"].to_list() == [None, None, None, "a", "b"]
    a, b = footprints(bdf)[3:]
    assert z_values(a) == {-600.0}
    assert z_values(b) == {100.0}


def test_parts_and_holes_survive() -> None:
    two = {"id": "two", "class": "IfcBeam", "container": "storey"}
    two["mesh"] = merged(box(0, 0, 0, 10, 10, 10), box(20, 0, 0, 30, 10, 10))
    ring = {"id": "ring", "class": "IfcSlab", "container": "storey"}
    ring["mesh"] = merged(
        box(0, 0, 0, 30, 10, 5),
        box(0, 20, 0, 30, 30, 5),
        box(0, 10, 0, 10, 20, 5),
        box(20, 10, 0, 30, 20, 5),
    )
    bdf = simple_features(model([*SPATIAL, two, ring]))
    multi, holed = footprints(bdf)[3:]
    assert multi.geom_type == "MultiPolygon"
    assert len(multi.geoms) == 2
    assert multi.area == 200
    assert holed.geom_type == "Polygon"
    assert len(holed.interiors) == 1
    assert holed.area == 30 * 30 - 10 * 10
    assert z_values(multi) == z_values(holed) == {3000.0}
    assert shapely.is_valid(multi) and shapely.is_valid(holed)


def test_vertical_mesh_has_an_empty_footprint() -> None:
    quad = {"id": "quad", "class": "IfcWall", "container": "storey"}
    quad["mesh"] = mesh([[0, 0, 0], [10, 0, 0], [10, 0, 5], [0, 0, 5]], [[0, 1, 2], [0, 2, 3]])
    bdf = simple_features(model([*SPATIAL, quad]))
    assert bdf.frame["footprint"].null_count() == 3
    footprint = footprints(bdf)[3]
    assert footprint.is_empty
    assert footprint.geom_type == "Polygon"
    assert footprint.has_z
    assert bdf.frame["footprint_ref"][3] == "storey"


def test_slanted_triangles_project_to_their_shadow() -> None:
    vertices = np.array([[0, 0, 0], [4, 0, 3], [0, 3, 10], [4, 3, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    outline = plan_outline(vertices, faces)
    assert outline.geom_type == "Polygon"
    assert outline.area == 12
    assert not outline.has_z
    assert plan_outline(vertices, np.zeros((0, 3), dtype=int)).is_empty


def test_sliver_triangles_are_left_out() -> None:
    # A 10 by 5 wall whose top edge leans 1e-12 out of its plane: vertical up to rounding
    vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 1e-12, 5], [0, 0, 5]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    assert plan_outline(vertices, faces).is_empty
    vertices[2, 1] = 1e-6  # a plan area of 5e-6 is above 1e-9 times the extent squared
    assert plan_outline(vertices, faces).area == pytest.approx(5e-6)


def test_clamp_keeps_an_outline_inside_its_box() -> None:
    low, high = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    inside = shapely.box(1, 1, 9, 9)
    assert _clamp(inside, low, high) is inside
    assert _clamp(shapely.Polygon(), low, high).is_empty
    outside = shapely.box(-1e-9, 2, 10 + 1e-9, 8)
    clamped = _clamp(outside, low, high)
    assert clamped.bounds == (0, 2, 10, 8)
    assert clamped.is_valid


def test_simple_features_replaces_its_columns() -> None:
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(0, 0, 0, 1, 1, 1)
    once = simple_features(model([*SPATIAL, column]))
    twice = simple_features(once)
    assert twice.frame.columns == once.frame.columns
    assert twice.equals(once)


def test_simple_features_checks_its_input() -> None:
    bdf = model(SPATIAL)
    with pytest.raises(ValueError, match="lightbim=True"):
        simple_features(bdf, kind=pl.lit("mass"))
    with pytest.raises(ValueError, match="bbox"):
        simple_features(bdf.with_frame(bdf.frame.drop("bbox")))
    duplicated = bdf.with_frame(pl.concat([bdf.frame, bdf.frame]))
    with pytest.raises(ValueError, match="unique"):
        simple_features(duplicated)


def test_a_frame_filtered_down_to_its_elements_is_refused() -> None:
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(0, 0, 0, 1, 1, 1)
    bdf = model([*SPATIAL, column])
    columns_only = bdf.with_frame(bdf.frame.filter(pl.col("class") == "IfcColumn"))
    with pytest.raises(ValueError, match="'storey'"):
        simple_features(columns_only)
    # Filtering the elements while keeping the spatial rows, or filtering afterwards, is fine
    elements = bdf.with_frame(bdf.frame.filter(pl.col("class") != "IfcOpeningElement"))
    assert simple_features(elements).frame["footprint_ref"].to_list()[-1] == "storey"
    after = simple_features(bdf).frame.filter(pl.col("class") == "IfcColumn")
    assert after["footprint_ref"].to_list() == ["storey"]


def check_footprints(bdf: BimDataFrame) -> pl.DataFrame:
    """Asserts the invariants of every footprint and returns a frame of their measures."""
    # Without properties: filtering a frame copies them, 5.5 GB for the architectural model
    frame = bdf.frame.drop("properties", strict=False)
    assert frame["footprint"].null_count() == frame["mesh"].null_count()
    assert frame.filter(pl.col("mesh").is_null())["footprint"].null_count() == len(
        frame.filter(pl.col("mesh").is_null())
    )
    assert frame["footprint_ref"].null_count() == frame["mesh"].null_count()
    with_mesh = frame.filter(pl.col("mesh").is_not_null())
    geometries = shapely.from_wkb(with_mesh["footprint"].to_numpy())
    assert shapely.is_valid(geometries).all()
    assert shapely.has_z(geometries).all()
    assert set(shapely.get_type_id(geometries)) <= {3, 6}
    bounds = shapely.bounds(geometries)
    box = with_mesh["bbox"].struct.unnest()
    inside = bounds[~shapely.is_empty(geometries)]
    boxes = box.filter(~pl.Series(shapely.is_empty(geometries)))
    assert (inside[:, 0] >= boxes["xmin"].to_numpy()).all()
    assert (inside[:, 1] >= boxes["ymin"].to_numpy()).all()
    assert (inside[:, 2] <= boxes["xmax"].to_numpy()).all()
    assert (inside[:, 3] <= boxes["ymax"].to_numpy()).all()
    levels = frame.select(container=pl.col("id"), level=pl.col("elevation"))
    refs = with_mesh.join(levels, on="container", how="left", maintain_order="left")
    expected_ref = (
        pl.when(pl.col("level").is_not_null()).then(pl.col("container")).otherwise(pl.col("id"))
    )
    expected_z = (
        pl.when(pl.col("level").is_not_null())
        .then(pl.col("level"))
        .otherwise(pl.col("bbox").struct.field("zmin"))
    )
    refs = refs.with_columns(expected_ref=expected_ref, expected_z=expected_z)
    assert refs.filter(pl.col("footprint_ref") != pl.col("expected_ref")).is_empty()
    coordinates = shapely.get_coordinates(geometries, include_z=True, return_index=True)
    z_per_row = pl.DataFrame({"row": coordinates[1], "z": coordinates[0][:, 2]}).group_by("row")
    z_range = z_per_row.agg(pl.col("z").min().alias("zmin"), pl.col("z").max().alias("zmax"))
    z_range = z_range.sort("row")
    expected = refs["expected_z"].to_numpy()[z_range["row"].to_numpy()]
    assert z_range["zmin"].to_numpy() == pytest.approx(expected)
    assert z_range["zmax"].to_numpy() == pytest.approx(expected)
    return with_mesh.select("id", "name", "class", "container", "footprint_ref").with_columns(
        geom_type=pl.Series(shapely.get_type_id(geometries)),
        area=pl.Series(shapely.area(geometries)),
        empty=pl.Series(shapely.is_empty(geometries)),
    )


def test_sixty5_structural_footprints(sixty5_structural_features: BimDataFrame) -> None:
    measures = check_footprints(sixty5_structural_features)
    assert len(measures) == 2647
    # Openings carry the mesh of their void and are outlined like any other mesh
    assert len(measures.filter(pl.col("class") == "IfcOpeningElement")) == 1284
    assert dict(measures["geom_type"].value_counts().iter_rows()) == {3: 2644, 6: 3}
    assert not measures["empty"].any()
    assert measures["area"].min() > 0
    # Every element sits in a storey with an elevation, so every footprint lies at its storey
    assert measures.filter(pl.col("footprint_ref") != pl.col("container")).is_empty()
    columns = measures.filter(pl.col("name").str.contains("kolom vierkant_gen_lxb 500x800"))
    assert len(columns) > 0
    assert columns["area"].to_numpy() == pytest.approx(500 * 800)


def test_sixty5_structural_filtered_to_its_elements(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame.drop("properties")
    elements = frame.filter(pl.col("class") != "IfcOpeningElement")
    bdf = simple_features(sixty5_structural.with_frame(elements))
    assert len(bdf.frame) == 2687 - 1284
    assert bdf.frame["footprint"].null_count() == 40
    assert bdf.frame.filter(pl.col("class") == "IfcOpeningElement").is_empty()
    columns = sixty5_structural.with_frame(frame.filter(pl.col("class") == "IfcColumn"))
    with pytest.raises(ValueError, match="keep the spatial rows"):
        simple_features(columns)


def test_sixty5_prefab_facade_footprints() -> None:
    bdf = simple_features(read_ifc(data.fetch(data.SIXTY5_PREFAB_FACADE)))
    measures = check_footprints(bdf)
    assert len(measures) == 1055
    # One Jordahl console is two parts that touch along an edge in plan, up to rounding: the
    # faces that bridged them have a plan area of 7e-10 and are slivers
    assert dict(measures["geom_type"].value_counts().iter_rows()) == {3: 1054, 6: 1}
    assert not measures["empty"].any()
    assert measures.filter(pl.col("footprint_ref") != pl.col("container")).is_empty()


@pytest.mark.slow
def test_sixty5_architectural_footprints() -> None:
    """The ArchiCAD export, with the 5.5 GB properties Struct dropped first (see CLAUDE.md)."""
    bdf = read_ifc(data.fetch(data.SIXTY5_ARCHITECTURAL))
    bdf = simple_features(bdf.with_frame(bdf.frame.drop("properties")))
    measures = check_footprints(bdf)
    assert len(measures) == 16_760
    assert dict(measures["geom_type"].value_counts().iter_rows()) == {3: 15_697, 6: 1063}
    assert measures["empty"].sum() == 3
    assert measures.filter(~pl.col("empty"))["area"].min() > 0

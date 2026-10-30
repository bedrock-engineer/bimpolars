"""read_ifc on a small IFC4 model built with the ifcopenshell API: the cases the Sixty5 models
do not cover (metres, a nameless style with transparency, a storey placed at z = 0 with its
level in Elevation, an element in no container, a space, an IfcMapConversion)."""

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.georeference
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.style
import ifcopenshell.api.unit
import numpy as np
import polars as pl
import pytest
import shapely

from bimpolars import BimDataFrame, LinearUnit, linear_unit, read_ifc
from bimpolars.ifc import _items, _profile_outline


@pytest.fixture(scope="module")
def synthetic_model() -> ifcopenshell.file:
    f = ifcopenshell.api.project.create_file(version="IFC4")
    project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="Synthetic")
    ifcopenshell.api.unit.assign_unit(f, length={"is_metric": True, "raw": "METERS"})
    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model
    )
    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingStorey", name="Level 1")
    ifcopenshell.api.aggregate.assign_object(f, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(f, relating_object=site, products=[building])
    ifcopenshell.api.aggregate.assign_object(f, relating_object=building, products=[storey])
    ifcopenshell.api.geometry.edit_object_placement(f, product=storey)
    storey.Elevation = 3.0
    space = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSpace", name="Room")
    ifcopenshell.api.aggregate.assign_object(f, relating_object=storey, products=[space])

    wall = ifcopenshell.api.root.create_entity(f, ifc_class="IfcWall", name="Wall")
    representation = ifcopenshell.api.geometry.add_wall_representation(
        f, context=body, length=5.0, height=3.0, thickness=0.2
    )
    ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=representation)
    ifcopenshell.api.geometry.edit_object_placement(f, product=wall)
    ifcopenshell.api.spatial.assign_container(f, relating_structure=storey, products=[wall])
    style = ifcopenshell.api.style.add_style(f, name=None)
    ifcopenshell.api.style.add_surface_style(
        f,
        style=style,
        ifc_class="IfcSurfaceStyleShading",
        attributes={
            "SurfaceColour": {"Name": None, "Red": 1.0, "Green": 0.0, "Blue": 0.0},
            "Transparency": 0.25,
        },
    )
    ifcopenshell.api.style.assign_representation_styles(
        f, shape_representation=representation, styles=[style]
    )
    pset = ifcopenshell.api.pset.add_pset(f, product=wall, name="Pset_WallCommon")
    ifcopenshell.api.pset.edit_pset(
        f, pset=pset, properties={"LoadBearing": True, "FireRating": "REI60"}
    )
    ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingElementProxy", name="Orphan")
    ifcopenshell.api.georeference.add_georeferencing(f)
    ifcopenshell.api.georeference.edit_georeferencing(
        f,
        projected_crs={"Name": "EPSG:28992"},
        coordinate_operation={
            "Eastings": 160071.5,
            "Northings": 384156.5,
            "OrthogonalHeight": 18.6,
        },
    )
    return f


@pytest.fixture(scope="module")
def synthetic(synthetic_model: ifcopenshell.file) -> BimDataFrame:
    with pytest.warns(UserWarning, match="IfcMapConversion"):
        return read_ifc(synthetic_model, threads=1)


def test_rows(synthetic: BimDataFrame) -> None:
    frame = synthetic.frame
    assert sorted(frame["name"]) == ["Building", "Level 1", "Orphan", "Room", "Site", "Wall"]
    by_name = {row["name"]: row for row in frame.iter_rows(named=True)}
    assert by_name["Wall"]["container"] == by_name["Level 1"]["id"]
    assert by_name["Room"]["container"] == by_name["Level 1"]["id"]
    assert by_name["Level 1"]["container"] == by_name["Building"]["id"]
    assert by_name["Orphan"]["container"] is None
    assert by_name["Site"]["container"] is None
    assert by_name["Orphan"]["mesh"] is None
    assert by_name["Wall"]["bbox"] == {
        "xmin": 0.0, "ymin": 0.0, "zmin": 0.0, "xmax": 5.0, "ymax": 0.2, "zmax": 3.0
    }  # fmt: skip


def test_storey_level_falls_back_to_the_elevation_attribute(synthetic: BimDataFrame) -> None:
    storey = synthetic.frame.filter(pl.col("class") == "IfcBuildingStorey")
    assert storey["elevation"].to_list() == [3.0]


def test_nameless_style_gets_a_fallback_name(
    synthetic: BimDataFrame, synthetic_model: ifcopenshell.file
) -> None:
    wall = synthetic.frame.filter(pl.col("name") == "Wall").row(0, named=True)
    (style,) = wall["mesh"]["styles"]
    (surface_style,) = synthetic_model.by_type("IfcSurfaceStyle")
    assert style["name"] == f"IfcSurfaceStyle-{surface_style.id()}"
    assert style["rgba"] == [1.0, 0.0, 0.0, 0.75]
    assert style["face_count"] == len(wall["mesh"]["faces"]) // 3


def test_unit_and_properties(synthetic: BimDataFrame) -> None:
    assert linear_unit(synthetic.crs) == LinearUnit("metre", 1.0)
    assert synthetic.crs["name"] == "Synthetic model coordinates"
    wall = synthetic.frame.filter(pl.col("name") == "Wall").row(0, named=True)
    assert wall["properties"]["Pset_WallCommon"] == {"LoadBearing": True, "FireRating": "REI60"}


def test_an_asymmetric_i_section_has_flanges_of_their_own() -> None:
    # IFC2X3 derives IfcAsymmetricIShapeProfileDef from IfcIShapeProfileDef, IFC4 does not and
    # renames the bottom flange: both give a 300 wide bottom flange 20 thick and a 150 wide top
    # flange 10 thick on a web of 10, 500 deep, centred on the bounding box
    ifc2x3 = ifcopenshell.file(schema="IFC2X3").create_entity(
        "IfcAsymmetricIShapeProfileDef",
        ProfileType="AREA",
        OverallWidth=300.0,
        OverallDepth=500.0,
        WebThickness=10.0,
        FlangeThickness=20.0,
        TopFlangeWidth=150.0,
        TopFlangeThickness=10.0,
    )
    ifc4 = ifcopenshell.file(schema="IFC4").create_entity(
        "IfcAsymmetricIShapeProfileDef",
        ProfileType="AREA",
        BottomFlangeWidth=300.0,
        OverallDepth=500.0,
        WebThickness=10.0,
        BottomFlangeThickness=20.0,
        TopFlangeWidth=150.0,
        TopFlangeThickness=10.0,
    )
    area = 300 * 20 + 150 * 10 + 10 * (500 - 20 - 10)
    centroid_y = (300 * 20 * -240 + 150 * 10 * 245 + 10 * 470 * 5) / area  # web -230..240
    for profile in (ifc2x3, ifc4):
        outline = _profile_outline(profile)
        assert outline.bounds == (-150, -250, 150, 250)
        assert outline.area == pytest.approx(area)
        assert outline.centroid.y == pytest.approx(centroid_y)
        assert outline.intersection(shapely.box(-200, 240, 200, 250)).area == 150 * 10
    # Without TopFlangeThickness the top flange is as thick as the bottom one
    ifc4.TopFlangeThickness = None
    assert _profile_outline(ifc4).area == pytest.approx(300 * 20 + 150 * 20 + 10 * 460)


def test_booleans_keep_what_is_part_of_the_element() -> None:
    f = ifcopenshell.file(schema="IFC4")

    def solid(name: str) -> ifcopenshell.entity_instance:
        profile = f.createIfcRectangleProfileDef("AREA", name, None, 1.0, 1.0)
        return f.create_entity("IfcExtrudedAreaSolid", SweptArea=profile, Depth=1.0)

    a, b, c = solid("a"), solid("b"), solid("c")
    cut = f.createIfcHalfSpaceSolid(f.createIfcPlane(), False)
    clipping = f.createIfcBooleanClippingResult("DIFFERENCE", b, cut)
    union = f.createIfcBooleanResult("UNION", a, clipping)
    intersection = f.createIfcCsgSolid(f.createIfcBooleanResult("INTERSECTION", c, a))
    found = [
        (item.SweptArea.ProfileName, clipped)
        for _, item, clipped in _items([union, intersection], np.eye(4))
    ]
    # A union gives both operands, a clipping or an intersection its first, flagged clipped
    assert found == [("a", False), ("b", True), ("c", True)]

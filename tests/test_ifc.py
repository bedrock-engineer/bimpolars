"""read_ifc on the Sixty5 models; the numbers come from ifcopenshell 0.9.0.post1."""

import ifcopenshell
import numpy as np
import polars as pl
import pytest

from bimpolars import BBOX_DTYPE, MESH_DTYPE, BimDataFrame, data, read_ifc, read_parquet
from bimpolars.frame import MODEL_COLUMNS
from bimpolars.ifc import DEFAULT_RGBA

SPATIAL = ["IfcSite", "IfcBuilding", "IfcBuildingStorey"]


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


def test_columns(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame
    assert frame.columns == list(MODEL_COLUMNS)
    assert frame.schema["class"] == pl.Categorical
    assert frame.schema["mesh"] == MESH_DTYPE
    assert frame.schema["bbox"] == BBOX_DTYPE
    assert frame.schema["native_geom"] == pl.Null


def test_one_row_per_product(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame
    assert len(frame) == 2687
    assert frame["id"].n_unique() == 2687
    counts = dict(frame["class"].value_counts().iter_rows())
    assert sum(counts[c] for c in SPATIAL) == 21
    assert counts["IfcBuildingStorey"] == 19
    assert counts["IfcGrid"] == 19
    assert sum(n for c, n in counts.items() if c not in SPATIAL and c != "IfcGrid") == 2647
    assert counts["IfcOpeningElement"] == 1284
    assert counts["IfcColumn"] == 387
    assert counts["IfcBeam"] == 354
    assert counts["IfcPile"] == 146


def test_rows_without_geometry_have_null_mesh_and_bbox(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame
    without = frame.filter(pl.col("mesh").is_null())
    assert set(without["class"].unique()) == {*SPATIAL, "IfcGrid"}
    assert len(without) == 40
    assert without["bbox"].null_count() == 40
    with_mesh = frame.filter(pl.col("mesh").is_not_null())
    assert len(with_mesh) == 2647
    assert with_mesh["bbox"].null_count() == 0


def test_styles_match_the_surface_styles(
    sixty5_structural: BimDataFrame, sixty5_structural_model: ifcopenshell.file
) -> None:
    styles = (
        sixty5_structural.frame.select(pl.col("mesh").struct.field("styles"))
        .drop_nulls()
        .explode("styles")
        .unnest("styles")
    )
    distinct = styles.group_by("name").agg(pl.col("rgba").n_unique().alias("colours"))
    assert len(distinct) == 15
    assert distinct["colours"].max() == 1
    in_file = {style.Name for style in sixty5_structural_model.by_type("IfcSurfaceStyle")}
    assert len(in_file) == 16
    named = set(distinct["name"].drop_nulls())
    assert len(named) == 14
    assert named <= in_file
    unstyled = styles.filter(pl.col("name").is_null())
    assert unstyled["rgba"].unique().to_list() == [list(DEFAULT_RGBA)]
    rgba = styles["rgba"].to_numpy()
    assert rgba.min() >= 0 and rgba.max() <= 1
    assert styles["face_count"].min() > 0


def test_every_face_has_exactly_one_style(sixty5_structural: BimDataFrame) -> None:
    faces = style_faces(sixty5_structural.frame)
    assert faces["n_faces"].sum() == 199_578
    assert faces.filter(pl.col("n_faces") != pl.col("n_faces_styled")).is_empty()
    assert faces["n_styles"].min() >= 1


def test_faces_are_sorted_by_style(sixty5_structural: BimDataFrame) -> None:
    row = sixty5_structural.frame.filter(pl.col("mesh").struct.field("styles").list.len() > 1).row(
        0, named=True
    )
    mesh = row["mesh"]
    faces = np.asarray(mesh["faces"]).reshape(-1, 3)
    vertices = np.asarray(mesh["vertices"]).reshape(-1, 3)
    assert faces.max() < len(vertices)
    start = 0
    for style in mesh["styles"]:
        block = faces[start : start + style["face_count"]]
        assert len(block) == style["face_count"]
        start += style["face_count"]
    assert start == len(faces)


def test_bbox_matches_the_mesh(sixty5_structural: BimDataFrame) -> None:
    rows = sixty5_structural.frame.filter(pl.col("mesh").is_not_null()).head(20)
    for row in rows.iter_rows(named=True):
        vertices = np.asarray(row["mesh"]["vertices"]).reshape(-1, 3)
        bbox = row["bbox"]
        assert [bbox["xmin"], bbox["ymin"], bbox["zmin"]] == vertices.min(axis=0).tolist()
        assert [bbox["xmax"], bbox["ymax"], bbox["zmax"]] == vertices.max(axis=0).tolist()


def test_model_extent_is_in_millimetres(sixty5_structural: BimDataFrame) -> None:
    bbox = sixty5_structural.frame.select(
        pl.col("bbox").struct.field("xmin").min(),
        pl.col("bbox").struct.field("xmax").max(),
        pl.col("bbox").struct.field("zmin").min(),
        pl.col("bbox").struct.field("zmax").max(),
    ).row(0)
    assert bbox[0] == pytest.approx(-2240.04, abs=0.01)
    assert bbox[1] == pytest.approx(33585.0)
    assert bbox[2] == pytest.approx(-30400.0)
    assert bbox[3] == pytest.approx(56640.0)


def test_container_points_at_the_spatial_structure(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame
    by_class = {c: frame.filter(pl.col("class") == c) for c in SPATIAL}
    site, building = by_class["IfcSite"], by_class["IfcBuilding"]
    storeys = by_class["IfcBuildingStorey"]
    assert site["container"].to_list() == [None]
    assert building["container"].to_list() == [site["id"][0]]
    assert storeys["container"].unique().to_list() == [building["id"][0]]
    others = frame.filter(~pl.col("class").is_in(SPATIAL))
    assert others["container"].null_count() == 0
    assert set(others["container"]) <= set(storeys["id"])


def test_elevation_on_storeys(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame
    storeys = frame.filter(pl.col("class") == "IfcBuildingStorey").sort("elevation")
    assert storeys["elevation"].null_count() == 0
    assert storeys["elevation"].head(4).to_list() == [0.0, 4500.0, 7700.0, 10900.0]
    assert frame.filter(pl.col("class") != "IfcBuildingStorey")["elevation"].null_count() == 2668


def test_properties(sixty5_structural: BimDataFrame) -> None:
    frame = sixty5_structural.frame
    dtype = frame.schema["properties"]
    assert isinstance(dtype, pl.Struct)
    psets = {field.name: field.dtype for field in dtype.fields}
    assert isinstance(psets["Pset_ColumnCommon"], pl.Struct)
    assert psets["Pset_ColumnCommon"].to_schema()["LoadBearing"] == pl.Boolean
    assert psets["BaseQuantities"].to_schema()["GrossVolume"] == pl.Float64
    # IfcLogical gives True, False or "UNKNOWN": a type that differs between rows is a String
    assert psets["Pset_BuildingStoreyCommon"].to_schema()["AboveGround"] == pl.String
    columns = frame.filter(pl.col("class") == "IfcColumn").select(
        load_bearing=pl.col("properties")
        .struct.field("Pset_ColumnCommon")
        .struct.field("LoadBearing"),
        volume=pl.col("properties").struct.field("BaseQuantities").struct.field("GrossVolume"),
    )
    assert columns["load_bearing"].all()
    assert columns["volume"].min() > 0
    site = frame.filter(pl.col("class") == "IfcSite")
    assert site["properties"].null_count() == 0
    assert all(value is None for value in site["properties"][0].values())


def test_crs_is_a_plain_engineering_crs_in_millimetres(sixty5_structural: BimDataFrame) -> None:
    crs = sixty5_structural.crs
    assert crs["type"] == "EngineeringCRS"
    assert crs["name"] == "MM16169 model coordinates"
    axes = crs["coordinate_system"]["axis"]
    assert [axis["direction"] for axis in axes] == ["east", "north", "up"]
    millimetre = {"type": "LinearUnit", "name": "millimetre", "conversion_factor": 0.001}
    assert all(axis["unit"] == millimetre for axis in axes)


def test_parquet_round_trip(sixty5_structural: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "sixty5.parquet"
    sixty5_structural.write_parquet(path)
    back = read_parquet(path)
    assert back.equals(sixty5_structural)
    assert back.crs == sixty5_structural.crs
    assert back.frame.schema == sixty5_structural.frame.schema
    assert "bimpolars" in pl.read_parquet_metadata(path)


def test_read_ifc_from_a_path_equals_read_from_the_model(
    sixty5_structural: BimDataFrame, sixty5_structural_model: ifcopenshell.file
) -> None:
    from_path = read_ifc(data.fetch(data.SIXTY5_STRUCTURAL), threads=2)
    assert from_path.equals(sixty5_structural)


def test_prefab_facade() -> None:
    bdf = read_ifc(data.fetch(data.SIXTY5_PREFAB_FACADE))
    frame = bdf.frame
    counts = dict(frame["class"].value_counts().iter_rows())
    assert counts["IfcBuildingStorey"] == 18
    assert counts["IfcBuildingElementProxy"] == 702
    assert counts["IfcWall"] == 353
    assert len(frame.filter(~pl.col("class").is_in(SPATIAL))) == 1055
    faces = style_faces(frame)
    assert faces.filter(pl.col("n_faces") != pl.col("n_faces_styled")).is_empty()
    assert bdf.crs["coordinate_system"]["axis"][0]["unit"]["name"] == "millimetre"


@pytest.mark.slow
def test_architectural() -> None:
    """The ArchiCAD export: 16,784 products and 191 per-type property sets with 35,068
    properties, so the dense properties Struct alone takes about 5.5 GB (see CLAUDE.md)."""
    bdf = read_ifc(data.fetch(data.SIXTY5_ARCHITECTURAL))
    # select the light columns first: a filter on the whole frame copies the properties
    frame = bdf.frame.select("id", "class", "mesh", "bbox")
    assert len(frame) == 16_784
    assert frame["id"].n_unique() == len(frame)
    assert len(frame.filter(pl.col("class") == "IfcBuildingStorey")) > 0
    with_mesh = frame.filter(pl.col("mesh").is_not_null())
    assert len(with_mesh) == 16_760
    assert with_mesh["bbox"].null_count() == 0
    faces = style_faces(frame)
    assert faces["n_faces"].sum() == 1_671_713
    assert faces.filter(pl.col("n_faces") != pl.col("n_faces_styled")).is_empty()
    assert len(bdf.frame.schema["properties"].to_schema()) == 191

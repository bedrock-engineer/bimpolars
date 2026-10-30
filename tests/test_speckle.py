"""read_speckle on the Revit Advanced Sample versions (arch, str, mep) with their bundles.

The numbers come from the v3 objects specklepy 2026.9.1 projected from the bundles (the
bucket's ``speckle-v3-objects.parquet`` of 2026-10-29: four versions, 44,452 objects, an
element's own geometry kept when its definition is empty) and from the bundles themselves
(rvextract, schema 1.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import shapely
import shapely.errors

from bimpolars import (
    BBOX_DTYPE,
    MESH_DTYPE,
    BimDataFrame,
    data,
    linear_unit,
    read_ifc,
    read_parquet,
    read_speckle,
    simple_features,
    version_objects,
)
from bimpolars.frame import MODEL_COLUMNS
from bimpolars.speckle import LEVEL_CLASS, OBJECT_SET
from tests.synthetic import run_without, style_faces

INSTANCE_TYPE = "Speckle.Core.Models.Instances.InstanceProxy"
IFC_GUID_SET = "properties.Parameters.Instance Parameters.IFC Parameters"
IFC_VERSION = "5cb009bb77"
"""The fourth version of the Parquet: rvt-advanced-sample-arch.ifc received by Speckle."""
MEP_STAIR = "e1d4bdfb-2688-47c1-a386-3b48116e5fd2-0008cb48"
"""The stair (Stairs - Massiv - Stufen Naturstein) whose outline needs plan_outline's retry."""


@pytest.fixture(scope="module")
def objects_path() -> Path:
    return data.fetch(data.RVT_SPECKLE_V3_OBJECTS)


@pytest.fixture(scope="module")
def arch_objects(objects_path: Path) -> list[dict]:
    return version_objects(objects_path, data.RVT_SPECKLE_ARCH.version_id)


@pytest.fixture(scope="module")
def arch(arch_objects: list[dict]) -> BimDataFrame:
    return read_speckle(arch_objects, data.fetch_bundle(data.RVT_SPECKLE_ARCH.bundle))


@pytest.fixture(scope="module")
def arch_bare(arch_objects: list[dict]) -> BimDataFrame:
    return read_speckle(arch_objects)


@pytest.fixture(scope="module")
def str_objects(objects_path: Path) -> list[dict]:
    return version_objects(objects_path, data.RVT_SPECKLE_STR.version_id)


@pytest.fixture(scope="module")
def structural(str_objects: list[dict]) -> BimDataFrame:
    return read_speckle(str_objects, data.fetch_bundle(data.RVT_SPECKLE_STR.bundle))


@pytest.fixture(scope="module")
def mep_objects(objects_path: Path) -> list[dict]:
    return version_objects(objects_path, data.RVT_SPECKLE_MEP.version_id)


@pytest.fixture(scope="module")
def mep(mep_objects: list[dict]) -> BimDataFrame:
    return read_speckle(mep_objects, data.fetch_bundle(data.RVT_SPECKLE_MEP.bundle))


def extent(frame: pl.DataFrame) -> np.ndarray:
    bbox = pl.col("bbox").struct
    return np.array(
        frame.select(
            bbox.field("xmin").min(),
            bbox.field("ymin").min(),
            bbox.field("zmin").min(),
            bbox.field("xmax").max(),
            bbox.field("ymax").max(),
            bbox.field("zmax").max(),
        ).row(0)
    )


def faces_by_row(frame: pl.DataFrame) -> dict[str, int]:
    """The face count of every row that has a mesh."""
    n_faces = pl.col("mesh").struct.field("faces").list.len() // 3
    return dict(frame.filter(pl.col("mesh").is_not_null()).select("id", n_faces).iter_rows())


def own_geometry_and_placements(objects: list[dict]) -> dict[str, list[int]]:
    """The elements that have geometry of their own and a placement, with the number of
    objects in each placement's definition."""
    root = objects[0]
    definitions = {
        proxy["applicationId"]: proxy.get("objects") or []
        for proxy in root["instanceDefinitionProxies"]
    }
    content = {member for members in definitions.values() for member in members}
    own: set[str] = set()
    placed: dict[str, list[int]] = {}
    for obj in objects[1:]:
        app_id = obj.get("applicationId")
        if app_id in content:
            continue
        base = app_id.split("-instance-")[0] if app_id else obj["id"]
        if obj.get("speckle_type") == INSTANCE_TYPE:
            placed.setdefault(base, []).append(len(definitions[obj["definitionId"]]))
        elif obj.get("displayValue"):
            own.add(base)
    return {base: sizes for base, sizes in placed.items() if base in own}


def test_version_objects(
    arch_objects: list[dict], str_objects: list[dict], mep_objects: list[dict], objects_path: Path
) -> None:
    assert len(arch_objects) == 13_194
    root = arch_objects[0]
    assert root["applicationId"] == "artifact-root"
    assert root["units"] == "mm"
    assert len(root["__closure"]) == 13_193
    assert {obj["id"] for obj in arch_objects[1:]} == set(root["__closure"])
    assert len(root["instanceDefinitionProxies"]) == 765
    assert len(str_objects) == 5355
    assert len(mep_objects) == 16_173
    # the IFC export of the arch model, received by Speckle's importer: in metres, no bundle
    ifc_objects = version_objects(objects_path, IFC_VERSION)
    assert len(ifc_objects) == 9730
    assert ifc_objects[0]["units"] == "m"
    with pytest.raises(ValueError, match="not in the objects table"):
        version_objects(objects_path, "nope")


def test_fetch_bundle_gives_the_folder_of_tables() -> None:
    folder = data.fetch_bundle(data.RVT_SPECKLE_ARCH.bundle)
    assert folder.is_dir()
    names = {path.name for path in folder.glob("*.parquet")}
    assert names == {
        f"advanced_sample_project_arch.{table}.parquet" for table in data.BUNDLE_TABLES
    }


def test_columns_and_crs(arch: BimDataFrame) -> None:
    frame = arch.frame
    assert frame.columns == list(MODEL_COLUMNS)
    assert frame.schema["class"] == pl.Categorical
    assert frame.schema["mesh"] == MESH_DTYPE
    assert frame.schema["bbox"] == BBOX_DTYPE
    assert frame.schema["native_geom"] == pl.Null
    assert arch.crs["type"] == "EngineeringCRS"
    assert arch.crs["name"] == "advanced_sample_project_arch model coordinates"
    assert arch.crs["datum"]["name"] == "Revit project base point"
    assert linear_unit(arch.crs).name == "millimetre"
    assert frame["id"].n_unique() == len(frame)


def test_arch_rows_and_instances(arch: BimDataFrame, arch_bare: BimDataFrame) -> None:
    frame = arch.frame
    # 5761 elements with an object (396 with geometry of their own, 5365 placed as instances)
    # + 111 elements the bundle alone knows + 5 levels
    assert len(frame) == 5877
    assert len(arch_bare.frame) == 5761
    assert frame["mesh"].is_not_null().sum() == 5568
    assert frame.filter(pl.col("mesh").is_null())["bbox"].null_count() == 309
    counts = dict(frame["class"].value_counts().iter_rows())
    assert counts["Curtain Wall Mullions"] == 3276
    assert counts["Curtain Panels"] == 1541
    assert counts["Walls"] == 180  # 146 with an object + the 34 curtain walls below
    assert counts["Doors"] == 101
    assert counts["Rooms"] == 91
    assert counts[LEVEL_CLASS] == 5
    assert INSTANCE_TYPE not in counts
    # the bundle's elements without an object: rows with a null mesh after the version's own
    bundle_only = frame.filter(
        ~pl.col("id").is_in(arch_bare.frame["id"].implode()) & (pl.col("class") != LEVEL_CLASS)
    )
    assert len(bundle_only) == 111
    assert bundle_only["mesh"].null_count() == 111
    assert dict(bundle_only["class"].value_counts().iter_rows()) == {
        "<Room Separation>": 76,
        "Walls": 34,
        "Structural Beam Systems": 1,
    }
    # the levels come first, then the version's elements in their order, then the bundle's
    n_levels, n_elements = 5, len(arch_bare.frame)
    assert (
        frame["id"][n_levels : n_levels + n_elements].to_list() == arch_bare.frame["id"].to_list()
    )
    # every instance gets its Revit category from the bundle; without it the class is the
    # speckle_type of the proxy
    bare_counts = dict(arch_bare.frame["class"].value_counts().iter_rows())
    assert bare_counts == {INSTANCE_TYPE: 5365, "Objects.Data.DataObject": 396}
    instances = arch_bare.frame.filter(pl.col("class") == INSTANCE_TYPE)["id"]
    with_category = frame.filter(pl.col("id").is_in(instances.implode()))
    assert len(with_category) == 5365
    assert with_category["class"].is_in([INSTANCE_TYPE]).sum() == 0
    assert frame["name"].null_count() == 0
    assert arch_bare.frame["name"].null_count() == 5365


def test_placements_fold_into_their_element(arch: BimDataFrame, arch_objects: list[dict]) -> None:
    ids = arch.frame["id"]
    assert ids.str.contains("-instance-").sum() == 0
    proxies = [obj for obj in arch_objects if obj["speckle_type"] == INSTANCE_TYPE]
    assert len(proxies) == 5817
    bases = pl.Series("base", [obj["applicationId"].split("-instance-")[0] for obj in proxies])
    multiple = bases.value_counts().filter(pl.col("count") > 1)["base"]
    assert len(multiple) == 18
    for base in multiple:
        assert (ids == base).sum() == 1
    # the stair placed 22 times, each placement a definition of its own
    stair = arch.frame.filter(pl.col("id") == "90ca1a0f-0050-420c-86e3-1efbab1aba65-00057f0b").row(
        0, named=True
    )
    assert stair["class"] == "Stairs"
    assert len(stair["mesh"]["faces"]) // 3 == 1872
    # the 267 nested instances and the 1744 definition DataObjects are not rows
    assert ids.str.starts_with("nested-inst-").sum() == 0
    assert ids.str.starts_with("def-geo-").sum() == 0


def test_own_geometry_and_placements(
    arch: BimDataFrame,
    arch_objects: list[dict],
    structural: BimDataFrame,
    str_objects: list[dict],
    mep_objects: list[dict],
) -> None:
    """specklepy's projection puts an element's geometry on the element itself when its
    definition is empty (the 841 str elements an older projection left without a mesh), and
    for eight parapet walls keeps both: the wall body on the element and its cap, a wall
    sweep, in a placed definition. Nothing is counted twice: without the placements the 841
    rows keep every face, and each parapet wall loses exactly the cap's 76."""
    both = own_geometry_and_placements(str_objects)
    assert len(both) == 841
    assert set(map(tuple, both.values())) == {(0,)}
    without = read_speckle([obj for obj in str_objects if obj["speckle_type"] != INSTANCE_TYPE])
    with_faces, without_faces = faces_by_row(structural.frame), faces_by_row(without.frame)
    assert all(with_faces[base] == without_faces[base] for base in both)
    assert sum(with_faces.values()) - sum(without_faces.values()) == 4712  # the 55 placed-only

    both = own_geometry_and_placements(mep_objects)
    assert len(both) == 10
    assert set(map(tuple, both.values())) == {(0,)}

    both = own_geometry_and_placements(arch_objects)
    assert len(both) == 75
    walls = sorted(base for base, sizes in both.items() if sizes != [0])
    assert len(walls) == 8
    assert all(both[base] == [1] for base in walls)
    rows = arch.frame.filter(pl.col("id").is_in(walls))
    assert set(rows["name"]) == {"Walls - Parapet Wall"}
    without = read_speckle(
        [
            obj
            for obj in arch_objects
            if obj["speckle_type"] != INSTANCE_TYPE
            or obj["applicationId"].split("-instance-")[0] not in walls
        ]
    )
    with_faces, without_faces = faces_by_row(arch.frame), faces_by_row(without.frame)
    assert all(with_faces[base] - without_faces[base] == 76 for base in walls)
    # the body spans z 11,400..12,000 (as in the IFC); the cap sits on top, up to 12,066
    z = pl.col("bbox").struct
    body = without.frame.filter(pl.col("id").is_in(walls))
    assert body.select(z.field("zmin").min(), z.field("zmax").max()).row(0) == pytest.approx(
        (11400.0, 12000.0), abs=0.5
    )
    assert rows.select(z.field("zmin").min(), z.field("zmax").max()).row(0) == pytest.approx(
        (11400.0, 12066.0), abs=0.5
    )


def test_meshes_and_styles(arch: BimDataFrame) -> None:
    faces = style_faces(arch.frame)
    assert faces["n_faces"].sum() == 253_971
    assert faces.filter(pl.col("n_faces") != pl.col("n_faces_styled")).is_empty()
    vertices = arch.frame.select(pl.col("mesh").struct.field("vertices").list.len().sum() // 3)
    assert vertices.item() == 376_356
    styles = (
        arch.frame.select(pl.col("mesh").struct.field("styles"))
        .drop_nulls()
        .explode("styles")
        .unnest("styles")
    )
    assert styles.select("name", "rgba").n_unique() == 42
    # the definition meshes get their material from the bundle's envelope; the rest from the
    # root's renderMaterialProxies; 77,238 faces have neither
    by_name = dict(styles.group_by("name").agg(pl.col("face_count").sum()).iter_rows())
    assert by_name[None] == 77_238
    assert by_name["Metal - Aluminium, Black-Anodized"] == 40_572
    assert by_name["Glass"] == 15_524
    glass = styles.filter(pl.col("name") == "Glass")["rgba"][0]
    assert glass[3] < 1.0  # translucent
    rgba = styles["rgba"].to_numpy()
    assert rgba.min() >= 0 and rgba.max() <= 1


def test_levels_and_containers(arch: BimDataFrame) -> None:
    frame = arch.frame
    levels = frame.filter(pl.col("class") == LEVEL_CLASS)
    assert levels["id"].to_list() == ["level-3", "level-71", "level-72", "level-84", "level-785"]
    assert levels["name"].to_list() == [
        "01 - Entry Level",
        "02 - Floor",
        "03 - Floor",
        "Roof",
        "Parapet",
    ]
    assert levels["elevation"].to_list() == pytest.approx([0.0, 3800.0, 7600.0, 11400.0, 12000.0])
    assert levels["mesh"].null_count() == 5
    assert levels["container"].null_count() == 5
    elements = frame.filter(pl.col("class") != LEVEL_CLASS)
    # three topographies, two slab edges, two masses, a beam and the bundle-only beam system
    assert elements["container"].null_count() == 9
    assert set(elements["container"].drop_nulls()) <= set(levels["id"])
    assert elements["elevation"].null_count() == len(elements)
    features = simple_features(arch)
    at_level = features.frame.filter(pl.col("footprint_ref").str.starts_with("level-"))
    assert len(at_level) == 5560
    assert features.frame["footprint"].null_count() == 309


def test_properties(arch: BimDataFrame, arch_bare: BimDataFrame) -> None:
    frame = arch.frame
    psets = frame.schema["properties"].to_schema()
    assert len(psets) == 102
    assert sum(len(pset.to_schema()) for pset in psets.values()) == 656
    assert psets[OBJECT_SET].to_schema() == {"family": pl.String, "type": pl.String}
    # the eav gives elementId as a double (130846.0), as specklepy's projection does
    assert psets["properties"].to_schema()["elementId"] == pl.Float64
    assert psets["properties"].to_schema()["builtInCategory"] == pl.String
    dimensions = psets["properties.Parameters.Instance Parameters.Dimensions"].to_schema()
    assert dimensions["Volume"] == pl.Float64
    walls = frame.filter(pl.col("class") == "Walls").select(
        family=pl.col("properties").struct.field(OBJECT_SET).struct.field("family"),
        volume=pl.col("properties")
        .struct.field("properties.Parameters.Instance Parameters.Dimensions")
        .struct.field("Volume"),
        width=pl.col("properties")
        .struct.field("properties.Parameters.Type Parameters.Construction")
        .struct.field("Width"),
    )
    assert walls["family"].null_count() == 0
    assert set(walls["family"]) == {"Basic Wall", "Curtain Wall"}
    assert walls["volume"].null_count() == 34  # the curtain walls, rows from the bundle alone
    assert walls["volume"].min() > 0
    assert walls["width"].null_count() < len(walls)  # type parameters join through the type
    # the instances carry parameters only through the bundle
    bare = arch_bare.frame.schema["properties"].to_schema()
    assert len(bare) == 50
    assert sum(len(pset.to_schema()) for pset in bare.values()) == 288
    assert frame["properties"].estimated_size() < 25_000_000  # 19.1 MB


def test_extent_agrees_with_the_ifc_export(arch: BimDataFrame) -> None:
    """Revit exports the IFC in shared coordinates, so the Speckle model (at the project base
    point) moves by the bundle's sharedCoordinates transform: a 9.2 m, 22.6 m shift."""
    model = pl.read_parquet(data.fetch(f"{data.RVT_SPECKLE_ARCH.bundle}.eav.model.parquet"))
    transform = model.filter(
        pl.col("path") == "modelPlacement.options.sharedCoordinates.transform"
    )["value_string"][0]
    matrix = np.array([float(value) for value in transform.split(",")]).reshape(4, 4)
    assert np.allclose(matrix[:3, :3], np.eye(3))
    shift = np.concatenate([matrix[:3, 3], matrix[:3, 3]])
    assert shift[:2] == pytest.approx([9201.999, 22635.061], abs=0.001)

    ifc = read_ifc(data.fetch(data.RVT_ADVANCED_SAMPLE_ARCH)).frame
    speckle = extent(arch.frame) + shift
    from_ifc = extent(ifc)
    # the plan extent and the lowest point (the topography in both) agree to the micrometre;
    # the top differs by 89 mm: a roof the IFC export has no geometry for
    assert speckle[[0, 1, 3, 4]] == pytest.approx(from_ifc[[0, 1, 3, 4]], abs=0.001)
    assert speckle[2] == pytest.approx(from_ifc[2], abs=0.001)
    assert speckle[5] == pytest.approx(from_ifc[5], abs=100)

    # element by element, through the IfcGUID Revit writes into both
    with_guid = arch.frame.select(
        "id",
        "bbox",
        guid=pl.col("properties").struct.field(IFC_GUID_SET).struct.field("IfcGUID"),
    ).filter(pl.col("bbox").is_not_null() & pl.col("guid").is_not_null())
    matched = with_guid.join(
        ifc.select(guid=pl.col("id"), ifc_bbox=pl.col("bbox")).filter(
            pl.col("ifc_bbox").is_not_null()
        ),
        on="guid",
    )
    assert len(matched) == 5412
    ours = np.array(matched.select(pl.col("bbox").struct.unnest()).rows()) + shift
    theirs = np.array(matched.select(pl.col("ifc_bbox").struct.unnest()).rows())
    deviation = np.abs(ours - theirs).max(axis=1)
    # Revit's IFC export tessellates wall ends, openings and family instances differently
    # from the bundle, by up to 322 mm: the parapet walls, whose placed cap overruns the wall
    # body at the corners (test_own_geometry_and_placements), and a few railings
    assert np.median(deviation) < 1e-6
    assert (deviation < 1).mean() > 0.85
    assert (deviation < 10).mean() > 0.97
    assert deviation.max() < 500


def test_structural(structural: BimDataFrame) -> None:
    frame = structural.frame
    # 918 elements with an object + 17 the bundle alone knows (13 beam systems, 2 trusses,
    # 2 reinforcements) + 4 levels
    assert len(frame) == 939
    assert frame.filter(pl.col("class") == LEVEL_CLASS)["elevation"].to_list() == pytest.approx(
        [0.0, 3800.0, 7600.0, 11400.0]
    )
    # every element with an object has a mesh: the 841 placed with an empty definition carry
    # their own geometry (test_own_geometry_and_placements)
    assert frame["mesh"].is_not_null().sum() == 918
    meshless = frame.filter(pl.col("mesh").is_null() & (pl.col("class") != LEVEL_CLASS))
    assert dict(meshless["class"].value_counts().iter_rows()) == {
        "Structural Beam Systems": 13,
        "Structural Trusses": 2,
        "Structural Area Reinforcement": 1,
        "Structural Path Reinforcement": 1,
    }
    assert meshless["container"].null_count() == 17  # none of them is on a level
    counts = dict(frame["class"].value_counts().iter_rows())
    assert counts["Structural Framing"] == 397
    assert counts["Structural Foundations"] == 294
    assert counts["Structural Columns"] == 203
    assert INSTANCE_TYPE not in counts
    faces = style_faces(frame)
    assert faces["n_faces"].sum() == 68_714
    assert faces.filter(pl.col("n_faces") != pl.col("n_faces_styled")).is_empty()
    styles = frame.select(pl.col("mesh").struct.field("styles")).drop_nulls().explode("styles")
    assert styles.unnest("styles").select("name", "rgba").n_unique() == 5
    assert structural.crs["name"] == "advanced_sample_project_str model coordinates"


def test_mep(mep: BimDataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = mep.frame
    # 7294 elements with an object + 1331 the bundle alone knows (1045 wires, 187 circuits,
    # 45 duct systems, 33 walls, 8 conduit runs, 6 piping systems, 7 rooms and separations)
    # + 4 levels
    assert len(frame) == 8629
    assert frame["mesh"].is_not_null().sum() == 6942
    counts = dict(frame["class"].value_counts().iter_rows())
    assert counts["Ducts"] == 724
    assert counts["Pipes"] == 491
    assert counts[LEVEL_CLASS] == 4
    assert INSTANCE_TYPE not in counts
    assert style_faces(frame)["n_faces"].sum() == 420_077
    # the model sits at Revit's Munich default: the geolocation is not used, the CRS is plain
    assert mep.crs["type"] == "EngineeringCRS"
    # six conceptual masses with geometry of their own lie up to 77 m west and 122 m south of
    # the building, whose other elements start at x -19.4 m, y -13.7 m
    assert extent(frame)[[0, 1]] == pytest.approx([-76774.748, -121732.265], abs=0.001)
    building = frame.filter(pl.col("class") != "Mass")
    assert extent(building)[[0, 1]] == pytest.approx([-19363.0, -13733.766], abs=0.001)
    # the stair MEP_STAIR (720 faces) makes GEOS's floating-point overlay fail once;
    # plan_outline falls back to the fixed-precision one
    failed: list[int] = []
    union_all = shapely.union_all

    def counting(geometries: object, *args: object, **kwargs: object) -> object:
        try:
            return union_all(geometries, *args, **kwargs)
        except shapely.errors.GEOSException:
            failed.append(len(geometries))  # type: ignore[arg-type]
            raise

    monkeypatch.setattr(shapely, "union_all", counting)
    features = simple_features(mep)
    assert failed == [234]  # the stair's triangles that are not slivers
    assert features.frame["footprint"].null_count() == frame["mesh"].null_count()
    stair = features.frame.filter(pl.col("id") == MEP_STAIR)
    assert stair["name"].to_list() == ["Stairs - Massiv - Stufen Naturstein"]
    assert len(stair["mesh"][0]["faces"]) // 3 == 720
    outline = shapely.from_wkb(stair["footprint"][0])
    assert outline.geom_type == "Polygon" and outline.is_valid and outline.area > 1e7


def test_parquet_round_trip(structural: BimDataFrame, tmp_path: Path) -> None:
    path = tmp_path / "str.parquet"
    structural.write_parquet(path)
    back = read_parquet(path)
    assert back.equals(structural)
    assert back.frame.schema == structural.frame.schema


def test_the_core_imports_without_specklepy() -> None:
    lines = run_without(
        "specklepy",
        "import bimpolars, bimpolars.data, bimpolars.speckle\n"
        "print(bimpolars.read_speckle.__module__)\n",
    )
    assert lines == ["bimpolars.speckle"]

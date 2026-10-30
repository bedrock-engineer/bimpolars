"""bimpolars.geobim: the derived Engineering CRS, set_crs and to_crs, on synthetic models and
on the Sixty5 structural model georeferenced as in the Sixty5 GeoBIM notebook."""

from __future__ import annotations

import json
import math
import warnings
from typing import Any, NamedTuple

import ifcopenshell
import ifcopenshell.util.placement
import numpy as np
import polars as pl
import pytest
import shapely

pyproj = pytest.importorskip("pyproj")

from bimpolars import (  # noqa: E402
    BBOX_DTYPE,
    MESH_DTYPE,
    NATIVE_GEOM_DTYPE,
    BimDataFrame,
    linear_unit,
    read_parquet,
    simple_features,
)
from bimpolars.frame import MODEL_COLUMNS  # noqa: E402
from bimpolars.geobim import (  # noqa: E402
    AFFINE_TOLERANCE,
    affine_transformation,
    derived_engineering_crs,
    linearized_site_to_ecef,
    set_crs,
    to_crs,
)
from bimpolars.geoparquet import GEO_METADATA_KEY, geo_metadata  # noqa: E402
from bimpolars.lightbim import QA_DTYPE  # noqa: E402
from tests.synthetic import SPATIAL, box, model, run_without, sweep, translation  # noqa: E402


class Site(NamedTuple):
    """The georeferencing of a model on EPSG:7415, as the notebooks define it."""

    name: str
    easting: float
    northing: float
    orthometric_height: float
    rotation: float
    unit: float

    def to_rd(self, points: np.ndarray) -> np.ndarray:
        """The IfcMapConversion in closed form: site units to RD New + NAP metres."""
        theta = math.radians(self.rotation)
        x, y, z = np.asarray(points, dtype=float).T
        return np.column_stack(
            [
                self.easting + self.unit * (x * math.cos(theta) - y * math.sin(theta)),
                self.northing + self.unit * (x * math.sin(theta) + y * math.cos(theta)),
                self.orthometric_height + self.unit * z,
            ]
        )

    def crs(self) -> dict[str, Any]:
        unit = {0.001: "millimetre", 1.0: "metre"}[self.unit]
        return derived_engineering_crs(
            "EPSG:7415",
            self.easting,
            self.northing,
            self.orthometric_height,
            self.rotation,
            unit=unit,
            name=self.name,
        )


# IFC-GeoReferencer export of the Sixty5 architectural model, EPSG:7415, with site z = 0 at
# the AHN4 ground level at the origin (sixty5_geobim.py): the x-axis is (0.6379617540893054,
# -0.7700680491484481) in the grid, so the rotation is atan2(ordinate, abscissa) = -50.36 deg
SIXTY5 = Site(
    "Sixty5",
    easting=160071.5,
    northing=384156.5,
    orthometric_height=18.625,
    rotation=math.degrees(math.atan2(-0.7700680491484481, 0.6379617540893054)),
    unit=0.001,
)

SITES = [
    pytest.param(SIXTY5, id="sixty5"),
    pytest.param(
        None,
        id="amsterdam-noord",
        marks=pytest.mark.skip(
            reason="TODO(bimpolars#6): the Amsterdam Noord georeferencing (easting, northing, "
            "orthometric height and rotation from amsterdam_noord_geobim.py's table) is still "
            "to come from joost; fill in a Site here"
        ),
    ),
]

SITE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [10_000.0, 0.0, 0.0],
        [0.0, 20_000.0, 0.0],
        [-9_202.0, -22_635.0, 4_100.0],
        [123_456.7, -98_765.4, 56_789.0],
    ]
)


@pytest.fixture(scope="module")
def sixty5_rd(sixty5_structural_features: BimDataFrame) -> BimDataFrame:
    """The Sixty5 structural model with footprints, moved to RD New + NAP."""
    return to_crs(set_crs(sixty5_structural_features, SIXTY5.crs()), "EPSG:7415")


def vertices(bdf: BimDataFrame) -> np.ndarray:
    return (
        bdf.frame["mesh"].struct.field("vertices").explode().drop_nulls().to_numpy().reshape(-1, 3)
    )


def footprints(bdf: BimDataFrame) -> np.ndarray:
    return shapely.from_wkb(bdf.frame["footprint"].to_numpy())


COLUMN_FOOTPRINT = shapely.Polygon(
    [(10, 20, 3000), (510, 20, 3000), (510, 820, 3000), (10, 820, 3000)]
)


def plain_column_model() -> BimDataFrame:
    """A 500 x 800 x 2800 mm column on the storey of SPATIAL, in a plain EngineeringCRS."""
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(10, 20, 3200, 510, 820, 6000)
    return model([*SPATIAL, column])


def column_model(footprint: shapely.Polygon | None = None) -> BimDataFrame:
    """The column model georeferenced as Sixty5, with the column's footprint when given."""
    bdf = set_crs(plain_column_model(), SIXTY5.crs())
    if footprint is None:
        return bdf
    wkb = shapely.to_wkb(footprint, flavor="iso")
    column = pl.Series("footprint", [None, None, None, wkb], dtype=pl.Binary)
    return bdf.with_frame(bdf.frame.with_columns(column))


def test_derived_engineering_crs_is_a_compound_of_two_derived_crss() -> None:
    crs = SIXTY5.crs()
    assert crs["type"] == "CompoundCRS"
    assert crs["name"] == "Sixty5 Engineering CRS"
    grid, height = crs["components"]
    assert grid["type"] == "DerivedProjectedCRS"
    assert grid["base_crs"]["id"] == {"authority": "EPSG", "code": 28992}
    assert grid["conversion"]["method"]["id"] == {"authority": "EPSG", "code": 9624}
    assert [p["name"] for p in grid["conversion"]["parameters"]] == [
        "A0",
        "A1",
        "A2",
        "B0",
        "B1",
        "B2",
    ]
    assert height["type"] == "DerivedVerticalCRS"
    assert height["base_crs"]["id"] == {"authority": "EPSG", "code": 5709}
    assert height["conversion"]["method"]["id"] == {"authority": "EPSG", "code": 9616}
    assert height["conversion"]["parameters"][0]["value"] == -18.625
    assert linear_unit(crs).name == "millimetre"
    json.dumps(crs)

    parsed = pyproj.CRS.from_json_dict(crs)
    assert parsed.type_name == "Compound CRS"
    assert [axis.unit_name for axis in parsed.axis_info] == ["millimetre"] * 3
    assert [axis.unit_conversion_factor for axis in parsed.axis_info] == [0.001] * 3


@pytest.mark.parametrize("site", SITES)
def test_derived_engineering_crs_reproduces_the_notebook_transform(site: Site) -> None:
    to_rd = pyproj.Transformer.from_crs(site.crs(), "EPSG:7415", always_xy=True)
    moved = np.column_stack(to_rd.transform(*SITE_POINTS.T))
    assert np.abs(moved - site.to_rd(SITE_POINTS)).max() < 1e-6
    assert moved[0] == pytest.approx(
        [site.easting, site.northing, site.orthometric_height], abs=1e-6
    )
    back = np.column_stack(
        pyproj.Transformer.from_crs("EPSG:7415", site.crs(), always_xy=True).transform(*moved.T)
    )
    assert np.abs(back - SITE_POINTS).max() < 1e-6 / site.unit


def test_derived_engineering_crs_checks_its_arguments() -> None:
    with pytest.raises(ValueError, match="compound"):
        derived_engineering_crs("EPSG:28992", 0, 0, 0, 0, unit="metre")
    with pytest.raises(ValueError, match="compound"):
        derived_engineering_crs("EPSG:4937", 0, 0, 0, 0, unit="metre")
    with pytest.raises(ValueError, match="unknown unit"):
        derived_engineering_crs("EPSG:7415", 0, 0, 0, 0, unit="furlong")
    with pytest.raises(ValueError, match="cannot read"):
        derived_engineering_crs("not a crs", 0, 0, 0, 0, unit="metre")


def test_set_crs_labels_without_moving() -> None:
    bdf = plain_column_model()
    labelled = set_crs(bdf, SIXTY5.crs())
    assert labelled.frame is bdf.frame
    assert labelled.crs == SIXTY5.crs()
    assert bdf.crs["type"] == "EngineeringCRS"

    from_epsg = set_crs(bdf, "EPSG:7415")
    assert from_epsg.crs["id"] == {"authority": "EPSG", "code": 7415}
    assert set_crs(bdf, pyproj.CRS("EPSG:7415")).crs == from_epsg.crs
    with pytest.raises(ValueError, match="three"):
        set_crs(bdf, "EPSG:28992")
    with pytest.raises(ValueError, match="cannot read"):
        set_crs(bdf, "not a crs")


def test_to_crs_moves_the_geometry_columns_together() -> None:
    bdf = column_model(COLUMN_FOOTPRINT)
    moved = to_crs(bdf, "EPSG:7415")
    frame = moved.frame

    assert frame.columns == [*MODEL_COLUMNS, "footprint"]
    assert frame.schema["mesh"] == MESH_DTYPE
    assert frame.schema["bbox"] == BBOX_DTYPE
    assert frame.schema["native_geom"] == pl.Null
    assert frame["mesh"].null_count() == 3
    assert frame["bbox"].null_count() == 3
    assert frame["footprint"].null_count() == 3
    assert frame["id"].to_list() == bdf.frame["id"].to_list()
    assert moved.crs["id"] == {"authority": "EPSG", "code": 7415}

    expected = SIXTY5.to_rd(vertices(bdf))
    assert np.abs(vertices(moved) - expected).max() < 1e-6
    mesh = frame["mesh"][3]
    assert mesh["faces"] == bdf.frame["mesh"][3]["faces"]
    assert mesh["styles"] == bdf.frame["mesh"][3]["styles"]
    # The box is rotated by 50 degrees, so its bounding box is that of the moved vertices,
    # not the moved corners of the old one
    bbox = frame["bbox"][3]
    assert [bbox[k] for k in ("xmin", "ymin", "zmin")] == pytest.approx(
        expected.min(axis=0), abs=1e-6
    )
    assert [bbox[k] for k in ("xmax", "ymax", "zmax")] == pytest.approx(
        expected.max(axis=0), abs=1e-6
    )
    assert bbox["xmax"] - bbox["xmin"] > 0.8  # a 0.5 x 0.8 m box turned 50 degrees spans 0.93 m

    footprint = footprints(moved)[3]
    assert footprint.has_z
    assert footprint.area == pytest.approx(0.5 * 0.8, abs=1e-6)
    z = shapely.get_coordinates(footprint, include_z=True)[:, 2]
    assert np.abs(z - (18.625 + 3.0)).max() < 1e-9
    assert frame["elevation"].to_list() == [None, None, pytest.approx(18.625 + 3.0), None]


def inside_bbox(footprint: shapely.Geometry, bbox: dict[str, float]) -> bool:
    xmin, ymin, xmax, ymax = footprint.bounds
    return (
        bbox["xmin"] <= xmin
        and bbox["ymin"] <= ymin
        and xmax <= bbox["xmax"]
        and ymax <= bbox["ymax"]
    )


def test_to_crs_moves_bbox_from_its_corners_without_a_mesh() -> None:
    """A frame with the mesh column selected away keeps a bbox: the box around the moved
    corners of the old one, which the footprint stays inside."""
    bdf = column_model(COLUMN_FOOTPRINT)
    bdf = bdf.with_frame(bdf.frame.drop("mesh"))
    moved = to_crs(bdf, "EPSG:7415")
    frame = moved.frame
    assert frame.columns == bdf.frame.columns
    assert frame.schema["bbox"] == BBOX_DTYPE
    assert frame["bbox"].null_count() == 3

    old = bdf.frame["bbox"][3]
    corners = np.array(
        [
            [x, y, z]
            for x in (old["xmin"], old["xmax"])
            for y in (old["ymin"], old["ymax"])
            for z in (old["zmin"], old["zmax"])
        ]
    )
    expected = SIXTY5.to_rd(corners)
    bbox = frame["bbox"][3]
    assert [bbox[k] for k in ("xmin", "ymin", "zmin")] == pytest.approx(
        expected.min(axis=0), abs=1e-6
    )
    assert [bbox[k] for k in ("xmax", "ymax", "zmax")] == pytest.approx(
        expected.max(axis=0), abs=1e-6
    )
    # The corners of a box are its vertices, so this is the bbox the mesh would have given
    with_mesh = to_crs(column_model(COLUMN_FOOTPRINT), "EPSG:7415").frame["bbox"][3]
    assert [bbox[k] for k in bbox] == pytest.approx([with_mesh[k] for k in with_mesh], abs=1e-9)
    footprint = footprints(moved)[3]
    assert footprint.has_z
    assert footprint.area == pytest.approx(0.5 * 0.8, abs=1e-6)
    assert inside_bbox(footprint, bbox)

    # Without a bbox either, the footprint moves as it is
    bare = to_crs(bdf.with_frame(bdf.frame.drop("bbox")), "EPSG:7415")
    assert bare.frame.columns == [c for c in bdf.frame.columns if c != "bbox"]
    assert footprints(bare)[3].equals_exact(footprint, 1e-9)


def test_clamping_a_footprint_to_its_bbox_keeps_z() -> None:
    """A footprint that spills out of its bbox is pulled in on x and y and keeps its z."""
    wide = shapely.Polygon(
        [(-1000, -1000, 3000), (2000, -1000, 3000), (2000, 2000, 3000), (-1000, 2000, 3000)]
    )
    moved = to_crs(column_model(wide), "EPSG:7415")
    footprint = footprints(moved)[3]
    bbox = moved.frame["bbox"][3]
    assert footprint.has_z and not footprint.is_empty
    assert inside_bbox(footprint, bbox)
    # The wide square covered the box, so a corner of it lands on each side of the bbox
    assert footprint.bounds == pytest.approx(
        (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]), abs=1e-9
    )
    z = shapely.get_coordinates(footprint, include_z=True)[:, 2]
    assert np.abs(z - (18.625 + 3.0)).max() < 1e-9


@pytest.mark.filterwarnings("ignore:the best transformation")
def test_footprints_moved_to_a_geographic_crs_keep_z(
    sixty5_structural_features: BimDataFrame,
) -> None:
    """Every Sixty5 footprint in OGC:CRS84h is a Polygon Z at its storey's moved level, inside
    its bbox, and the GeoParquet metadata lists Z types only."""
    geographic = to_crs(set_crs(sixty5_structural_features, SIXTY5.crs()), "OGC:CRS84h")
    frame = geographic.frame
    moved = footprints(geographic)
    has = np.array([f is not None and not f.is_empty for f in moved])
    assert has.sum() == 2647
    assert shapely.has_z(moved[has]).all()
    levels = dict(frame.select("id", "elevation").iter_rows())
    refs = frame["footprint_ref"].to_numpy()
    for ref, footprint in zip(refs[has], moved[has], strict=True):
        z = shapely.get_coordinates(footprint, include_z=True)[:, 2]
        # The level moves at the plan centre of the model; the footprint at its own place,
        # where the height differs by what the geoid does over 100 m
        assert np.abs(z - levels[ref]).max() < 0.01
    bboxes = [bbox for bbox, present in zip(frame["bbox"], has, strict=True) if present]
    for footprint, bbox in zip(moved[has], bboxes, strict=True):
        assert inside_bbox(footprint, bbox)
    geo = geo_metadata(frame, geographic.crs)
    assert geo is not None
    assert sorted(geo["columns"]["footprint"]["geometry_types"]) == ["MultiPolygon Z", "Polygon Z"]


def test_to_crs_warns_when_the_best_transformation_needs_missing_grids() -> None:
    """Without the NSGI grids PROJ falls back to a Helmert transformation; to_crs says so."""
    bdf = column_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        group = pyproj.transformer.TransformerGroup("EPSG:7415", "OGC:CRS84h", always_xy=True)
        to_crs(bdf, "OGC:CRS84h")
        to_crs(bdf, "EPSG:7415")  # to the base CRS: an exact conversion, nothing to warn about
    about_grids = [w for w in caught if str(w.message).startswith("the best transformation")]
    if group.best_available:
        assert not about_grids
    else:
        assert len(about_grids) == 1
        message = str(about_grids[0].message)
        assert "nl_nsgi_rdtrans2018.tif" in message and "set_network_enabled" in message
        assert about_grids[0].filename == __file__  # it points at the caller of to_crs


def test_to_crs_of_sixty5_puts_the_model_where_the_notebook_does(
    sixty5_structural_model: ifcopenshell.file,
    sixty5_structural_features: BimDataFrame,
    sixty5_rd: BimDataFrame,
) -> None:
    frame = sixty5_rd.frame
    assert pyproj.CRS.from_json_dict(sixty5_rd.crs) == pyproj.CRS("EPSG:7415")
    assert frame.schema == sixty5_structural_features.frame.schema
    assert frame["mesh"].null_count() == 40
    assert frame["footprint"].null_count() == 40

    # Every vertex lands where the closed-form IfcMapConversion puts it
    assert (
        np.abs(vertices(sixty5_rd) - SIXTY5.to_rd(vertices(sixty5_structural_features))).max()
        < 1e-6
    )

    # Pile heads as sixty5_piles.py extracts them: head = tip + Depth along the extrusion,
    # from the pile's placement and the solid's position. The moved pile's bounding box is
    # centred on the head in plan (to a tessellation of the circle) and tops out at it.
    heads: dict[str, np.ndarray] = {}
    for pile in sixty5_structural_model.by_type("IfcPile"):
        solid = pile.Representation.Representations[0].Items[0]
        assert solid.is_a("IfcExtrudedAreaSolid")
        placement = ifcopenshell.util.placement.get_local_placement(pile.ObjectPlacement)
        matrix = placement @ ifcopenshell.util.placement.get_axis2placement(solid.Position)
        tip = matrix @ np.array([0.0, 0.0, 0.0, 1.0])
        direction = matrix[:3, :3] @ np.array(solid.ExtrudedDirection.DirectionRatios)
        heads[pile.GlobalId] = tip[:3] + solid.Depth * direction
    assert len(heads) == 146
    piles = frame.filter(pl.col("class") == "IfcPile").select("id", "bbox").to_dicts()
    expected = SIXTY5.to_rd(np.array([heads[row["id"]] for row in piles]))
    bbox = [row["bbox"] for row in piles]
    centres = np.array([[(b["xmin"] + b["xmax"]) / 2, (b["ymin"] + b["ymax"]) / 2] for b in bbox])
    assert np.abs(centres - expected[:, :2]).max() < 1e-3
    assert np.abs(np.array([b["zmax"] for b in bbox]) - expected[:, 2]).max() < 1e-6

    # The model sits at Strijp-S: within 100 m of the site origin, between NAP -12 and +76 m
    extent = frame.select(
        pl.col("bbox").struct.field("xmin").min(),
        pl.col("bbox").struct.field("ymin").min(),
        pl.col("bbox").struct.field("zmin").min(),
        pl.col("bbox").struct.field("xmax").max(),
        pl.col("bbox").struct.field("ymax").max(),
        pl.col("bbox").struct.field("zmax").max(),
    ).row(0)
    assert abs(extent[0] - SIXTY5.easting) < 100 and abs(extent[3] - SIXTY5.easting) < 100
    assert abs(extent[1] - SIXTY5.northing) < 100 and abs(extent[4] - SIXTY5.northing) < 100
    assert -12 < extent[2] < extent[5] < 76

    # Storey levels move with the model, and the footprints keep lying at them
    levels = sixty5_structural_features.frame["elevation"].drop_nulls().to_numpy()
    moved_levels = frame["elevation"].drop_nulls().to_numpy()
    assert np.abs(moved_levels - (SIXTY5.orthometric_height + 0.001 * levels)).max() < 1e-9
    moved = footprints(sixty5_rd)
    storeys = dict(frame.select("id", "elevation").iter_rows())
    for ref, footprint in zip(frame["footprint_ref"], moved, strict=True):
        if footprint is None or footprint.is_empty:
            continue
        z = shapely.get_coordinates(footprint, include_z=True)[:, 2]
        assert np.abs(z - storeys[ref]).max() < 1e-9
    # and inside their bbox, the GeoParquet covering
    bounds = shapely.bounds(moved)
    low = frame.select(
        pl.col("bbox").struct.field("xmin"), pl.col("bbox").struct.field("ymin")
    ).to_numpy()
    high = frame.select(
        pl.col("bbox").struct.field("xmax"), pl.col("bbox").struct.field("ymax")
    ).to_numpy()
    has = ~np.isnan(bounds[:, 0])
    assert (bounds[has, :2] >= low[has]).all() and (bounds[has, 2:] <= high[has]).all()


def test_georeferenced_model_writes_geoparquet_in_the_target_crs(
    sixty5_rd: BimDataFrame, tmp_path
) -> None:
    path = tmp_path / "sixty5_rd.parquet"
    sixty5_rd.write_parquet(path)
    geo = json.loads(pl.read_parquet_metadata(path)[GEO_METADATA_KEY])
    assert pyproj.CRS.from_json_dict(geo["columns"]["footprint"]["crs"]) == pyproj.CRS("EPSG:7415")
    assert geo["columns"]["footprint"]["bbox"][0] == pytest.approx(SIXTY5.easting, abs=1)
    assert read_parquet(path).equals(sixty5_rd)


def native_geom_model() -> BimDataFrame:
    """The column model with the column's sweep in native_geom, as read_ifc fills it."""
    bdf = column_model()
    profile = shapely.box(-250, -400, 250, 400)
    native = pl.Series(
        "native_geom",
        [None, None, None, [sweep(translation(260, 420, 3200), profile, 2800)]],
        dtype=NATIVE_GEOM_DTYPE,
    )
    return bdf.with_frame(bdf.frame.with_columns(native))


@pytest.mark.filterwarnings("ignore:the best transformation")
def test_a_non_affine_target_drops_native_geom() -> None:
    bdf = native_geom_model()
    geographic = to_crs(bdf, "OGC:CRS84h")
    assert geographic.frame.schema["native_geom"] == pl.Null
    assert geographic.frame["native_geom"].null_count() == 4
    assert geographic.frame.columns == bdf.frame.columns
    # A deviation in degrees means nothing: it is nulled, the rest of lightbim_qa stays
    qa = to_crs(simple_features(bdf, lightbim=True), "OGC:CRS84h").frame["lightbim_qa"]
    assert qa.dtype == QA_DTYPE and qa.null_count() == 3
    assert qa.struct.field("deviation").null_count() == 4
    assert qa[3]["source"] == "extrusion"
    lon, lat, _ = vertices(geographic).T
    assert (abs(lon - 5.46) < 0.01).all() and (abs(lat - 51.45) < 0.01).all()
    assert geographic.crs["type"] == "GeographicCRS"
    assert affine_transformation(bdf, "OGC:CRS84h") is None
    assert affine_transformation(bdf, "EPSG:4978") is None

    # Another projection is not affine either: UTM zone 31N over RD New
    utm = to_crs(bdf, "EPSG:25831+5709")
    assert utm.frame.schema["native_geom"] == pl.Null
    assert affine_transformation(bdf, "EPSG:25831+5709") is None


def test_an_affine_target_keeps_native_geom() -> None:
    bdf = native_geom_model()
    matrix = affine_transformation(bdf, "EPSG:7415")
    assert matrix is not None
    homogeneous = np.c_[SITE_POINTS, np.ones(len(SITE_POINTS))]
    assert (
        np.abs((homogeneous @ matrix.T)[:, :3] - SIXTY5.to_rd(SITE_POINTS)).max() < AFFINE_TOLERANCE
    )
    # The matrix is composed onto the sweep's placement: the profile and the depth stay in
    # the sweep's millimetres and the placement, scale included, takes them to RD metres
    rd = to_crs(bdf, "EPSG:7415")
    assert rd.frame.schema["native_geom"] == NATIVE_GEOM_DTYPE
    assert rd.frame["native_geom"].null_count() == 3
    (moved,) = rd.frame["native_geom"][3]
    (original,) = bdf.frame["native_geom"][3]
    placement = np.array(moved["placement"]).reshape(4, 4)
    origin = placement @ np.array([0.0, 0.0, 0.0, 1.0])
    assert origin[:3] == pytest.approx(SIXTY5.to_rd([[260, 420, 3200]])[0], abs=1e-6)
    assert np.linalg.norm(placement[:3, :3], axis=0) == pytest.approx(0.001)
    assert moved["profile"] == original["profile"] and moved["depth"] == 2800.0
    assert moved["direction"] == [0.0, 0.0, 1.0]
    # The LightBIM axis of the moved sweep is the moved LightBIM axis
    rd_first = simple_features(rd, lightbim=True)
    site_first = to_crs(simple_features(bdf, lightbim=True), "EPSG:7415")
    for moved_bdf in (rd_first, site_first):
        assert moved_bdf.frame["lightbim_qa"][3]["source"] == "extrusion"
    # The deviation is a length in the CRS's unit: half the diagonal of the 500 x 800 column,
    # in millimetres on the site and in metres in RD, whichever was done first (to 10 nm:
    # pyproj moves the vertices, which the reduction in RD measures, to about 5 nm)
    site_deviation = simple_features(bdf, lightbim=True).frame["lightbim_qa"][3]["deviation"]
    assert site_deviation == pytest.approx(math.hypot(250, 400))
    for moved_bdf in (rd_first, site_first):
        deviation = moved_bdf.frame["lightbim_qa"].struct.field("deviation")
        assert deviation[3] == pytest.approx(math.hypot(0.25, 0.4), abs=1e-8)
        assert deviation.null_count() == 3
    with pytest.raises(TypeError, match="QA_DTYPE"):
        to_crs(bdf.with_frame(bdf.frame.with_columns(lightbim_qa=pl.lit(1.0))), "EPSG:7415")
    axis_rd_first = shapely.from_wkb(rd_first.frame["lightbim"][3])
    axis_site_first = shapely.from_wkb(site_first.frame["lightbim"][3])
    assert shapely.get_coordinates(axis_rd_first, include_z=True) == pytest.approx(
        shapely.get_coordinates(axis_site_first, include_z=True), abs=1e-6
    )
    # A native_geom of another layout cannot be placed; an empty column passes through
    placeholder = bdf.with_frame(
        bdf.frame.with_columns(pl.Series("native_geom", [None, None, None, {"depth": 2800.0}]))
    )
    with pytest.raises(TypeError, match="NATIVE_GEOM_DTYPE"):
        to_crs(placeholder, "EPSG:7415")
    empty = bdf.with_frame(
        bdf.frame.with_columns(pl.Series("native_geom", [None] * 4, dtype=pl.Null))
    )
    assert to_crs(empty, "EPSG:7415").frame.schema["native_geom"] == pl.Null
    # Back to the site CRS is affine too, and a round trip
    back = to_crs(to_crs(empty, "EPSG:7415"), SIXTY5.crs())
    assert np.abs(vertices(back) - vertices(empty)).max() < 1e-6 / SIXTY5.unit
    assert back.crs == SIXTY5.crs()


def test_to_crs_refuses_what_pyproj_cannot_do() -> None:
    plain = plain_column_model()
    with pytest.raises(ValueError, match="set_crs"):
        to_crs(plain, "EPSG:7415")
    georeferenced = set_crs(plain, SIXTY5.crs())
    with pytest.raises(ValueError, match="three"):
        to_crs(georeferenced, "EPSG:28992")
    with pytest.raises(ValueError, match="cannot read"):
        to_crs(georeferenced, "no such crs")


@pytest.mark.filterwarnings("ignore:the best transformation")
def test_linearized_site_to_ecef() -> None:
    matrix = linearized_site_to_ecef(SIXTY5.crs())
    # The reference: the closed-form georeferencing to RD New + NAP, then pyproj to ECEF
    rd_to_ecef = pyproj.Transformer.from_crs("EPSG:7415", "EPSG:4978", always_xy=True)

    def ecef(site_m: np.ndarray) -> np.ndarray:
        return np.column_stack(rd_to_ecef.transform(*SIXTY5.to_rd(site_m / SIXTY5.unit).T))

    origin = ecef(np.zeros((1, 3)))[0]
    assert matrix.shape == (4, 4)
    assert matrix[:3, 3] == pytest.approx(origin, abs=1e-6)
    assert matrix[3].tolist() == [0, 0, 0, 1]
    assert matrix[:3, :3] == pytest.approx(ecef(np.eye(3)).T - origin[:, None], abs=1e-6)
    # The columns are one metre along the site axes: of unit length up to the scale of RD New
    # at Eindhoven (about 1.0001) and orthogonal
    axes = matrix[:3, :3]
    assert np.linalg.norm(axes, axis=0) == pytest.approx([1, 1, 1], abs=2e-4)
    assert np.abs(axes.T @ axes - np.eye(3)).max() < 2e-4
    # The linearization holds to a tenth of a millimetre over 10 m (the earth curves 8 um)
    site_m = np.array([[10.0, -10.0, 10.0]])
    assert np.abs(matrix @ [*site_m[0], 1] - [*ecef(site_m)[0], 1]).max() < 1e-4


@pytest.mark.filterwarnings("ignore:the best transformation")
def test_a_site_in_millimetres_and_one_in_metres_land_on_the_same_spot() -> None:
    """PROJ orders the vertical conversion of a derived CRS after a 3D Helmert, which then
    reads millimetres as metres; to_crs goes through the base CRS instead."""
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(10, 20, 3200, 510, 820, 50_000)
    in_mm = set_crs(model([*SPATIAL, column]), SIXTY5.crs())
    column["mesh"] = box(0.01, 0.02, 3.2, 0.51, 0.82, 50.0)
    in_m = set_crs(model([*SPATIAL, column]), SIXTY5._replace(unit=1.0).crs())
    for target in ("OGC:CRS84h", "EPSG:4978"):
        from_mm, from_m = vertices(to_crs(in_mm, target)), vertices(to_crs(in_m, target))
        assert np.abs(from_mm - from_m).max() < (1e-8 if target == "OGC:CRS84h" else 1e-3)
    assert (
        np.abs(
            linearized_site_to_ecef(SIXTY5.crs())
            - linearized_site_to_ecef(SIXTY5._replace(unit=1.0).crs())
        ).max()
        < 1e-3
    )


def test_the_core_imports_without_pyproj() -> None:
    lines = run_without(
        "pyproj",
        "import bimpolars\n"
        "import bimpolars.data, bimpolars.features, bimpolars.geoparquet, bimpolars.gltf\n"
        "import bimpolars.ifc, bimpolars.speckle\n"
        "print(bimpolars.engineering_crs('ok')['name'])\n"
        "try:\n"
        "    import bimpolars.geobim\n"
        "except ImportError as error:\n"
        "    print(error)\n",
    )
    assert lines[0] == "ok"
    assert "bimpolars[geobim]" in lines[1]

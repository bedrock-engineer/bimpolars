import io
import json

import polars as pl
import pyproj
import pytest

from bimpolars import BBOX_DTYPE, MESH_DTYPE, BimDataFrame, engineering_crs, read_parquet
from bimpolars.frame import METADATA_KEY


@pytest.fixture
def small() -> BimDataFrame:
    mesh = {
        "vertices": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        "faces": [0, 1, 2, 1, 3, 2],
        "styles": [
            {"name": "Concrete", "rgba": [0.5, 0.5, 0.5, 1.0], "face_count": 1},
            {"name": None, "rgba": [0.7, 0.7, 0.7, 1.0], "face_count": 1},
        ],
    }
    frame = pl.DataFrame(
        [
            pl.Series("id", ["storey", "slab"]),
            pl.Series("name", ["Level 0", None]),
            pl.Series("class", ["IfcBuildingStorey", "IfcSlab"], dtype=pl.Categorical),
            pl.Series(
                "properties",
                [None, {"Pset_SlabCommon": {"LoadBearing": True}}],
                dtype=pl.Struct({"Pset_SlabCommon": pl.Struct({"LoadBearing": pl.Boolean})}),
            ),
            pl.Series("container", [None, "storey"]),
            pl.Series("elevation", [0.0, None]),
            pl.Series("native_geom", [None, None], dtype=pl.Null),
            pl.Series("mesh", [None, mesh], dtype=MESH_DTYPE),
            pl.Series(
                "bbox",
                [
                    None,
                    {"xmin": 0.0, "ymin": 0.0, "zmin": 0.0, "xmax": 1.0, "ymax": 1.0, "zmax": 0.0},
                ],
                dtype=BBOX_DTYPE,
            ),
        ]
    )
    return BimDataFrame(frame, engineering_crs("Small model", unit="millimetre"))


def test_frame_and_crs(small: BimDataFrame) -> None:
    assert len(small) == 2
    assert small.projjson["type"] == "EngineeringCRS"
    assert "Small model" in repr(small)
    assert "Small model" in small._repr_html_()


def test_crs_is_copied(small: BimDataFrame) -> None:
    crs = engineering_crs("Original")
    bdf = BimDataFrame(small.frame, crs)
    crs["name"] = "Changed"
    assert bdf.projjson["name"] == "Original"


def test_equals(small: BimDataFrame) -> None:
    assert small.equals(BimDataFrame(small.frame, small.projjson))
    assert not small.equals(small.with_frame(small.frame.head(1)))
    assert not small.equals(BimDataFrame(small.frame, engineering_crs("Other")))
    assert not small.equals(small.frame)


def test_with_frame_keeps_the_crs(small: BimDataFrame) -> None:
    slabs = small.with_frame(small.frame.filter(pl.col("class") == "IfcSlab"))
    assert len(slabs) == 1
    assert slabs.projjson == small.projjson


def test_parquet_round_trip_through_a_path(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "small.parquet"
    small.write_parquet(path)
    back = read_parquet(path)
    assert back.equals(small)
    assert back.frame.schema == small.frame.schema
    metadata = json.loads(pl.read_parquet_metadata(path)[METADATA_KEY])
    assert metadata == {"version": 1, "crs": small.projjson}


def test_parquet_round_trip_through_a_buffer(small: BimDataFrame) -> None:
    buffer = io.BytesIO()
    small.write_parquet(buffer)
    buffer.seek(0)
    assert BimDataFrame.read_parquet(buffer).equals(small)


def test_parquet_without_metadata_is_refused(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "plain.parquet"
    small.frame.write_parquet(path)
    with pytest.raises(ValueError, match="bimpolars"):
        read_parquet(path)


def test_parquet_of_another_layout_version_is_refused(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "future.parquet"
    metadata = {METADATA_KEY: json.dumps({"version": 2, "crs": small.projjson})}
    small.frame.write_parquet(path, metadata=metadata)
    with pytest.raises(ValueError, match="layout version 2"):
        read_parquet(path)


def test_write_parquet_keeps_the_callers_metadata(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "tagged.parquet"
    small.write_parquet(path, metadata={"source": "test"})
    metadata = pl.read_parquet_metadata(path)
    assert metadata["source"] == "test"
    assert json.loads(metadata[METADATA_KEY])["crs"] == small.projjson
    assert read_parquet(path).equals(small)


def test_constructor_checks_its_arguments(small: BimDataFrame) -> None:
    with pytest.raises(TypeError):
        BimDataFrame([], small.projjson)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pyproj cannot read"):
        BimDataFrame(small.frame, "not a CRS")
    with pytest.raises(ValueError):
        BimDataFrame(small.frame, {"name": "no type"})


def test_constructor_takes_a_crs_in_any_form(small: BimDataFrame) -> None:
    rd_nap = pyproj.CRS("EPSG:7415")
    for crs in ("EPSG:7415", 7415, rd_nap, rd_nap.to_wkt()):
        assert BimDataFrame(small.frame, crs).crs == rd_nap

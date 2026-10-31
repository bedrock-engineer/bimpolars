import io
import json

import polars as pl
import pyproj
import pytest

import bimpolars as bpl
from bimpolars import BBOX_DTYPE, MESH_DTYPE, BimDataFrame, engineering_crs, read_parquet
from bimpolars.frame import CRS_COLUMN, CRS_DTYPE, METADATA_KEY, MODEL_COLUMNS, crs_text


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


def test_the_crs_is_a_column(small: BimDataFrame) -> None:
    assert len(small) == 2
    assert small.frame.columns == list(MODEL_COLUMNS)
    assert small.frame.schema[CRS_COLUMN] == CRS_DTYPE
    assert small.frame[CRS_COLUMN].to_list() == [crs_text(small.projjson)] * 2
    assert small.projjson["type"] == "EngineeringCRS"
    assert small.crs == pyproj.CRS.from_json_dict(small.projjson)


def test_a_frame_with_the_column_needs_no_crs(small: BimDataFrame) -> None:
    again = BimDataFrame(small.frame)
    assert again.equals(small)
    assert again.projjson == small.projjson
    text = small.frame.with_columns(pl.col(CRS_COLUMN).cast(pl.String))
    assert BimDataFrame(text).equals(small)  # String becomes Categorical
    with pytest.raises(ValueError, match="no bim_crs column"):
        BimDataFrame(small.frame.drop(CRS_COLUMN))
    with pytest.raises(TypeError, match="PROJJSON text"):
        BimDataFrame(small.frame.with_columns(bim_crs=pl.lit(1)))


def test_a_crs_must_agree_with_the_column(small: BimDataFrame) -> None:
    assert BimDataFrame(small.frame, small.projjson).equals(small)
    with pytest.raises(ValueError, match=r"'Other'.*set_crs"):
        BimDataFrame(small.frame, engineering_crs("Other"))


def test_crs_is_copied(small: BimDataFrame) -> None:
    crs = engineering_crs("Original")
    bdf = BimDataFrame(small.frame.drop(CRS_COLUMN), crs)
    crs["name"] = "Changed"
    assert bdf.projjson["name"] == "Original"


def test_one_crs_or_an_error_that_names_them(small: BimDataFrame) -> None:
    other = bpl.set_crs(small, "EPSG:7415")
    mixed = bpl.concat([small, other])
    assert len(mixed) == 4
    with pytest.raises(ValueError, match=r"2 CRSs: 'Small model', 'Amersfoort / RD New \+ NAP"):
        _ = mixed.crs
    with pytest.raises(ValueError, match="group_by"):
        _ = mixed.projjson
    nulls = BimDataFrame(small.frame.with_columns(bim_crs=pl.lit(None, dtype=pl.String)))
    with pytest.raises(ValueError, match="2 of 2 rows without a CRS"):
        _ = nulls.crs
    with pytest.raises(ValueError, match="no rows"):
        _ = small.head(0).crs
    # Each CRS's rows on their own have their CRS again
    by_crs = dict(mixed.group_by(CRS_COLUMN, maintain_order=True).agg(pl.len()).iter_rows())
    assert list(by_crs.values()) == [2, 2]
    assert mixed.filter(pl.col("id") == "slab").head(1).crs == small.crs
    assert "2 CRSs" in repr(mixed)


def test_polars_methods_work_on_the_model(small: BimDataFrame) -> None:
    slabs = small.filter(pl.col("class") == "IfcSlab")
    assert isinstance(slabs, BimDataFrame)
    assert slabs.crs == small.crs
    assert len(slabs) == 1
    assert isinstance(small.with_columns(pl.lit(1).alias("one")), BimDataFrame)
    assert small["id"].to_list() == ["storey", "slab"]
    assert isinstance(small[:1], BimDataFrame)
    assert small.columns == small.frame.columns
    assert small.height == 2
    # A result without bim_crs, a Series or a scalar comes back as it is
    assert type(small.select("id", "name")) is pl.DataFrame
    assert type(small.get_column("id")) is pl.Series
    assert small.select(pl.len()).item() == 2
    # A BimDataFrame argument goes in as its frame
    joined = small.join(slabs.select("id", CRS_COLUMN, slab=pl.lit(True)), on=["id", CRS_COLUMN])
    assert isinstance(joined, BimDataFrame)
    assert joined["slab"].to_list() == [True]
    assert small.join(slabs, on="id").columns[-1] == f"{CRS_COLUMN}_right"
    assert "filter" in dir(small)


def test_arrow_readers_see_the_frame(small: BimDataFrame) -> None:
    assert pl.DataFrame(small).equals(small.frame)
    assert BimDataFrame(pl.DataFrame(small)).equals(small)
    pyarrow = pytest.importorskip("pyarrow")
    assert pyarrow.table(small).num_rows == 2


def test_the_display_shows_the_crs_name_once(small: BimDataFrame) -> None:
    assert "Small model" in repr(small)
    assert CRS_COLUMN not in repr(small)
    html = small._repr_html_()
    assert html.count("Small model") == 1
    assert CRS_COLUMN not in html


def test_concat_keeps_the_crs_of_every_row(small: BimDataFrame) -> None:
    both = bpl.concat([small, small])
    assert both.crs == small.crs
    assert len(both) == 4
    with_footprint = small.with_columns(extra=pl.lit(1))
    assert bpl.concat([small, with_footprint], how="diagonal")["extra"].to_list() == [
        None,
        None,
        1,
        1,
    ]
    with pytest.raises(TypeError, match="BimDataFrame"):
        bpl.concat([small, small.frame])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one"):
        bpl.concat([])


def test_equals(small: BimDataFrame) -> None:
    assert small.equals(BimDataFrame(small.frame))
    assert not small.equals(small.head(1))
    assert not small.equals(bpl.set_crs(small, engineering_crs("Other")))
    assert not small.equals(small.frame)


def test_with_frame_gives_a_frame_without_the_column_this_crs(small: BimDataFrame) -> None:
    slabs = small.with_frame(small.frame.filter(pl.col("class") == "IfcSlab"))
    assert len(slabs) == 1
    assert slabs.projjson == small.projjson
    bare = small.with_frame(small.frame.drop(CRS_COLUMN))
    assert bare.equals(small)
    other = bpl.set_crs(small, "EPSG:7415")
    assert small.with_frame(other.frame).crs == other.crs  # the column wins


def test_parquet_round_trip_through_a_path(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "small.parquet"
    small.write_parquet(path)
    back = read_parquet(path)
    assert back.equals(small)
    assert back.frame.schema == small.frame.schema
    metadata = json.loads(pl.read_parquet_metadata(path)[METADATA_KEY])
    assert metadata == {"version": 2}


def test_parquet_round_trip_through_a_buffer(small: BimDataFrame) -> None:
    buffer = io.BytesIO()
    small.write_parquet(buffer)
    buffer.seek(0)
    assert BimDataFrame.read_parquet(buffer).equals(small)


def test_parquet_of_a_mixed_frame_without_geometry(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "mixed.parquet"
    mixed = bpl.concat([small, bpl.set_crs(small, "EPSG:7415")])
    mixed.write_parquet(path)
    assert read_parquet(path).equals(mixed)


def test_parquet_of_layout_version_1_gets_the_column(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "version1.parquet"
    metadata = {METADATA_KEY: json.dumps({"version": 1, "crs": small.projjson})}
    small.frame.drop(CRS_COLUMN).write_parquet(path, metadata=metadata)
    assert read_parquet(path).equals(small)


def test_parquet_with_the_column_needs_no_metadata(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "plain.parquet"
    small.frame.write_parquet(path)
    assert read_parquet(path).equals(small)


def test_parquet_without_a_crs_is_refused(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "plain.parquet"
    small.frame.drop(CRS_COLUMN).write_parquet(path)
    with pytest.raises(ValueError, match="bim_crs"):
        read_parquet(path)


def test_parquet_of_another_layout_version_is_refused(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "future.parquet"
    small.frame.write_parquet(path, metadata={METADATA_KEY: json.dumps({"version": 3})})
    with pytest.raises(ValueError, match="layout version 3"):
        read_parquet(path)


def test_write_parquet_keeps_the_callers_metadata(small: BimDataFrame, tmp_path) -> None:
    path = tmp_path / "tagged.parquet"
    small.write_parquet(path, metadata={"source": "test"})
    metadata = pl.read_parquet_metadata(path)
    assert metadata["source"] == "test"
    assert json.loads(metadata[METADATA_KEY]) == {"version": 2}
    assert read_parquet(path).equals(small)


def test_constructor_checks_its_arguments(small: BimDataFrame) -> None:
    bare = small.frame.drop(CRS_COLUMN)
    with pytest.raises(TypeError):
        BimDataFrame([], small.projjson)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pyproj cannot read"):
        BimDataFrame(bare, "not a CRS")
    with pytest.raises(ValueError):
        BimDataFrame(bare, {"name": "no type"})


def test_constructor_takes_a_crs_in_any_form(small: BimDataFrame) -> None:
    rd_nap = pyproj.CRS("EPSG:7415")
    for crs in ("EPSG:7415", 7415, rd_nap, rd_nap.to_wkt()):
        assert BimDataFrame(small.frame.drop(CRS_COLUMN), crs).crs == rd_nap

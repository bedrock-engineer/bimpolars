"""GeoParquet: the geo metadata and what geopandas and DuckDB make of the written file."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import shapely

from bimpolars import BimDataFrame, read_parquet, simple_features
from bimpolars.crs import CRS84H
from bimpolars.frame import METADATA_KEY
from bimpolars.geoparquet import (
    COLUMN_CRS,
    GEO_METADATA_KEY,
    GEOMETRY_COLUMNS,
    GEOPARQUET_VERSION,
    _wkb_geometry_types,
    geo_metadata,
)
from tests.synthetic import SPATIAL, box, model


@pytest.fixture
def small() -> BimDataFrame:
    column = {"id": "column", "class": "IfcColumn", "container": "storey"}
    column["mesh"] = box(10, 20, 3200, 510, 820, 6000)
    slab = {"id": "slab", "class": "IfcSlab", "container": "storey"}
    slab["mesh"] = box(-1000, -1000, 2800, 1000, 1000, 3200)
    return simple_features(model([*SPATIAL, column, slab]))


def test_a_frame_without_geometry_has_no_geo_metadata(small: BimDataFrame) -> None:
    assert geo_metadata(small.frame.drop("footprint"), small.crs) is None


def test_geo_metadata(small: BimDataFrame) -> None:
    geo = geo_metadata(small.frame, small.crs)
    assert geo is not None
    assert geo["version"] == GEOPARQUET_VERSION
    assert geo["primary_column"] == "footprint"
    assert list(geo["columns"]) == ["footprint"]
    column = geo["columns"]["footprint"]
    assert column["encoding"] == "WKB"
    assert column["geometry_types"] == ["Polygon Z"]
    assert column["crs"] == small.crs
    assert column["bbox"] == [-1000.0, -1000.0, 1000.0, 1000.0]
    assert column["covering"] == {
        "bbox": {
            "xmin": ["bbox", "xmin"],
            "ymin": ["bbox", "ymin"],
            "xmax": ["bbox", "xmax"],
            "ymax": ["bbox", "ymax"],
        }
    }
    json.dumps(geo)


def lon_lat_square(lon: float, lat: float, z: float, size: float = 1e-5) -> shapely.Polygon:
    return shapely.Polygon(
        [(lon, lat, z), (lon + size, lat, z), (lon + size, lat + size, z), (lon, lat + size, z)]
    )


def with_geog_column(
    small: BimDataFrame, geometries: list[shapely.Geometry | None]
) -> pl.DataFrame:
    wkb = [None if g is None else shapely.to_wkb(g, flavor="iso") for g in geometries]
    return small.frame.with_columns(pl.Series("geog", wkb, dtype=pl.Binary))


def test_geog_is_the_primary_column_in_its_own_crs(small: BimDataFrame, tmp_path: Path) -> None:
    """A geog column (bimpolars.geobim.with_geog) comes first, in OGC:CRS84h, with the bounds
    of its own coordinates and without the model-CRS bbox as its covering."""
    assert GEOMETRY_COLUMNS == ("geog", "footprint", "lightbim")
    assert COLUMN_CRS == {"geog": CRS84H}
    squares = [
        None,
        None,
        None,
        lon_lat_square(5.46, 51.44, 60.0),
        lon_lat_square(5.47, 51.45, 61.0),
    ]
    frame = with_geog_column(small, squares)
    geo = geo_metadata(frame, small.crs)
    assert geo is not None
    assert geo["primary_column"] == "geog"
    assert list(geo["columns"]) == ["geog", "footprint"]
    geog = geo["columns"]["geog"]
    assert geog["encoding"] == "WKB"
    assert geog["geometry_types"] == ["Polygon Z"]
    assert geog["crs"] == CRS84H
    assert geog["bbox"] == pytest.approx([5.46, 51.44, 5.47001, 51.45001])
    assert "covering" not in geog
    footprint = geo["columns"]["footprint"]
    assert footprint["crs"] == small.crs
    assert footprint["bbox"] == [-1000.0, -1000.0, 1000.0, 1000.0]
    assert footprint["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    json.dumps(geo)

    # Without a geometry, or with empty ones only, geog has no bounds
    nothing = geo_metadata(with_geog_column(small, [None] * 5), small.crs)
    assert nothing is not None and "bbox" not in nothing["columns"]["geog"]
    assert nothing["columns"]["geog"]["geometry_types"] == []
    empty = shapely.from_wkt("POLYGON Z EMPTY")
    empties = geo_metadata(with_geog_column(small, [None, None, None, empty, empty]), small.crs)
    assert empties is not None and "bbox" not in empties["columns"]["geog"]
    assert empties["columns"]["geog"]["geometry_types"] == ["Polygon Z"]
    # An empty geometry next to a real one does not widen the bounds
    mixed = geo_metadata(with_geog_column(small, [None, None, None, squares[3], empty]), small.crs)
    assert mixed is not None
    assert mixed["columns"]["geog"]["bbox"] == pytest.approx([5.46, 51.44, 5.46001, 51.44001])
    # geog alone is a GeoParquet geometry too
    alone = geo_metadata(frame.drop("footprint"), small.crs)
    assert alone is not None and list(alone["columns"]) == ["geog"]

    # Written, read back, and read by geopandas with geog as the active geometry
    bdf = small.with_frame(frame)
    path = tmp_path / "geog.parquet"
    bdf.write_parquet(path)
    assert json.loads(pl.read_parquet_metadata(path)[GEO_METADATA_KEY]) == geo
    assert read_parquet(path).equals(bdf)
    geopandas = pytest.importorskip("geopandas")
    pyproj = pytest.importorskip("pyproj")
    gdf = geopandas.read_parquet(path)
    assert gdf.geometry.name == "geog"
    assert gdf.crs == pyproj.CRS("OGC:CRS84h")
    assert gdf["footprint"].crs == pyproj.CRS.from_json_dict(small.crs)
    assert gdf.geometry.dropna().has_z.all()
    assert list(gdf.geometry.dropna().bounds.iloc[0]) == pytest.approx(
        [5.46, 51.44, 5.46001, 51.44001]
    )


def test_geo_metadata_without_a_bbox_column(small: BimDataFrame) -> None:
    geo = geo_metadata(small.frame.drop("bbox"), small.crs)
    assert geo is not None
    column = geo["columns"]["footprint"]
    assert "covering" not in column
    assert "bbox" not in column
    renamed = small.frame.with_columns(
        pl.col("bbox").struct.rename_fields(["a", "b", "c", "d", "e", "f"])
    )
    geo = geo_metadata(renamed, small.crs)
    assert geo is not None
    assert "covering" not in geo["columns"]["footprint"]
    with pytest.raises(TypeError, match="Binary"):
        geo_metadata(small.frame.with_columns(footprint=pl.lit("POLYGON EMPTY")), small.crs)


def test_wkb_geometry_types() -> None:
    multipolygon = shapely.MultiPolygon([shapely.box(0, 0, 1, 1)])
    polygon_z = shapely.to_wkb(shapely.force_3d(shapely.box(0, 0, 1, 1), z=2), flavor="iso")
    multi_z = shapely.to_wkb(shapely.force_3d(multipolygon, z=2), flavor="iso")
    point_big_endian = shapely.to_wkb(shapely.Point(1, 2), byte_order=0)
    series = pl.Series([polygon_z, None, multi_z, polygon_z, point_big_endian], dtype=pl.Binary)
    assert _wkb_geometry_types(series) == ["Point", "Polygon Z", "MultiPolygon Z"]
    assert _wkb_geometry_types(pl.Series([None], dtype=pl.Binary)) == []
    assert _wkb_geometry_types(pl.Series([], dtype=pl.Binary)) == []
    with_m = shapely.to_wkb(shapely.from_wkt("POINT M (1 2 3)"), flavor="iso")
    with pytest.raises(ValueError, match="M coordinates"):
        _wkb_geometry_types(pl.Series([with_m], dtype=pl.Binary))
    with pytest.raises(ValueError, match="not WKB"):
        _wkb_geometry_types(pl.Series([b"PAR1"], dtype=pl.Binary))
    with pytest.raises(ValueError, match="unknown"):
        _wkb_geometry_types(pl.Series([b"\x01\x63\x00\x00\x00"], dtype=pl.Binary))


def test_extended_wkb_is_refused() -> None:
    # GeoParquet's WKB is ISO WKB: the PostGIS flavour with the Z, M or SRID flag is refused
    with_z = shapely.to_wkb(shapely.force_3d(shapely.box(0, 0, 1, 1), z=2), flavor="extended")
    with_srid = shapely.to_wkb(shapely.set_srid(shapely.Point(1, 2), 28992), include_srid=True)
    for extended in (with_z, with_srid):
        with pytest.raises(ValueError, match="extended WKB"):
            _wkb_geometry_types(pl.Series([extended], dtype=pl.Binary))
    plain = shapely.to_wkb(shapely.box(0, 0, 1, 1), flavor="extended")  # 2D: no flag set
    assert _wkb_geometry_types(pl.Series([plain], dtype=pl.Binary)) == ["Polygon"]


def test_write_parquet_writes_geoparquet(small: BimDataFrame, tmp_path: Path) -> None:
    path = tmp_path / "small.parquet"
    small.write_parquet(path)
    metadata = pl.read_parquet_metadata(path)
    assert json.loads(metadata[GEO_METADATA_KEY]) == geo_metadata(small.frame, small.crs)
    assert json.loads(metadata[METADATA_KEY])["crs"] == small.crs
    back = read_parquet(path)
    assert back.equals(small)
    assert back.frame.schema["footprint"] == pl.Binary
    plain = tmp_path / "plain.parquet"
    small.with_frame(small.frame.drop("footprint", "footprint_ref")).write_parquet(plain)
    assert GEO_METADATA_KEY not in pl.read_parquet_metadata(plain)


def test_wkb_is_a_byte_array_column(small: BimDataFrame, tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "small.parquet"
    small.write_parquet(path)
    schema = pq.read_schema(path)
    assert str(schema.field("footprint").type) in ("binary", "large_binary")
    assert GEO_METADATA_KEY.encode() in pq.read_metadata(path).metadata


def test_geopandas_reads_the_crs_and_the_bbox_covering(
    sixty5_structural_features: BimDataFrame, tmp_path: Path
) -> None:
    geopandas = pytest.importorskip("geopandas")
    pyproj = pytest.importorskip("pyproj")
    path = tmp_path / "sixty5.parquet"
    sixty5_structural_features.write_parquet(path)
    columns = ["id", "class", "footprint_ref", "footprint", "bbox"]
    gdf = geopandas.read_parquet(path, columns=columns)
    assert len(gdf) == 2687
    assert gdf.geometry.name == "footprint"
    assert gdf.crs == pyproj.CRS.from_json_dict(sixty5_structural_features.crs)
    assert gdf.crs.axis_info[0].unit_name == "millimetre"
    assert gdf.geometry.isna().sum() == 40
    assert gdf.geometry.dropna().has_z.all()
    assert set(gdf.geometry.dropna().geom_type) == {"Polygon", "MultiPolygon"}
    # The bbox covering: a window of 5 m around a column of the first storey
    frame = sixty5_structural_features.frame
    column = frame.filter(pl.col("class") == "IfcColumn").row(0, named=True)["bbox"]
    window = (
        column["xmin"] - 5000,
        column["ymin"] - 5000,
        column["xmax"] + 5000,
        column["ymax"] + 5000,
    )
    in_window = geopandas.read_parquet(path, columns=columns, bbox=window)
    bbox = pl.col("bbox").struct
    expected = frame.filter(
        (bbox.field("xmin") <= window[2])
        & (bbox.field("ymin") <= window[3])
        & (bbox.field("xmax") >= window[0])
        & (bbox.field("ymax") >= window[1])
    )
    assert 0 < len(in_window) < len(gdf)
    assert set(in_window["id"]) == set(expected["id"])
    assert in_window.geometry.intersects(shapely.box(*window)).all()


def test_duckdb_spatial_reads_the_footprints(
    sixty5_structural_features: BimDataFrame, tmp_path: Path
) -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    try:
        connection.install_extension("spatial")
        connection.load_extension("spatial")
    except duckdb.Error as error:  # the extension repository is not reachable everywhere
        pytest.skip(f"DuckDB spatial extension unavailable: {error}")
    path = tmp_path / "sixty5.parquet"
    sixty5_structural_features.write_parquet(path)
    source = f"read_parquet('{path.as_posix()}')"
    types = connection.execute(
        f"SELECT ST_GeometryType(footprint) AS type, count(*) AS n FROM {source} "
        "WHERE footprint IS NOT NULL GROUP BY 1"
    ).fetchall()
    assert {str(kind): n for kind, n in types} == {"POLYGON": 2644, "MULTIPOLYGON": 3}
    nulls = connection.execute(f"SELECT count(*) FROM {source} WHERE footprint IS NULL")
    assert nulls.fetchone() == (40,)
    covered = connection.execute(
        f"SELECT count(*) FROM {source} WHERE footprint IS NOT NULL AND NOT "
        "ST_Within(footprint, ST_MakeEnvelope(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))"
    )
    assert covered.fetchone() == (0,)
    geo = connection.execute(
        f"SELECT decode(value) FROM parquet_kv_metadata('{path.as_posix()}') "
        "WHERE decode(key) = 'geo'"
    ).fetchone()
    assert geo is not None
    assert json.loads(geo[0])["primary_column"] == "footprint"

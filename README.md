# bimpolars

Building Information Models in polars DataFrames with geometry columns.

A `BimDataFrame` is one BIM model: a [polars](https://pola.rs) DataFrame with one row per
element, with or without geometry, and one model coordinate reference system stored as
[PROJJSON](https://proj.org/en/stable/specifications/projjson.html). Meshes and bounding boxes
are plain nested columns and footprints are WKB, so the whole model filters, joins and
aggregates like any other DataFrame and round-trips through Parquet, as GeoParquet once it has
footprints.

```python
import polars as pl
import bimpolars

bdf = bimpolars.read_ifc("sixty5_O-S1-CON-structural-engineering.ifc")
bdf.frame.schema
# id: String, name: String, class: Categorical, properties: Struct, container: String,
# elevation: Float64, native_geom: Null, mesh: Struct, bbox: Struct

columns = bdf.frame.filter(pl.col("class") == "IfcColumn")
columns.select(
    "id",
    "name",
    load_bearing=pl.col("properties").struct.field("Pset_ColumnCommon").struct.field("LoadBearing"),
    triangles=pl.col("mesh").struct.field("faces").list.len() // 3,
)

# Outline the elements, not the voids of the openings; the storeys stay to place the footprints
elements = bdf.frame.filter(pl.col("class") != "IfcOpeningElement")
bdf = bimpolars.simple_features(bdf.with_frame(elements))
bdf.frame.schema  # + footprint_ref: String, footprint: Binary (ISO WKB)
bdf.write_parquet("sixty5.parquet")  # GeoParquet: footprint is the primary geometry
bimpolars.read_parquet("sixty5.parquet").equals(bdf)  # True, CRS included

import geopandas

geopandas.read_parquet("sixty5.parquet", columns=["id", "class", "footprint"]).crs
# <Engineering CRS: {"$schema": ...}>  Name: MM16169 model coordinates, millimetre axes
```

## Install

```sh
uv add bimpolars  # or: pip install bimpolars
```

The core depends on polars, numpy, shapely and ifcopenshell. Georeferencing (pyproj) and
Speckle (specklepy) will come as the extras `[geobim]` and `[speckle]`.

## Functions

- `read_ifc(source, *, threads=None)`: reads every product of an IFC into a `BimDataFrame`,
  tessellated by ifcopenshell in the model's own coordinates and length unit. The columns:
  - `id`, `name`, `class` (Categorical IFC class). Every `IfcProduct` is a row, openings
    included: an `IfcOpeningElement` row carries the solid mesh of the void it cuts, so filter
    `class != "IfcOpeningElement"` before outlining or rendering elements.
  - `properties`: a Struct of property and quantity sets (the type's included), each a Struct
    of values. A property whose type differs between rows becomes a String.
  - `container`: the `id` of the spatial element the row sits in (the storey for most
    elements, the building for a storey; null for the site, for elements in no container and
    for ports); `elevation`: the storey's level in model coordinates.
  - `native_geom`: null for now; `mesh`: `{vertices, faces, styles}` with flat vertices, faces
    sorted by style, and each style's `name`, `rgba` and `face_count`; `bbox`:
    `{xmin, ymin, zmin, xmax, ymax, zmax}`. Rows without geometry have a null `mesh` and `bbox`.
- `simple_features(bdf)`: adds the plan footprint of every row with a mesh, in two columns:
  - `footprint`: the outline of the mesh in plan, the union of its triangles projected to the
    XY plane, as `POLYGON Z` or `MULTIPOLYGON Z` in ISO WKB (`Binary`), lying flat at the
    level of the `footprint_ref` row. Holes and separate parts survive. A mesh with no plan
    area (all triangles vertical) gets `POLYGON Z EMPTY`; a row without a mesh gets null.
    Every mesh is outlined, an `IfcOpeningElement`'s void included: filter openings out first
    for a plan of the elements.
  - `footprint_ref`: the `id` of the row whose level the footprint lies at: the row's
    `container` when that has an `elevation` (the storey, for most elements), else the row
    itself at the lowest z of its mesh.
  Every row's `container` must be in the frame: filter the elements, but keep the spatial
  rows, or `simple_features` raises. Decode the column with
  `shapely.from_wkb(bdf.frame["footprint"].to_numpy())`.
- `plan_outline(vertices, faces)`: the plan outline of one triangle mesh as a 2D shapely
  Polygon or MultiPolygon, the union of its triangles projected to XY, for a mesh outside a
  frame. Slivers (triangles with less plan area than 1e-9 of the extent squared) are left
  out.
- `BimDataFrame(frame, crs)`: the thin wrapper with `.frame`, `.crs`, `.with_frame()`,
  `.equals()`, `.write_parquet()` and `BimDataFrame.read_parquet()`.
- `read_parquet(source)`: reads a `BimDataFrame` back, CRS included. The CRS sits under the
  `bimpolars` key of the Parquet file metadata as `{"version": 1, "crs": <PROJJSON>}`.
- `engineering_crs(name, *, unit, datum)`: a plain EngineeringCRS as PROJJSON in a given
  length unit (a known name or a `LinearUnit(name, conversion_factor)`), which a model without
  georeferencing gets. `linear_unit(crs)` reads the unit back.
- `properties_series(rows)`: the Struct column of property sets that `read_ifc` uses, with the
  dtype per property inferred across rows, for any reader with named sets of values.
- `bimpolars.data.fetch(key, *, download=True)`: the path of a test or example model, downloaded
  from `http://r2.eu.bedrock.engineer/bimpolars/` on first use into `bimpolars.data.data_dir()`:
  `BIMPOLARS_DATA_DIR`, else `$XDG_CACHE_HOME/bimpolars`, else `~/.cache/bimpolars`.

## GeoParquet

`write_parquet` on a frame with a `footprint` column writes
[GeoParquet 1.1](https://geoparquet.org): the file metadata gets a `geo` entry
(`bimpolars.geoparquet.geo_metadata`) with `footprint` as the primary geometry column, WKB
encoded, its geometry types, the model CRS as PROJJSON and, when the frame has its `bbox`
column, that column as the `covering` of the footprints (its `xmin`, `ymin`, `xmax` and `ymax`
fields; `zmin` and `zmax` describe the mesh). Readers that understand GeoParquet then find the
footprints with their CRS, and can skip row groups with the bbox:

```python
geopandas.read_parquet("sixty5.parquet", columns=["id", "footprint"], bbox=(0, 0, 10_000, 10_000))
```

```sql
LOAD spatial;  -- DuckDB
SELECT id, ST_Area(footprint) FROM read_parquet('sixty5.parquet') WHERE bbox.xmax < 10000;
```

`mesh` and `native_geom` are not GeoParquet geometries; they share the model CRS implicitly.

## Develop

```sh
uv sync
uv run pytest            # BIMPOLARS_DATA_DIR=/path/to/mirror to use local test data
uv run pytest -m slow    # only the slow tests, such as the 343 MB architectural model
uv run mypy src
uv run ruff check && uv run ruff format --check
```

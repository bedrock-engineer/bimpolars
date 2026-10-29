# bimpolars

Building Information Models in polars DataFrames with geometry columns. A `BimDataFrame` is a
polars DataFrame with one row per element and one model CRS as PROJJSON (`src/bimpolars/frame.py`).
Readers fill the columns in `MODEL_COLUMNS`; `read_ifc` (`src/bimpolars/ifc.py`) is the first one.
`simple_features` (`src/bimpolars/features.py`) adds the `FEATURE_COLUMNS` `footprint_ref` and
`footprint`; `src/bimpolars/geoparquet.py` builds the `geo` metadata that makes the written
Parquet GeoParquet.

## Rules

- `uv` project with hatchling, `src/bimpolars/` layout, `py.typed`. Python >= 3.12.
- `uv run pytest`, `uv run mypy src`, `uv run ruff check` and `uv run ruff format --check` must
  pass. Tests that read a model of hundreds of MB are marked `slow` and run only with
  `uv run pytest -m slow`.
- Typed code with Google-style docstrings (ruff enforces `D` with the google convention).
- The core imports neither pyproj nor specklepy. The plan: #6 adds the extra `[geobim]`
  (pyproj) and #10 the extra `[speckle]` (specklepy), each in its own module. Until then
  pyproj is a dev dependency only, used to check the PROJJSON. CRSs are PROJJSON dicts
  (`src/bimpolars/crs.py`) so the core can build and store them without pyproj. shapely is a
  core dependency (the footprints need it).
- Geometry stays in the model's own coordinates and length unit (millimetres for most IFCs);
  the CRS carries the unit. Never convert to metres silently.
- Rows are every `IfcProduct`. On the structural model 1,284 of the 2,647 element rows are
  `IfcOpeningElement` with the solid mesh of the void, so glTF (#8) and LightBIM (#11) filter
  `class != "IfcOpeningElement"`, and a caller filters them out before `simple_features` for
  a plan of the elements (`simple_features` outlines every mesh, openings included, because
  GeoParquet needs `footprint` null exactly when `bbox` is). Such a filter keeps the spatial
  rows: `simple_features` raises when a row's `container` has no row in the frame, rather
  than silently placing the footprint at the element's own level. `container` is null for
  `IfcDistributionPort` rows (ports hang off their element, not a spatial element).
- The inference of property dtypes across rows lives in `src/bimpolars/properties.py`
  (`properties_series`), for `read_speckle` (#9) to reuse.
- Simple-feature columns (`footprint`, later `geog` and the LightBIM geometry) are ISO WKB with
  Z in a `Binary` column, so that GeoParquet readers (DuckDB spatial, geopandas, GDAL) take them
  as they are; shapely decodes them. Meshes stay nested columns. A geometry column that is
  listed in `geoparquet.GEOMETRY_COLUMNS` gets `geo` metadata on write, the first present one
  as the primary; `bbox` is the covering (x and y only: its z range is the mesh's).
- A row's `footprint` is null exactly when its `mesh` is null; a mesh without plan area gets
  `POLYGON Z EMPTY`. The footprint's bounds stay inside the row's `bbox`.
- Keep the README's function list current when a public function is added.

## Test data

Models live in the bucket `http://r2.eu.bedrock.engineer/bimpolars/` and are read through
`bimpolars.data.fetch(key)`, which looks in `bimpolars.data.data_dir()` first (the environment
variable `BIMPOLARS_DATA_DIR`, else `$XDG_CACHE_HOME/bimpolars`, else `~/.cache/bimpolars`)
and downloads only what is missing.
Set `BIMPOLARS_DATA_DIR` to a local mirror with the same keys when the bucket is unreachable:

```sh
BIMPOLARS_DATA_DIR=/path/to/mirror uv run pytest
```

Keys and the known numbers for assertions (ifcopenshell 0.9.0.post1):

- `sixty5_O-S1-CON-structural-engineering.ifc`: 2687 IfcProduct = 2647 elements (1284 openings)
  + 21 spatial (site, building, 19 storeys) + 19 grids; 199,578 triangles; 15 styles (14
  IfcSurfaceStyle names of the 16 in the file, plus faces without a style); IFC2X3 in mm.
  Footprints: 2644 `Polygon Z` + 3 `MultiPolygon Z` (openings included), none empty, every
  element in a storey with an elevation; the `kolom vierkant_gen_lxb 500x800` columns have a
  plan area of 400,000.
- `sixty5_O-S1-BWK-BIM-prefab-facade.ifc`: 1055 elements, 18 storeys; 1054 `Polygon Z` + 1
  `MultiPolygon Z` (a console whose two parts touch along an edge, up to rounding).
- `sixty5_O-S1-BWK-BIM-architectural.ifc`: 343 MB, `slow`. ArchiCAD export: 16,784 products
  (16,760 with a mesh), 1,671,713 triangles, 191 property sets (one per library part) with
  35,068 distinct properties. The dense `properties` Struct is 35,068 columns of 16,784 rows,
  about 5.5 GB in memory (String columns cost 16 bytes per row whether null or not), so
  `read_ifc` peaks near 8 GB and takes two minutes; without that column it needs 2.5 GB and
  85 s. Select the columns you need before filtering such a frame. See #15 before changing
  the layout of `properties`. Footprints take 40 s once `properties` is dropped: 15,697
  `Polygon Z`, 1063 `MultiPolygon Z`, 3 empty (two 2-triangle railings and a stair landing).
  The sliver rule of `plan_outline` (triangles under 1e-9 of the extent squared are left
  out) decides 25 of these: without it GEOS's robustness fallback leaves hairline gaps in
  24 door and furniture outlines, and the Jordahl console's two parts become one Polygon.
- `rvt-advanced-sample-arch.ifc`: 5696 elements, 5 storeys.
- `rvt-advanced-sample/speckle-v3-objects.parquet` and
  `rvt-advanced-sample/speckle-bundle/<model id>/<version id>/*.parquet`: Speckle data for
  `read_speckle` (#9, not written yet).

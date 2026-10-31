# The frame: rows and columns

A `BimDataFrame` (`src/bimpolars/frame.py`) is a polars DataFrame with one row per element and
the BIM CRS of each row in its `bim_crs` column, parsed as PROJJSON (`bdf.projjson`) and as a
`pyproj.CRS` (`bdf.crs`). The readers fill the columns in `MODEL_COLUMNS`, `bim_crs` last;
`simple_features` adds the `FEATURE_COLUMNS` ([features.md](features.md)), `lightbim=True`
the `LIGHTBIM_COLUMNS` ([lightbim.md](lightbim.md)) and `geobim.with_geog` the `geog` column
([geobim.md](geobim.md)).

## The bim_crs column

- `bim_crs` (`frame.CRS_COLUMN`) is the one place the CRS lives (#42): the wrapper keeps none.
  It holds compact PROJJSON text (`frame.crs_text`) and covers `mesh`, `bbox`, `native_geom`,
  `elevation`, `footprint` and `lightbim`, never `geog` (OGC:CRS84h).
- Its dtype is `pl.Categorical` (`frame.CRS_DTYPE`); the constructor casts a String column.
  A String column read back from Parquet holds a copy per row: 428 MB for 200,000 rows of
  EPSG:7415's 2.2 kB PROJJSON, against 2.3 MB as a Categorical, and its `unique()` is about
  25 times slower.
- The constructor checks that pyproj reads every value (`frame._crs_of`, which parses a text
  once per process), so a display never meets text that is no CRS.
- `bdf.projjson` and `bdf.crs` read the column on every call, with no cache on the wrapper:
  an in-place polars method (`extend`, `insert_column`) changes the frame under it. Values
  that pyproj reads as exactly the same CRS (`is_exact_same`, not `==`, which calls two
  engineering CRSs that differ only by name equal) count as one. They raise a ValueError that
  names the CRSs when the column holds several, has nulls, or the frame has no rows (a filter
  that matches nothing). Code that needs the CRS reads them, and so refuses a mixed frame;
  code that only carries rows (`simple_features`) does not, and a frame without rows goes
  through `to_crs`, `with_geog` and `write_parquet` (with a null GeoParquet `crs`) unchanged.
- Code that builds a frame keeps `bim_crs` and wraps it with `BimDataFrame(frame)`;
  `frame.with_crs_column` puts one CRS on every row (the constructor, `set_crs`, `to_crs`),
  after the last model column when the frame has none yet, so a layout-1 file and a fresh
  model get one column order. `concat` stacks models and keeps each row's CRS.
- `__getattr__` and `__getitem__` hand polars through and wrap a DataFrame result whose
  `bim_crs` is still Categorical (`null_count`, `describe` and `implode` turn it into
  something else and come back as polars); `__contains__` and `__iter__` go over the
  columns, like polars; `__arrow_c_stream__` serves `pl.DataFrame(bdf)`, pyarrow and DuckDB.
  The public functions take a polars DataFrame with `bim_crs` too (`frame.as_bim_dataframe`),
  because `pipe` and a `group_by` loop hand out plain frames.
  marimo picks a table by `isinstance` or narwhals and ignores the Arrow stream, so
  `__narwhals_dataframe__` hands narwhals the compliant frame of `nw.from_native(frame)`.
  narwhals has no public way for a wrapper to do that, so it reads the private
  `_compliant_frame`; `test_narwhals_and_marimo_tables_read_the_frame` pins it. A cell whose
  output is `bdf` shows `_display_`: the CRS name above a table whose `format_mapping` shows
  `bim_crs` as CRS names and the WKB columns as WKT (`repr` and `_repr_html_` show the names
  too); the values stay PROJJSON and WKB, so a selection wraps again with `BimDataFrame`.
- Parquet layout 2: the `bimpolars` metadata entry is `{"version": 2}` and the CRS is in the
  column; a file of layout 1 (`{"version": 1, "crs": ...}`) gets the column on read.

## Rows

Both readers give one row per element: `read_ifc` a row per `IfcProduct`, `read_speckle` a row
per element of the version and of its bundle ([speckle.md](speckle.md)).

On the structural model 1,284 of the 2,647 element rows are `IfcOpeningElement` with the solid
mesh of the void, so glTF and LightBIM filter `class != "IfcOpeningElement"`, and a caller
filters them out before `simple_features` for a plan of the elements (`simple_features`
outlines every mesh, openings included, because GeoParquet needs `footprint` null exactly when
`bbox` is). Such a filter keeps the spatial rows: `simple_features` raises when a row's
`container` has no row in the frame, rather than silently placing the footprint at the
element's own level. `container` is null for `IfcDistributionPort` rows (ports hang off their
element, not a spatial element).

## Properties

The inference of property dtypes across rows lives in `src/bimpolars/properties.py`
(`properties_series`); `read_ifc` and `read_speckle` both use it. The dense `properties` Struct
of a model with many property sets costs gigabytes (the architectural model in
[test-data.md](test-data.md)): see #15 before changing its layout.

## Meshes

`src/bimpolars/mesh.py` holds the `Mesh` the readers end with, `Style`, `styled_mesh` (faces
sorted by style), `mesh_frame`, which builds the `mesh` and `bbox` columns from it, and
`mesh_arrays`, which reads a `mesh` column back into flat arrays with per-row offsets. Code that
reads a `mesh` column uses `mesh_arrays` (#18: features and lightbim do; geobim, gltf and
speckle still walk the column themselves).

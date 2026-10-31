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
- `bdf.projjson` and `bdf.crs` parse the column's one value and cache it on the wrapper. They
  raise a ValueError that names the CRSs when the column holds several, has nulls, or the
  frame has no rows (a filter that matches nothing). Code that needs the CRS reads them, and
  so refuses a mixed frame; code that only carries rows (`simple_features`) does not.
- Code that builds a frame keeps `bim_crs` and wraps it with `BimDataFrame(frame)`;
  `geobim._with_crs` puts one CRS on every row (`set_crs`, `to_crs`). `concat` stacks models
  and keeps each row's CRS.
- `__getattr__` and `__getitem__` hand polars through and wrap a DataFrame result that kept
  `bim_crs`; `__arrow_c_stream__` serves `pl.DataFrame(bdf)`, pyarrow and DuckDB.
  `mo.ui.table(bdf)` still fails: marimo picks a table by `isinstance` or narwhals, and
  neither reads the Arrow stream, so pass `bdf.frame` there. A cell whose output is `bdf`
  shows `_display_`: the CRS name above a table without `bim_crs`.
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

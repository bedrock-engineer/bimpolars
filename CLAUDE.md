# bimpolars

Building Information Models in polars DataFrames with geometry columns. A `BimDataFrame` is a
polars DataFrame with one row per element and one model CRS as PROJJSON (`src/bimpolars/frame.py`).
Readers fill the columns in `MODEL_COLUMNS`; `read_ifc` (`src/bimpolars/ifc.py`) is the first one.

## Rules

- `uv` project with hatchling, `src/bimpolars/` layout, `py.typed`. Python >= 3.12.
- `uv run pytest`, `uv run mypy src`, `uv run ruff check` and `uv run ruff format --check` must
  pass. Tests that read a model of hundreds of MB are marked `slow` and run only with
  `uv run pytest -m slow`.
- Typed code with Google-style docstrings (ruff enforces `D` with the google convention).
- The core imports neither pyproj nor specklepy. The plan: #6 adds the extra `[geobim]`
  (pyproj, shapely) and #10 the extra `[speckle]` (specklepy), each in its own module. Until
  then pyproj is a dev dependency only, used to check the PROJJSON. CRSs are PROJJSON dicts
  (`src/bimpolars/crs.py`) so the core can build and store them without pyproj.
- Geometry stays in the model's own coordinates and length unit (millimetres for most IFCs);
  the CRS carries the unit. Never convert to metres silently.
- Rows are every `IfcProduct`. On the structural model 1,284 of the 2,647 element rows are
  `IfcOpeningElement` with the solid mesh of the void, so footprints (#5), glTF (#8) and
  LightBIM (#11) filter `class != "IfcOpeningElement"`. `container` is null for
  `IfcDistributionPort` rows (ports hang off their element, not a spatial element).
- The inference of property dtypes across rows lives in `src/bimpolars/properties.py`
  (`properties_series`), for `read_speckle` (#9) to reuse.
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
- `sixty5_O-S1-BWK-BIM-prefab-facade.ifc`: 1055 elements, 18 storeys.
- `sixty5_O-S1-BWK-BIM-architectural.ifc`: 343 MB, `slow`. ArchiCAD export: 16,784 products,
  1.67 M triangles, 191 property sets (one per library part) with 35,068 distinct properties,
  so the dense `properties` Struct alone takes about 2.7 GB in memory; `read_ifc` needs about
  4 GB and two minutes.
- `rvt-advanced-sample-arch.ifc`: 5696 elements, 5 storeys.
- `rvt-advanced-sample/speckle-v3-objects.parquet` and
  `rvt-advanced-sample/speckle-bundle/<model id>/<version id>/*.parquet`: Speckle data for
  `read_speckle` (#9, not written yet).

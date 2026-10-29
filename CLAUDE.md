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
- The core imports neither pyproj nor specklepy. They belong to the extras `[geobim]` (pyproj,
  shapely) and `[speckle]` (specklepy), each in its own module. CRSs are PROJJSON dicts
  (`src/bimpolars/crs.py`) so the core can build and store them without pyproj.
- Geometry stays in the model's own coordinates and length unit (millimetres for most IFCs);
  the CRS carries the unit. Never convert to metres silently.
- Keep the README's function list current when a public function is added.

## Test data

Models live in the bucket `http://r2.eu.bedrock.engineer/bimpolars/` and are read through
`bimpolars.data.fetch(key)`, which looks in the folder named by the environment variable
`BIMPOLARS_DATA_DIR` first (default `~/.cache/bimpolars`) and downloads only what is missing.
Set `BIMPOLARS_DATA_DIR` to a local mirror with the same keys when the bucket is unreachable:

```sh
BIMPOLARS_DATA_DIR=/path/to/mirror uv run pytest
```

Keys and the known numbers for assertions (ifcopenshell 0.9.0.post1):

- `sixty5_O-S1-CON-structural-engineering.ifc`: 2687 IfcProduct = 2647 elements (1284 openings)
  + 21 spatial (site, building, 19 storeys) + 19 grids; 199,578 triangles; 15 styles (14
  IfcSurfaceStyle names of the 16 in the file, plus faces without a style); IFC2X3 in mm.
- `sixty5_O-S1-BWK-BIM-prefab-facade.ifc`: 1055 elements, 18 storeys.
- `sixty5_O-S1-BWK-BIM-architectural.ifc`: 343 MB, `slow`.
- `rvt-advanced-sample-arch.ifc`: 5696 elements, 5 storeys.
- `rvt-advanced-sample/speckle-v3-objects.parquet` and
  `rvt-advanced-sample/speckle-bundle/<model id>/<version id>/*.parquet`: Speckle data for
  `read_speckle`.

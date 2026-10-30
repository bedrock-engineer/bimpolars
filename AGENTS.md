# bimpolars

Building Information Models in polars DataFrames with geometry columns: a `BimDataFrame` has
one row per element and one model CRS as PROJJSON. Readers (`read_ifc`, `read_speckle`) build
it; `simple_features`, `bimpolars.geobim`, `bimpolars.gltf` and GeoParquet take it further. The
README lists the public functions: keep that list current when you add one.

## Checks

A `uv` project (hatchling, `src/bimpolars/` layout, `py.typed`, Python >= 3.12). These pass on
every commit:

```sh
uv run pytest
uv run mypy src
uv run ruff check
uv run ruff format --check
```

Tests that read a model of hundreds of MB are marked `slow` and run only with
`uv run pytest -m slow`.

## Rules

- Typed code with Google-style docstrings (ruff enforces `D` with the google convention).
- Geometry stays in the model's own coordinates and length unit (millimetres for most IFCs);
  the CRS carries the unit. Never convert to metres silently.
- The core imports neither ifcopenshell, pyproj nor specklepy. Each lives in one module behind
  an extra: ifcopenshell in `bimpolars.ifc` (`[ifc]`), which `bimpolars.__getattr__` imports
  when `read_ifc` is first looked up, so the other modules leave it out of their imports;
  pyproj in `bimpolars.geobim` (`[geobim]`); specklepy in `bimpolars.bundle` (`[speckle]`),
  while `bimpolars.speckle` reads v3 objects as plain JSON. CRSs are PROJJSON dicts
  (`src/bimpolars/crs.py`) so that the core builds and stores them without pyproj; shapely is
  a core dependency (the footprints need it). Three tests import the core in a subprocess with
  `sys.modules[...] = None` to keep it that way:
  `tests/test_ifc_synthetic.py::test_the_core_imports_without_ifcopenshell`,
  `tests/test_geobim.py::test_the_core_imports_without_pyproj` and
  `tests/test_speckle.py::test_the_core_imports_without_specklepy`.

## Docs

Read the doc of an area before you change its code or write its tests:

- [The frame](docs/agents/frame.md): rows and columns, openings and `container`, property
  dtypes, the `mesh` column.
- [read_ifc](docs/agents/ifc.md): the `attributes` set and the `native_geom` sweeps.
- [Speckle](docs/agents/speckle.md): `read_speckle`, bundles and `bundle_objects`.
- [Footprints and GeoParquet](docs/agents/features.md): `simple_features`, `plan_outline`, the
  WKB columns and the `geo` metadata.
- [Georeferencing](docs/agents/geobim.md): CRSs, PROJ transformations and the NSGI grids,
  `to_crs`, `geog`.
- [LightBIM](docs/agents/lightbim.md): the structural-analysis members.
- [glTF](docs/agents/gltf.md): the binary glTF writers.
- [Test data](docs/agents/test-data.md): the bucket, a local mirror when it is unreachable,
  and the known numbers of each model for assertions.

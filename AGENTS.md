# bimpolars

Building Information Models in polars DataFrames with geometry columns: a `BimDataFrame` has
one row per element and one model CRS, stored as PROJJSON (`bdf.projjson`) and handed out as a
`pyproj.CRS` (`bdf.crs`). Readers (`read_ifc`, `read_speckle`) build
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
- The core imports neither ifcopenshell nor specklepy. Each lives in one module behind an
  extra: ifcopenshell in `bimpolars.ifc` (`[ifc]`), which `bimpolars.__getattr__` imports when
  `read_ifc` is first looked up, so the other modules leave it out of their imports; specklepy
  in `bimpolars.bundle` (`[speckle]`), while `bimpolars.speckle` reads v3 objects as plain
  JSON. Two tests import the core in a subprocess with `sys.modules[...] = None` to keep it
  that way: `tests/test_ifc_synthetic.py::test_the_core_imports_without_ifcopenshell` and
  `tests/test_speckle.py::test_the_core_imports_without_specklepy`.
- pyproj and shapely are core dependencies (#31). A CRS goes in in any form pyproj reads
  (`crs.CRSLike`, read with `crs.resolve_crs`; a `unit=` also takes a BimDataFrame, through
  `crs.as_linear_unit`) and is stored as PROJJSON (`src/bimpolars/crs.py`), which the
  GeoParquet and glTF metadata carry: code that writes or compares the stored CRS uses
  `bdf.projjson`, code that asks PROJ uses `bdf.crs`. CRSs handed back to the user
  (`bdf.crs`, `site_crs`) are `pyproj.CRS`; the PROJJSON builders
  `engineering_crs` and `CRS84H` stay PROJJSON. The geobim functions and `bimpolars.data` are
  on the package itself, so `import bimpolars as bpl` is all a notebook needs.
- The Pages site (`.github/workflows/pages.yml`) is specklestuff's demo page (its `site/`),
  built from specklestuff's main with that repository's lock of bimpolars, so it shows the
  same page as specklestuff's Pages under bimpolars' header (the workflow's `SITE=bimpolars`
  picks specklestuff's `site/intro/bimpolars.html`); nothing else of it lives here. A page of bimpolars' own, or one built from this repository's
  main, needs that workflow to change.

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

# bimpolars

Building Information Models in polars DataFrames with geometry columns. A `BimDataFrame` is a
polars DataFrame with one row per element and one model CRS as PROJJSON (`src/bimpolars/frame.py`).
Readers fill the columns in `MODEL_COLUMNS`; `read_ifc` (`src/bimpolars/ifc.py`) is the first one.
`simple_features` (`src/bimpolars/features.py`) adds the `FEATURE_COLUMNS` `footprint_ref` and
`footprint`; `src/bimpolars/geoparquet.py` builds the `geo` metadata that makes the written
Parquet GeoParquet. `src/bimpolars/geobim.py` (the `[geobim]` extra, pyproj) georeferences a
model: `derived_engineering_crs`, `set_crs`, `to_crs`, `with_geog` (the `geog` column: the
footprints in OGC:CRS84h), `affine_transformation` and `linearized_site_to_ecef`.
`src/bimpolars/gltf.py` writes the meshes as binary glTF:
`element_meshes_glb` (one primitive per style) and `triangles_glb` (any triangles).
`src/bimpolars/speckle.py` reads a Speckle version from its v3 objects: `version_objects`
(Parquet or SQLite copy) and `read_speckle` (instances expanded, the bundle's categories,
parameters and levels joined by applicationId). `src/bimpolars/mesh.py` holds the `Mesh` the
readers end with and `mesh_frame`, which builds the `mesh` and `bbox` columns from it.

## Rules

- `uv` project with hatchling, `src/bimpolars/` layout, `py.typed`. Python >= 3.12.
- `uv run pytest`, `uv run mypy src`, `uv run ruff check` and `uv run ruff format --check` must
  pass. Tests that read a model of hundreds of MB are marked `slow` and run only with
  `uv run pytest -m slow`.
- Typed code with Google-style docstrings (ruff enforces `D` with the google convention).
- The core imports neither pyproj nor specklepy. pyproj lives in `bimpolars.geobim` behind
  the extra `[geobim]` (#6); #10 adds the extra `[speckle]` (specklepy) in its own module
  (`bimpolars.bundle`), while `bimpolars.speckle` (#9) reads v3 objects as plain JSON.
  `tests/test_geobim.py::test_the_core_imports_without_pyproj` and
  `tests/test_speckle.py::test_the_core_imports_without_specklepy` import the core in a
  subprocess with `sys.modules[...] = None` to keep it that way. CRSs are PROJJSON dicts
  (`src/bimpolars/crs.py`) so the core can build and store them without pyproj. shapely is a
  core dependency (the footprints need it).
- `to_crs` moves `mesh`, the WKB columns of `geobim.WKB_COLUMNS` (`footprint`, `lightbim`
  once #11 adds it; never `geog`, which is always OGC:CRS84h) and `elevation` (as a point at
  the plan centre of the model), recomputes `bbox` from the moved meshes (from the eight
  moved corners of the old `bbox` when the frame has no `mesh` column), and keeps the
  footprints inside their `bbox` with `geobim._clamp_xy`, which clips x and y and keeps z
  (`shapely.transform` drops z unless `include_z=True`). It nulls `native_geom` when the
  transformation is not affine over the model (`affine_transformation` gives the 4x4 matrix
  when it is; #11 uses it to place its sweeps and replaces the `NotImplementedError` for a
  filled column). Targets must have three axes (`EPSG:7415`, never `EPSG:28992`): the model's
  unit becomes the target's, and a model stays in one unit per axis.
- PROJ applies the deriving conversion of a DerivedVerticalCRS after the geodetic step of the
  horizontal part, so a 3D Helmert transformation (the fallback without grids) reads a
  millimetre site's heights as metres: 0.7 m of horizontal error at 50 m height, and an ECEF
  linearization tilted by 0.8 degrees. `geobim._transformer` therefore goes through the base
  CRS of a derived compound CRS; keep every transformation on that path.
- Without the NSGI grids (`nl_nsgi_rdtrans2018.tif`, `nl_nsgi_nlgeo2018.tif`; cdn.proj.org is
  blocked from the cloud containers, and the PROJ network is off) PROJ falls back to a Helmert
  transformation that is good to about a metre and a ballpark vertical step that passes NAP
  heights off as ellipsoidal heights. `geobim._transformer` warns (a `UserWarning` pointing
  at the caller) when `TransformerGroup(...).best_available` is False; the `description` of
  a lazy `Transformer` does not say which operation runs. A test of a transformation to
  ETRS89 must state that tolerance or check the group.
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
  (`properties_series`); `read_ifc` and `read_speckle` both use it.
- `read_speckle` rows are the objects under the root's `elements` (collections traversed;
  definition content, which specklepy names `def-geo-<geometry index>` and
  `nested-inst-<n>`, left out), keyed by `applicationId` with `-instance-N` stripped, so a
  placement folds into its element's row and the same mesh (listed as a Mesh and as the
  DataObject that displays it) counts once. `class` = the object's `category`, else the
  bundle's category, else `speckle_type`. The property set of a value is the dotted path of
  its group (`properties`, `properties.Parameters.Instance Parameters.Dimensions`, and
  `object` for the object's own fields), the same for the object's bag and the bundle's eav
  `path`; an eav row with a number and its text gives the number. Levels become rows
  (`level-<node id>`, class `Levels`, `elevation` in model units) and `container` points at
  them, so `simple_features` works on the result. Styles: the root's `renderMaterialProxies`
  (first proxy that targets the mesh, the element or the instance wins), then the envelope's
  `HAS_MATERIAL` for definition geometry. The CRS is the root's `units`, never the Revit
  geolocation. No specklepy: the SGEO geometry of a bundle is not decoded here (#10).
- `plan_outline` retries a union that GEOS's floating-point overlay fails on (one MEP stair
  mesh of 1376 faces does) with `grid_size` = the sliver fraction times the extent, which is
  the fixed-precision overlay that cannot fail.
- Simple-feature columns (`footprint`, `geog` and later the LightBIM geometry) are ISO WKB with
  Z in a `Binary` column, so that GeoParquet readers (DuckDB spatial, geopandas, GDAL) take them
  as they are; shapely decodes them. Meshes stay nested columns. A geometry column that is
  listed in `geoparquet.GEOMETRY_COLUMNS` (`geog` before `footprint`) gets `geo` metadata on
  write, the first present one as the primary. `bbox` is the covering of the columns in the
  model CRS: the `geo` entry points readers at the column's x and y fields, which every
  footprint lies within, so that a spatial filter can skip row groups. A GeoParquet covering
  has no z slot, and the column's `zmin..zmax` is the mesh's, which a footprint flat at its
  storey's elevation often lies outside of; the column keeps all six.
- `geog` (`geobim.with_geog`) is `footprint` through `geobim._transformer` to OGC:CRS84h
  (`crs.CRS84H`, the PROJJSON written out so that the core can label the column), null exactly
  when `footprint` is, `POLYGON Z EMPTY` where the footprint is; `to_crs` never moves it
  (`geoparquet.COLUMN_CRS` lists the columns with a CRS of their own, `geobim.WKB_COLUMNS`
  leaves them out). Its `geo` entry has `CRS84H` as the CRS, the bounds of its own
  coordinates as the file `bbox` (decoded with shapely on write) and no covering, since
  `bbox` is in the model CRS; geopandas's `bbox=` filter then refuses the file (it needs a
  covering on the primary column). pyogrio writes a CRS as WKT1, which cannot say a 3D
  geographic CRS, so `GeoDataFrame.to_file` on a `geog` frame needs
  `set_crs("OGC:CRS84", allow_override=True)` first; `to_json()` works as it is. The Sixty5
  site origin is at 5.46015 E, 51.44648 N (Strijp-S); the model lies within 0.002 degrees.
- A row's `footprint` is null exactly when its `mesh` is null; a mesh without plan area gets
  `POLYGON Z EMPTY`. The footprint's bounds stay inside the row's `bbox`.
- glTF (`gltf.py`) is hand-written glb 2.0, no glTF library in the core: vertices in metres
  (scaled by the CRS unit) and z up in the accessors, a root node with `Y_UP_ROTATION` (named
  after the CRS; `extras.crs` is the model CRS in its own unit and `extras.unit` says the data
  is in metres) that makes the scene y-up as the spec wants, and a child node translated to
  the centre of the written vertices' bounding box (not the `bbox` column: openings and
  meshless rows are left out) holding the mesh so that float32 stays precise far from the
  origin. One primitive and material per `(name, rgba)` style, sorted by name with
  the unstyled last; `IfcOpeningElement` rows left out. `normals=True` (default) writes
  non-indexed triangles with flat normals (3 vertices per face: ifcopenshell welds vertices
  by position, so averaged vertex normals would shade a box smooth); `normals=False` writes
  the `mesh` column as it is, indexed, without normals. The tests read the glb back with
  trimesh (dev dependency) and with a small reader of the chunks and accessors.
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
  Georeferencing (IFC-GeoReferencer export of the architectural model, EPSG:7415): the site
  origin at E 160071.5, N 384156.5, NAP +18.625 (AHN4 at the origin), the x-axis
  (0.6379617540893054, -0.7700680491484481) in the grid, so the rotation is -50.36 degrees;
  `to_crs` to EPSG:7415 takes 0.2 s and puts the model within 100 m of the origin between NAP
  -11.8 and +75.3 m, the 146 pile heads (tip + Depth along the extrusion, as in
  `sixty5_piles.py`) within 0.2 mm in plan of the centre of their moved `bbox` and at its
  `zmax`. glTF: the 1363 element rows that are not openings hold 89,740 vertices, 177,130
  faces and 12 distinct `(name, rgba)` styles, so `element_meshes_glb` writes 12 primitives
  with 531,390 flat-shaded vertices (12.8 MB, 0.6 s; 89,740 vertices and 3.2 MB with
  `normals=False`).
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
- `rvt-advanced-sample-arch.ifc`: 5696 elements, 5 storeys; 5734 rows (30 ports with a null
  `container`, 8 spatial), extent x -107,474..211,591, y -53,399..98,364, z -1829..12,485 mm
  in shared coordinates (the survey point), 49 s to read.
- `rvt-advanced-sample/speckle-v3-objects.parquet` (`data.RVT_SPECKLE_V3_OBJECTS`: arch
  12,915 objects, str 2291, mep 16,139; each version's root first, all `projected` from the
  bundles with specklepy 2026.9.1) and the bundles
  `rvt-advanced-sample/speckle-bundle/<model id>/<version id>/<name>.<table>.parquet`
  (`data.RVT_SPECKLE_ARCH/STR/MEP`, 18 tables each, `data.fetch_bundle`). `read_speckle` with
  the bundle, all in mm at the project base point: arch 5766 rows = 5761 elements (321 with
  geometry of their own, 5440 instances, all 5440 with a category; 18 elements have several
  placements, a stair 22) + 5 levels (0, 3800, 7600, 11,400, 12,000), 5501 meshes, 372,804
  vertices, 251,301 faces, 41 styles (174,063 faces styled), 96 property sets with 633 fields
  (7.8 MB), 1.6 s; the IFC export sits 9201.999 mm east and 22,635.061 mm north of it, and
  after that shift the plan extents agree to the micrometre (z max differs 89 mm: a roof
  without IFC geometry) and of 5347 elements matched by IfcGUID 85 % agree within 1 mm, 97 %
  within 10 mm, the worst wall by 430 mm. str: 922 rows (918 + 4 levels), only 77 meshes
  (841 instances of empty definitions, #10), 34,882 faces, 4 styles. mep: 7298 rows (7294 +
  4 levels), 6932 meshes, 419,305 faces, 86 sets with 1016 fields (12.4 MB), 3.8 s;
  `simple_features` 6 s.

# bimpolars

Building Information Models in polars DataFrames with geometry columns. A `BimDataFrame` is a polars
DataFrame with one row per element and one model CRS as PROJJSON (`src/bimpolars/frame.py`). Readers
fill the columns in `MODEL_COLUMNS`; `read_ifc` (`src/bimpolars/ifc.py`) is the first one.
`simple_features` (`src/bimpolars/features.py`) adds the `FEATURE_COLUMNS` `footprint_ref` and
`footprint`, and with `lightbim=True` the `LIGHTBIM_COLUMNS` of `src/bimpolars/lightbim.py`
(`lightbim`, `lightbim_kind`, `lightbim_qa`); `src/bimpolars/geoparquet.py` builds the `geo`
metadata that makes the written Parquet GeoParquet. `src/bimpolars/geobim.py` (the `[geobim]` extra,
pyproj) georeferences a model: `derived_engineering_crs`, `set_crs`, `to_crs`, `with_geog` (the
`geog` column: the footprints in OGC:CRS84h), `affine_transformation` and
`linearized_site_to_ecef`. `src/bimpolars/gltf.py` writes the meshes as binary glTF:
`element_meshes_glb` (one primitive per style) and `triangles_glb` (any triangles).
`src/bimpolars/speckle.py` reads a Speckle version from its v3 objects: `version_objects` (Parquet
or SQLite copy) and `read_speckle` (instances expanded, the bundle's categories, parameters and
levels joined by applicationId). `src/bimpolars/mesh.py` holds the `Mesh` the readers end with,
`Style`, `styled_mesh` (faces sorted by style), `mesh_frame`, which builds the `mesh` and `bbox`
columns from it, and `mesh_arrays`, which reads a `mesh` column back into flat arrays with per-row
offsets (#18: features and lightbim use it; geobim, gltf and speckle still walk the column
themselves).

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
- `to_crs` moves `mesh`, the WKB columns of `geobim.WKB_COLUMNS` (`footprint`, `lightbim`;
  never `geog`, which is always OGC:CRS84h) and `elevation` (as a point at the plan centre of
  the model), recomputes `bbox` from the moved meshes (from the eight moved corners of the
  old `bbox` when the frame has no `mesh` column), and keeps the WKB columns inside their
  `bbox` with `features._clamp_xy`, which clips x and y and keeps z (`shapely.transform`
  drops z unless `include_z=True`). It nulls `native_geom` when the transformation is not
  affine over the model, and otherwise composes the 4x4 matrix of `affine_transformation`
  onto each sweep's `placement` (`geobim._place_sweeps`): the matrix carries the unit scale,
  so the profile and the depth stay in the sweep's own unit. `lightbim_qa.deviation` is a
  length: `geobim._scale_deviations` scales it by the matrix's cube-root determinant, or
  nulls it when there is no matrix. Targets must have three axes
  (`EPSG:7415`, never `EPSG:28992`): the model's unit becomes the target's, and a model stays
  in one unit per axis.
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
  (`properties_series`); `read_ifc` and `read_speckle` both use it. `read_ifc` adds the set
  `properties.ATTRIBUTES_PSET` (`attributes`) with the `PredefinedType` of
  `ifcopenshell.util.element.get_predefined_type` (the type's, `ObjectType` for USERDEFINED),
  because the LightBIM class table keys `IfcSlab.BASESLAB` (a raft) on it.
- `native_geom` (`frame.NATIVE_GEOM_DTYPE`, a List of `SWEEP_DTYPE`) holds the
  `IfcExtrudedAreaSolid` body items `read_ifc` can read: mapped items resolved recursively, a
  DIFFERENCE or INTERSECTION `IfcBooleanResult` reduced to its first operand (`clipped`), a
  UNION to both operands, the profile as 2D ISO WKB in the sweep's XY plane (rectangle and
  hollow, circle and hollow, I-shape and asymmetric I-shape in IFC2X3's and IFC4's attribute
  names, arbitrary closed with voids; polylines, composite curves, trimmed circles and lines,
  indexed poly curves; arcs at 11.25 degrees as the notebook did), `placement` row-major from
  sweep to model coordinates. Other items (breps, revolved and tapered sweeps) and profiles it
  cannot outline are skipped, so a list can cover part of a body; null without any.
  `read_speckle` leaves it Null.
- LightBIM (`lightbim.lightbim_features`, #11): the member comes from the caller's `kind`
  expression, then `lightbim.CLASSES` (IFC class, `Class.PREDEFINEDTYPE`, Revit category; a None
  entry means no member: openings, spaces, reinforcement), then the shape test (tightest of a
  world box, a plan-PCA upright box and a 3D-PCA box, snapped within 5 degrees; plate test
  before line test, `SLENDERNESS` 4). The table's role holds only with the table's member (a
  `kind` member drops it), and only the table's supports turn into piles by shape: the caller's
  `kind` always wins. `load_bearing`: `LoadBearing` of a `Pset_` set, then of any set, then
  Revit's `Structural`, then the table. The axis or plane comes from the largest sweep of
  `native_geom` (curve: the extrusion through the profile centroid, unless the profile is a
  slender side view; surface: the extrusion is the normal when no longer than the profile's
  short side, else the profile's short axis), else from the box; the extent always comes from
  the mesh (the curve spans the vertices' projection, a surface is `plan_outline` of the mesh in
  the plane's own coordinates, so walls get holes). A support lies at `bbox.zmin`, a mass at the
  divergence-theorem centre of gravity (the box centre when that falls outside the box: open or
  inside-out meshes, 4 of them on the Revit samples). `shapely.area` and `is_valid` are 2D: a
  wall's vertical polygon has no plan area, so tests measure it in its own plane.
- `read_speckle` rows are the objects under the root's `elements` (collections traversed;
  definition content, which specklepy names `def-geo-<geometry index>` and
  `nested-inst-<n>`, left out), keyed by `applicationId` with `-instance-N` stripped, so a
  placement folds into its element's row and the same mesh (listed as a Mesh and as the
  DataObject that displays it) counts once. With a bundle, its `eav.objects` that the version
  has no object for (curtain walls, room separations, MEP systems, wires, circuits) get rows
  with a null `mesh`, so both readers give one row per element. `class` = the object's
  `category`, else the bundle's category, else `speckle_type` (null when the object has none).
  The property set of a value is the dotted path of its group (`properties`,
  `properties.Parameters.Instance Parameters.Dimensions`, and `object` for the object's own
  fields), the same for the object's bag and the bundle's eav `path`; an eav row with a number
  and its text gives the number; values keep the unit the eav `unit` column states (lengths
  mm, areas m², volumes m³), never scaled. Levels become rows (`level-<node id>`, class
  `Levels`, `elevation` in model units) and `container` points at them, so `simple_features`
  works on the result. Styles: the root's `renderMaterialProxies` (first proxy that targets
  the mesh, the element or the instance wins), then the envelope's `HAS_MATERIAL` for
  definition geometry. Every mesh of an element carries the element's applicationId, so a
  layered wall or ceiling gets its first layer's material on all faces (#20 is the per-layer
  follow-up through the envelope's `DISPLAY` order). The envelope's kinds and relations are
  looked up by name in `envelope.node_kinds` and `envelope.rel_types`, with schema 1.5's
  numbers as the fallback. The CRS is the root's `units`, else the bundle's, else the meshes'
  most common, else metres; never the Revit geolocation. No specklepy: the SGEO geometry of a
  bundle is not decoded here (#10).
- `plan_outline` retries a union that GEOS's floating-point overlay fails on (one MEP stair
  mesh of 720 faces does: `e1d4bdfb-2688-47c1-a386-3b48116e5fd2-0008cb48`, pinned in
  `test_mep`) with `grid_size` = the sliver fraction times the extent, which is the
  fixed-precision overlay that cannot fail.
- Simple-feature columns (`footprint`, `geog` and the LightBIM geometry `lightbim`) are ISO WKB with
  Z in a `Binary` column, so that GeoParquet readers (DuckDB spatial, geopandas, GDAL) take them
  as they are; shapely decodes them. Meshes stay nested columns. A geometry column that is
  listed in `geoparquet.GEOMETRY_COLUMNS` (`geog`, `footprint`, `lightbim`) gets `geo` metadata on
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
  `native_geom`: 2651 sweeps on 2630 rows (beams 350 of 354, columns 379 of 387 through mapped
  items, walls 296 of 296 + 0 of the 3 brep IfcWalls, slabs 69 of 70, footings 105 of 106 with
  106 sweeps, piles 146, openings 1284, the origin proxy 21); 31 clipped (11 beams, 20 walls).
  `attributes.PredefinedType`: slabs 21 BASESLAB + 49 FLOOR. LightBIM (1.3 s after the 4 s of
  the footprints): 1363 members (the openings, spatial rows and grids are null): 354 beams,
  387 columns and 146 piles as curves, 49 floors and 299 walls as surfaces (15 walls a
  MultiPolygon: cut through by an opening), 106 footings and 21 base slabs (rafts) as
  supports, the origin proxy a mass by shape. Deviations: curves below 700 (piles 300),
  surfaces 100..255, supports their height. 89 beams are steel Z sections 200..363 mm long
  extruded along their length with a 424 mm high section. ifcopenshell tessellates the round
  piles 599.51 x 599.76 mm, so their `bbox` centre is 0.24 mm off the circle's centre.
- `sixty5_O-S1-BWK-BIM-prefab-facade.ifc`: 1055 elements, 18 storeys; 1054 `Polygon Z` + 1
  `MultiPolygon Z` (a console whose two parts touch along an edge, up to rounding).
- `sixty5_O-S1-BWK-BIM-architectural.ifc`: 343 MB, `slow`. ArchiCAD export: 16,784 products
  (16,760 with a mesh), 1,671,713 triangles, 191 property sets (one per library part) with
  35,068 distinct properties, plus the `attributes` set. The dense `properties` Struct is
  35,068 columns of 16,784 rows, about 5.5 GB in memory (String columns cost 16 bytes per row
  whether null or not), so `read_ifc` peaks near 8 GB and takes two minutes (5.4 minutes in a
  4-CPU cloud container); without that column it needs 2.5 GB and 85 s. `native_geom`: 2694
  sweeps on 2681 rows (all 948 IfcWallStandardCase, 1143 of 4147 IfcCovering; the 6478
  IfcWall are breps), 1 MB, 12 s of the read; the predefined types 1.3 s. Select the
  columns you need before filtering such a frame. See #15 before changing the layout of
  `properties`. Footprints take 40 s once `properties` is dropped: 15,697
  `Polygon Z`, 1063 `MultiPolygon Z`, 3 empty (two 2-triangle railings and a stair landing).
  The sliver rule of `plan_outline` (triangles under 1e-9 of the extent squared are left
  out) decides 25 of these: without it GEOS's robustness fallback leaves hairline gaps in
  24 door and furniture outlines, and the Jordahl console's two parts become one Polygon.
- `rvt-advanced-sample-arch.ifc`: 5696 elements, 5 storeys; 5734 rows (30 ports with a null
  `container`, 8 spatial), extent x -107,474..211,591, y -53,399..98,364, z -1829..12,485 mm
  in shared coordinates (the survey point), 49 s to read.
- `rvt-advanced-sample/speckle-v3-objects.parquet` (`data.RVT_SPECKLE_V3_OBJECTS`, 16.7 MB,
  44,452 objects in four versions, each version's root first: arch `d5e5bacdde` 13,194
  objects, str `b66058b4d1` 5355, mep `4413abe507` 16,173, all `projected` from the bundles
  with specklepy 2026.9.1 so that an element whose definition is empty carries its own
  geometry; and `5cb009bb77`, `rvt-advanced-sample-arch.ifc` received by Speckle's IFC
  importer, 9730 objects in metres without a bundle) and the bundles
  `rvt-advanced-sample/speckle-bundle/<model id>/<version id>/<name>.<table>.parquet`
  (`data.RVT_SPECKLE_ARCH/STR/MEP`, 18 tables each, `data.fetch_bundle`). `read_speckle` with
  the bundle, all in mm at the project base point: arch 5877 rows = 5761 elements with an
  object (396 with geometry of their own, 5365 instances, all with a category from the
  bundle; 18 elements have several placements, a stair 22) + 111 elements the bundle alone
  knows (34 curtain walls, 76 room separations, a beam system; null `mesh`) + 5 levels (0,
  3800, 7600, 11,400, 12,000), 5568 meshes, 376,356 vertices, 253,971 faces, 42 styles
  (176,733 faces styled), 102 property sets with 656 fields (19 MB), 3 s; the IFC export sits
  9201.999 mm east and 22,635.061 mm north of it, and after that shift the plan extents agree
  to the micrometre (z max differs 89 mm: a roof without IFC geometry) and of 5412 elements
  matched by IfcGUID 85 % agree within 1 mm, 97 % within 10 mm, the worst by 322 mm. Eight
  parapet walls have both geometry of their own (the body, within 0.5 mm of the IFC) and a
  placement of a 76-face definition (the cap, a wall sweep, which overruns the body at the
  corners: the 322 mm); the other 67 arch, all 841 str and all 10 mep elements with own
  geometry and a placement point at an empty definition, so nothing is counted twice
  (`test_own_geometry_and_placements`). str: 939 rows (918 with an object, every one with a
  mesh, + 17 bundle-only + 4 levels), 68,714 faces, 5 styles, 39 sets with 357 fields (2 MB),
  0.7 s. mep: 8629 rows (7294 + 1331 bundle-only: 1045 wires, 187 circuits, 45 duct systems,
  33 walls, 8 conduit runs, 6 piping systems, 7 rooms and separations; + 4 levels), 6942
  meshes, 420,077 faces, 30 styles, 91 sets with 1098 fields (58 MB), 3.4 s;
  `simple_features` 7 s; six conceptual masses (`Mass`, geometry of their own) lie up to 77 m
  west and 122 m south of the building, whose other elements span x -19.4..53.1 m, y
  -13.7..41.9 m.

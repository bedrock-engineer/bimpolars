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
# elevation: Float64, native_geom: List(Struct), mesh: Struct, bbox: Struct

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

# The structural-analysis model: beams and columns as lines, slabs and walls as midplanes
analysis = bimpolars.simple_features(bdf, lightbim=True)
analysis.frame.select("id", "lightbim", "lightbim_kind", "lightbim_qa")

from pathlib import Path

Path("sixty5.glb").write_bytes(bimpolars.element_meshes_glb(bdf))  # glTF, one material per style

import geopandas

geopandas.read_parquet("sixty5.parquet", columns=["id", "class", "footprint"]).crs
# <Engineering CRS: {"$schema": ...}>  Name: MM16169 model coordinates, millimetre axes
```

The demo page, [bedrock-engineer.github.io/bimpolars](https://bedrock-engineer.github.io/bimpolars/),
shows the Sixty5 structural model read with `read_ifc` and placed on earth with
`derived_engineering_crs` in [specklewidgets](https://github.com/bedrock-engineer/specklestuff)'
GeoSpeckleViewer, among PDOK's 3D BAG, beside a map of the clicked element's footprint (its
`geog`). The page is built from specklestuff's `site/` (`.github/workflows/pages.yml`).

## Install

```sh
uv add bimpolars  # or: pip install bimpolars
uv add "bimpolars[ifc]"  # with ifcopenshell, for read_ifc
uv add "bimpolars[speckle]"  # with specklepy, for Speckle bundles
```

The core depends on polars, numpy, shapely and pyproj, so georeferencing needs no extra.
Reading IFC (ifcopenshell) is the extra `[ifc]`: `import bimpolars` does not import
ifcopenshell, `bimpolars.read_ifc` does on first use. Reading a Speckle version from its v3
objects needs no specklepy; projecting a Speckle bundle to those objects does, and is the extra
`[speckle]` (see [Speckle](#speckle)).

A CRS goes in in any form pyproj reads: PROJJSON, a `pyproj.CRS`, `"EPSG:7415"`, an EPSG code
or WKT (`bimpolars.CRSLike`). A `BimDataFrame` stores it as PROJJSON (`bdf.projjson`), which
is what its GeoParquet and glTF metadata carry, and hands it out as a `pyproj.CRS` (`bdf.crs`),
like `GeoDataFrame.crs`.

## Functions

- `read_ifc(source, *, threads=None)`: reads every product of an IFC into a `BimDataFrame`,
  tessellated by ifcopenshell in the model's own coordinates and length unit. The columns:
  - `id`, `name`, `class` (Categorical IFC class). Every `IfcProduct` is a row, openings
    included: an `IfcOpeningElement` row carries the solid mesh of the void it cuts, so filter
    `class != "IfcOpeningElement"` before outlining or rendering elements.
  - `properties`: a Struct of property and quantity sets (the type's included), each a Struct
    of values. A property whose type differs between rows becomes a String. The set
    `attributes` holds the entity's `PredefinedType` (its type's, or `ObjectType` when it is
    USERDEFINED), which tells a base slab from a floor.
  - `container`: the `id` of the spatial element the row sits in (the storey for most
    elements, the building for a storey; null for the site, for elements in no container and
    for ports); `elevation`: the storey's level in model coordinates.
  - `native_geom` (`NATIVE_GEOM_DTYPE`): the extruded solids (`IfcExtrudedAreaSolid`) of the
    body, mapped items resolved and both operands of a union kept, as a list of sweeps:
    `placement` (a row-major 4x4 matrix from the sweep's coordinates to the model CRS),
    `profile` (a 2D WKB polygon with its voids), `profile_type`, `profile_name`, `depth`,
    `direction` and `clipped` (the IFC cut the solid afterwards, by a difference or an
    intersection). Breps and other solids are left out, so a list can describe part of a body;
    null when there is no sweep. LightBIM takes a member's axis or midplane from it.
  - `mesh`: `{vertices, faces, styles}` with flat vertices, faces sorted by style, and each
    style's `name`, `rgba` and `face_count`; `bbox`: `{xmin, ymin, zmin, xmax, ymax, zmax}`.
    Rows without geometry have a null `mesh` and `bbox`.
- `simple_features(bdf, *, lightbim=False, kind=None)`: adds the plan footprint of every row
  with a mesh, in two columns, and with `lightbim=True` the three columns of
  [LightBIM](#lightbim):
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
- `bimpolars.lightbim.lightbim_features(bdf, *, kind=None)`: the LightBIM columns on their own,
  for a frame that has its footprints. The class table is `bimpolars.lightbim.CLASSES`.
- `plan_outline(vertices, faces)`: the plan outline of one triangle mesh as a 2D shapely
  Polygon or MultiPolygon, the union of its triangles projected to XY, for a mesh outside a
  frame. Slivers (triangles with less plan area than 1e-9 of the extent squared) are left
  out.
- `element_meshes_glb(bdf, *, normals=True)`: the meshes as binary glTF (`bytes`), in metres,
  one primitive and material per style (`name` and `rgba` of the mesh's `styles`), without
  the `IfcOpeningElement` rows. See [glTF](#gltf).
- `triangles_glb(groups)`: any triangles in metres, z up, as flat-shaded binary glTF: per
  material name an `(n, 3, 3)` array of corners and an RGB or RGBA colour in 0..1.
- `BimDataFrame(frame, crs)`: the thin wrapper, with the CRS in any form, with `.frame`,
  `.crs` (a `pyproj.CRS`), `.projjson`, `.with_frame()`,
  `.equals()`, `.write_parquet()` and `BimDataFrame.read_parquet()`.
- `read_parquet(source)`: reads a `BimDataFrame` back, CRS included. The CRS sits under the
  `bimpolars` key of the Parquet file metadata as `{"version": 1, "crs": <PROJJSON>}`.
- `engineering_crs(name, *, unit, datum)`: a plain EngineeringCRS as PROJJSON in a given
  length unit (a known name, a `LinearUnit(name, conversion_factor)`, or a CRS or a model whose
  unit it copies), which a model without georeferencing gets. `bdf.crs.axis_info[0].unit_name`
  reads the unit back.
- `properties_series(rows)`: the Struct column of property sets that `read_ifc` and
  `read_speckle` use, with the dtype per property inferred across rows, for any reader with
  named sets of values.
- `read_speckle(objects, bundle=None, *, name=None)`: reads the v3 objects of a Speckle version
  into a `BimDataFrame`, instances expanded, with the Revit categories, parameters and levels
  of the version's bundle when one is given. See [Speckle](#speckle).
- `version_objects(source, version)`: the v3 objects of one version, root first, from a local
  copy: a Parquet file with `version_id`, `object_id`, `is_root` and `content` columns, or a
  SQLite database with `versions(version_id, v3_root_id)` and `objects(hash, content)`.
- `bimpolars.bundle.bundle_objects(bundle)`: the v3 objects of a bundle-only version, projected
  from its bundle (a folder of `<name>.<table>.parquet` files, or the bundle's URL in the
  bimpolars bucket) with specklepy, the `[speckle]` extra; the placed elements keep the
  geometry they carry themselves and definition geometry keeps its materials.
  `bimpolars.bundle.bundle_model(bundle)` opens such a bundle as a specklepy `Model`. See
  [Speckle](#speckle).
- `bimpolars.data.fetch(key, *, download=True)`: the path of a test or example model, downloaded
  from `http://r2.eu.bedrock.engineer/bimpolars/` on first use into `bimpolars.data.data_dir()`:
  `BIMPOLARS_DATA_DIR`, else `$XDG_CACHE_HOME/bimpolars`, else `~/.cache/bimpolars`. The key
  may be given as its URL in the bucket. `bimpolars.data.fetch_bundle(prefix)` fetches the
  tables of a Speckle bundle (`bimpolars.data.BUNDLE_TABLES`) the same way and gives their
  folder; a bundle's overflow geometry shards (`<name>.geometries.<N>.parquet`, past 1.5 GiB
  of geometry) are not among them, so read such a bundle from a complete folder.
- `bimpolars.data.bucket_key(key_or_url)`: the key of a URL in the bucket (`http` or `https`,
  percent-encoded or not), which `fetch` and `fetch_bundle` use; a URL outside the bucket, or
  a key that would leave the data folder (absolute, a drive or `..`), raises `ValueError`.
- `bimpolars.speckle.bundle_folder(bundle)`: the local folder of a bundle given as a folder or
  as its URL in the bucket (fetched with `fetch_bundle`), as `read_speckle` and
  `bimpolars.bundle` take it.

## Georeferencing

The georeferencing functions live in `bimpolars.geobim` and on the package itself. A model from
a reader sits in its own coordinates; georeferencing says where that is on earth:

```python
import bimpolars as bpl

bdf = bpl.simple_features(bpl.read_ifc("sixty5.ifc"))
# Sixty5: the site origin at RD New (160071.5, 384156.5), NAP +18.625 m, the site's y-axis
# 50.36 degrees clockwise from grid north, in the model's own unit (millimetres)
crs = bpl.derived_engineering_crs(
    "EPSG:7415", 160071.5, 384156.5, 18.625, -50.36, unit=bdf, name="Sixty5"
)
site = bpl.set_crs(bdf, crs)  # the same geometry, now a georeferenced Engineering CRS
rd = bpl.to_crs(site, "EPSG:7415")  # meshes, footprints, bbox and levels in RD New + NAP metres
rd.write_parquet("sixty5_rd.parquet")  # GeoParquet with EPSG:7415
geographic = bpl.to_crs(site, "OGC:CRS84h")  # the whole model in lon, lat, ellipsoidal height

site = bpl.with_geog(site)  # + geog: the footprints in OGC:CRS84h, next to the site's own
site.write_parquet("sixty5_geog.parquet")  # GeoParquet with geog as the primary geometry
geopandas.read_parquet("sixty5_geog.parquet", columns=["id", "class", "geog"]).to_json()  # GeoJSON
```

- `derived_engineering_crs(base_crs, easting, northing, orthometric_height, rotation, *, unit, name)`:
  the Engineering CRS of a site as a `pyproj.CRS`, derived from a compound of a projected and a
  vertical CRS such as EPSG:7415 (RD New + NAP height): a DerivedProjectedCRS with an Affine
  parametric transformation (EPSG:9624) and a DerivedVerticalCRS with a Vertical Offset
  (EPSG:9616), with the axes in the site's `unit` (`unit=bdf` copies the model's). The site
  origin lands on `(easting, northing, orthometric_height)` and the site's y-axis points `rotation` degrees
  counterclockwise from grid north (for an IfcMapConversion:
  `degrees(atan2(XAxisOrdinate, XAxisAbscissa))`). PROJ then chains site → base → anything.
- `set_crs(bdf, crs)`: labels the model with a CRS (PROJJSON, a `pyproj.CRS` or user input such
  as `"EPSG:7415"`) without moving the geometry, like `GeoDataFrame.set_crs`. The CRS must have
  three axes.
- `to_crs(bdf, crs)`: moves the model to another CRS with one pyproj transformation, like
  `GeoDataFrame.to_crs`: the `mesh` vertices, the WKB columns in the model CRS (`footprint`
  and `lightbim`; never `geog`) and the storey `elevation`s (at the plan centre of the
  model) move together, `bbox` is recomputed from the moved meshes (from the moved corners of
  the old `bbox` when the frame has no `mesh` column) and the footprints are kept inside it.
  The target must have three axes, so that the meshes keep one unit per axis: `"EPSG:7415"`,
  not `"EPSG:28992"`.
  `native_geom` keeps its sweeps only under an affine transformation (to the base CRS of the
  Engineering CRS, for instance): the matrix is composed onto each sweep's `placement`, scale
  included, and the profile and depth stay in the sweep's own unit. A geographic, geocentric or
  differently projected target nulls it. `lightbim_qa.deviation`, a length, is scaled by the
  same matrix into the target's unit (1500 mm becomes 1.5 m in EPSG:7415), and nulled when the
  transformation is not affine. A derived CRS is passed through its base CRS, because PROJ
  otherwise applies a 3D Helmert transformation to heights still in the site's unit.
- `with_geog(bdf)`: adds `geog`, the `footprint` of every row in OGC:CRS84h (longitude and
  latitude in degrees, ellipsoidal height in metres on WGS 84: `bimpolars.crs.CRS84H`), as
  ISO WKB, null exactly when `footprint` is. That is the CRS GeoJSON means by its coordinates,
  so the column exports to GeoJSON as it is (`GeoDataFrame.to_json()`, `shapely.to_geojson`,
  DuckDB's `ST_AsGeoJSON`), and the written GeoParquet takes it as the primary geometry (see
  [GeoParquet](#geoparquet)). The model keeps its own CRS and `footprint`, and `to_crs` never
  moves `geog`. A model in a plain EngineeringCRS raises: `set_crs` it first. GDAL's WKT1
  cannot say a 3D geographic CRS, so `GeoDataFrame.to_file` (pyogrio) refuses the column's
  CRS; label the frame `OGC:CRS84` first (`gdf.set_crs("OGC:CRS84", allow_override=True)`),
  which keeps the heights.
- `affine_transformation(bdf, crs)`: the 4x4 matrix `to_crs` would apply to a placement, or
  None when the transformation is not affine over the model (within `AFFINE_TOLERANCE`, 1 µm).
- `linearized_site_to_ecef(crs)`: the 4x4 matrix from site metres to ECEF metres (EPSG:4978),
  linearized at the site origin, for CesiumJS and 3D Tiles.

Transformations to ETRS89 (`EPSG:4937`, `OGC:CRS84h`, ECEF) should follow RDNAPTRANS™2018,
which PROJ does only with the NSGI grids `nl_nsgi_rdtrans2018.tif` and `nl_nsgi_nlgeo2018.tif`
(install them, or enable the PROJ network). Without them PROJ falls back to a Helmert
transformation that is good to about a metre, and passes NAP heights off as ellipsoidal heights;
`to_crs` warns when that happens.

## GeoParquet

`write_parquet` on a frame with a `geog`, `footprint` or `lightbim` column writes
[GeoParquet 1.1](https://geoparquet.org): the file metadata gets a `geo` entry
(`bimpolars.geoparquet.geo_metadata`) that lists each of them WKB encoded with its geometry
types and its CRS as PROJJSON: `footprint` in the model CRS, `geog` in OGC:CRS84h. The first
one present is the primary geometry column (`bimpolars.geoparquet.GEOMETRY_COLUMNS`): `geog`
when `with_geog` added it, so that a reader that takes the primary column has longitude and
latitude, else `footprint`. When the frame has its `bbox` column, the entry also names that
column as the *covering* of the footprints: a pointer that tells a reader that each row's
footprint lies within `bbox.xmin..xmax` and `bbox.ymin..ymax`, so that a spatial filter can
skip row groups. A GeoParquet covering has those four slots and no z; the `bbox` column itself
keeps all six fields. The entry's own bounding box, one for the whole file, is x and y only
for another reason: the footprints lie flat at their storey's elevation, which is often
outside the mesh's `zmin..zmax`, so the mesh's z range would not contain them. `geog` is in
degrees, which `bbox` does not cover, so its entry gets the bounds of its own coordinates and
no covering (`geopandas.read_parquet(..., bbox=...)` filters on the primary column's covering
and refuses a file whose primary column has none). Readers that understand GeoParquet then
find the footprints with their CRS, and can skip row groups with the bbox:

```python
geopandas.read_parquet("sixty5.parquet", columns=["id", "footprint"], bbox=(0, 0, 10_000, 10_000))
```

```sql
LOAD spatial;  -- DuckDB
SELECT id, ST_Area(footprint) FROM read_parquet('sixty5.parquet') WHERE bbox.xmax < 10000;
```

`lightbim` is a GeoParquet geometry column too, listed after `footprint` with the same CRS and
covering: every reduction lies within its row's `bbox`. `mesh` and `native_geom` are not GeoParquet
geometries; they share the model CRS implicitly.

## LightBIM

`simple_features(bdf, lightbim=True)` reduces every element the way a structural analysis model
does, next to its footprint:

```python
bdf = bimpolars.simple_features(bimpolars.read_ifc("sixty5.ifc"), lightbim=True)
bdf.frame.select(
    "name",
    pl.col("lightbim_kind").struct.unnest(),  # member, role, load_bearing
    pl.col("lightbim_qa").struct.unnest(),  # rule, source, deviation
)
```

- `lightbim`: ISO WKB with Z in the model CRS. The `member` of `lightbim_kind` sets it: a
  `curve` (beam, column, brace, pile) is a `LINESTRING Z` along its axis, a `surface` (slab,
  wall, plate, roof) a `POLYGON Z` on its midplane with the holes and parts of its mesh (a wall
  with windows), a `support` (footing, raft) the footprint at its bearing level (the bottom of
  the mesh), a `mass` the footprint at the height of its centre of gravity.
- `lightbim_kind`: `{member, role, load_bearing}` with `member` and `role` Enums. The member
  comes from, in order: the `kind` expression the caller passes (a Struct of any of the three
  fields, or a String member, null where it has no say); the class table
  `bimpolars.lightbim.CLASSES` (IFC classes, with the predefined type where it matters:
  `IfcSlab.BASESLAB` is a raft, and Revit categories); the shape of the mesh (the tightest of
  an upright box and a box along its principal axes, snapped to the model axes within 5
  degrees: a surface when the smallest extent is 4 times smaller than the middle one, else a
  curve when the middle is 4 times smaller than the largest, else a mass). A role the table
  leaves open (`IfcMember`, Revit `Structural Framing`), or that a `kind` member other than
  the table's drops, follows the angle of the axis; `load_bearing` comes from the
  `LoadBearing` of a `Pset_` set (`Pset_WallCommon`), else of any other set, else the Revit
  `Structural` parameter, else the table. Openings, spaces and reinforcement are no members:
  their columns are null, as they are for rows without a mesh. The class table and
  `load_bearing` read `properties`: keep its sets `attributes` and `Pset_*Common` (or Revit's
  `Structural`) when you slim a large frame, since without `attributes` a base slab is a floor.
- `lightbim_qa`: `{rule, source, deviation}`. `rule` says what decided the member (`kind`,
  `class` or `shape`), `source` what placed the axis or plane (`extrusion` from `native_geom`,
  `extrusion, cuts ignored` when the IFC clipped the sweep, or `mesh` from the box around it),
  and `deviation` how far, in CRS units, the farthest vertex of the mesh lies from the axis
  (half the section's diagonal) or the plane (half the thickness), or above the bearing level
  (the height of a footing), or from the centre of gravity's height (a mass; the centre of
  the box around the mesh when the mesh is not closed).

The axis or the plane comes from the element's largest sweep where `native_geom` has one,
else from the box; the extent always comes from the mesh, so a clipped beam ends where its
mesh ends. On the Sixty5 structural model every beam, column and pile is a curve on its axis,
every floor and wall a surface on its midplane, and every footing and base slab a support at
its bottom; the reduction takes 1.3 s on top of the 4 s of the footprints.

**Known limitations.** One straight axis or one flat plane per element: walls with a bent
midline (L, T, curved) and folded slabs or curved beams are reduced wrongly
([bimpolars#3](https://github.com/bedrock-engineer/bimpolars/issues/3)). Only extruded sweeps
are read from the IFC; other bodies place their axis from the box around the mesh.

## glTF

`element_meshes_glb` writes the meshes as [binary glTF 2.0](https://registry.khronos.org/glTF/)
(`.glb`), the format CesiumJS, three.js, Blender and most 3D viewers load:

```python
from pathlib import Path

structure = bdf.with_frame(bdf.frame.filter(pl.col("class").is_in(["IfcColumn", "IfcBeam"])))
Path("structure.glb").write_bytes(bimpolars.element_meshes_glb(structure))
```

- One primitive and one material per style: the faces of every mesh are grouped by the
  `(name, rgba)` of their style, so a viewer colours the model as its authoring tool did
  (translucent styles get `alphaMode` BLEND). `IfcOpeningElement` rows are left out: their
  mesh is the void they cut.
- Metres, from the model's unit (`bdf.crs.axis_info[0]`), as glTF requires. The root node
  carries the model CRS as `extras.crs` and, next to it, `extras.unit` (`"metre"`): the CRS
  keeps its own unit, the vertex data under the node does not.
- The vertex data keeps the model's z-up coordinates; the root node, named after the CRS,
  rotates the scene to glTF's y-up (`bimpolars.gltf.Y_UP_ROTATION`, the 3D Tiles convention:
  model `(x, y, z)` is scene `(x, z, -y)`). A child node at the centre of the written
  vertices' bounding box holds the mesh, so that the float32 vertices stay precise for a model
  far from the origin. In CesiumJS, place the model with
  `Model.fromGltfAsync({url, modelMatrix, forwardAxis: Cesium.Axis.X})` (the default up axis)
  and the matrix of `linearized_site_to_ecef`.
- `normals=True` (the default) writes flat-shaded triangles: every triangle gets its own
  three vertices with the face normal, because the meshes share vertices between faces of
  different orientation. `normals=False` writes the vertices and faces as the `mesh` column
  holds them, indexed and about a quarter of the size, without normals: a glTF reader then
  computes flat normals (the specification requires it), though CesiumJS renders such a model
  unlit and three.js smooths it.

## Speckle

`read_speckle` reads a [Speckle](https://speckle.systems) version without specklepy, from the
version's v3 objects: the JSON graph of a root collection, `DataObject`s with their
`displayValue` meshes, and instances (`InstanceProxy`, `InstanceDefinitionProxy`). The objects
come from a local copy of a Speckle download (`version_objects`), or from the version's
*bundle*, the Parquet tables Speckle 4 publishes instead, projected with specklepy by
`bimpolars.bundle.bundle_objects` (the `[speckle]` extra).

```python
import polars as pl
import bimpolars
from bimpolars import data

arch = data.RVT_SPECKLE_ARCH  # the Revit Advanced Sample, architecture
objects = bimpolars.version_objects(data.fetch(data.RVT_SPECKLE_V3_OBJECTS), arch.version_id)
bdf = bimpolars.read_speckle(objects, data.fetch_bundle(arch.bundle))
bdf.frame.filter(pl.col("class") == "Walls").select(
    "id",
    "name",
    "container",  # the level: a row with class "Levels" and the elevation
    family=pl.col("properties").struct.field("object").struct.field("family"),
    volume=pl.col("properties")
    .struct.field("properties.Parameters.Instance Parameters.Dimensions")
    .struct.field("Volume"),
)

from bimpolars.bundle import bundle_objects  # the [speckle] extra

bundle = f"{data.DATA_URL}{arch.bundle}"  # the bundle's folder, or its URL in the bucket
same = bimpolars.read_speckle(bundle_objects(bundle), bundle)
same.equals(bdf)  # True: the bucket's objects were projected the same way
```

- One row per element: the objects under the root's `elements`, collections traversed, the
  objects that belong to an instance definition left out. An `InstanceProxy` is expanded in
  place: its definition's meshes (nested instances included), moved by its transform and
  scaled from their `units` to the model's. A placement named `<applicationId>-instance-N`
  folds into the row of `<applicationId>`, so an element placed several times is one row with
  one merged mesh. `id` is the `applicationId` (the Revit UniqueId), `name` the object's name.
- With a bundle, the elements the bundle knows but the version has no object for (Revit
  elements the projection leaves out: curtain walls, room separations, MEP systems, wires and
  circuits) get rows too, with the bundle's category, name, properties and level and a null
  `mesh`, so that `read_speckle` and `read_ifc` both give one row per element of the model.
- `class`: the object's `category`, else the Revit category the bundle has for it, else the
  `speckle_type`. The bundle is where the instances get theirs: specklepy's projection gives an
  `InstanceProxy` nothing but its transform.
- `properties`: a Struct of sets, each a group of the property tree named by its dotted path:
  `properties` holds the values directly in the object's property bag (`elementId`, `typeId`),
  `properties.Parameters.Instance Parameters.Dimensions` a group's values (`Volume`), and
  `object` the object's own fields (`family`, `type`). The path is the bundle's eav `path`
  column, so a value is found at `f"{set}.{name}"`. The bundle's values (type parameters
  included, instance values winning) fill the rows the objects leave bare. A property is a
  number only when the bundle has a number for every value of it (`elementId`); otherwise it is
  text, also where the object carries a number, so a Mark `"129"` stays text next to `"104A"`
  and `name` never becomes a number. A value keeps the unit Revit states for its parameter
  (the eav `unit` column: lengths in millimetres, areas in square metres, volumes in cubic
  metres, the `Volume` above included); it is not scaled to the model unit.
- With a bundle, the levels (the envelope's level nodes) become rows of class `Levels` with
  their `elevation` and the id `level-<node id>`, and every element's `container` points at
  its level, so `simple_features` places the footprints at the level. Without one, `container`
  and `elevation` are null.
- `mesh` and `bbox` in the model's units; quads and n-gons are fan-triangulated. A style is the
  `RenderMaterial` the root's `renderMaterialProxies` give the mesh, its element or its
  instance; a definition's mesh without one gets, with a bundle, the material the envelope
  attaches to the bundle geometry it came from (`def-geo-<geometry index>`). The first proxy
  that targets an element wins when several do, and specklepy gives every mesh of an element
  the element's applicationId, so a layered wall, floor or ceiling (one proxy per layer) is
  painted in its first layer's material; a material per layer needs the bundle's `DISPLAY`
  order (bimpolars#20).
- The CRS is a plain engineering CRS in the root's `units` (else the bundle's, else the unit
  most meshes name, else metres), anchored at Revit's project base point: the model's own
  coordinates. The geolocation
  of a Revit model is never used (the MEP sample sits at Revit's Munich default). Revit exports
  IFC in shared coordinates, so to compare with `read_ifc`, move the Speckle model by the
  bundle's `modelPlacement.options.sharedCoordinates.transform`; on the RVT Advanced Sample
  the plan extents then agree to the micrometre and 85 % of the elements matched by IfcGUID
  within 1 mm.
- `bundle_objects` projects a bundle on disk with specklepy's `Model.to_base()`, serialized as
  a Speckle send would, and restores two things that projection loses. An element placed as
  an instance keeps the geometry it carries itself: Revit puts structural framing, columns and
  foundations on the element with an empty definition (841 of the 935 elements of the
  structural sample), which `to_base` leaves without geometry; it becomes a DataObject under
  the element's id, the first placement is renamed `<id>-instance-0`, and the two fold into
  one row. And the geometry of an instance definition gets the material the envelope attaches
  to it, so family instances are coloured without the bundle at hand. The bucket's
  `speckle-v3-objects.parquet` was projected this way, so both routes give equal frames.

## Develop

```sh
uv sync
uv run pytest            # BIMPOLARS_DATA_DIR=/path/to/mirror to use local test data
uv run pytest -m slow    # only the slow tests, such as the 343 MB architectural model
uv run mypy src
uv run ruff check && uv run ruff format --check
```

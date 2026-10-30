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
# elevation: Float64, native_geom: Null, mesh: Struct, bbox: Struct

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

from pathlib import Path

Path("sixty5.glb").write_bytes(bimpolars.element_meshes_glb(bdf))  # glTF, one material per style

import geopandas

geopandas.read_parquet("sixty5.parquet", columns=["id", "class", "footprint"]).crs
# <Engineering CRS: {"$schema": ...}>  Name: MM16169 model coordinates, millimetre axes
```

## Install

```sh
uv add bimpolars  # or: pip install bimpolars
uv add "bimpolars[geobim]"  # with pyproj, for georeferencing
uv add "bimpolars[speckle]"  # with specklepy, for Speckle bundles
```

The core depends on polars, numpy, shapely and ifcopenshell. Georeferencing (pyproj) is the
extra `[geobim]`. Reading a Speckle version from its v3 objects needs no specklepy; projecting
a Speckle bundle to those objects does, and is the extra `[speckle]` (see [Speckle](#speckle)).

## Functions

- `read_ifc(source, *, threads=None)`: reads every product of an IFC into a `BimDataFrame`,
  tessellated by ifcopenshell in the model's own coordinates and length unit. The columns:
  - `id`, `name`, `class` (Categorical IFC class). Every `IfcProduct` is a row, openings
    included: an `IfcOpeningElement` row carries the solid mesh of the void it cuts, so filter
    `class != "IfcOpeningElement"` before outlining or rendering elements.
  - `properties`: a Struct of property and quantity sets (the type's included), each a Struct
    of values. A property whose type differs between rows becomes a String.
  - `container`: the `id` of the spatial element the row sits in (the storey for most
    elements, the building for a storey; null for the site, for elements in no container and
    for ports); `elevation`: the storey's level in model coordinates.
  - `native_geom`: null for now; `mesh`: `{vertices, faces, styles}` with flat vertices, faces
    sorted by style, and each style's `name`, `rgba` and `face_count`; `bbox`:
    `{xmin, ymin, zmin, xmax, ymax, zmax}`. Rows without geometry have a null `mesh` and `bbox`.
- `simple_features(bdf)`: adds the plan footprint of every row with a mesh, in two columns:
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
- `plan_outline(vertices, faces)`: the plan outline of one triangle mesh as a 2D shapely
  Polygon or MultiPolygon, the union of its triangles projected to XY, for a mesh outside a
  frame. Slivers (triangles with less plan area than 1e-9 of the extent squared) are left
  out.
- `element_meshes_glb(bdf, *, normals=True)`: the meshes as binary glTF (`bytes`), in metres,
  one primitive and material per style (`name` and `rgba` of the mesh's `styles`), without
  the `IfcOpeningElement` rows. See [glTF](#gltf).
- `triangles_glb(groups)`: any triangles in metres, z up, as flat-shaded binary glTF: per
  material name an `(n, 3, 3)` array of corners and an RGB or RGBA colour in 0..1.
- `BimDataFrame(frame, crs)`: the thin wrapper with `.frame`, `.crs`, `.with_frame()`,
  `.equals()`, `.write_parquet()` and `BimDataFrame.read_parquet()`.
- `read_parquet(source)`: reads a `BimDataFrame` back, CRS included. The CRS sits under the
  `bimpolars` key of the Parquet file metadata as `{"version": 1, "crs": <PROJJSON>}`.
- `engineering_crs(name, *, unit, datum)`: a plain EngineeringCRS as PROJJSON in a given
  length unit (a known name or a `LinearUnit(name, conversion_factor)`), which a model without
  georeferencing gets. `linear_unit(crs)` reads the unit back.
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
  folder.

## Georeferencing

`bimpolars.geobim` needs pyproj (the `[geobim]` extra). A model from a reader sits in its own
coordinates; georeferencing says where that is on earth:

```python
import bimpolars
from bimpolars.geobim import derived_engineering_crs, set_crs, to_crs

bdf = bimpolars.simple_features(bimpolars.read_ifc("sixty5.ifc"))
# Sixty5: the site origin at RD New (160071.5, 384156.5), NAP +18.625 m, the site's y-axis
# 50.36 degrees clockwise from grid north, in the model's millimetres
crs = derived_engineering_crs(
    "EPSG:7415",
    160071.5,
    384156.5,
    18.625,
    -50.36,
    unit=bimpolars.linear_unit(bdf.crs),
    name="Sixty5",
)
site = set_crs(bdf, crs)  # the same geometry, now a georeferenced Engineering CRS
rd = to_crs(site, "EPSG:7415")  # meshes, footprints, bbox and storey levels in RD New + NAP metres
rd.write_parquet("sixty5_rd.parquet")  # GeoParquet with EPSG:7415
geog = to_crs(site, "OGC:CRS84h")  # longitude, latitude, ellipsoidal height
```

- `derived_engineering_crs(base_crs, easting, northing, orthometric_height, rotation, *, unit, name)`:
  the Engineering CRS of a site as PROJJSON, derived from a compound of a projected and a
  vertical CRS such as EPSG:7415 (RD New + NAP height): a DerivedProjectedCRS with an Affine
  parametric transformation (EPSG:9624) and a DerivedVerticalCRS with a Vertical Offset
  (EPSG:9616), with the axes in the site's `unit`. The site origin lands on
  `(easting, northing, orthometric_height)` and the site's y-axis points `rotation` degrees
  counterclockwise from grid north (for an IfcMapConversion:
  `degrees(atan2(XAxisOrdinate, XAxisAbscissa))`). PROJ then chains site → base → anything.
- `set_crs(bdf, crs)`: labels the model with a CRS (PROJJSON, a `pyproj.CRS` or user input such
  as `"EPSG:7415"`) without moving the geometry, like `GeoDataFrame.set_crs`. The CRS must have
  three axes.
- `to_crs(bdf, crs)`: moves the model to another CRS with one pyproj transformation, like
  `GeoDataFrame.to_crs`: the `mesh` vertices, the WKB columns in the model CRS (`footprint`,
  later `lightbim`; never `geog`) and the storey `elevation`s (at the plan centre of the
  model) move together, `bbox` is recomputed from the moved meshes (from the moved corners of
  the old `bbox` when the frame has no `mesh` column) and the footprints are kept inside it.
  The target must have three axes, so that the meshes keep one unit per axis: `"EPSG:7415"`,
  not `"EPSG:28992"`.
  `native_geom` keeps its placement only under an affine transformation (to the base CRS of
  the Engineering CRS, for instance); a geographic, geocentric or differently projected target
  nulls it. A derived CRS is passed through its base CRS, because PROJ otherwise applies a
  3D Helmert transformation to heights still in the site's unit.
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

`write_parquet` on a frame with a `footprint` column writes
[GeoParquet 1.1](https://geoparquet.org): the file metadata gets a `geo` entry
(`bimpolars.geoparquet.geo_metadata`) with `footprint` as the primary geometry column, WKB
encoded, its geometry types and the model CRS as PROJJSON. When the frame has its `bbox`
column, the entry also names that column as the *covering* of the footprints: a pointer that
tells a reader that each row's footprint lies within `bbox.xmin..xmax` and `bbox.ymin..ymax`,
so that a spatial filter can skip row groups. A GeoParquet covering has those four slots and
no z; the `bbox` column itself keeps all six fields. The entry's own bounding box, one for the
whole file, is x and y only for another reason: the footprints lie flat at their storey's
elevation, which is often outside the mesh's `zmin..zmax`, so the mesh's z range would not
contain them. Readers that understand GeoParquet then find the footprints with their CRS, and
can skip row groups with the bbox:

```python
geopandas.read_parquet("sixty5.parquet", columns=["id", "footprint"], bbox=(0, 0, 10_000, 10_000))
```

```sql
LOAD spatial;  -- DuckDB
SELECT id, ST_Area(footprint) FROM read_parquet('sixty5.parquet') WHERE bbox.xmax < 10000;
```

`mesh` and `native_geom` are not GeoParquet geometries; they share the model CRS implicitly.

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
- Metres, from the model's unit (`linear_unit(bdf.crs)`), as glTF requires. The root node
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
  included, instance values winning) fill the rows the objects leave bare. A value keeps the
  unit Revit states for its parameter (the eav `unit` column: lengths in millimetres, areas in
  square metres, volumes in cubic metres, the `Volume` above included); it is not scaled to the
  model unit.
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

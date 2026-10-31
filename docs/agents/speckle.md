# Speckle: read_speckle and bundles

`src/bimpolars/speckle.py` reads a Speckle version from its v3 objects as plain JSON, without
specklepy: `version_objects` (Parquet or SQLite copy) and `read_speckle` (instances expanded,
the bundle's categories, parameters and levels joined by applicationId).
`src/bimpolars/bundle.py` (the `[speckle]` extra, specklepy) builds those objects from a bundle
on disk, or in the bucket: `bundle_model` and `bundle_objects`. Say "built from the bundle",
not "projected" (specklepy's word for `Model.to_base()`, which in bimpolars clashes with
projected CRSs and footprints projected to XY).

## read_speckle rows

- Rows are the objects under the root's `elements` (collections traversed; definition content,
  which specklepy names `def-geo-<geometry index>` and `nested-inst-<n>`, left out), keyed by
  `applicationId` with `-instance-N` stripped, so a placement folds into its element's row and
  the same mesh (listed as a Mesh and as the DataObject that displays it) counts once.
- With a bundle, its `eav.objects` that the version has no object for (curtain walls, room
  separations, MEP systems, wires, circuits) get rows with a null `mesh`, so both readers give
  one row per element.
- `class` = the object's `category`, else the bundle's category, else `speckle_type` (null when
  the object has none).
- Levels become rows (`level-<node id>`, class `Levels`, `elevation` in model units) and
  `container` points at them, so `simple_features` works on the result.

## read_speckle values

- The property set of a value is the dotted path of its group (`properties`,
  `properties.Parameters.Instance Parameters.Dimensions`, and `object` for the object's own
  fields), the same for the object's bag and the bundle's eav `path`.
- An eav row with a number and its text gives the number; values keep the unit the eav `unit`
  column states (lengths mm, areas m², volumes m³), never scaled.
- Styles: the root's `renderMaterialProxies` (first proxy that targets the mesh, the element or
  the instance wins), then the envelope's `HAS_MATERIAL` for definition geometry. Every mesh of
  an element carries the element's applicationId, so a layered wall or ceiling gets its first
  layer's material on all faces (#20 is the per-layer follow-up through the envelope's
  `DISPLAY` order).
- The envelope's kinds and relations are looked up by name in `envelope.node_kinds` and
  `envelope.rel_types`, with schema 1.5's numbers as the fallback.
- The CRS is the root's `units`, else the bundle's, else the meshes' most common, else metres;
  never the Revit geolocation.
- `native_geom` is Null.

## Bundles

- A bundle's SGEO geometry is decoded by `bundle_objects`; `read_speckle` reads JSON only.
- A bundle is a folder, a mapping of tables, or its URL in the bucket (`data.bucket_key` strips
  `DATA_URL`, http or https, and refuses keys that leave the data folder; `bundle_folder`
  fetches the tables into the data folder; any other URL raises).
- `fetch_bundle` knows the 18 `BUNDLE_TABLES`, not the overflow geometry shards
  `<name>.geometries.<N>.parquet` specklepy writes past 1.5 GiB, so a bundle that large must
  come from a complete folder (a listing comes with the generic-bucket decision on #21).

## bundle_objects

`bundle_objects` is specklepy's `Model.to_base()` on the `Model` that `bundle_model` builds from
the folder (`_FolderModel`, whose `close()`, and so `with`, keeps the folder; specklepy's own
deletes it), with two fixes on the Base tree before serialization:

- The own geometry of an element that has placements (`base_projection._geometry_object`,
  private like `_materials`, hence the exact pin `specklepy[bundle]==2026.9.1`) is inserted
  before its first placement, which is renamed `<applicationId>-instance-0` (before the first
  `-instance-N` if there is no plain one, at the root if there is none; never dropped).
- The material proxies get `def-geo-<k>` targets for the definition geometry the envelope's
  `HAS_MATERIAL` styles (a geometry in a `DEFINES` relation without a `DISPLAY` owner), the
  proxies as `_materials` builds them, in the envelope's node order.

The serialization is specklepy's deprecated JSON pipeline (`BaseObjectSerializer` into a
`MemoryTransport`; only its own DeprecationWarning is silenced, by message), root first. On the
three RVT bundles every non-root object has the bucket's id (content hash) and the
`read_speckle` frames are equal (`tests/test_bundle.py`); the root differs in the proxies' ids
only, because the notebook wrote them as its own JSON.

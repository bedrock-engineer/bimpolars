# Test data

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
  columns you need before filtering such a frame (the layout of `properties`:
  [frame.md](frame.md)). Footprints take 40 s once `properties` is dropped: 15,697
  `Polygon Z`, 1063 `MultiPolygon Z`, 3 empty (two 2-triangle railings and a stair landing).
  The sliver rule of `plan_outline` (triangles under 1e-9 of the extent squared are left
  out) decides 25 of these: without it GEOS's robustness fallback leaves hairline gaps in
  24 door and furniture outlines, and the Jordahl console's two parts become one Polygon.
- `rvt-advanced-sample-arch.ifc`: 5696 elements, 5 storeys; 5734 rows (30 ports with a null
  `container`, 8 spatial), extent x -107,474..211,591, y -53,399..98,364, z -1829..12,485 mm
  in shared coordinates (the survey point), 49 s to read.
- `rvt-advanced-sample/speckle-v3-objects.parquet` (`data.RVT_SPECKLE_V3_OBJECTS`, 16.7 MB,
  44,452 objects in four versions, each version's root first: arch `d5e5bacdde` 13,194
  objects, str `b66058b4d1` 5355, mep `4413abe507` 16,173, and `5cb009bb77`,
  `rvt-advanced-sample-arch.ifc` received by Speckle's IFC importer, 9730 objects in metres;
  all built from the bundles by specklepy 2026.9.1 (`v3_route` "projected", specklepy's word),
  so that an element whose definition is empty carries its own geometry) and the bundles
  `rvt-advanced-sample/speckle-bundle/<model id>/<version id>/<name>.<table>.parquet`
  (`data.RVT_SPECKLE_ARCH/STR/MEP`, 18 tables each, and `data.RVT_SPECKLE_IFC`, 15: no
  `REVIT_TABLES`; `data.fetch_bundle`); `bundle_objects`
  reproduces each version from its bundle, object for object, in 3 to 4 s. `read_speckle` with
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
  -13.7..41.9 m. IFC: 5574 rows (5569 elements + 5 levels), 5512 with a mesh; the bundle has
  no `category`, so `class` is the `InstanceProxy` type or null. Text that reads as a number
  (#35): 8 IFC elements are named by one (`192754`..`192761`), 136 of the 167 arch `Mark`s
  and 107 of its 229 `Type Mark`s, 5069 of the 5220 mep `Mark`s (1348 of them numbers on the
  objects).

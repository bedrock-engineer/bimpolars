# The frame: rows and columns

A `BimDataFrame` (`src/bimpolars/frame.py`) is a polars DataFrame with one row per element and
one model CRS as PROJJSON (`src/bimpolars/crs.py`). The readers fill the columns in
`MODEL_COLUMNS`; `simple_features` adds the `FEATURE_COLUMNS` ([features.md](features.md)),
`lightbim=True` the `LIGHTBIM_COLUMNS` ([lightbim.md](lightbim.md)) and `geobim.with_geog` the
`geog` column ([geobim.md](geobim.md)).

## Rows

Both readers give one row per element: `read_ifc` a row per `IfcProduct`, `read_speckle` a row
per element of the version and of its bundle ([speckle.md](speckle.md)).

On the structural model 1,284 of the 2,647 element rows are `IfcOpeningElement` with the solid
mesh of the void, so glTF and LightBIM filter `class != "IfcOpeningElement"`, and a caller
filters them out before `simple_features` for a plan of the elements (`simple_features`
outlines every mesh, openings included, because GeoParquet needs `footprint` null exactly when
`bbox` is). Such a filter keeps the spatial rows: `simple_features` raises when a row's
`container` has no row in the frame, rather than silently placing the footprint at the
element's own level. `container` is null for `IfcDistributionPort` rows (ports hang off their
element, not a spatial element).

## Properties

The inference of property dtypes across rows lives in `src/bimpolars/properties.py`
(`properties_series`); `read_ifc` and `read_speckle` both use it. The dense `properties` Struct
of a model with many property sets costs gigabytes (the architectural model in
[test-data.md](test-data.md)): see #15 before changing its layout.

## Meshes

`src/bimpolars/mesh.py` holds the `Mesh` the readers end with, `Style`, `styled_mesh` (faces
sorted by style), `mesh_frame`, which builds the `mesh` and `bbox` columns from it, and
`mesh_arrays`, which reads a `mesh` column back into flat arrays with per-row offsets. Code that
reads a `mesh` column uses `mesh_arrays` (#18: features and lightbim do; geobim, gltf and
speckle still walk the column themselves).

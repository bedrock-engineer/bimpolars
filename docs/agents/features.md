# Footprints and GeoParquet

`simple_features` (`src/bimpolars/features.py`) adds the `FEATURE_COLUMNS` `footprint_ref` and
`footprint`, and with `lightbim=True` the LightBIM columns ([lightbim.md](lightbim.md)).
`src/bimpolars/geoparquet.py` builds the `geo` metadata that makes the written Parquet
GeoParquet.

## Footprints

- A row's `footprint` is null exactly when its `mesh` is null; a mesh without plan area gets
  `POLYGON Z EMPTY`. The footprint's bounds stay inside the row's `bbox`.
- `simple_features` outlines every mesh, openings included, and raises when a row's `container`
  has no row in the frame ([frame.md](frame.md)).
- `plan_outline` leaves out sliver triangles (under 1e-9 of the extent squared): without that
  rule GEOS's robustness fallback leaves hairline gaps in outlines ([test-data.md](test-data.md),
  the architectural model). It retries a union that GEOS's floating-point overlay fails on
  (one MEP stair mesh of 720 faces does: `e1d4bdfb-2688-47c1-a386-3b48116e5fd2-0008cb48`,
  pinned in `test_mep`) with `grid_size` = the sliver fraction times the extent, which is the
  fixed-precision overlay that cannot fail.

## Geometry columns

Simple-feature columns (`footprint`, `geog` and the LightBIM geometry `lightbim`) are ISO WKB
with Z in a `Binary` column, so that GeoParquet readers (DuckDB spatial, geopandas, GDAL) take
them as they are; shapely decodes them. Meshes stay nested columns.

A geometry column that is listed in `geoparquet.GEOMETRY_COLUMNS` (`geog`, `footprint`,
`lightbim`) gets `geo` metadata on write, the first present one as the primary. `bbox` is the
covering of the columns in the model CRS: the `geo` entry points readers at the column's x and
y fields, which every footprint lies within, so that a spatial filter can skip row groups. A
GeoParquet covering has no z slot, and the column's `zmin..zmax` is the mesh's, which a
footprint flat at its storey's elevation often lies outside of; the column keeps all six.

`geog` has a CRS of its own and no covering ([geobim.md](geobim.md)).

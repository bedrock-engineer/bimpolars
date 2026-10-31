# Georeferencing: CRSs, PROJ, to_crs and geog

`src/bimpolars/geobim.py` georeferences a model: `site_crs`, `set_crs`, `to_crs`,
`with_geog`, `affine_transformation` and `linearized_site_to_ecef`, all exported at the top
level too. pyproj is a core dependency: every CRS argument goes through `crs.resolve_crs`, and
a model stores its CRS as PROJJSON (`src/bimpolars/crs.py`).

## to_crs

- `to_crs` moves `mesh`, the WKB columns of `geobim.WKB_COLUMNS` (`footprint`, `lightbim`;
  never `geog`, which is always OGC:CRS84h) and `elevation` (as a point at the plan centre of
  the model).
- It recomputes `bbox` from the moved meshes (from the eight moved corners of the old `bbox`
  when the frame has no `mesh` column), and keeps the WKB columns inside their `bbox` with
  `features._clamp_xy`, which clips x and y and keeps z (`shapely.transform` drops z unless
  `include_z=True`).
- It nulls `native_geom` when the transformation is not affine over the model, and otherwise
  composes the 4x4 matrix of `affine_transformation` onto each sweep's `placement`
  (`geobim._place_sweeps`): the matrix carries the unit scale, so the profile and the depth
  stay in the sweep's own unit.
- `lightbim_qa.deviation` is a length: `geobim._scale_deviations` scales it by the matrix's
  cube-root determinant, or nulls it when there is no matrix.
- Targets have three axes (`EPSG:7415`, never `EPSG:28992`): the model's unit becomes the
  target's, and a model stays in one unit per axis.

## PROJ transformations

- Every transformation goes through `geobim._transformer`, which goes through the base CRS of a
  derived compound CRS. PROJ applies the deriving conversion of a DerivedVerticalCRS after the
  geodetic step of the horizontal part, so a 3D Helmert transformation (the fallback without
  grids) reads a millimetre site's heights as metres: 0.7 m of horizontal error at 50 m height,
  and an ECEF linearization tilted by 0.8 degrees.
- Without the NSGI grids (`nl_nsgi_rdtrans2018.tif`, `nl_nsgi_nlgeo2018.tif`; cdn.proj.org is
  blocked from the cloud containers, and the PROJ network is off) PROJ falls back to a Helmert
  transformation that is good to about a metre and a ballpark vertical step that passes NAP
  heights off as ellipsoidal heights. `geobim._transformer` warns (a `UserWarning` pointing at
  the caller) when `TransformerGroup(...).best_available` is False; the `description` of a lazy
  `Transformer` does not say which operation runs. A test of a transformation to ETRS89 states
  that tolerance or checks the group.

## geog

- `geog` (`geobim.with_geog`) is `footprint` through `geobim._transformer` to OGC:CRS84h
  (`crs.CRS84H`, the PROJJSON written out so that the metadata does not change with PROJ), null
  exactly when `footprint` is, `POLYGON Z EMPTY` where the footprint is.
- `to_crs` never moves it: `geoparquet.COLUMN_CRS` lists the columns with a CRS of their own,
  and `geobim.WKB_COLUMNS` leaves them out.
- Its `geo` entry has `CRS84H` as the CRS, the bounds of its own coordinates as the file `bbox`
  (decoded with shapely on write) and no covering, since `bbox` is in the model CRS; geopandas's
  `bbox=` filter then refuses the file (it needs a covering on the primary column).
- pyogrio writes a CRS as WKT1, which cannot say a 3D geographic CRS, so
  `GeoDataFrame.to_file` on a `geog` frame needs `set_crs("OGC:CRS84", allow_override=True)`
  first; `to_json()` works as it is.
- The Sixty5 site origin is at 5.46015 E, 51.44648 N (Strijp-S); the model lies within 0.002
  degrees.

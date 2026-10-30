# LightBIM

`lightbim.lightbim_features` (`src/bimpolars/lightbim.py`) reduces the elements to
structural-analysis members: the `LIGHTBIM_COLUMNS` `lightbim`, `lightbim_kind` and
`lightbim_qa`, which `simple_features(..., lightbim=True)` adds. Openings get no member
([frame.md](frame.md)).

## The member

- The member comes from the caller's `kind` expression, then `lightbim.CLASSES` (IFC class,
  `Class.PREDEFINEDTYPE`, Revit category; a None entry means no member: openings, spaces,
  reinforcement), then the shape test.
- The shape test takes the tightest of a world box, a plan-PCA upright box and a 3D-PCA box,
  snapped within 5 degrees; the plate test runs before the line test, `SLENDERNESS` 4.
- The table's role holds only with the table's member (a `kind` member drops it), and only the
  table's supports turn into piles by shape: the caller's `kind` always wins.
- `load_bearing`: `LoadBearing` of a `Pset_` set, then of any set, then Revit's `Structural`,
  then the table.

## The geometry

- The axis or plane comes from the largest sweep of `native_geom` ([ifc.md](ifc.md)), else from
  the box. Curve: the extrusion through the profile centroid, unless the profile is a slender
  side view. Surface: the extrusion is the normal when no longer than the profile's short side,
  else the profile's short axis.
- The extent always comes from the mesh: the curve spans the vertices' projection, a surface is
  `plan_outline` of the mesh in the plane's own coordinates, so walls get holes.
- A support lies at `bbox.zmin`, a mass at the divergence-theorem centre of gravity (the box
  centre when that falls outside the box: open or inside-out meshes, 4 of them on the Revit
  samples).
- `shapely.area` and `is_valid` are 2D: a wall's vertical polygon has no plan area, so tests
  measure it in its own plane.

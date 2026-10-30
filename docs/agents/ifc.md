# read_ifc

`read_ifc` (`src/bimpolars/ifc.py`) reads an IFC with ifcopenshell, the `[ifc]` extra:
`bimpolars.__getattr__` imports the module when `read_ifc` is first looked up. Its rows are
every `IfcProduct`, openings included ([frame.md](frame.md)).

## Attributes

`read_ifc` adds the set `properties.ATTRIBUTES_PSET` (`attributes`) with the `PredefinedType`
of `ifcopenshell.util.element.get_predefined_type` (the type's, `ObjectType` for USERDEFINED),
because the LightBIM class table keys `IfcSlab.BASESLAB` (a raft) on it.

## native_geom

`native_geom` (`frame.NATIVE_GEOM_DTYPE`, a List of `SWEEP_DTYPE`) holds the
`IfcExtrudedAreaSolid` body items `read_ifc` can read:

- Mapped items resolved recursively; a DIFFERENCE or INTERSECTION `IfcBooleanResult` reduced
  to its first operand (`clipped`), a UNION to both operands.
- The profile as 2D ISO WKB in the sweep's XY plane: rectangle and hollow, circle and hollow,
  I-shape and asymmetric I-shape in IFC2X3's and IFC4's attribute names, arbitrary closed with
  voids; polylines, composite curves, trimmed circles and lines, indexed poly curves; arcs at
  11.25 degrees as the notebook did.
- `placement` row-major from sweep to model coordinates.

Other items (breps, revolved and tapered sweeps) and profiles it cannot outline are skipped, so
a list can cover part of a body; null without any. `read_speckle` leaves it Null. `to_crs`
moves the placements ([geobim.md](geobim.md)) and LightBIM takes its axes and planes from the
largest sweep ([lightbim.md](lightbim.md)).

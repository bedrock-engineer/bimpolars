# glTF

`src/bimpolars/gltf.py` writes the meshes as binary glTF: `element_meshes_glb` (one primitive
per style) and `triangles_glb` (any triangles). It is hand-written glb 2.0, with no glTF library
in the core.

- Vertices in metres (scaled by the CRS unit) and z up in the accessors.
- A root node with `Y_UP_ROTATION` (named after the CRS; `extras.crs` is the BIM CRS in its
  own unit and `extras.unit` says the data is in metres) makes the scene y-up as the spec
  wants.
- A child node translated to the centre of the written vertices' bounding box (not the `bbox`
  column: openings and meshless rows are left out) holds the mesh, so that float32 stays
  precise far from the origin.
- One primitive and material per `(name, rgba)` style, sorted by name with the unstyled last;
  `IfcOpeningElement` rows left out.
- `normals=True` (default) writes non-indexed triangles with flat normals (3 vertices per face:
  ifcopenshell welds vertices by position, so averaged vertex normals would shade a box
  smooth); `normals=False` writes the `mesh` column as it is, indexed, without normals.
- The tests read the glb back with trimesh (dev dependency) and with a small reader of the
  chunks and accessors.

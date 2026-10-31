"""LightBIM: the structural-analysis reduction of every element, as simple features.

A structural analysis model reduces an element to what carries the load: a beam or a column
to the line along its axis, a slab or a wall to the surface in its midplane, a footing to the
plane it bears on, anything else to its footprint at the height of its centre of gravity.
``bimpolars.simple_features`` with ``lightbim=True`` adds that reduction next to the
footprint, in the three columns of ``LIGHTBIM_COLUMNS``:

- ``lightbim``: the reduction as ISO WKB with Z in the BIM CRS (``LIGHTBIM_DTYPE``): a
  ``LINESTRING Z`` for a curve member, a ``POLYGON Z`` or ``MULTIPOLYGON Z`` for the others.
- ``lightbim_kind`` (``KIND_DTYPE``): ``member`` (curve, surface, support or mass, which
  sets the geometry), ``role`` (beam, column, brace, pile, slab, wall, plate, roof, footing,
  raft, stair or ramp) and ``load_bearing``.
- ``lightbim_qa`` (``QA_DTYPE``): ``rule``, which decided the member (``kind``, the
  caller's expression; ``class``, the table ``CLASSES``; ``shape``, the box test),
  ``source``, what placed the reduction (``extrusion`` from ``native_geom``, with
  ``, cuts ignored`` when the source clipped the sweep, or ``mesh``), and ``deviation``, the
  farthest any vertex of the mesh lies from the reduction's axis or plane, in CRS units: half
  the diagonal of a curve member's section, the half-thickness of a slab, the height of a
  footing above its bearing level, a mass's farthest vertex from its centre of gravity's height.

The member comes from, in order: the ``kind`` expression the caller passes, the class table
(IFC classes with their predefined type, and Revit categories), then the shape: the tightest
of an upright box and the box around the mesh's principal axes, which is a surface when its
smallest extent is ``SLENDERNESS`` times smaller than the middle one (the plate test), else
a curve when the middle extent is that much smaller than the largest (the line test), else a
mass. The geometry's axis or plane comes from the element's native sweep when ``native_geom``
has one (the centroid line of the profile, or its midplane), else from the box; its extent
always comes from the mesh, so that a clipped sweep ends where the element ends and a wall with
openings gets its holes.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import polars as pl
import shapely

from bimpolars.features import _clamp_xy, plan_outline
from bimpolars.frame import NATIVE_GEOM_DTYPE, BimDataFrame, as_bim_dataframe
from bimpolars.mesh import mesh_arrays
from bimpolars.properties import ATTRIBUTES_PSET

MEMBERS: tuple[str, ...] = ("curve", "surface", "support", "mass")
"""The kinds of member, which set the geometry of the reduction."""

ROLES: tuple[str, ...] = (
    "beam",
    "column",
    "brace",
    "pile",
    "slab",
    "wall",
    "plate",
    "roof",
    "footing",
    "raft",
    "stair",
    "ramp",
)
"""The structural roles a member can have."""

MEMBER_DTYPE = pl.Enum(MEMBERS)
ROLE_DTYPE = pl.Enum(ROLES)
KIND_DTYPE = pl.Struct({"member": MEMBER_DTYPE, "role": ROLE_DTYPE, "load_bearing": pl.Boolean})
"""The dtype of ``lightbim_kind``."""

QA_DTYPE = pl.Struct({"rule": pl.String, "source": pl.String, "deviation": pl.Float64})
"""The dtype of ``lightbim_qa``."""

LIGHTBIM_DTYPE = pl.Binary
"""The dtype of ``lightbim``: ISO WKB with Z coordinates."""

LIGHTBIM_COLUMNS: tuple[str, ...] = ("lightbim", "lightbim_kind", "lightbim_qa")
"""The columns :func:`lightbim_features` adds, in order."""

SLENDERNESS = 4.0
"""The ratio between two extents of the box around a mesh that makes it a plate or a line."""

SNAP_ANGLE = 5.0
"""A principal axis within this many degrees of a model axis snaps to it: vertices bunched
up at notches and chamfers tilt the principal axes of a straight element slightly."""

ORIENTATION_ANGLE = 10.0
"""An axis within this many degrees of vertical belongs to a column, of horizontal to a beam;
a normal within it of vertical belongs to a slab, of horizontal to a wall."""

_COS_SNAP = math.cos(math.radians(SNAP_ANGLE))
_COS_UPRIGHT = math.cos(math.radians(ORIENTATION_ANGLE))
_SIN_FLAT = math.sin(math.radians(ORIENTATION_ANGLE))
_REQUIRED_COLUMNS = ("class", "mesh", "bbox", "footprint")
_KEY_SEPARATOR = "."

_Floats = npt.NDArray[np.float64]


class Kind(NamedTuple):
    """What the class table says about a class: any field may be None (undecided)."""

    member: str | None
    role: str | None
    load_bearing: bool | None


CLASSES: dict[str, Kind | None] = {
    # IFC classes, with the resolved PredefinedType after a dot where it changes the kind
    "IfcBeam": Kind("curve", "beam", None),
    "IfcColumn": Kind("curve", "column", None),
    "IfcMember": Kind("curve", None, None),
    "IfcMember.BRACE": Kind("curve", "brace", None),
    "IfcPile": Kind("curve", "pile", True),
    "IfcSlab": Kind("surface", "slab", None),
    "IfcSlab.BASESLAB": Kind("support", "raft", True),
    "IfcSlab.ROOF": Kind("surface", "roof", None),
    "IfcRoof": Kind("surface", "roof", None),
    "IfcPlate": Kind("surface", "plate", None),
    "IfcWall": Kind("surface", "wall", None),
    "IfcWallStandardCase": Kind("surface", "wall", None),
    "IfcWallElementedCase": Kind("surface", "wall", None),
    "IfcCurtainWall": Kind("surface", "wall", False),
    "IfcFooting": Kind("support", "footing", True),
    "IfcFooting.FOOTING_BEAM": Kind("curve", "beam", True),
    "IfcFooting.CAISSON_FOUNDATION": Kind("curve", "pile", True),
    "IfcStair": Kind("mass", "stair", None),
    "IfcStairFlight": Kind("mass", "stair", None),
    "IfcRamp": Kind("mass", "ramp", None),
    "IfcRampFlight": Kind("mass", "ramp", None),
    "IfcOpeningElement": None,
    "IfcSpace": None,
    "IfcAnnotation": None,
    "IfcReinforcingBar": None,
    "IfcReinforcingMesh": None,
    "IfcTendon": None,
    "IfcTendonAnchor": None,
    "IfcMechanicalFastener": None,
    # Revit categories, the class of read_speckle rows
    "Structural Columns": Kind("curve", "column", True),
    "Columns": Kind("curve", "column", None),
    "Structural Framing": Kind("curve", None, True),
    "Structural Foundations": Kind("support", "footing", True),
    "Floors": Kind("surface", "slab", None),
    "Walls": Kind("surface", "wall", None),
    "Roofs": Kind("surface", "roof", None),
    "Stairs": Kind("mass", "stair", None),
    "Ramps": Kind("mass", "ramp", None),
    "Rooms": None,
    "Areas": None,
    "Structural Rebar": None,
    "Structural Area Reinforcement": None,
    "Structural Path Reinforcement": None,
    "Structural Fabric Reinforcement": None,
    "Structural Rebar Couplers": None,
    "Structural Connections": None,
    "Structural Stiffeners": None,
}
"""The kind per source class: IFC classes (``IfcSlab``), IFC classes with their predefined
type (``IfcSlab.BASESLAB``, which wins over ``IfcSlab``) and Revit categories (``Walls``).

A None field is undecided and falls through: a ``role`` to the orientation of the member (an
``IfcMember`` or Revit ``Structural Framing`` is a column, a beam or a brace by the angle of its
axis), ``load_bearing`` to the ``LoadBearing`` property (IFC) or the ``Structural`` parameter
(Revit). A None entry is a class that is no member at all (an opening's void, a space;
reinforcement, connections and stiffeners, which an analysis model carries in its sections):
its rows get null LightBIM columns. A ``support`` from this table whose shape is a vertical
line (a pile exported as a footing) becomes a ``curve`` with the role ``pile``; a support the
``kind`` expression gives stays one. Edit the dict, or pass ``kind`` to
:func:`lightbim_features`, to change the table.
"""


def lightbim_features(
    bdf: BimDataFrame | pl.DataFrame, *, kind: pl.Expr | None = None
) -> pl.DataFrame:
    """The LightBIM columns of a frame that has its footprints.

    Args:
        bdf (BimDataFrame | pl.DataFrame): A BimDataFrame with ``class``, ``mesh``, ``bbox`` and the
            ``footprint`` of ``bimpolars.simple_features``. ``properties`` and ``native_geom``
            (``bimpolars.NATIVE_GEOM_DTYPE``) are used when present. Keep the ``properties`` sets
            ``attributes`` (the IFC ``PredefinedType``, which the class table keys rafts on: without
            it an ``IfcSlab.BASESLAB`` is a floor), the ``Pset_*Common`` sets (``LoadBearing``) and
            Revit's ``Structural`` parameter when slimming a frame: without them ``load_bearing`` is
            the table's or null.
        kind (pl.Expr | None): An expression over the frame that decides the kind of a row ahead of
            the class table and the shape test. It evaluates to a Struct with any of the fields
            ``member``, ``role`` and ``load_bearing``, or to a String ``member``; a null field falls
            through to the next rule, so an expression can set the kind of a few rows
            (``pl.when(...).then(pl.lit({"member": "curve", "role": "beam"}))``) or the role of all
            walls. A member that differs from the table's drops the table's role: the orientation
            decides it (a curve or a surface) or it stays null.

    Returns:
        pl.DataFrame: A frame with the columns of ``LIGHTBIM_COLUMNS``, one row per row of ``bdf``:
        null for a row without a mesh and for a class that is no member.

    Raises:
        ValueError: If a required column is missing.
        ValueError: If ``kind`` gives a member or a role that is not in ``MEMBERS`` or ``ROLES``.
        TypeError: If ``kind`` evaluates to something other than a Struct or a String.
        TypeError: If ``native_geom`` has another layout than ``bimpolars.NATIVE_GEOM_DTYPE``.
    """
    frame = as_bim_dataframe(bdf).frame
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"the LightBIM reduction needs the columns {', '.join(missing)}")
    decided = _decide(frame, kind)
    members = decided["member"].to_list()
    roles = decided["role"].to_list()
    load_bearing = decided["load_bearing"].to_list()
    rules = decided["rule"].to_list()
    skip = decided["skip"].to_numpy()
    sweeps = _sweeps(frame)
    arrays = mesh_arrays(frame["mesh"])
    boxes = np.column_stack(
        [
            frame["bbox"].struct.field(field).to_numpy()
            for field in ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
        ]
    )
    footprints = frame["footprint"].to_list()

    geometries: list[bytes | None] = [None] * len(frame)
    kinds: list[dict[str, Any] | None] = [None] * len(frame)
    qa: list[dict[str, Any] | None] = [None] * len(frame)
    for row in arrays.present.tolist():
        if skip[row]:
            continue
        vertices, faces = arrays.row(row)
        reduction = _reduce(
            vertices,
            faces,
            boxes[row],
            members[row],
            roles[row],
            rules[row],
            sweeps[row] if sweeps is not None else None,
            footprints[row],
        )
        geometries[row] = shapely.to_wkb(reduction.geometry, flavor="iso")
        kinds[row] = {
            "member": reduction.member,
            "role": reduction.role,
            "load_bearing": load_bearing[row],
        }
        qa[row] = {
            "rule": reduction.rule,
            "source": reduction.source,
            "deviation": reduction.deviation,
        }
    return pl.DataFrame(
        [
            pl.Series("lightbim", geometries, dtype=LIGHTBIM_DTYPE),
            pl.Series("lightbim_kind", kinds, dtype=KIND_DTYPE),
            pl.Series("lightbim_qa", qa, dtype=QA_DTYPE),
        ]
    )


class _Reduction(NamedTuple):
    geometry: shapely.Geometry
    member: str
    role: str | None
    rule: str
    source: str
    deviation: float


class _Line(NamedTuple):
    """A point and a unit vector: the axis of a curve member or the normal of a surface."""

    point: _Floats
    axis: _Floats


class _Box(NamedTuple):
    """An oriented box around a mesh, its axes sorted by extent, largest first."""

    axes: _Floats
    centre: _Floats
    extents: _Floats


def _reduce(
    vertices: _Floats,
    faces: npt.NDArray[np.int64],
    bbox: _Floats,
    member: str | None,
    role: str | None,
    rule: str,
    sweep: dict[str, Any] | None,
    footprint: bytes | None,
) -> _Reduction:
    """Reduces one mesh to its LightBIM feature."""
    box: _Box | None = None
    line: _Line | None = None
    shape_line: _Line | None = None
    source = "mesh"
    if member is None:
        box = _tightest_box(vertices)
        member, axis = _shape(box)
        shape_line = _Line(box.centre, axis)
    elif member == "support" and rule == "class":
        # A pile exported as a footing: a support whose shape is a vertical line. A support
        # the caller's kind gives stays one (a tall pedestal)
        box = _tightest_box(vertices)
        shape_member, axis = _shape(box)
        if shape_member == "curve" and abs(axis[2]) >= _COS_UPRIGHT:
            member, role, rule = "curve", "pile", "shape"
            shape_line = _Line(box.centre, axis)
    if member in ("curve", "surface"):
        if sweep is not None:
            line = _sweep_line(sweep, member)
            if line is not None:
                source = "extrusion, cuts ignored" if sweep["clipped"] else "extrusion"
        if line is None:
            line = shape_line or _box_line(box or _tightest_box(vertices), member, role)
    if member == "curve":
        assert line is not None
        geometry, deviation = _curve(vertices, line)
    elif member == "surface":
        assert line is not None
        geometry, deviation = _surface(vertices, faces, line)
    elif member == "support":
        geometry = shapely.force_3d(_plan(footprint), z=bbox[2])
        deviation = float((vertices[:, 2] - bbox[2]).max())
    else:
        z = _centre_of_gravity(vertices, faces)[2]
        geometry = shapely.force_3d(_plan(footprint), z=z)
        deviation = float(np.abs(vertices[:, 2] - z).max())
    if role is None and rule != "shape" and line is not None:
        role = _role_from_orientation(member, line.axis)
    geometry = _clamp_xy(geometry, bbox[:2], bbox[3:5])
    return _Reduction(geometry, member, role, rule, source, deviation)


def _plan(footprint: bytes | None) -> shapely.Geometry:
    """A row's footprint in 2D: the plan outline its support or mass lies flat in."""
    if footprint is None:  # simple_features gives every row with a mesh a footprint
        return shapely.Polygon()
    return shapely.force_2d(shapely.from_wkb(footprint))


def _decide(frame: pl.DataFrame, kind: pl.Expr | None) -> pl.DataFrame:
    """Per row, the member, role and load_bearing decided ahead of the shape, and the rule.

    Returns:
        pl.DataFrame: A frame with ``member`` and ``role`` (String, null when the shape or the
        orientation decides), ``load_bearing`` (Boolean), ``rule`` (``kind``, ``class`` or
        ``shape``) and ``skip`` (True for a class that is no member).
    """
    override = _override(frame, kind)
    table = _table(frame)
    sourced = _load_bearing(frame)
    member = pl.coalesce("kind_member", "class_member")
    return pl.concat([override, table, sourced.to_frame()], how="horizontal").select(
        member=member,
        # The table's role belongs to the table's member: a curve slab or a mass wall has none
        role=pl.coalesce(
            "kind_role", pl.when(member == pl.col("class_member")).then(pl.col("class_role"))
        ),
        load_bearing=pl.coalesce("kind_load_bearing", "sourced", "class_load_bearing"),
        rule=pl.when(pl.col("kind_member").is_not_null())
        .then(pl.lit("kind"))
        .when(pl.col("class_member").is_not_null())
        .then(pl.lit("class"))
        .otherwise(pl.lit("shape")),
        skip=pl.col("class_skip") & pl.col("kind_member").is_null(),
    )


def _override(frame: pl.DataFrame, kind: pl.Expr | None) -> pl.DataFrame:
    """The caller's ``kind`` expression as the columns kind_member, kind_role, kind_load_bearing."""
    n = len(frame)
    empty = pl.DataFrame(
        [
            pl.Series("kind_member", [None] * n, dtype=pl.String),
            pl.Series("kind_role", [None] * n, dtype=pl.String),
            pl.Series("kind_load_bearing", [None] * n, dtype=pl.Boolean),
        ]
    )
    if kind is None:
        return empty
    values = frame.with_columns(kind.alias("__kind"))["__kind"]  # a literal broadcasts
    if isinstance(values.dtype, pl.Struct):
        fields = {field.name for field in values.dtype.fields}
        unknown = fields - {"member", "role", "load_bearing"}
        if unknown:
            raise ValueError(
                f"kind gives the fields {sorted(unknown)}; it may give member, role and "
                "load_bearing"
            )
        parts = {name: values.struct.field(name) for name in fields}
    elif values.dtype in (pl.String, pl.Categorical) or isinstance(values.dtype, pl.Enum):
        parts = {"member": values}
    else:
        raise TypeError(
            f"kind must evaluate to a Struct of member, role and load_bearing or to a String "
            f"member, not to {values.dtype}"
        )
    columns = []
    for name, allowed in (("member", MEMBERS), ("role", ROLES)):
        if name in parts:
            part = parts[name].cast(pl.String)
            bad = part.drop_nulls().filter(~part.drop_nulls().is_in(allowed))
            if not bad.is_empty():
                raise ValueError(f"kind gives the {name} {bad[0]!r}; it must be one of {allowed}")
            columns.append(part.alias(f"kind_{name}"))
    if "load_bearing" in parts:
        columns.append(parts["load_bearing"].cast(pl.Boolean).alias("kind_load_bearing"))
    return empty.with_columns(columns)


def _table(frame: pl.DataFrame) -> pl.DataFrame:
    """The class table's say per row: class_member, class_role, class_load_bearing, class_skip."""
    classes = frame["class"].cast(pl.String)
    predefined = _predefined_type(frame)
    keys = (
        classes
        if predefined is None
        else pl.when(predefined.is_not_null())
        .then(classes + _KEY_SEPARATOR + predefined)
        .otherwise(classes)
    )
    rows = pl.DataFrame({"class": classes}).with_columns(key=keys)
    entries = pl.DataFrame(
        {
            "key": list(CLASSES),
            "member": [None if kind is None else kind.member for kind in CLASSES.values()],
            "role": [None if kind is None else kind.role for kind in CLASSES.values()],
            "load_bearing": [
                None if kind is None else kind.load_bearing for kind in CLASSES.values()
            ],
            "skip": [kind is None for kind in CLASSES.values()],
        },
        schema={
            "key": pl.String,
            "member": pl.String,
            "role": pl.String,
            "load_bearing": pl.Boolean,
            "skip": pl.Boolean,
        },
    ).with_columns(found=pl.lit(True))
    by_key = rows.join(entries, on="key", how="left", maintain_order="left")
    by_class = rows.join(
        entries, left_on="class", right_on="key", how="left", maintain_order="left"
    )
    take_key = pl.col("found").fill_null(False)
    return by_key.select(
        [
            pl.when(take_key).then(pl.col(name)).otherwise(by_class[name]).alias(f"class_{name}")
            for name in ("member", "role", "load_bearing")
        ]
        + [
            pl.when(take_key)
            .then(pl.col("skip"))
            .otherwise(by_class["skip"])
            .fill_null(False)
            .alias("class_skip")
        ]
    )


def _predefined_type(frame: pl.DataFrame) -> pl.Series | None:
    """The IFC predefined type per row, from the ``attributes`` property set, if any."""
    attributes = _property_set(frame, ATTRIBUTES_PSET)
    if attributes is None or "PredefinedType" not in attributes.to_schema():
        return None
    return frame["properties"].struct.field(ATTRIBUTES_PSET).struct.field("PredefinedType")


def _load_bearing(frame: pl.DataFrame) -> pl.Series:
    """Whether the source says a row bears load: IFC ``LoadBearing``, Revit ``Structural``.

    The first set that says it wins, in this order: the ``LoadBearing`` of the IFC ``Pset_``
    sets (``Pset_WallCommon``), the ``LoadBearing`` of any other set (ArchiCAD and Revit
    exports carry their own), then Revit's ``Structural`` parameter.
    """
    candidates: list[tuple[int, pl.Series]] = []
    if "properties" in frame.columns and isinstance(frame.schema["properties"], pl.Struct):
        properties = frame["properties"]
        for field in frame.schema["properties"].fields:
            if not isinstance(field.dtype, pl.Struct):
                continue
            names = field.dtype.to_schema()
            values = properties.struct.field(field.name)
            if "LoadBearing" in names:
                rank = 0 if field.name.startswith("Pset_") else 1
                candidates.append((rank, _as_boolean(values.struct.field("LoadBearing"))))
            elif "Structural" in names and field.name.endswith("Structural"):
                candidates.append((2, _as_boolean(values.struct.field("Structural"))))
    candidates.sort(key=lambda candidate: candidate[0])  # stable: Struct order within a rank
    if not candidates:
        return pl.Series("sourced", [None] * len(frame), dtype=pl.Boolean)
    return pl.DataFrame({f"c{i}": c for i, (_, c) in enumerate(candidates)}).select(
        sourced=pl.coalesce(pl.all())
    )["sourced"]


def _as_boolean(values: pl.Series) -> pl.Series:
    """Reads a property as a Boolean: a bool, a number (non-zero) or Yes/No text."""
    if values.dtype == pl.Boolean:
        return values
    if values.dtype.is_numeric():
        return values != 0
    if values.dtype == pl.String:
        lowered = values.str.to_lowercase()
        return pl.select(
            pl.when(lowered.is_in(["yes", "true", "1"]))
            .then(True)
            .when(lowered.is_in(["no", "false", "0"]))
            .then(False)
            .otherwise(None)
            .alias(values.name)
        ).to_series()
    return pl.Series(values.name, [None] * len(values), dtype=pl.Boolean)


def _property_set(frame: pl.DataFrame, name: str) -> pl.Struct | None:
    """The dtype of a property set in the ``properties`` Struct, if the frame has it."""
    if "properties" not in frame.columns:
        return None
    dtype = frame.schema["properties"]
    if not isinstance(dtype, pl.Struct):
        return None
    field = dtype.to_schema().get(name)
    return field if isinstance(field, pl.Struct) else None


def _sweeps(frame: pl.DataFrame) -> list[dict[str, Any] | None] | None:
    """Per row, the largest sweep of ``native_geom`` (by profile area times depth), if any."""
    if "native_geom" not in frame.columns or frame.schema["native_geom"] == pl.Null:
        return None
    if frame.schema["native_geom"] != NATIVE_GEOM_DTYPE:
        raise TypeError(
            f"native_geom has the dtype {frame.schema['native_geom']}, not "
            "bimpolars.NATIVE_GEOM_DTYPE; the LightBIM reduction reads only that layout"
        )
    largest: list[dict[str, Any] | None] = []
    for sweeps in frame["native_geom"].to_list():
        if not sweeps:
            largest.append(None)
            continue
        best, best_volume = None, -1.0
        for sweep in sweeps:
            volume = abs(shapely.from_wkb(sweep["profile"]).area * sweep["depth"])
            if volume > best_volume:
                best, best_volume = sweep, volume
        largest.append(best)
    return largest


def _sweep_line(sweep: dict[str, Any], member: str) -> _Line | None:
    """The axis of a curve member or the midplane of a surface, from its sweep.

    The axis of a curve member is the sweep direction through the profile's centroid. Only a
    beam extruded across its width, whose profile is its side view (a slender profile,
    ``SLENDERNESS`` times longer than it is wide and than the depth), gets the profile's
    long axis at mid-depth instead; a short stub keeps its extrusion as the axis. The normal
    of a surface is the sweep direction when the extrusion is no longer than the profile's
    short side (a slab), else the profile's short axis (a wall); the plane passes through the
    middle of the sweep.

    Returns:
        _Line | None: None when the profile is degenerate.
    """
    placement = np.asarray(sweep["placement"], dtype=float).reshape(4, 4)
    profile = shapely.from_wkb(sweep["profile"])
    if profile.is_empty or profile.area == 0.0:
        return None
    linear = placement[:3, :3]
    extrusion = linear @ (np.asarray(sweep["direction"], dtype=float) * float(sweep["depth"]))
    length = float(np.linalg.norm(extrusion))
    if length == 0.0:
        return None
    envelope = shapely.oriented_envelope(profile)
    if envelope.geom_type != "Polygon":
        return None
    corners = np.asarray(envelope.exterior.coords)[:4]
    edges = np.roll(corners, -1, axis=0) - corners
    sides = np.linalg.norm(edges[:2], axis=1)
    short, long = int(np.argmin(sides)), int(np.argmax(sides))
    if sides[long] == 0.0:
        return None

    def lift(xy: npt.ArrayLike) -> _Floats:
        x, y = np.asarray(xy, dtype=float)
        return np.asarray((placement @ np.array([x, y, 0.0, 1.0]))[:3] + extrusion / 2)

    def direction(xy: npt.ArrayLike) -> _Floats:
        x, y = np.asarray(xy, dtype=float)
        vector = linear @ np.array([x, y, 0.0])
        return np.asarray(vector / np.linalg.norm(vector))

    long_side = float(np.linalg.norm(linear @ np.append(edges[long], 0.0)))
    short_side = float(np.linalg.norm(linear @ np.append(edges[short], 0.0)))
    if member == "curve":
        sideways = long_side >= SLENDERNESS * max(short_side, length)
        if not sideways:
            return _Line(lift(profile.centroid.coords[0]), extrusion / length)
        return _Line(lift(corners.mean(axis=0)), direction(edges[long]))
    if length <= short_side:
        return _Line(lift(profile.centroid.coords[0]), extrusion / length)
    return _Line(lift(corners.mean(axis=0)), direction(edges[short]))


def _box_line(box: _Box, member: str, role: str | None) -> _Line:
    """The axis (longest extent) or the normal (smallest extent) of the box around a mesh.

    A wall's normal is horizontal: when the smallest extent of the box is vertical (a wall
    that is wider than high, or an L-shaped one, bimpolars#3), the shorter of the horizontal
    extents gives the normal instead.
    """
    if member == "curve":
        return _Line(box.centre, box.axes[0])
    normal = box.axes[2]
    if role == "wall" and abs(normal[2]) > 0.5:
        horizontal = [i for i in range(3) if abs(box.axes[i][2]) <= 0.5]
        if horizontal:
            normal = box.axes[max(horizontal)]
    return _Line(box.centre, normal)


def _tightest_box(vertices: _Floats) -> _Box:
    """The tightest of the boxes around a mesh: upright, or along its principal axes.

    Three boxes are tried: one along the model axes, an upright one turned in plan to the
    principal axis of the vertices' plan positions, and one along the three principal axes
    of the vertices (an inclined brace). Principal axes within ``SNAP_ANGLE`` of a model
    axis snap to it. A later box wins only when it is at least 5 % smaller.
    """
    mean = vertices.mean(axis=0)
    centred = vertices - mean
    candidates = [np.eye(3)]
    plan = _principal_direction(centred[:, :2])
    candidates.append(np.array([[plan[0], plan[1], 0.0], [-plan[1], plan[0], 0.0], [0, 0, 1.0]]))
    if len(vertices) >= 3:
        candidates.append(_snapped(np.linalg.svd(centred, full_matrices=False)[2]))
    best: tuple[float, _Floats, _Floats, _Floats] | None = None
    for axes in candidates:
        projected = centred @ axes.T
        low, high = projected.min(axis=0), projected.max(axis=0)
        volume = float(np.prod(high - low))
        if best is None or volume < 0.95 * best[0]:
            best = (volume, axes, low, high)
    _, axes, low, high = best  # type: ignore[misc]
    extents = high - low
    order = np.argsort(-extents, kind="stable")
    centre = mean + axes.T @ ((low + high) / 2)
    return _Box(axes[order], centre, extents[order])


def _principal_direction(plan: _Floats) -> _Floats:
    """The principal axis of 2D points, snapped to a model axis within ``SNAP_ANGLE``."""
    if len(plan) < 2 or not plan.any():
        return np.array([1.0, 0.0])
    axis = np.linalg.svd(plan, full_matrices=False)[2][0]
    if abs(axis[0]) >= _COS_SNAP:
        return np.array([1.0, 0.0])
    if abs(axis[1]) >= _COS_SNAP:
        return np.array([0.0, 1.0])
    return np.asarray(axis / np.linalg.norm(axis))


def _snapped(axes: _Floats) -> _Floats:
    """Snaps principal axes within ``SNAP_ANGLE`` of a model axis to it, kept orthonormal."""
    snapped: list[_Floats] = []
    for axis in axes:
        nearest = int(np.argmax(np.abs(axis)))
        if abs(axis[nearest]) >= _COS_SNAP:
            axis = np.sign(axis[nearest]) * np.eye(3)[nearest]
        for earlier in snapped:
            axis = axis - (axis @ earlier) * earlier
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            # Two axes snapped to one model axis: take the model axis least aligned so far
            spent = np.abs(np.array(snapped)).sum(axis=0)
            axis = np.eye(3)[int(np.argmin(spent))]
            for earlier in snapped:
                axis = axis - (axis @ earlier) * earlier
            norm = float(np.linalg.norm(axis))
        snapped.append(axis / norm)
    return np.array(snapped)


def _shape(box: _Box) -> tuple[str, _Floats]:
    """The member the box test gives, with its normal (surface) or axis (curve or mass)."""
    largest, middle, smallest = box.extents
    if middle > 0.0 and smallest * SLENDERNESS <= middle:
        return "surface", box.axes[2]
    if largest > 0.0 and middle * SLENDERNESS <= largest:
        return "curve", box.axes[0]
    return "mass", box.axes[0]


def _curve(vertices: _Floats, line: _Line) -> tuple[shapely.Geometry, float]:
    """The axis line, from the first to the last vertex along it, and the mesh's half-width."""
    relative = vertices - line.point
    along = relative @ line.axis
    across = relative - np.outer(along, line.axis)
    deviation = float(np.linalg.norm(across, axis=1).max())
    ends = line.point + np.outer([along.min(), along.max()], line.axis)
    return shapely.LineString(ends), deviation


def _surface(
    vertices: _Floats, faces: npt.NDArray[np.int64], line: _Line
) -> tuple[shapely.Geometry, float]:
    """The mesh's outline projected onto the plane, holes included, and its half-thickness."""
    normal = line.axis
    s = np.eye(3)[int(np.argmin(np.abs(normal)))]
    s = s - (s @ normal) * normal
    s /= np.linalg.norm(s)
    t = np.cross(normal, s)
    relative = vertices - line.point
    depth = relative @ normal
    deviation = float(np.abs(depth).max())
    local = np.column_stack([relative @ s, relative @ t, depth])
    outline = plan_outline(local, faces)

    def lift(coordinates: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return line.point + np.outer(coordinates[:, 0], s) + np.outer(coordinates[:, 1], t)

    return shapely.transform(shapely.force_3d(outline), lift, include_z=True), deviation


def _centre_of_gravity(vertices: _Floats, faces: npt.NDArray[np.int64]) -> _Floats:
    """The centre of gravity of a closed mesh (divergence theorem); the box centre otherwise.

    An open or inconsistently wound mesh (a planting, a railing of loose panels) can still
    enclose a signed volume, around a centre that lies outside the mesh: a centre outside the
    box around the vertices falls back to the box centre too.
    """
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    box_centre = (low + high) / 2
    if len(faces) == 0:
        return np.asarray(box_centre)
    a, b, c = (vertices[faces[:, i]] - box_centre for i in range(3))
    tetrahedra = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    volume = float(tetrahedra.sum())
    if abs(volume) <= 1e-6 * float(np.prod(np.maximum(high - low, 1e-300))):
        return np.asarray(box_centre)
    moment = (tetrahedra[:, None] * (a + b + c) / 4).sum(axis=0)
    centre = box_centre + moment / volume
    if (np.abs(centre - box_centre) > (high - low) / 2).any():
        return np.asarray(box_centre)
    return np.asarray(centre)


def _role_from_orientation(member: str, axis: _Floats) -> str | None:
    """A role from the angle of a curve member's axis or a surface's normal with vertical."""
    upright = abs(axis[2])
    if member == "curve":
        if upright >= _COS_UPRIGHT:
            return "column"
        return "beam" if upright <= _SIN_FLAT else "brace"
    if member == "surface":
        if upright >= _COS_UPRIGHT:
            return "slab"
        return "wall" if upright <= _SIN_FLAT else "plate"
    return None

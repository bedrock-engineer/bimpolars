"""Property sets as a polars Struct column, with one dtype per property across the rows.

Every reader has the same problem: each element carries named sets of properties (IFC
property and quantity sets, Speckle properties) whose values are typed per element, while a
column needs one dtype. ``properties_series`` infers it from the values: a property whose
values agree across rows gets their dtype, a property whose values disagree becomes a String.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from polars.datatypes import DataTypeClass

ATTRIBUTES_PSET = "attributes"
"""The property set a reader fills with an entity's own attributes that name its kind: for
IFC, ``PredefinedType`` (the element's or its type's, ``ObjectType`` when USERDEFINED)."""

type Kind = tuple[bool, str | None]
"""How a value is typed: whether it is a list, and the scalar kind of it or of its items."""

_SCALAR_DTYPES: dict[str, pl.DataType] = {
    "bool": pl.Boolean(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "str": pl.String(),
}


def properties_series(
    rows: Sequence[Mapping[str, Mapping[str, Any]]], name: str = "properties"
) -> pl.Series:
    """Builds a Struct column from the property sets of every row.

    Each set becomes a Struct field holding a Struct of its properties. A property gets the
    dtype of its values when they agree across rows: Boolean, Int64, Float64, String, or a
    List of one of those; integers and floats together widen to Float64. When the values
    disagree, or when a value is nested (a dict, a list of lists), the property becomes a
    String with the value's text, JSON for containers. A row without a set has a null for it,
    so a row without any set is a Struct of nulls. Without any property at all the column has
    the Null dtype.

    The Struct is dense: every set's every property is a column over all rows. A model whose
    exporter writes one set per type with hundreds of properties each (ArchiCAD does) gets
    tens of thousands of columns, which cost memory in proportion even though most values
    are null.

    Args:
        rows (Sequence[Mapping[str, Mapping[str, Any]]]): Per row, its sets by name, each a mapping
            of property name to value.
        name (str): The column name.

    Returns:
        pl.Series: A Series of ``len(rows)`` Structs.
    """
    kinds: dict[str, dict[str, set[Kind]]] = {}
    for row in rows:
        for pset, values in row.items():
            pset_kinds = kinds.setdefault(pset, {})
            for key, value in values.items():
                if value is not None:
                    pset_kinds.setdefault(key, set()).add(_kind(value))
    dtypes = {
        pset: {key: _dtype(key_kinds) for key, key_kinds in pset_kinds.items() if key_kinds}
        for pset, pset_kinds in kinds.items()
    }
    dtypes = {pset: fields for pset, fields in dtypes.items() if fields}
    if not dtypes:
        return pl.Series(name, [None] * len(rows), dtype=pl.Null)
    # Column by column: a model has hundreds of sets on thousands of rows, most of them absent
    # on most rows, so each set is built from the rows that have it and gathered into place.
    # to_struct shares the gathered columns rather than copying them.
    columns = []
    for pset, fields in dtypes.items():
        positions = [index for index, row in enumerate(rows) if pset in row]
        present = [rows[index][pset] for index in positions]
        struct = pl.DataFrame(
            [
                pl.Series(key, [_coerce(values.get(key), dtype) for values in present], dtype=dtype)
                for key, dtype in fields.items()
            ]
        ).to_struct(pset)
        lookup = np.full(len(rows), -1, dtype=np.int64)
        lookup[positions] = np.arange(len(positions))
        columns.append(struct.gather(pl.Series(lookup).replace(-1, None).cast(pl.UInt32)))
    return pl.DataFrame(columns).to_struct(name)


def _kind(value: object) -> Kind:
    """Classifies a value: a list of one scalar kind, or a scalar."""
    if isinstance(value, list | tuple):
        return (True, _one_kind({_scalar_kind(item) for item in value if item is not None}))
    return (False, _scalar_kind(value))


def _scalar_kind(value: object) -> str:
    """The kind of a scalar: bool, int, float, str, or other for anything nested or unknown."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "other"


def _one_kind(kinds: set[str]) -> str | None:
    """The scalar kind a set of kinds agree on: int and float widen to float, else other."""
    if not kinds:
        return None
    if kinds <= {"int", "float"}:
        return "float" if "float" in kinds else "int"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "other"


def _dtype(kinds: set[Kind]) -> pl.DataType:
    """The polars dtype for the kinds a property has across rows."""
    list_kinds = {is_list for is_list, _ in kinds}
    if len(list_kinds) != 1:
        return pl.String()
    scalar = _one_kind({kind for _, kind in kinds if kind is not None})
    if scalar == "other":
        return pl.String()
    dtype = _SCALAR_DTYPES[scalar] if scalar is not None else pl.String()
    return pl.List(dtype) if list_kinds == {True} else dtype


def _coerce(value: Any, dtype: pl.DataType | DataTypeClass) -> Any:
    """Converts a property value to what polars expects for ``dtype``."""
    if value is None:
        return None
    if dtype == pl.String():
        if isinstance(value, str):
            return value
        if isinstance(value, list | tuple | dict):
            return json.dumps(value, default=str)
        return str(value)
    if dtype == pl.Float64():
        return float(value)
    if isinstance(dtype, pl.List):
        return [_coerce(item, dtype.inner) for item in value]
    return value

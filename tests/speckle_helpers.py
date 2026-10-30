"""Helpers the Speckle tests share: reading the shape of a version's v3 objects."""

from __future__ import annotations

import polars as pl

INSTANCE_TYPE = "Speckle.Core.Models.Instances.InstanceProxy"


def faces_by_row(frame: pl.DataFrame) -> dict[str, int]:
    """The face count of every row that has a mesh."""
    n_faces = pl.col("mesh").struct.field("faces").list.len() // 3
    return dict(frame.filter(pl.col("mesh").is_not_null()).select("id", n_faces).iter_rows())


def own_geometry_and_placements(objects: list[dict]) -> dict[str, list[int]]:
    """The elements that have geometry of their own and a placement, with the number of
    objects in each placement's definition."""
    root = objects[0]
    definitions = {
        proxy["applicationId"]: proxy.get("objects") or []
        for proxy in root["instanceDefinitionProxies"]
    }
    content = {member for members in definitions.values() for member in members}
    own: set[str] = set()
    placed: dict[str, list[int]] = {}
    for obj in objects[1:]:
        app_id = obj.get("applicationId")
        if app_id in content:
            continue
        base = app_id.split("-instance-")[0] if app_id else obj["id"]
        if obj.get("speckle_type") == INSTANCE_TYPE:
            placed.setdefault(base, []).append(len(definitions[obj["definitionId"]]))
        elif obj.get("displayValue"):
            own.add(base)
    return {base: sizes for base, sizes in placed.items() if base in own}


def material_targets(root: dict) -> dict[str, list[str]]:
    """The applicationIds each RenderMaterialProxy of a root targets, by the proxy's."""
    return {
        proxy["applicationId"]: list(proxy.get("objects") or [])
        for proxy in root.get("renderMaterialProxies") or []
    }

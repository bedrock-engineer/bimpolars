"""bundle_objects on the Revit Advanced Sample bundles, against the bucket's v3 objects.

The bucket's ``speckle-v3-objects.parquet`` holds the three versions as the reference notebook
projected them (specklepy 2026.9.1 with the same two fixes), so the projection here must give
the same objects, and ``read_speckle`` the same frame, with the bundle named by its bucket URL."""

from __future__ import annotations

import dataclasses
import shutil
import warnings
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("specklepy")

from specklepy.objects.data_objects import DataObject
from specklepy.objects.models.collections.collection import Collection

from bimpolars import data, read_speckle, version_objects
from bimpolars.bundle import (
    DEFINITION_GEOMETRY_PREFIX,
    FIRST_PLACEMENT_SUFFIX,
    _keep_element_geometry,
    bundle_model,
    bundle_objects,
)
from bimpolars.speckle import LEVEL_CLASS
from tests.speckle_helpers import (
    INSTANCE_TYPE,
    material_targets,
    own_geometry_and_placements,
)

DATA_OBJECT_TYPE = "Objects.Data.DataObject"
VERSIONS = {"arch": data.RVT_SPECKLE_ARCH, "str": data.RVT_SPECKLE_STR, "mep": data.RVT_SPECKLE_MEP}
N_OBJECTS = {"arch": 13_194, "str": 5355, "mep": 16_173}
N_ROWS = {"arch": 5877, "str": 939, "mep": 8629}
N_MESHES = {"arch": 5568, "str": 918, "mep": 6942}
N_BUNDLE_OBJECTS = {"arch": 5872, "str": 935, "mep": 8625}
N_OWN_GEOMETRY_AND_PLACED = {"arch": 75, "str": 841, "mep": 10}
"""Elements with geometry of their own and a placement: what the plain projection drops."""
N_STYLED_DEFINITION_GEOMETRIES = {"arch": 719, "str": 27, "mep": 610}
"""Definition geometries the envelope gives a material: what the plain projection leaves grey."""


def bucket_url(version: data.SpeckleVersion) -> str:
    """The URL of the bundle's key prefix in the bucket, as a user without the mirror has it."""
    return f"{data.DATA_URL}{version.bundle}"


@pytest.fixture(scope="module", params=list(VERSIONS))
def which(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def version(which: str) -> data.SpeckleVersion:
    return VERSIONS[which]


@pytest.fixture(scope="module")
def objects(version: data.SpeckleVersion) -> list[dict]:
    return bundle_objects(bucket_url(version))


@pytest.fixture(scope="module")
def bucket_objects(version: data.SpeckleVersion) -> list[dict]:
    return version_objects(data.fetch(data.RVT_SPECKLE_V3_OBJECTS), version.version_id)


@pytest.fixture(scope="module")
def plain(version: data.SpeckleVersion) -> Collection:
    """specklepy's own projection of the bundle, without the two fixes."""
    return bundle_model(data.fetch_bundle(version.bundle)).to_base()


def element_objects(objects: list[dict]) -> set[str]:
    """The applicationIds of the DataObjects that are elements, not definition geometry."""
    return {
        obj["applicationId"]
        for obj in objects
        if obj.get("speckle_type") == DATA_OBJECT_TYPE
        and not obj["applicationId"].startswith(DEFINITION_GEOMETRY_PREFIX)
    }


def placements(root: Collection) -> list[str]:
    """The applicationIds of the InstanceProxy objects under a Base tree, depth first."""
    found: list[str] = []
    for child in root.elements:
        if isinstance(child, Collection):
            found += placements(child)
        elif child.speckle_type == INSTANCE_TYPE:
            found.append(child.applicationId)
    return found


def plain_element_objects(root: Collection) -> set[str]:
    """The same for specklepy's own projection, before serialization."""
    found: set[str] = set()
    for child in root.elements:
        if isinstance(child, Collection):
            found |= plain_element_objects(child)
        elif isinstance(child, DataObject) and not child.applicationId.startswith(
            DEFINITION_GEOMETRY_PREFIX
        ):
            found.add(child.applicationId)
    return found


def test_bundle_objects_match_the_bucket(
    objects: list[dict], bucket_objects: list[dict], which: str
) -> None:
    assert len(objects) == N_OBJECTS[which] == len(bucket_objects)
    root = objects[0]
    assert root["applicationId"] == "artifact-root"
    assert root["units"] == "mm"
    assert root["version"] == 4
    assert set(root["__closure"]) == {obj["id"] for obj in objects[1:]}
    # every object but the root is the bucket's, byte for byte: the id is the content hash
    assert {obj["id"] for obj in objects[1:]} == {obj["id"] for obj in bucket_objects[1:]}
    assert Counter(obj.get("speckle_type") for obj in objects) == Counter(
        obj.get("speckle_type") for obj in bucket_objects
    )
    theirs_root = bucket_objects[0]
    assert root["instanceDefinitionProxies"] == theirs_root["instanceDefinitionProxies"]
    # the root differs in its material proxies only: the reference notebook wrote them as
    # its own JSON, with other ids and no totalChildrenCount, in the same order and with
    # the same targets and material values
    mine, theirs = material_targets(root), material_targets(theirs_root)
    assert list(mine) == list(theirs)
    assert {name: sorted(targets) for name, targets in mine.items()} == {
        name: sorted(targets) for name, targets in theirs.items()
    }

    def value(proxy: dict) -> dict:
        return {
            key: item
            for key, item in proxy["value"].items()
            if key not in {"id", "totalChildrenCount"}
        }

    assert [value(proxy) for proxy in root["renderMaterialProxies"]] == [
        value(proxy) for proxy in theirs_root["renderMaterialProxies"]
    ]


def test_read_speckle_from_the_bucket_url(
    objects: list[dict], bucket_objects: list[dict], version: data.SpeckleVersion, which: str
) -> None:
    url = bucket_url(version)
    mine = read_speckle(objects, url)
    theirs = read_speckle(bucket_objects, data.fetch_bundle(version.bundle))
    assert mine.frame.schema == theirs.frame.schema
    assert mine.equals(theirs)
    assert len(mine.frame) == N_ROWS[which]
    assert mine.frame["mesh"].is_not_null().sum() == N_MESHES[which]
    assert mine.frame["id"].str.contains("-instance-").sum() == 0


def test_placed_elements_keep_their_geometry(
    objects: list[dict], plain: Collection, which: str
) -> None:
    both = own_geometry_and_placements(objects)
    assert len(both) == N_OWN_GEOMETRY_AND_PLACED[which]
    if which == "arch":
        # eight parapet walls place a cap (a one-object definition) on their own body
        assert Counter(tuple(sizes) for sizes in both.values()) == {(0,): 67, (1,): 8}
    else:
        assert set(map(tuple, both.values())) == {(0,)}
    # exactly their first placements are renamed, so that the element's id is free for its
    # own geometry; the other placements keep their -instance-N
    renamed = {
        obj["applicationId"]
        for obj in objects
        if obj.get("speckle_type") == INSTANCE_TYPE
        and obj["applicationId"].endswith(FIRST_PLACEMENT_SUFFIX)
    }
    assert renamed == {f"{base}{FIRST_PLACEMENT_SUFFIX}" for base in both}
    # specklepy's own projection has no DataObject for them
    without = plain_element_objects(plain)
    assert element_objects(objects) == without | set(both)
    assert without.isdisjoint(both)


def test_definition_geometry_carries_its_materials(
    objects: list[dict], plain: Collection, version: data.SpeckleVersion, which: str
) -> None:
    targets = material_targets(objects[0])
    definition_geometry = {
        obj["applicationId"]
        for obj in objects
        if obj.get("speckle_type") == DATA_OBJECT_TYPE
        and obj["applicationId"].startswith(DEFINITION_GEOMETRY_PREFIX)
    }
    styled = {
        target
        for names in targets.values()
        for target in names
        if target.startswith(DEFINITION_GEOMETRY_PREFIX)
    }
    assert len(styled) == N_STYLED_DEFINITION_GEOMETRIES[which]
    assert styled <= definition_geometry  # no proxy targets geometry that is not there
    assert not any(
        target.startswith(DEFINITION_GEOMETRY_PREFIX)
        for proxy in plain["renderMaterialProxies"]
        for target in proxy.objects
    )
    # so the objects alone style the instances as the objects with the bundle do, where
    # read_speckle falls back on the envelope
    bare = read_speckle(objects)
    full = read_speckle(objects, bucket_url(version))
    elements = full.frame.filter(pl.col("id").is_in(bare.frame["id"].implode()))
    assert elements["id"].to_list() == bare.frame["id"].to_list()
    assert elements["mesh"].equals(bare.frame["mesh"])
    assert full.frame.filter(pl.col("class") == LEVEL_CLASS).height > 0


def test_bundle_model(version: data.SpeckleVersion, which: str) -> None:
    folder = data.fetch_bundle(version.bundle)
    model = bundle_model(folder)
    assert Path(model.directory) == folder
    assert model.units == "mm"
    assert len(model.objects) == N_BUNDLE_OBJECTS[which]
    assert len(model.levels) == {"arch": 5, "str": 4, "mep": 4}[which]
    assert model.properties["modelPlacement.source"] == "projectBasePoint"
    assert model.geometries  # read from the folder


def test_closing_a_bundle_model_keeps_its_folder(tmp_path: Path) -> None:
    """specklepy's Model deletes its folder on close(), and so at the end of `with model:`;
    the folder of bundle_model is the user's (the data cache, or a shared mirror)."""
    copy = tmp_path / "str"
    shutil.copytree(data.fetch_bundle(data.RVT_SPECKLE_STR.bundle), copy)
    with bundle_model(copy) as model:
        n_geometries = len(model.geometries)
    assert n_geometries > 0
    model.close()
    assert sorted(path.name for path in copy.iterdir()) == sorted(
        f"advanced_sample_project_str.{table}.parquet" for table in data.BUNDLE_TABLES
    )
    assert len(bundle_objects(copy)) == N_OBJECTS["str"]


def test_only_the_json_pipeline_warning_is_silenced() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        objects = bundle_objects(data.fetch_bundle(data.RVT_SPECKLE_STR.bundle))
        warnings.warn("another deprecation", DeprecationWarning, stacklevel=1)
    assert len(objects) == N_OBJECTS["str"]
    assert [str(warning.message) for warning in caught] == ["another deprecation"]


def test_own_geometry_without_its_first_placement() -> None:
    """When an element's first placement has no instance node, specklepy names its other
    placements -instance-1, ... and none after the element: the own geometry goes before
    the first of them. Without any placement in the tree it goes to the root."""
    model = bundle_model(data.fetch_bundle(data.RVT_SPECKLE_STR.bundle))
    bundle = model.bundle
    edges = bundle.relations.display_instance_edges
    by_object: dict[int, list] = {}
    for edge in edges:
        by_object.setdefault(edge.src, []).append(edge)
    # two str elements with own geometry and one placement each
    shifted, unplaced = [
        object_k
        for object_k, placed in sorted(by_object.items())
        if len(placed) == 1 and bundle.relations.display_by_object(object_k)
    ][:2]
    # `shifted`: a second placement on a copy of its instance node, then the first node
    # dropped, so that the remaining placement is -instance-1
    (first,) = by_object[shifted]
    second = max(bundle.nodes) + 1
    bundle.nodes[second] = bundle.nodes[first.dst]
    edges.append(dataclasses.replace(first, dst=second, ord=first.ord + 1))
    del bundle.nodes[first.dst]
    # `unplaced`: its only instance node dropped
    del bundle.nodes[by_object[unplaced][0].dst]
    root = model.to_base()
    _keep_element_geometry(model, root)
    shifted_id = bundle.object_app_ids[shifted]
    unplaced_id = bundle.object_app_ids[unplaced]
    ids = [child.applicationId for child in root.elements if not isinstance(child, Collection)]
    names = placements(root)
    assert f"{shifted_id}-instance-1" in names
    assert shifted_id not in names and f"{shifted_id}{FIRST_PLACEMENT_SUFFIX}" not in names
    assert unplaced_id not in names
    assert unplaced_id in ids  # appended to the root
    flat: list[str] = []

    def walk(collection: Collection) -> None:
        for child in collection.elements:
            if isinstance(child, Collection):
                walk(child)
            else:
                flat.append(child.applicationId)

    walk(root)
    assert flat.index(shifted_id) == flat.index(f"{shifted_id}-instance-1") - 1
    assert flat.count(shifted_id) == 1 and flat.count(unplaced_id) == 1


def test_bundle_locations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        bundle_model(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match=r"eav\.objects"):
        bundle_model(tmp_path)
    with pytest.raises(ValueError, match="not in the bucket"):
        bundle_objects("https://example.com/bundles/advanced_sample_project_arch")

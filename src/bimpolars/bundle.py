"""Builds the v3 objects ``bimpolars.read_speckle`` reads from a Speckle bundle.

A Speckle version is published either as v3 objects (the JSON graph of
``bimpolars.speckle``) or, since Speckle 4, as a *bundle*: Parquet tables for the
properties (``eav``), the scene graph (``envelope``) and the geometry (``geometries``, SGEO
blobs). A bundle-only version has no v3 objects on the server; specklepy's
``Model.to_base()`` builds the legacy object tree from the bundle (its module
``specklepy.bundle.base_projection`` calls this a projection). ``bundle_objects`` builds
it from a bundle on disk or in the bimpolars bucket, and restores two things ``to_base``
loses: an element that is placed as an instance keeps the geometry it carries
itself, and the geometry of an instance definition keeps its materials.

This module needs specklepy: install the ``[speckle]`` extra. The core of bimpolars does not
import it; ``bimpolars.read_speckle`` reads the objects this module gives as plain JSON.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any

try:
    from specklepy.bundle.bundle_reader import read_bundle
    from specklepy.bundle.model import Model
    from specklepy.objects.base import Base
    from specklepy.objects.models.collections.collection import Collection
    from specklepy.objects.proxies import RenderMaterialProxy
    from specklepy.serialization.base_object_serializer import BaseObjectSerializer
    from specklepy.transports.memory import MemoryTransport
except ImportError as error:
    raise ImportError(
        "bimpolars.bundle needs specklepy: install the extra bimpolars[speckle]"
    ) from error

try:
    # specklepy's builders of an element's own geometry as a DataObject, and of the
    # material nodes to RenderMaterialProxies; private, which is why the extra pins specklepy
    from specklepy.bundle.base_projection import _geometry_object, _materials
except ImportError as error:
    raise ImportError(
        "bimpolars.bundle needs specklepy 2026.9.1, whose Model.to_base() it extends: "
        f"install the extra bimpolars[speckle] ({error})"
    ) from error

from bimpolars.speckle import SpeckleObject, bundle_folder

DEFINITION_GEOMETRY_PREFIX = "def-geo-"
"""The applicationId specklepy gives the geometry of an instance definition: the bundle's
geometry index after it."""

MATERIAL_PREFIX = "mat-"
"""The applicationId of a material's RenderMaterialProxy: the envelope's node id after it."""

FIRST_PLACEMENT_SUFFIX = "-instance-0"
"""What :func:`bundle_objects` appends to the applicationId of an element's first placement
when the element keeps geometry of its own under the plain applicationId."""

_JSON_PIPELINE_DEPRECATION = r".*JSON based send/receive pipelines are deprecated"
"""The text of the DeprecationWarning of specklepy's JSON pipeline (``MemoryTransport``,
``BaseObjectSerializer``), which :func:`bundle_objects` uses on purpose."""

_PLACEMENT = re.compile(r"^(?P<base>.+)-instance-\d+$")
"""A placement after the first: the element's applicationId plus ``-instance-N``."""


class _FolderModel(Model):  # type: ignore[misc]
    """A ``Model`` over a folder it does not own: closing it keeps the folder.

    specklepy's ``Model`` owns the temporary folder ``receive`` downloads into, and
    ``close()`` (also at the end of ``with model:``) deletes it. A bundle opened from the
    user's folder must survive that.
    """

    def close(self) -> None:
        """Marks the model closed and leaves its folder alone."""
        self._closed = True


def bundle_model(bundle: str | os.PathLike[str]) -> Model:
    """Opens a Speckle bundle as a specklepy ``Model``, from its folder, without a server.

    specklepy's ``receive`` downloads a version's bundle into a temporary folder and wraps it
    in a ``Model``; this does the same for a bundle that is already on disk, so that the
    model's objects, levels, materials, definitions and geometry (``model.objects``,
    ``model.levels``, ...) can be read, or built into v3 objects with ``bundle_objects``.

    Args:
        bundle (str | os.PathLike[str]): The folder holding the bundle's ``<name>.<table>.parquet``
            files, or the URL of the bundle's key prefix in the bimpolars bucket
            (``<DATA_URL><prefix>``, as ``bimpolars.data.fetch_bundle`` takes it), fetched into the
            data folder.

    Returns:
        Model: The model, without project, model or version id. Closing it, or leaving a ``with``
        block, keeps the folder, unlike a model specklepy's ``receive`` downloaded; its
        geometry stays readable from the folder afterwards.

    Raises:
        ValueError: If the folder does not exist, or the URL is not in the bucket.
        FileNotFoundError: If the folder lacks ``eav.objects``, ``eav.paths`` or ``eav.eav``.
        FileNotFoundError: If the folder lacks ``envelope.nodes`` or ``envelope.relations``.
        urllib.error.URLError: If downloading a table of a bundle named by its URL fails.
    """
    folder = bundle_folder(bundle)
    files = sorted(str(path) for path in folder.iterdir() if path.is_file())
    return _FolderModel(
        project_id="",
        model_id="",
        version_id="",
        directory=str(folder),
        files=files,
        bundle=read_bundle(str(folder)),
    )


def bundle_objects(bundle: str | os.PathLike[str]) -> list[SpeckleObject]:
    """Builds the v3 objects of a version from its Speckle bundle, root first.

    The objects are what specklepy's ``Model.to_base()`` builds, serialized as a Speckle
    send would (DataObjects and meshes detached, long arrays in DataChunks) and parsed again:
    a root ``Collection`` with its ``__closure``, the elements as ``DataObject`` objects with their
    ``displayValue`` meshes, ``InstanceProxy`` placements with their transforms, the
    ``instanceDefinitionProxies`` and ``renderMaterialProxies`` on the root, and the bundle's
    ``units``. The list goes straight into ``bimpolars.read_speckle``, with the same
    bundle for the categories, parameters and levels, and matches what
    ``bimpolars.version_objects`` gives for a version built this way.

    Two things the plain ``to_base`` loses are restored:

    - An element with instance placements keeps the geometry it carries itself. Revit puts
      structural framing, columns and foundations in model coordinates on the element and
      gives them an empty instance definition, so ``to_base`` leaves them without geometry
      (841 of the 935 elements of the structural sample). Here that geometry becomes a
      DataObject under the element's applicationId, listed right before its first placement,
      which is renamed ``<applicationId>-instance-0``; ``bimpolars.read_speckle`` folds
      both into the element's row. An element that has both, such as a parapet wall whose
      cap is a placed sweep, keeps both.
    - The geometry of an instance definition (``def-geo-<geometry index>``) gets the material
      the envelope attaches to it: ``to_base`` gives materials only to the geometry of an
      element, so instances of families would render in a viewer's default grey. The
      ``renderMaterialProxies`` target those ids too, in the envelope's order of materials.

    Args:
        bundle (str | os.PathLike[str]): The bundle's folder, or the URL of its key prefix in the
            bimpolars bucket, as ``bundle_model`` takes it.

    Returns:
        list[SpeckleObject]: The parsed objects, the root first and its closure after it.

    Raises:
        ValueError: If the folder does not exist, or the URL is not in the bucket.
        FileNotFoundError: If the folder lacks a table the bundle reader requires.
        FileNotFoundError: If the folder lacks the geometry shards.
        urllib.error.URLError: If downloading a table of a bundle named by its URL fails.
    """
    model = bundle_model(bundle)
    root = model.to_base()
    _keep_element_geometry(model, root)
    _add_definition_materials(model, root)
    with warnings.catch_warnings():
        # the JSON pipeline is deprecated since specklepy 2026.9; it is the one we want
        warnings.filterwarnings("ignore", _JSON_PIPELINE_DEPRECATION, DeprecationWarning)
        memory = MemoryTransport()
        root_id, _ = BaseObjectSerializer(write_transports=[memory]).traverse_base(root)
    objects: dict[str, str] = memory.objects
    return [
        json.loads(objects[root_id]),
        *(json.loads(text) for object_id, text in objects.items() if object_id != root_id),
    ]


def _keep_element_geometry(model: Model, root: Collection) -> None:
    """Puts the own geometry of the placed elements back, before their first placement.

    specklepy names an element's first placement after the element and the others
    ``<applicationId>-instance-N``. The first is renamed ``-instance-0`` so that the
    element's own geometry can take the plain applicationId. When the first placement is
    missing (its instance node is not in the envelope), the geometry goes before the first
    placement there is, which already leaves the plain applicationId free; an element with
    no placement in the tree at all gets its geometry at the root.
    """
    bundle = model.bundle
    elements: dict[str, Base] = {}
    for object_k in sorted({edge.src for edge in bundle.relations.display_instance_edges}):
        app_id = bundle.object_app_ids.get(object_k)
        if app_id is None:
            continue
        element = _geometry_object(model, object_k, app_id)
        if element is not None:
            elements[app_id] = element
    if not elements:
        return

    def insert(collection: Collection) -> None:
        children: list[Any] = []
        for child in collection.elements:
            if isinstance(child, Collection):
                insert(child)
            app_id = getattr(child, "applicationId", None)
            if isinstance(app_id, str):
                if app_id in elements:
                    children.append(elements.pop(app_id))
                    child.applicationId = child.id = f"{app_id}{FIRST_PLACEMENT_SUFFIX}"
                elif (match := _PLACEMENT.match(app_id)) and match.group("base") in elements:
                    children.append(elements.pop(match.group("base")))
            children.append(child)
        collection.elements = children

    insert(root)
    root.elements.extend(elements.values())


def _add_definition_materials(model: Model, root: Collection) -> None:
    """Makes the material proxies target the definition geometry the envelope styles."""
    bundle = model.bundle
    relations = bundle.relations
    owned = relations.object_by_geometry()
    in_definitions = {
        geometry_k for members in relations.defines_by_definition.values() for geometry_k in members
    }
    targets: dict[int, list[str]] = {}
    for geometry_k, material_k in relations.material_by_geometry.items():
        if geometry_k in owned or geometry_k not in in_definitions:
            continue
        geometry = model.geometries.get(geometry_k)
        if geometry is not None and geometry.is_sgeo:  # to_base built an object for it
            targets.setdefault(material_k, []).append(f"{DEFINITION_GEOMETRY_PREFIX}{geometry_k}")
    if not targets:
        return
    existing = {
        str(proxy.applicationId): proxy
        for proxy in getattr(root, "renderMaterialProxies", None) or []
    }
    proxies: list[RenderMaterialProxy] = []
    # every material node as specklepy's to_base builds it, in the envelope's order; the
    # ones to_base already attached to an element's geometry come from the root
    for node_k, fresh in _materials(bundle.nodes).items():
        proxy = existing.get(str(fresh.applicationId), fresh)
        for target in targets.get(node_k, []):
            if target not in proxy.objects:
                proxy.objects.append(target)
        if proxy.objects:
            proxies.append(proxy)
    root["renderMaterialProxies"] = proxies


__all__ = [
    "DEFINITION_GEOMETRY_PREFIX",
    "FIRST_PLACEMENT_SUFFIX",
    "MATERIAL_PREFIX",
    "bundle_model",
    "bundle_objects",
]

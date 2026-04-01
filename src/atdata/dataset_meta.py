"""DatasetMeta parameter object for bundling shared metadata fields.

Reduces parameter explosion across ``Index.insert_dataset``,
``Index.write_samples``, and ``DatasetPublisher.publish*`` by collecting
the six metadata fields (name, schema_ref, description, tags, license,
metadata) into a single dataclass.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class DatasetMeta:
    """Metadata for publishing or indexing a dataset.

    Bundle common fields shared across ``write_samples``,
    ``insert_dataset``, and atmosphere publication.

    Args:
        name: Human-readable name for the dataset.
        schema_ref: Optional schema reference (AT URI or local ref).
        description: Optional dataset description.
        tags: Optional tags for discovery.
        license: Optional SPDX license identifier.
        metadata: Optional arbitrary metadata dict.

    Examples:
        >>> meta = DatasetMeta(name="mnist", tags=["vision"])
        >>> meta.name
        'mnist'
    """

    name: str
    schema_ref: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    license: str | None = None
    metadata: dict | None = None


def _resolve_meta(
    meta: DatasetMeta | None = None,
    *,
    name: str | None = None,
    schema_ref: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    license: str | None = None,
    metadata: dict | None = None,
) -> DatasetMeta:
    """Normalize a ``DatasetMeta`` from either a meta object or flat kwargs.

    When both *meta* and explicit kwargs are provided, explicit kwargs
    override the corresponding fields in *meta* (explicit wins).

    Args:
        meta: Optional pre-built metadata object.
        name: Dataset name (required if *meta* is ``None``).
        schema_ref: Optional schema reference override.
        description: Optional description override.
        tags: Optional tags override.
        license: Optional license override.
        metadata: Optional metadata dict override.

    Returns:
        Resolved ``DatasetMeta`` instance.

    Raises:
        TypeError: If neither *name* nor *meta* is provided.
    """
    if meta is None:
        if name is None:
            raise TypeError(
                "Either 'meta' or 'name' must be provided."
            )
        return DatasetMeta(
            name=name,
            schema_ref=schema_ref,
            description=description,
            tags=tags,
            license=license,
            metadata=metadata,
        )

    # Build overrides from explicit kwargs (only non-None values win)
    overrides: dict = {}
    if name is not None:
        overrides["name"] = name
    if schema_ref is not None:
        overrides["schema_ref"] = schema_ref
    if description is not None:
        overrides["description"] = description
    if tags is not None:
        overrides["tags"] = tags
    if license is not None:
        overrides["license"] = license
    if metadata is not None:
        overrides["metadata"] = metadata

    if overrides:
        return dataclasses.replace(meta, **overrides)
    return meta

"""Promotion workflow for migrating datasets from local to atmosphere.

This module provides functionality to promote locally-indexed datasets to the
ATProto atmosphere network. This enables sharing datasets with the broader
federation while maintaining schema consistency.

Examples:
    >>> from atdata.local import Index, Repo
    >>> from atdata.atmosphere import AtmosphereClient, AtmosphereIndex
    >>> from atdata.promote import promote_to_atmosphere
    >>>
    >>> # Setup
    >>> local_index = Index()
    >>> client = AtmosphereClient()
    >>> client.login("handle.bsky.social", "app-password")
    >>>
    >>> # Promote a dataset
    >>> entry = local_index.get_dataset("my-dataset")
    >>> at_uri = promote_to_atmosphere(entry, local_index, client)
"""

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .local import LocalDatasetEntry, Index
    from .atmosphere import AtmosphereClient
    from ._protocols import AbstractDataStore, Packable


def _find_existing_schema(
    client: "AtmosphereClient",
    name: str,
    version: str,
) -> str | None:
    """Check if a schema with the given name and version already exists.

    Args:
        client: Authenticated atmosphere client.
        name: Schema name to search for.
        version: Schema version to match.

    Returns:
        AT URI of existing schema if found, None otherwise.
    """
    from .atmosphere import SchemaLoader

    loader = SchemaLoader(client)
    for record in loader.list_all():
        rec_value = record.get("value", record)
        if rec_value.get("name") == name and rec_value.get("version") == version:
            return record.get("uri", "")
    return None


def _find_or_publish_schema(
    sample_type: "Type[Packable]",
    version: str,
    client: "AtmosphereClient",
    description: str | None = None,
) -> str:
    """Find existing schema or publish a new one.

    Checks if a schema with the same name and version already exists on the
    user's atmosphere repository. If found, returns the existing URI to avoid
    duplicates. Otherwise, publishes a new schema record.

    Args:
        sample_type: The PackableSample subclass to publish.
        version: Semantic version string.
        client: Authenticated atmosphere client.
        description: Optional schema description.

    Returns:
        AT URI of the schema (existing or newly published).
    """
    from .atmosphere import SchemaPublisher

    schema_name = f"{sample_type.__module__}.{sample_type.__name__}"

    # Check for existing schema
    existing = _find_existing_schema(client, schema_name, version)
    if existing:
        return existing

    # Publish new schema
    publisher = SchemaPublisher(client)
    uri = publisher.publish(
        sample_type,
        version=version,
        description=description,
    )
    return str(uri)


def promote_to_atmosphere(
    local_entry: "LocalDatasetEntry",
    local_index: "Index",
    atmosphere_client: "AtmosphereClient",
    *,
    data_store: "AbstractDataStore | None" = None,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    license: str | None = None,
) -> str:
    """Promote a local dataset to the atmosphere network.

    This function takes a locally-indexed dataset and publishes it to ATProto,
    making it discoverable on the federated atmosphere network.

    Args:
        local_entry: The LocalDatasetEntry to promote.
        local_index: Local index containing the schema for this entry.
        atmosphere_client: Authenticated AtmosphereClient.
        data_store: Optional data store for copying data to new location.
            If None, the existing data_urls are used as-is.
        name: Override name for the atmosphere record. Defaults to local name.
        description: Optional description for the dataset.
        tags: Optional tags for discovery.
        license: Optional license identifier.

    Returns:
        AT URI of the created atmosphere dataset record.

    Raises:
        KeyError: If schema not found in local index.
        ValueError: If local entry has no data URLs.

    Examples:
        >>> entry = local_index.get_dataset("mnist-train")
        >>> uri = promote_to_atmosphere(entry, local_index, client)
        >>> print(uri)
        at://did:plc:abc123/ac.foundation.dataset.datasetIndex/...
    """
    from .atmosphere import DatasetPublisher
    from ._schema_codec import schema_to_type

    # Validate entry has data
    if not local_entry.data_urls:
        raise ValueError(f"Local entry '{local_entry.name}' has no data URLs")

    # Get schema from local index
    schema_ref = local_entry.schema_ref
    schema_record = local_index.get_schema(schema_ref)

    # Reconstruct sample type from schema
    sample_type = schema_to_type(schema_record)
    schema_version = schema_record.get("version", "1.0.0")

    # Find or publish schema on atmosphere (deduplication)
    atmosphere_schema_uri = _find_or_publish_schema(
        sample_type,
        schema_version,
        atmosphere_client,
        description=schema_record.get("description"),
    )

    # Determine data URLs
    if data_store is not None:
        # Copy data to new storage location
        # Create a temporary Dataset to write through the data store
        from .dataset import Dataset

        # Build WDS URL from data_urls
        if len(local_entry.data_urls) == 1:
            wds_url = local_entry.data_urls[0]
        else:
            # Use brace notation for multiple URLs
            wds_url = " ".join(local_entry.data_urls)

        ds = Dataset[sample_type](wds_url)
        prefix = f"promoted/{local_entry.name}"
        data_urls = data_store.write_shards(ds, prefix=prefix)
    else:
        # Use existing URLs as-is
        data_urls = local_entry.data_urls

    # Publish dataset record to atmosphere
    publisher = DatasetPublisher(atmosphere_client)
    uri = publisher.publish_with_urls(
        urls=data_urls,
        schema_uri=atmosphere_schema_uri,
        name=name or local_entry.name,
        description=description,
        tags=tags,
        license=license,
        metadata=local_entry.metadata,
    )

    return str(uri)


__all__ = [
    "promote_to_atmosphere",
]

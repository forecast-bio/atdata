#!/usr/bin/env python3
"""Demonstration of atdata.atmosphere ATProto integration.

This script demonstrates how to use the atmosphere module to publish
and discover datasets on the AT Protocol network.

Usage:
    # Dry run (no actual ATProto connection):
    python atmosphere_demo.py

    # With actual ATProto connection:
    python atmosphere_demo.py --handle your.handle.social --password your-app-password

Requirements:
    pip install atdata[atmosphere]

Note:
    Use an app-specific password, not your main Bluesky password.
    Create app passwords at: https://bsky.app/settings/app-passwords
"""

import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    AtmosphereClient,
    AtmosphereIndex,
    PDSBlobStore,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    DatasetLoader,
    AtUri,
)
from atdata import BlobSource


# =============================================================================
# Define sample types using @packable decorator
# =============================================================================

@atdata.packable
class ImageSample:
    """A sample containing image data with metadata."""
    image: NDArray
    label: str
    confidence: float


@atdata.packable
class TextEmbeddingSample:
    """A sample containing text with embedding vectors."""
    text: str
    embedding: NDArray
    source: str


# =============================================================================
# Demo functions
# =============================================================================

def demo_type_introspection():
    """Demonstrate how atmosphere introspects PackableSample types."""
    print("\n" + "=" * 60)
    print("Type Introspection Demo")
    print("=" * 60)

    # Show what information is available from a PackableSample type
    print(f"\nSample type: {ImageSample.__name__}")
    print(f"Is dataclass: {is_dataclass(ImageSample)}")

    print("\nFields:")
    for field in fields(ImageSample):
        print(f"  - {field.name}: {field.type}")

    # Create a sample instance
    sample = ImageSample(
        image=np.random.rand(224, 224, 3).astype(np.float32),
        label="cat",
        confidence=0.95,
    )

    print("\nSample instance:")
    print(f"  image shape: {sample.image.shape}")
    print(f"  image dtype: {sample.image.dtype}")
    print(f"  label: {sample.label}")
    print(f"  confidence: {sample.confidence}")

    # Demonstrate serialization
    packed = sample.packed
    print(f"\nSerialized size: {len(packed):,} bytes")

    # Round-trip
    restored = ImageSample.from_bytes(packed)
    print(f"Round-trip successful: {np.allclose(sample.image, restored.image)}")


def demo_at_uri_parsing():
    """Demonstrate AT URI parsing."""
    print("\n" + "=" * 60)
    print("AT URI Parsing Demo")
    print("=" * 60)

    # Example AT URIs
    uris = [
        "at://did:plc:abc123/ac.foundation.dataset.sampleSchema/xyz789",
        "at://alice.bsky.social/ac.foundation.dataset.record/my-dataset",
    ]

    for uri_str in uris:
        print(f"\nParsing: {uri_str}")
        uri = AtUri.parse(uri_str)
        print(f"  Authority:  {uri.authority}")
        print(f"  Collection: {uri.collection}")
        print(f"  Rkey:       {uri.rkey}")
        print(f"  Roundtrip:  {str(uri)}")


def demo_schema_record_building():
    """Demonstrate building schema records from PackableSample types."""
    print("\n" + "=" * 60)
    print("Schema Record Building Demo")
    print("=" * 60)

    from atdata.atmosphere._types import SchemaRecord, FieldDef, FieldType

    # Build a schema record manually (what SchemaPublisher does internally)
    schema = SchemaRecord(
        name="ImageSample",
        version="1.0.0",
        description="A sample containing image data with metadata",
        fields=[
            FieldDef(
                name="image",
                field_type=FieldType(kind="ndarray", dtype="float32", shape=[224, 224, 3]),
                optional=False,
            ),
            FieldDef(
                name="label",
                field_type=FieldType(kind="primitive", primitive="str"),
                optional=False,
            ),
            FieldDef(
                name="confidence",
                field_type=FieldType(kind="primitive", primitive="float"),
                optional=False,
            ),
        ],
    )

    # Convert to ATProto record format
    record = schema.to_record()

    print("\nSchema record structure:")
    print(f"  $type: {record['$type']}")
    print(f"  name: {record['name']}")
    print(f"  version: {record['version']}")
    print(f"  description: {record.get('description', 'N/A')}")
    print(f"  fields: {len(record['fields'])} fields")

    for field in record["fields"]:
        print(f"    - {field['name']}: {field['fieldType']}")


def demo_mock_client():
    """Demonstrate the AtmosphereClient interface with a mock."""
    print("\n" + "=" * 60)
    print("Mock Client Demo (no network)")
    print("=" * 60)

    from unittest.mock import Mock, MagicMock

    # Create a mock atproto client
    mock_atproto = Mock()
    mock_atproto.me = MagicMock()
    mock_atproto.me.did = "did:plc:demo123456789"
    mock_atproto.me.handle = "demo.bsky.social"

    # Mock the login response
    mock_profile = Mock()
    mock_profile.did = "did:plc:demo123456789"
    mock_profile.handle = "demo.bsky.social"
    mock_atproto.login.return_value = mock_profile

    # Mock create_record response
    mock_response = Mock()
    mock_response.uri = "at://did:plc:demo123456789/ac.foundation.dataset.sampleSchema/abc123"
    mock_atproto.com.atproto.repo.create_record.return_value = mock_response

    # Create our client with the mock
    client = AtmosphereClient(_client=mock_atproto)
    client.login("demo.bsky.social", "fake-password")

    print(f"\nAuthenticated as: {client.handle}")
    print(f"DID: {client.did}")

    # Demonstrate schema publishing with mock
    publisher = SchemaPublisher(client)
    uri = publisher.publish(
        ImageSample,
        name="ImageSample",
        version="1.0.0",
        description="Demo image sample type",
    )

    print(f"\nPublished schema at: {uri}")
    print(f"  Authority: {uri.authority}")
    print(f"  Collection: {uri.collection}")
    print(f"  Rkey: {uri.rkey}")


def demo_live_connection(handle: str, password: str):
    """Demonstrate actual ATProto connection.

    Args:
        handle: Bluesky handle (e.g., 'alice.bsky.social')
        password: App-specific password
    """
    print("\n" + "=" * 60)
    print("Live ATProto Connection Demo")
    print("=" * 60)

    # Create client and authenticate
    print(f"\nConnecting as {handle}...")
    client = AtmosphereClient()
    client.login(handle, password)

    print("Authenticated!")
    print(f"  DID: {client.did}")
    print(f"  Handle: {client.handle}")

    # Publish a schema
    print("\nPublishing ImageSample schema...")
    schema_publisher = SchemaPublisher(client)
    schema_uri = schema_publisher.publish(
        ImageSample,
        name="ImageSample",
        version="1.0.0",
        description="Demo: Image sample with label and confidence",
    )
    print(f"  Schema URI: {schema_uri}")

    # List schemas we've published
    print("\nListing your published schemas...")
    schema_loader = SchemaLoader(client)
    schemas = schema_loader.list_all(limit=10)
    print(f"  Found {len(schemas)} schema(s)")
    for schema in schemas:
        print(f"    - {schema.get('name', 'Unknown')}: v{schema.get('version', '?')}")

    # Publish a dataset record (pointing to example URLs)
    print("\nPublishing dataset record (external URL storage)...")
    dataset_publisher = DatasetPublisher(client)
    dataset_uri = dataset_publisher.publish_with_urls(
        urls=["s3://example-bucket/demo-data-{000000..000009}.tar"],
        schema_uri=str(schema_uri),
        name="Demo Image Dataset",
        description="Example dataset demonstrating atmosphere publishing",
        tags=["demo", "images", "atdata"],
        license="MIT",
    )
    print(f"  Dataset URI: {dataset_uri}")

    # List datasets
    print("\nListing your published datasets...")
    dataset_loader = DatasetLoader(client)
    datasets = dataset_loader.list_all(limit=10)
    print(f"  Found {len(datasets)} dataset(s)")
    for ds in datasets:
        print(f"    - {ds.get('name', 'Unknown')}")
        print(f"      Schema: {ds.get('schemaRef', 'N/A')}")
        tags = ds.get('tags', [])
        if tags:
            print(f"      Tags: {', '.join(tags)}")


def demo_blob_storage(handle: str, password: str):
    """Demonstrate blob storage for smaller datasets.

    ATProto supports blob storage (up to 50MB per blob by default, configurable).
    This is useful for smaller datasets that don't need external storage.

    Args:
        handle: Bluesky handle (e.g., 'alice.bsky.social')
        password: App-specific password
    """
    import io
    import webdataset as wds

    print("\n" + "=" * 60)
    print("Blob Storage Demo")
    print("=" * 60)

    # Create client and authenticate
    print(f"\nConnecting as {handle}...")
    client = AtmosphereClient()
    client.login(handle, password)
    print(f"Authenticated as {client.handle}")

    # Define a simple sample type for this demo
    @atdata.packable
    class DemoSample:
        id: int
        text: str

    # Create sample instances using the @packable type
    samples = [
        DemoSample(id=0, text="Hello from blob storage!"),
        DemoSample(id=1, text="ATProto is decentralized."),
        DemoSample(id=2, text="atdata makes ML data easy."),
    ]

    # Create a WebDataset tar in memory using proper as_wds serialization
    print("\nCreating small dataset in memory...")
    tar_buffer = io.BytesIO()
    with wds.writer.TarWriter(tar_buffer) as sink:
        for sample in samples:
            sink.write(sample.as_wds)

    tar_data = tar_buffer.getvalue()
    print(f"  Created tar with {len(samples)} samples ({len(tar_data):,} bytes)")

    # Publish schema
    print("\nPublishing schema...")
    schema_publisher = SchemaPublisher(client)
    schema_uri = schema_publisher.publish(DemoSample, version="1.0.0")
    print(f"  Schema URI: {schema_uri}")

    # Publish dataset with blob storage
    print("\nUploading data as blob and publishing dataset...")
    dataset_publisher = DatasetPublisher(client)
    dataset_uri = dataset_publisher.publish_with_blobs(
        blobs=[tar_data],
        schema_uri=str(schema_uri),
        name="Blob Storage Demo Dataset",
        description="Small dataset stored directly in ATProto blobs",
        tags=["demo", "blob-storage"],
    )
    print(f"  Dataset URI: {dataset_uri}")

    # Verify storage type
    print("\nVerifying blob storage...")
    dataset_loader = DatasetLoader(client)
    storage_type = dataset_loader.get_storage_type(str(dataset_uri))
    print(f"  Storage type: {storage_type}")

    # Get blob URLs
    blob_urls = dataset_loader.get_blob_urls(str(dataset_uri))
    print(f"  Blob URLs: {len(blob_urls)} blob(s)")
    for url in blob_urls:
        # Truncate URL for display
        print(f"    {url[:80]}...")

    # Load and iterate over data
    print("\nLoading and iterating over blob data...")
    ds = dataset_loader.to_dataset(str(dataset_uri), DemoSample)
    for batch in ds.ordered():
        print(f"  Sample id={batch.id}, text={batch.text}")

    print("\nBlob storage demo complete!")


def demo_pds_blob_store(handle: str, password: str):
    """Demonstrate PDSBlobStore for decentralized dataset storage.

    PDSBlobStore is the recommended way to store datasets as ATProto blobs.
    It provides automatic shard management and integrates with AtmosphereIndex.

    Args:
        handle: Bluesky handle (e.g., 'alice.bsky.social')
        password: App-specific password
    """
    import webdataset as wds

    print("\n" + "=" * 60)
    print("PDSBlobStore Demo (Recommended Approach)")
    print("=" * 60)

    # Create client and authenticate
    print(f"\nConnecting as {handle}...")
    client = AtmosphereClient()
    client.login(handle, password)
    print(f"Authenticated as {client.handle}")

    # Create PDSBlobStore and AtmosphereIndex
    print("\nSetting up PDSBlobStore and AtmosphereIndex...")
    store = PDSBlobStore(client)
    index = AtmosphereIndex(client, data_store=store)
    print("  Store and index configured")

    # Define a sample type
    @atdata.packable
    class FeatureSample:
        features: NDArray
        label: int
        source: str

    # Create sample data
    print("\nCreating sample data...")
    samples = [
        FeatureSample(
            features=np.random.randn(64).astype(np.float32),
            label=i % 5,
            source="demo",
        )
        for i in range(50)
    ]

    # Write to temporary tar file
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, "demo.tar")

        with wds.writer.TarWriter(tar_path) as sink:
            for i, s in enumerate(samples):
                sink.write({**s.as_wds, "__key__": f"{i:06d}"})

        print(f"  Created tar with {len(samples)} samples")

        # Create dataset
        dataset = atdata.Dataset[FeatureSample](tar_path)

        # Publish schema
        print("\nPublishing schema...")
        schema_uri = index.publish_schema(
            FeatureSample,
            version="1.0.0",
            description="Demo feature vectors",
        )
        print(f"  Schema URI: {schema_uri}")

        # Publish dataset with blob storage
        print("\nPublishing dataset (shards uploaded as blobs)...")
        entry = index.insert_dataset(
            dataset,
            name=f"pds-blob-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            schema_ref=schema_uri,
            description="Dataset stored using PDSBlobStore",
            tags=["demo", "pds-blob-store"],
        )

        print(f"  Dataset URI: {entry.uri}")
        print(f"  Blob URLs: {len(entry.data_urls)} shard(s)")
        for url in entry.data_urls:
            print(f"    {url[:70]}...")

    # Load back from blobs
    print("\nLoading dataset from blobs...")
    source = store.create_source(entry.data_urls)
    print(f"  Created BlobSource with {len(source.blob_refs)} blob(s)")

    loaded_ds = atdata.Dataset[FeatureSample](source)
    count = 0
    for batch in loaded_ds.ordered():
        count += 1
        print(f"  Sample {count}: label={batch.label}, features shape={batch.features.shape}")
        if count >= 3:
            print("  ...")
            break

    print("\nPDSBlobStore demo complete!")
    print("  - Shards stored as ATProto blobs in your PDS")
    print("  - No external storage required")
    print("  - Fully decentralized!")


def demo_dataset_loading():
    """Demonstrate loading a dataset from an ATProto record."""
    print("\n" + "=" * 60)
    print("Dataset Loading Demo (conceptual)")
    print("=" * 60)

    print("""
Once you have published a dataset, others can load it like this:

    from atdata.atmosphere import AtmosphereClient, DatasetLoader

    client = AtmosphereClient()
    # Note: reading public records doesn't require authentication

    loader = DatasetLoader(client)

    # Get the dataset record
    record = loader.get("at://did:plc:abc123/ac.foundation.dataset.record/xyz")

    # Check storage type (external URLs or ATProto blobs)
    storage_type = loader.get_storage_type(uri)
    print(f"Storage type: {storage_type}")

    # For external URL storage:
    if storage_type == "external":
        urls = loader.get_urls(uri)
        print(f"Dataset URLs: {urls}")

    # For blob storage:
    elif storage_type == "blobs":
        blob_urls = loader.get_blob_urls(uri)
        print(f"Blob URLs: {blob_urls}")

    # to_dataset() handles both storage types automatically:
    dataset = loader.to_dataset(
        "at://did:plc:abc123/ac.foundation.dataset.record/xyz",
        sample_type=ImageSample,
    )

    # Now iterate as usual
    for batch in dataset.shuffled(batch_size=32):
        images = batch.image  # (32, 224, 224, 3)
        labels = batch.label  # list of 32 strings
        process(images, labels)
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate atdata.atmosphere ATProto integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--handle",
        help="Bluesky handle for live demo (e.g., alice.bsky.social)",
    )
    parser.add_argument(
        "--password",
        help="App-specific password for live demo",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("atdata.atmosphere Demo")
    print("=" * 60)
    print(f"\nTime: {datetime.now().isoformat()}")
    print(f"atdata version: {atdata.__name__}")

    # Always run these demos (no network required)
    demo_type_introspection()
    demo_at_uri_parsing()
    demo_schema_record_building()
    demo_mock_client()
    demo_dataset_loading()

    # Run live demos if credentials provided
    if args.handle and args.password:
        demo_live_connection(args.handle, args.password)
        demo_pds_blob_store(args.handle, args.password)  # Recommended approach
        demo_blob_storage(args.handle, args.password)     # Legacy approach
    else:
        print("\n" + "=" * 60)
        print("Live Demo Skipped")
        print("=" * 60)
        print("\nTo run with actual ATProto connection:")
        print("  python atmosphere_demo.py --handle your.handle --password your-app-password")
        print("\nCreate app passwords at: https://bsky.app/settings/app-passwords")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

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
import sys
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    AtmosphereClient,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    DatasetLoader,
    AtUri,
)


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

    print(f"\nSample instance:")
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

    print(f"Authenticated!")
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
    print("\nPublishing dataset record...")
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

    # Get the WebDataset URLs
    urls = loader.get_urls("at://did:plc:abc123/ac.foundation.dataset.record/xyz")
    print(f"Dataset URLs: {urls}")

    # If you have the sample type class, create a Dataset directly
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

    # Run live demo if credentials provided
    if args.handle and args.password:
        demo_live_connection(args.handle, args.password)
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

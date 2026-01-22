#!/usr/bin/env python3
"""Demonstration of promoting local datasets to the atmosphere network.

This script demonstrates the workflow for migrating datasets from local
Redis/S3 storage to the federated ATProto atmosphere network.

Usage:
    # Dry run with mocks (no external services required):
    python promote_workflow.py

    # With actual ATProto connection:
    python promote_workflow.py --handle your.handle --password your-app-password

Requirements:
    pip install atdata[atmosphere]

Note:
    Use an app-specific password, not your main Bluesky password.
    Create app passwords at: https://bsky.app/settings/app-passwords
"""

import argparse
from datetime import datetime
from unittest.mock import Mock

from numpy.typing import NDArray

import atdata
from atdata.promote import promote_to_atmosphere


# =============================================================================
# Define sample types
# =============================================================================

@atdata.packable
class ExperimentSample:
    """A sample from a scientific experiment."""
    measurement: NDArray
    timestamp: float
    sensor_id: str


# =============================================================================
# Demo functions
# =============================================================================

def demo_promotion_concept():
    """Explain the promotion workflow concept."""
    print("\n" + "=" * 60)
    print("Promotion Workflow Overview")
    print("=" * 60)

    print("""
The promotion workflow moves datasets from local storage to the atmosphere:

    LOCAL                           ATMOSPHERE
    -----                           ----------
    Redis Index                     ATProto PDS
    S3 Storage            -->       (same S3 or new location)
    atdata://local/sampleSchema/...             at://did:plc:.../schema/...

Steps:
1. Retrieve dataset entry from LocalIndex
2. Get schema from local index
3. Find or publish schema on atmosphere (deduplication)
4. Optionally copy data to new storage location
5. Create dataset record on atmosphere
6. Return AT URI for the published dataset

Key features:
- Schema deduplication: Won't republish identical schemas
- Flexible data handling: Keep existing URLs or copy to new storage
- Metadata preservation: Local metadata carries over to atmosphere
""")


def demo_mock_promotion():
    """Demonstrate promotion with mocked services."""
    print("\n" + "=" * 60)
    print("Mock Promotion Demo")
    print("=" * 60)

    from atdata.local import LocalDatasetEntry

    # Create a mock local entry
    local_entry = LocalDatasetEntry(
        _name="experiment-2024-001",
        _schema_ref="atdata://local/sampleSchema/__main__.ExperimentSample@1.0.0",
        _data_urls=[
            "s3://research-bucket/experiments/exp-2024-001/shard-000000.tar",
            "s3://research-bucket/experiments/exp-2024-001/shard-000001.tar",
        ],
        _metadata={
            "experiment_date": "2024-01-15",
            "lab": "Physics Building Room 302",
            "principal_investigator": "Dr. Smith",
        },
    )

    print("\nLocal entry to promote:")
    print(f"  Name: {local_entry.name}")
    print(f"  Schema: {local_entry.schema_ref}")
    print(f"  URLs: {len(local_entry.data_urls)} shards")
    print(f"  Metadata: {local_entry.metadata}")

    # Create mock local index
    mock_index = Mock()
    mock_index.get_schema.return_value = {
        "name": "__main__.ExperimentSample",
        "version": "1.0.0",
        "description": "A sample from a scientific experiment",
        "fields": [
            {"name": "measurement", "fieldType": {"$type": "local#ndarray", "dtype": "float32"}, "optional": False},
            {"name": "timestamp", "fieldType": {"$type": "local#primitive", "primitive": "float"}, "optional": False},
            {"name": "sensor_id", "fieldType": {"$type": "local#primitive", "primitive": "str"}, "optional": False},
        ],
    }

    # Create mock atmosphere client
    mock_client = Mock()
    mock_client.did = "did:plc:demo123456789"

    # Mock the atmosphere modules
    from unittest.mock import patch

    with patch("atdata.promote._find_existing_schema") as mock_find:
        mock_find.return_value = None  # No existing schema

        with patch("atdata.atmosphere.SchemaPublisher") as MockSchemaPublisher:
            mock_schema_pub = MockSchemaPublisher.return_value
            mock_schema_uri = Mock(__str__=lambda s: "at://did:plc:demo123456789/ac.foundation.dataset.sampleSchema/exp001")
            mock_schema_pub.publish.return_value = mock_schema_uri

            with patch("atdata.atmosphere.DatasetPublisher") as MockDatasetPublisher:
                mock_ds_pub = MockDatasetPublisher.return_value
                mock_ds_uri = Mock(__str__=lambda s: "at://did:plc:demo123456789/ac.foundation.dataset.datasetIndex/exp2024001")
                mock_ds_pub.publish_with_urls.return_value = mock_ds_uri

                # Perform the promotion
                result = promote_to_atmosphere(
                    local_entry,
                    mock_index,
                    mock_client,
                    tags=["experiment", "physics", "2024"],
                    license="CC-BY-4.0",
                )

    print("\nPromotion result:")
    print(f"  AT URI: {result}")
    print("\nPublished:")
    print("  Schema: at://did:plc:demo123456789/.../exp001")
    print("  Dataset: at://did:plc:demo123456789/.../exp2024001")


def demo_schema_deduplication():
    """Demonstrate schema deduplication during promotion."""
    print("\n" + "=" * 60)
    print("Schema Deduplication Demo")
    print("=" * 60)

    from atdata.promote import _find_existing_schema
    from unittest.mock import patch

    mock_client = Mock()

    # Scenario 1: Schema already exists
    print("\nScenario 1: Schema already exists on atmosphere")
    with patch("atdata.atmosphere.SchemaLoader") as MockLoader:
        mock_loader = MockLoader.return_value
        mock_loader.list_all.return_value = [
            {
                "uri": "at://did:plc:abc/schema/existing",
                "value": {
                    "name": "mymodule.MySample",
                    "version": "1.0.0",
                }
            }
        ]

        result = _find_existing_schema(mock_client, "mymodule.MySample", "1.0.0")
        print("  Looking for: mymodule.MySample@1.0.0")
        print(f"  Found: {result}")
        print("  Action: Reuse existing schema (no republish)")

    # Scenario 2: Different version
    print("\nScenario 2: Same name but different version")
    with patch("atdata.atmosphere.SchemaLoader") as MockLoader:
        mock_loader = MockLoader.return_value
        mock_loader.list_all.return_value = [
            {
                "uri": "at://did:plc:abc/schema/v1",
                "value": {
                    "name": "mymodule.MySample",
                    "version": "1.0.0",  # v1.0.0 exists
                }
            }
        ]

        result = _find_existing_schema(mock_client, "mymodule.MySample", "2.0.0")  # Looking for v2.0.0
        print("  Looking for: mymodule.MySample@2.0.0")
        print(f"  Found: {result}")
        print("  Action: Publish new schema record")


def demo_data_migration_options():
    """Explain data migration options during promotion."""
    print("\n" + "=" * 60)
    print("Data Migration Options")
    print("=" * 60)

    print("""
When promoting, you can choose how to handle the data files:

Option A: Keep existing URLs (default)
-----------------------------------------
    promote_to_atmosphere(entry, index, client)

    - Data stays in original S3 location
    - Dataset record points to existing URLs
    - Fastest option, no data copying
    - Requires original storage to remain accessible

Option B: Copy to new S3 location
-----------------------------------------
    new_store = S3DataStore(creds, bucket='public-bucket')
    promote_to_atmosphere(entry, index, client, data_store=new_store)

    - Data is copied to new bucket
    - Dataset record points to new URLs
    - Good for moving from private to public storage

Option C: Use ATProto blobs (future)
-----------------------------------------
    # Not yet implemented
    promote_to_atmosphere(entry, index, client, data_store='pds-blobs')

    - Data uploaded as ATProto blobs
    - Self-contained in the PDS
    - Size limits apply (ATProto blob limits)
""")


def demo_live_promotion(handle: str, password: str):
    """Demonstrate actual promotion to ATProto."""
    print("\n" + "=" * 60)
    print("Live Promotion Demo")
    print("=" * 60)

    from atdata.atmosphere import AtmosphereClient
    from atdata.local import LocalDatasetEntry

    # Connect to atmosphere
    print(f"\nConnecting as {handle}...")
    client = AtmosphereClient()
    client.login(handle, password)
    print(f"Authenticated! DID: {client.did}")

    # Create a demo local entry (simulating a real local dataset)
    local_entry = LocalDatasetEntry(
        _name="demo-promoted-dataset",
        _schema_ref="atdata://local/sampleSchema/__main__.ExperimentSample@1.0.0",
        _data_urls=["s3://example-bucket/demo-data-{000000..000004}.tar"],
        _metadata={"promoted_from": "local_demo", "demo": True},
    )

    # Create a mock local index with our schema
    mock_index = Mock()
    mock_index.get_schema.return_value = {
        "name": "__main__.ExperimentSample",
        "version": "1.0.0",
        "fields": [
            {"name": "measurement", "fieldType": {"$type": "local#ndarray", "dtype": "float32"}, "optional": False},
            {"name": "timestamp", "fieldType": {"$type": "local#primitive", "primitive": "float"}, "optional": False},
            {"name": "sensor_id", "fieldType": {"$type": "local#primitive", "primitive": "str"}, "optional": False},
        ],
    }

    print("\nPromoting dataset to atmosphere...")
    result = promote_to_atmosphere(
        local_entry,
        mock_index,
        client,
        tags=["demo", "atdata"],
        license="MIT",
    )

    print("\nPromotion successful!")
    print(f"  AT URI: {result}")
    print("\nYou can now discover this dataset via:")
    print(f"  atdata.load_dataset('@{handle}/demo-promoted-dataset')")


def demo_full_workflow():
    """Show the complete local-to-atmosphere workflow."""
    print("\n" + "=" * 60)
    print("Complete Workflow Example")
    print("=" * 60)

    print("""
Here's a complete example of the local-to-atmosphere workflow:

    import atdata
    from atdata.local import LocalIndex, S3DataStore
    from atdata.atmosphere import AtmosphereClient
    from atdata.promote import promote_to_atmosphere
    import webdataset as wds

    # 1. Define your sample type
    @atdata.packable
    class MySample:
        features: NDArray
        label: str

    # 2. Create samples and write to local tar
    samples = [MySample(features=..., label=...) for ...]
    with wds.writer.TarWriter("local-data.tar") as sink:
        for i, sample in enumerate(samples):
            sink.write({**sample.as_wds, "__key__": f"{i:06d}"})

    # 3. Set up index with S3 data store and insert dataset
    s3_creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }
    store = S3DataStore(s3_creds, bucket='my-bucket')
    local_index = LocalIndex(data_store=store)  # Connects to Redis

    # Publish schema and insert dataset
    local_index.publish_schema(MySample, version="1.0.0")
    dataset = atdata.Dataset[MySample]("local-data.tar")
    entry = local_index.insert_dataset(dataset, name='my-dataset')

    print(f"Local CID: {entry.cid}")
    print(f"Local URLs: {entry.data_urls}")

    # 4. When ready to share, promote to atmosphere
    client = AtmosphereClient()
    client.login('myhandle.bsky.social', 'app-password')

    at_uri = promote_to_atmosphere(
        entry,
        local_index,
        client,
        tags=['ml', 'vision'],
        license='MIT',
    )

    print(f"Published at: {at_uri}")

    # 5. Others can now discover and load your dataset
    # ds = atdata.load_dataset('@myhandle.bsky.social/my-dataset')
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate local to atmosphere promotion workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--handle",
        help="Bluesky handle for live demo",
    )
    parser.add_argument(
        "--password",
        help="App-specific password for live demo",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("atdata Promotion Workflow Demo")
    print("=" * 60)
    print(f"\nTime: {datetime.now().isoformat()}")

    # Always run these demos (no external services required)
    demo_promotion_concept()
    demo_mock_promotion()
    demo_schema_deduplication()
    demo_data_migration_options()
    demo_full_workflow()

    # Run live demo if credentials provided
    if args.handle and args.password:
        demo_live_promotion(args.handle, args.password)
    else:
        print("\n" + "=" * 60)
        print("Live Demo Skipped")
        print("=" * 60)
        print("\nTo run with actual ATProto connection:")
        print("  python promote_workflow.py --handle your.handle --password your-app-password")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

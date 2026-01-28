#!/usr/bin/env python3
"""Demonstration of atdata local storage workflow.

This script demonstrates how to use the local module to store and index
datasets using Redis and S3-compatible storage.

Usage:
    # Dry run with mocks (no Redis/S3 required):
    python local_workflow.py

    # With actual Redis (requires redis-server running):
    python local_workflow.py --redis

    # With Redis and S3 (requires MinIO or AWS):
    python local_workflow.py --redis --s3-endpoint http://localhost:9000

Requirements:
    pip install atdata redis

Note:
    For S3 storage, you can use MinIO for local development:
    docker run -p 9000:9000 minio/minio server /data
"""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import atdata
from atdata.local import LocalIndex, LocalDatasetEntry, S3DataStore


# =============================================================================
# Define sample types
# =============================================================================

@atdata.packable
class TrainingSample:
    """A sample containing features and label for training."""
    features: NDArray
    label: int


@atdata.packable
class TextSample:
    """A sample containing text data."""
    text: str
    category: str


# =============================================================================
# Demo functions
# =============================================================================

def demo_local_dataset_entry():
    """Demonstrate LocalDatasetEntry creation and CID generation."""
    print("\n" + "=" * 60)
    print("LocalDatasetEntry Demo")
    print("=" * 60)

    # Create an entry
    entry = LocalDatasetEntry(
        _name="my-dataset",
        _schema_ref="atdata://local/sampleSchema/TrainingSample@1.0.0",
        _data_urls=["s3://bucket/data-000000.tar", "s3://bucket/data-000001.tar"],
        _metadata={"source": "example", "samples": 10000},
    )

    print(f"\nEntry name: {entry.name}")
    print(f"Schema ref: {entry.schema_ref}")
    print(f"Data URLs: {entry.data_urls}")
    print(f"Metadata: {entry.metadata}")
    print(f"CID: {entry.cid}")

    # Demonstrate CID determinism
    entry2 = LocalDatasetEntry(
        _name="different-name",  # Name doesn't affect CID
        _schema_ref="atdata://local/sampleSchema/TrainingSample@1.0.0",
        _data_urls=["s3://bucket/data-000000.tar", "s3://bucket/data-000001.tar"],
    )

    print("\nCID comparison (same content, different name):")
    print(f"  Entry 1 CID: {entry.cid}")
    print(f"  Entry 2 CID: {entry2.cid}")
    print(f"  Match: {entry.cid == entry2.cid}")


def demo_local_index_mock():
    """Demonstrate LocalIndex operations with mock data."""
    print("\n" + "=" * 60)
    print("LocalIndex Demo (mock)")
    print("=" * 60)

    # LocalIndex without Redis connection works for read operations
    _index = LocalIndex()  # noqa: F841 - demo instantiation

    print("\nLocalIndex created (no Redis connection)")
    print("Methods available:")
    print("  - index.insert_dataset(dataset, name='...')")
    print("  - index.get_dataset(name_or_cid)")
    print("  - index.list_datasets()")
    print("  - index.publish_schema(sample_type, version='1.0.0')")
    print("  - index.get_schema(ref)")
    print("  - index.list_schemas()")
    print("  - index.decode_schema(ref)  # Returns PackableSample class")


def demo_local_index_redis(redis_host: str = "localhost", redis_port: int = 6379):
    """Demonstrate LocalIndex with actual Redis."""
    print("\n" + "=" * 60)
    print("LocalIndex Demo (Redis)")
    print("=" * 60)

    from redis import Redis

    # Connect to Redis
    try:
        redis = Redis(host=redis_host, port=redis_port)
        redis.ping()
    except Exception as e:
        print(f"Could not connect to Redis: {e}")
        print("Skipping Redis demo.")
        return

    # Create index with Redis
    index = LocalIndex(redis=redis)
    print(f"\nConnected to Redis at {redis_host}:{redis_port}")

    # Publish a schema
    print("\nPublishing TrainingSample schema...")
    schema_ref = index.publish_schema(TrainingSample, version="1.0.0")
    print(f"  Schema ref: {schema_ref}")

    # List schemas
    print("\nListing schemas:")
    for schema in index.list_schemas():
        print(f"  - {schema.get('name', 'Unknown')} v{schema.get('version', '?')}")

    # Get schema and decode to type
    schema_record = index.get_schema(schema_ref)
    print(f"\nSchema record: {schema_record.get('name')}")
    print(f"  Fields: {[f['name'] for f in schema_record.get('fields', [])]}")

    # Decode schema back to a PackableSample class
    decoded_type = index.decode_schema(schema_ref)
    print(f"\nDecoded type: {decoded_type.__name__}")

    # Clean up test data
    for key in redis.scan_iter(match="LocalSchema:*"):
        redis.delete(key)
    print("\nCleaned up test schemas")


def demo_s3_datastore():
    """Demonstrate S3DataStore interface."""
    print("\n" + "=" * 60)
    print("S3DataStore Demo")
    print("=" * 60)

    # S3DataStore with mock credentials (won't actually connect)
    creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }

    store = S3DataStore(creds, bucket="my-bucket")

    print("\nS3DataStore created:")
    print(f"  Bucket: {store.bucket}")
    print(f"  Supports streaming: {store.supports_streaming()}")

    # read_url returns the URL unchanged (passthrough for WDS)
    url = "s3://my-bucket/data.tar"
    print(f"\nread_url passthrough: {store.read_url(url)}")


def demo_repo_workflow(tmp_path: Path):
    """Demonstrate full Repo workflow with local files."""
    import webdataset as wds

    print("\n" + "=" * 60)
    print("Repo Workflow Demo (local files)")
    print("=" * 60)

    # Create sample data
    samples = [
        TrainingSample(features=np.random.randn(10).astype(np.float32), label=i % 3)
        for i in range(100)
    ]

    print(f"\nCreated {len(samples)} training samples")

    # Create a Dataset and write to local tar file
    tar_path = tmp_path / "local-data-000000.tar"
    with wds.writer.TarWriter(str(tar_path)) as sink:
        for i, sample in enumerate(samples):
            sink.write({**sample.as_wds, "__key__": f"sample_{i:06d}"})

    print(f"Wrote samples to: {tar_path}")

    # Load the dataset back
    ds = atdata.Dataset[TrainingSample](str(tar_path))
    loaded = list(ds.ordered(batch_size=None))
    print(f"Loaded {len(loaded)} samples back")

    # Verify round-trip
    assert len(loaded) == len(samples)
    assert np.allclose(loaded[0].features, samples[0].features)
    print("Round-trip verification: PASSED")


def demo_load_dataset_with_index():
    """Demonstrate load_dataset with index parameter."""
    print("\n" + "=" * 60)
    print("load_dataset with Index Demo")
    print("=" * 60)

    print("""
The load_dataset() function supports an index parameter for both local
and atmosphere backends:

    # Local index lookup
    from atdata import load_dataset
    from atdata.local import LocalIndex

    index = LocalIndex()
    ds = load_dataset('@local/my-dataset', index=index, split='train')

    # The index resolves the dataset name to URLs and schema
    for batch in ds.shuffled(batch_size=32):
        process(batch)

    # Atmosphere lookup (via @handle/dataset syntax)
    ds = load_dataset('@alice.science/mnist', split='train')

    # This automatically:
    # 1. Resolves the handle to a DID
    # 2. Fetches the dataset record from the user's repository
    # 3. Gets the data URLs from the record
    # 4. Resolves the schema for type information
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate atdata local storage workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--redis",
        action="store_true",
        help="Run demos that require Redis",
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host (default: localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--s3-endpoint",
        help="S3 endpoint URL for live S3 demo",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("atdata.local Demo")
    print("=" * 60)
    print(f"\nTime: {datetime.now().isoformat()}")

    # Always run these demos (no external services required)
    demo_local_dataset_entry()
    demo_local_index_mock()
    demo_s3_datastore()
    demo_load_dataset_with_index()

    # Run with temp directory for file-based demos
    with tempfile.TemporaryDirectory() as tmp:
        demo_repo_workflow(Path(tmp))

    # Run Redis demo if requested
    if args.redis:
        demo_local_index_redis(args.redis_host, args.redis_port)
    else:
        print("\n" + "=" * 60)
        print("Redis Demo Skipped")
        print("=" * 60)
        print("\nTo run with Redis: python local_workflow.py --redis")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

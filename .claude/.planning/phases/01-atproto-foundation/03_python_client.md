# Python Client Library Architecture

## Overview

This document specifies the Python library extensions to `atdata` for ATProto integration. The goal is to add ATProto publishing and discovery capabilities while maintaining backward compatibility with existing code.

## Design Principles

- **Backward compatible**: Existing code continues to work unchanged
- **Optional integration**: ATProto features are opt-in
- **Pythonic API**: Follows Python conventions and `atdata` style
- **Type-safe**: Full type hints with generics
- **Testable**: Mockable dependencies, unit testable

## Module Structure

```
src/atdata/
  __init__.py          # Existing exports
  dataset.py           # Existing Dataset, PackableSample
  lens.py              # Existing Lens, LensNetwork
  _helpers.py          # Existing serialization helpers
  atproto/             # NEW: ATProto integration
    __init__.py        # Public API exports
    client.py          # ATProtoClient for auth/session
    schema.py          # Schema publishing/loading
    dataset.py         # Dataset publishing/loading
    lens.py            # Lens publishing/loading
    _lexicon.py        # Lexicon record builders
    _types.py          # Type definitions for records
```

## Core Components

### 1. ATProtoClient - Authentication & Session Management

**File**: `src/atdata/atproto/client.py`

```python
from typing import Optional
from atproto import Client as ATProtoSDKClient

class ATProtoClient:
    """Wrapper around atproto SDK client with atdata-specific helpers."""

    def __init__(self, client: Optional[ATProtoSDKClient] = None):
        """
        Initialize ATProto client.

        Args:
            client: Optional pre-configured atproto Client. If None, creates new client.
        """
        self._client = client or ATProtoSDKClient()
        self._session: Optional[dict] = None

    def login(self, handle: str, password: str) -> None:
        """Authenticate with ATProto PDS."""
        self._session = self._client.login(handle, password)

    def login_with_token(self, access_token: str, refresh_token: str) -> None:
        """Authenticate using existing tokens."""
        # Implementation
        pass

    @property
    def is_authenticated(self) -> bool:
        """Check if client has valid session."""
        return self._session is not None

    @property
    def did(self) -> str:
        """Get DID of authenticated user."""
        if not self._session:
            raise ValueError("Not authenticated")
        return self._session['did']

    # Low-level record operations
    def create_record(self, collection: str, record: dict) -> str:
        """Create a record and return its AT-URI."""
        # Implementation using self._client
        pass

    def get_record(self, uri: str) -> dict:
        """Fetch a record by AT-URI."""
        # Implementation
        pass

    def list_records(self, collection: str, did: Optional[str] = None) -> list[dict]:
        """List records in a collection."""
        # Implementation
        pass
```

**Usage**:
```python
from atdata.atproto import ATProtoClient

client = ATProtoClient()
client.login("alice.bsky.social", "password")
```

### 2. Schema Publishing & Loading

**File**: `src/atdata/atproto/schema.py`

```python
from typing import Type, TypeVar, get_type_hints
from dataclasses import fields, is_dataclass
import atdata
from .client import ATProtoClient
from ._lexicon import build_schema_record

ST = TypeVar('ST', bound=atdata.PackableSample)

class SchemaPublisher:
    """Handles publishing PackableSample schemas to ATProto."""

    def __init__(self, client: ATProtoClient):
        self.client = client

    def publish_schema(
        self,
        sample_type: Type[ST],
        *,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Publish a PackableSample schema to ATProto.

        Args:
            sample_type: The PackableSample class to publish
            name: Human-readable name (defaults to class name)
            version: Semantic version
            description: Human-readable description
            metadata: Arbitrary metadata dict

        Returns:
            AT-URI of the created schema record
        """
        if not self.client.is_authenticated:
            raise ValueError("Client must be authenticated")

        # Extract field information from dataclass
        schema_record = self._build_schema_record(
            sample_type, name, version, description, metadata
        )

        # Publish to ATProto
        uri = self.client.create_record("app.bsky.atdata.schema", schema_record)
        return uri

    def _build_schema_record(
        self,
        sample_type: Type[ST],
        name: Optional[str],
        version: str,
        description: Optional[str],
        metadata: Optional[dict]
    ) -> dict:
        """Build schema record dict from PackableSample class."""
        if not is_dataclass(sample_type):
            raise ValueError(f"{sample_type} must be a dataclass")

        field_defs = []
        type_hints = get_type_hints(sample_type)

        for field in fields(sample_type):
            field_type = type_hints[field.name]
            field_def = self._field_to_record(field.name, field_type)
            field_defs.append(field_def)

        return {
            "$type": "app.bsky.atdata.schema",
            "name": name or sample_type.__name__,
            "version": version,
            "description": description or "",
            "fields": field_defs,
            "metadata": metadata or {},
            "createdAt": datetime.now(timezone.utc).isoformat()
        }

    def _field_to_record(self, name: str, field_type) -> dict:
        """Convert Python type annotation to schema field record."""
        # Handle Optional types
        is_optional = False
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            args = field_type.__args__
            if type(None) in args:
                is_optional = True
                field_type = next(arg for arg in args if arg is not type(None))

        # Map Python types to schema types
        type_def = self._python_type_to_schema_type(field_type)

        return {
            "name": name,
            "type": type_def,
            "optional": is_optional
        }

    def _python_type_to_schema_type(self, python_type) -> dict:
        """Map Python type to schema type definition."""
        # Handle primitives
        if python_type is str:
            return {"kind": "primitive", "primitive": "str"}
        elif python_type is int:
            return {"kind": "primitive", "primitive": "int"}
        elif python_type is float:
            return {"kind": "primitive", "primitive": "float"}
        elif python_type is bool:
            return {"kind": "primitive", "primitive": "bool"}
        elif python_type is bytes:
            return {"kind": "primitive", "primitive": "bytes"}

        # Handle NDArray - this is the key special case
        # In atdata, NDArray is used as a type annotation
        if hasattr(python_type, '__origin__'):
            origin = python_type.__origin__
            if origin.__name__ == 'NDArray' or str(origin) == 'numpy.ndarray':
                # Extract dtype from annotation if available
                # For now, default to float32
                return {
                    "kind": "ndarray",
                    "dtype": "float32",  # TODO: extract from annotation
                    "shape": None
                }

        # If it's another PackableSample, create nested reference
        if is_dataclass(python_type) and issubclass(python_type, atdata.PackableSample):
            # This would require publishing the nested type first
            raise NotImplementedError("Nested PackableSample types not yet supported")

        raise ValueError(f"Unsupported type: {python_type}")

class SchemaLoader:
    """Handles loading PackableSample schemas from ATProto."""

    def __init__(self, client: ATProtoClient):
        self.client = client

    def get_schema(self, uri: str) -> dict:
        """Fetch a schema record by AT-URI."""
        record = self.client.get_record(uri)
        if record.get('$type') != 'app.bsky.atdata.schema':
            raise ValueError(f"Record at {uri} is not a schema record")
        return record

    def list_schemas(self, did: Optional[str] = None) -> list[dict]:
        """List available schema records."""
        return self.client.list_records("app.bsky.atdata.schema", did)
```

**Usage**:
```python
from atdata.atproto import ATProtoClient, SchemaPublisher

@atdata.packable
class MySample:
    image: NDArray
    label: str

client = ATProtoClient()
client.login("alice.bsky.social", "password")

publisher = SchemaPublisher(client)
schema_uri = publisher.publish_schema(
    MySample,
    description="My sample type",
    version="1.0.0"
)
print(f"Published schema at {schema_uri}")
```

### 3. Dataset Publishing & Loading

**File**: `src/atdata/atproto/dataset.py`

```python
from typing import Type, TypeVar, Optional
import msgpack
import atdata
from .client import ATProtoClient
from .schema import SchemaPublisher

ST = TypeVar('ST', bound=atdata.PackableSample)

class DatasetPublisher:
    """Handles publishing Dataset index records to ATProto."""

    def __init__(self, client: ATProtoClient):
        self.client = client
        self.schema_publisher = SchemaPublisher(client)

    def publish_dataset(
        self,
        dataset: atdata.Dataset[ST],
        *,
        name: str,
        schema_uri: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        auto_publish_schema: bool = True
    ) -> str:
        """
        Publish a dataset index record to ATProto.

        Args:
            dataset: The Dataset to publish
            name: Human-readable dataset name
            schema_uri: AT-URI of the schema record (required if auto_publish_schema=False)
            description: Human-readable description
            tags: Searchable tags
            license: License identifier (SPDX preferred)
            auto_publish_schema: If True and schema_uri not provided, publish schema automatically

        Returns:
            AT-URI of the created dataset record
        """
        if not self.client.is_authenticated:
            raise ValueError("Client must be authenticated")

        # Ensure schema is published
        if schema_uri is None:
            if not auto_publish_schema:
                raise ValueError("schema_uri required when auto_publish_schema=False")
            schema_uri = self.schema_publisher.publish_schema(dataset.sample_type)

        # Build dataset record
        dataset_record = {
            "$type": "app.bsky.atdata.dataset",
            "name": name,
            "schemaRef": schema_uri,
            "urls": [dataset.url],  # Single URL for now
            "description": description or "",
            "metadata": msgpack.packb(dataset.metadata),
            "tags": tags or [],
            "license": license or "",
            "createdAt": datetime.now(timezone.utc).isoformat()
        }

        # Add size information if available
        # (would need to iterate dataset or have metadata about size)

        # Publish to ATProto
        uri = self.client.create_record("app.bsky.atdata.dataset", dataset_record)
        return uri

class DatasetLoader:
    """Handles loading Datasets from ATProto records."""

    def __init__(self, client: ATProtoClient):
        self.client = client

    def load_dataset(self, uri: str) -> atdata.Dataset:
        """
        Load a Dataset from an ATProto record.

        Args:
            uri: AT-URI of the dataset record

        Returns:
            Dataset instance configured from the record
        """
        # Fetch the dataset record
        record = self.client.get_record(uri)
        if record.get('$type') != 'app.bsky.atdata.dataset':
            raise ValueError(f"Record at {uri} is not a dataset record")

        # For now, we still need the Python class for the sample type
        # In the future, this could use codegen
        # TODO: Implement dynamic type loading via codegen

        # Extract URLs and metadata
        urls = record['urls']
        metadata = msgpack.unpackb(record.get('metadata', b''))

        # We need the schema to instantiate the Dataset with correct type
        # This is a limitation - we need codegen to create the type dynamically
        # For now, raise an error
        raise NotImplementedError(
            "Loading datasets requires code generation to instantiate sample types. "
            f"Schema URI: {record['schemaRef']}\n"
            "Use the codegen tool to generate the Python class first."
        )

    def list_datasets(self, did: Optional[str] = None) -> list[dict]:
        """List available dataset records."""
        return self.client.list_records("app.bsky.atdata.dataset", did)

    def search_datasets(self, tags: Optional[list[str]] = None, query: Optional[str] = None) -> list[dict]:
        """
        Search for datasets.

        Args:
            tags: Filter by tags
            query: Text search query

        Returns:
            List of matching dataset records
        """
        # This would use AppView in production
        # For now, fetch all and filter client-side
        all_datasets = self.list_records("app.bsky.atdata.dataset")

        filtered = all_datasets
        if tags:
            filtered = [d for d in filtered if any(t in d.get('tags', []) for t in tags)]
        if query:
            filtered = [d for d in filtered if query.lower() in d.get('name', '').lower() or
                       query.lower() in d.get('description', '').lower()]

        return filtered
```

**Usage**:
```python
from atdata.atproto import ATProtoClient, DatasetPublisher

# Create dataset
dataset = atdata.Dataset[MySample](url="s3://bucket/data-{000000..000009}.tar")

# Publish
client = ATProtoClient()
client.login("alice.bsky.social", "password")

publisher = DatasetPublisher(client)
dataset_uri = publisher.publish_dataset(
    dataset,
    name="My Training Data",
    description="Training data for my model",
    tags=["computer-vision", "training"],
    license="MIT"
)
print(f"Published dataset at {dataset_uri}")
```

### 4. Lens Publishing

**File**: `src/atdata/atproto/lens.py`

```python
from typing import Callable, Optional
import inspect
from .client import ATProtoClient

class LensPublisher:
    """Handles publishing Lens transformations to ATProto."""

    def __init__(self, client: ATProtoClient):
        self.client = client

    def publish_lens(
        self,
        lens_getter: Callable,
        lens_putter: Callable,
        *,
        name: str,
        source_schema_uri: str,
        target_schema_uri: str,
        description: Optional[str] = None,
        code_repository: Optional[str] = None,
        code_commit: Optional[str] = None
    ) -> str:
        """
        Publish a Lens transformation to ATProto.

        Args:
            lens_getter: The getter function (Source -> Target)
            lens_putter: The putter function (Target, Source -> Source)
            name: Human-readable lens name
            source_schema_uri: AT-URI of source schema
            target_schema_uri: AT-URI of target schema
            description: What this transformation does
            code_repository: Git repository URL
            code_commit: Git commit hash

        Returns:
            AT-URI of the created lens record
        """
        if not self.client.is_authenticated:
            raise ValueError("Client must be authenticated")

        # Build lens record
        lens_record = {
            "$type": "app.bsky.atdata.lens",
            "name": name,
            "sourceSchema": source_schema_uri,
            "targetSchema": target_schema_uri,
            "description": description or "",
            "createdAt": datetime.now(timezone.utc).isoformat()
        }

        # Add code references
        if code_repository and code_commit:
            getter_name = lens_getter.__name__
            putter_name = lens_putter.__name__

            lens_record["getterCode"] = {
                "kind": "reference",
                "repository": code_repository,
                "commit": code_commit,
                "path": f"{getter_name}"  # Simplified - would need module path
            }
            lens_record["putterCode"] = {
                "kind": "reference",
                "repository": code_repository,
                "commit": code_commit,
                "path": f"{putter_name}"
            }
        else:
            # For initial version, we could store source code directly
            # But this is DANGEROUS - security review required
            raise NotImplementedError(
                "Inline code storage not yet supported. "
                "Please provide code_repository and code_commit."
            )

        # Publish to ATProto
        uri = self.client.create_record("app.bsky.atdata.lens", lens_record)
        return uri
```

## Extension to Existing Classes

### Adding ATProto methods to Dataset

**Approach**: Add methods directly to `Dataset` class in `src/atdata/dataset.py`

```python
class Dataset[ST: PackableSample]:
    # ... existing implementation ...

    def publish_to_atproto(
        self,
        client: 'ATProtoClient',  # Forward reference to avoid circular import
        *,
        name: str,
        **kwargs
    ) -> str:
        """
        Publish this dataset to ATProto.

        This is a convenience method that wraps DatasetPublisher.
        """
        from .atproto import DatasetPublisher
        publisher = DatasetPublisher(client)
        return publisher.publish_dataset(self, name=name, **kwargs)

    @classmethod
    def from_atproto(
        cls,
        client: 'ATProtoClient',
        uri: str
    ) -> 'Dataset':
        """
        Load a dataset from an ATProto record.

        Note: This requires the sample type to be available in Python.
        Use codegen to generate types from schema records.
        """
        from .atproto import DatasetLoader
        loader = DatasetLoader(client)
        return loader.load_dataset(uri)
```

**Usage**:
```python
# Publishing
dataset = atdata.Dataset[MySample](url="s3://...")
uri = dataset.publish_to_atproto(client, name="My Dataset")

# Loading (future, requires codegen)
dataset = atdata.Dataset.from_atproto(client, uri)
```

## Public API Exports

**File**: `src/atdata/atproto/__init__.py`

```python
from .client import ATProtoClient
from .schema import SchemaPublisher, SchemaLoader
from .dataset import DatasetPublisher, DatasetLoader
from .lens import LensPublisher

__all__ = [
    "ATProtoClient",
    "SchemaPublisher",
    "SchemaLoader",
    "DatasetPublisher",
    "DatasetLoader",
    "LensPublisher",
]
```

## Testing Strategy

### Unit Tests
- Mock `ATProtoClient` to avoid network calls
- Test schema record building from various PackableSample types
- Test error handling (auth failures, invalid types, etc.)

### Integration Tests
- Use ATProto test server or sandbox
- Test full publish/query cycle
- Verify record structure matches Lexicon

### Example Test
```python
import pytest
from unittest.mock import Mock
import atdata
from atdata.atproto import SchemaPublisher

@atdata.packable
class TestSample:
    field1: str
    field2: int

def test_schema_publisher():
    # Mock client
    mock_client = Mock()
    mock_client.is_authenticated = True
    mock_client.create_record = Mock(return_value="at://did:example/app.bsky.atdata.schema/abc123")

    # Publish schema
    publisher = SchemaPublisher(mock_client)
    uri = publisher.publish_schema(TestSample, version="1.0.0")

    # Verify
    assert uri == "at://did:example/app.bsky.atdata.schema/abc123"
    mock_client.create_record.assert_called_once()

    # Check the record structure
    call_args = mock_client.create_record.call_args
    collection, record = call_args[0]
    assert collection == "app.bsky.atdata.schema"
    assert record["name"] == "TestSample"
    assert len(record["fields"]) == 2
```

## Dependencies

**New dependencies** (to be added to `pyproject.toml`):

```toml
[project]
dependencies = [
    # ... existing ...
    "atproto>=0.0.40",  # ATProto Python SDK
]
```

## Implementation Checklist (Phase 2)

- [ ] Set up `atdata/atproto/` module structure
- [ ] Implement `ATProtoClient` wrapper
- [ ] Implement `SchemaPublisher` with type introspection
- [ ] Implement `DatasetPublisher`
- [ ] Implement `LensPublisher` (code reference only)
- [ ] Add convenience methods to `Dataset` class
- [ ] Write unit tests for all publishers
- [ ] Write integration tests with test server
- [ ] Update documentation with examples

## Future Enhancements

### Better NDArray Type Handling
- Parse `NDArray[DType, Shape]` annotations for accurate dtype/shape
- Support for shape constraints in schema

### Dynamic Type Loading
- Use codegen to create types at runtime from schema records
- Enable `Dataset.from_atproto()` without pre-existing Python classes

### Caching
- Cache schema lookups to avoid repeated network calls
- Local schema registry

### Batch Operations
- Publish multiple schemas/datasets in one call
- Bulk import/export

### AppView Integration
- Use AppView for fast search instead of client-side filtering
- Streaming results for large queries

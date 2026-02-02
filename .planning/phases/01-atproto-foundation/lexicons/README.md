# ATProto Lexicon Definitions for atdata

This directory contains the ATProto Lexicon JSON definitions for the distributed dataset federation system.

## Lexicons

### Core Record Types

1. **[ac.foundation.dataset.sampleSchema](ac.foundation.dataset.sampleSchema.json)**
   - Defines PackableSample-compatible sample types using JSON Schema
   - Supports versioning via rkey format: `{NSID}@{semver}`
   - Includes NDArray shim for ML/scientific data types
   - Example: [sampleSchema_example.json](../examples/sampleSchema_example.json)

2. **[ac.foundation.dataset.record](ac.foundation.dataset.record.json)**
   - Index records for WebDataset-backed datasets
   - Hybrid storage support (external URLs + PDS blobs)
   - References sampleSchema for type information
   - Examples:
     - [External storage](../examples/dataset_external_storage.json)
     - [Blob storage](../examples/dataset_blob_storage.json)

3. **[ac.foundation.dataset.lens](ac.foundation.dataset.lens.json)**
   - Bidirectional transformations between sample types
   - External code references (GitHub, tangled.org)
   - Language metadata for multi-language support
   - Example: [lens_example.json](../examples/lens_example.json)

### Query APIs

4. **[ac.foundation.dataset.getLatestSchema](ac.foundation.dataset.getLatestSchema.json)**
   - Query to get the latest version of a schema by NSID
   - Returns full record + all available versions
   - Handles the custom rkey versioning scheme

## Key Design Decisions

### 1. Namespace

All Lexicons use the `ac.foundation.dataset.*` namespace:
- `ac.foundation` - Organization namespace
- `dataset` - Domain (distributed datasets)
- Specific record types: `sampleSchema`, `record`, `lens`

### 2. Schema Versioning (rkey Convention)

**Custom rkey format**: `{NSID}@{semver}`

**Example**: `com.example.myschema@1.2.3`

- `{NSID}`: Permanent identifier for the schema type (e.g., `com.example.myschema`)
- `{semver}`: Semantic version (e.g., `1.2.3`)

**Benefits**:
- Immutable version records
- Easy to list all versions of a schema
- Natural query pattern via `getLatestSchema`
- Clear semantic versioning enforcement

**Implementation**: The sampleSchema Lexicon uses `"key": "any"` to support this custom format.

### 3. JSON Schema with NDArray Shim

**Decision**: Use standard JSON Schema for type definitions with a custom NDArray shim.

**NDArray Shim Structure**:
```json
{
  "$defs": {
    "ndarray": {
      "type": "object",
      "required": ["dtype", "shape", "data"],
      "properties": {
        "dtype": {
          "type": "string",
          "description": "Numpy dtype string (e.g., 'float32', 'uint8')"
        },
        "shape": {
          "type": "array",
          "items": {"type": "integer"},
          "description": "Array shape"
        },
        "data": {
          "type": "string",
          "format": "byte",
          "description": "Array data as base64-encoded bytes"
        }
      }
    }
  }
}
```

**Usage in schemas**:
```json
{
  "properties": {
    "image": {
      "$ref": "#/$defs/ndarray",
      "dtype": "uint8",
      "shape": [null, null, 3]
    }
  }
}
```

**Benefits**:
- Leverages JSON Schema ecosystem (validators, tooling)
- Custom NDArray handling for ML/scientific data
- Extensible via `schemaType` field (future: Protobuf, etc.)

### 4. Hybrid Storage

**Open union** for storage location:
- `storageExternal`: External URLs (S3, HTTP, IPFS, etc.)
- `storageBlobs`: ATProto PDS blobs

**Benefits**:
- Flexibility: Use external storage for large datasets
- Decentralization: Use blobs for small datasets or self-hosting
- AppView can proxy both types uniformly

### 5. External Code References

**Lenses use code references** instead of inline code for security:
- Repository URL (GitHub, tangled.org)
- Commit hash (immutability)
- Function path (e.g., `lenses/vision.py:rgb_to_grayscale`)

**Benefits**:
- Secure: No arbitrary code execution
- Verifiable: Commit hash ensures immutability
- Auditable: Users can review code before use

## Example Workflows

### Publishing a Schema

```python
from atdata.atproto import SchemaPublisher

@atdata.packable
class ImageSample:
    image: NDArray  # uint8, [H, W, 3]
    label: str

publisher = SchemaPublisher(client)
schema_uri = publisher.publish_schema(
    ImageSample,
    name="ImageSample",
    version="1.0.0",
    description="RGB image with label"
)
# Result: at://did:plc:abc123/ac.foundation.dataset.sampleSchema/imagesample@1.0.0
```

### Publishing a Dataset

```python
from atdata.atproto import DatasetPublisher

dataset = atdata.Dataset[ImageSample](
    url="s3://my-bucket/dataset-{000000..000009}.tar"
)

publisher = DatasetPublisher(client)
dataset_uri = publisher.publish_dataset(
    dataset,
    name="My Image Dataset",
    schema_uri=schema_uri,
    tags=["computer-vision", "training"]
)
```

### Discovering Datasets

```python
from atdata.atproto import DatasetLoader

loader = DatasetLoader(client)

# Search by tags
datasets = loader.search_datasets(tags=["computer-vision"])

# Load dataset
dataset = loader.load_dataset(datasets[0]['uri'])
```

## Migration & Versioning

### Publishing a New Schema Version

```python
# Publish v2.0.0 with migration lens
schema_uri_v2 = publisher.publish_schema(
    ImageSampleV2,
    name="ImageSample",
    version="2.0.0",
    previous_version=schema_uri_v1,
    migration_lens=migration_lens_uri
)
```

### Getting Latest Schema

```python
from atdata.atproto import query_latest_schema

latest = query_latest_schema(
    client,
    schema_id="imagesample"  # Just the NSID part
)
# Returns: {
#   "uri": "at://.../imagesample@2.0.0",
#   "version": "2.0.0",
#   "record": {...},
#   "allVersions": [...]
# }
```

## Validation

See [06_lexicon_validation.md](../decisions/06_lexicon_validation.md) for validation process.

### Quick Validation

```bash
# Validate Lexicon JSON (requires ATProto tooling)
atproto-lexicon validate ac.foundation.dataset.sampleSchema.json

# Validate example records
python scripts/validate_examples.py
```

## Future Extensions

### Potential Additional Lexicons

- `ac.foundation.dataset.collection` - Group multiple datasets
- `ac.foundation.dataset.benchmark` - Evaluation results on datasets
- `ac.foundation.dataset.attestation` - Formal correctness proofs for Lenses
- `ac.foundation.dataset.verification` - Trusted DID attestations

### Schema Type Extensions

Current: `"schemaType": "jsonschema"`

Future possibilities:
- `"schemaType": "protobuf"` - Protocol Buffers definitions
- `"schemaType": "avro"` - Apache Avro schemas
- Custom domain-specific schema languages

## References

- Planning documents: `../*.md`
- Design decisions: `../decisions/*.md`
- Architectural assessment: `../decisions/assessment.md`
- ATProto Lexicon spec: https://atproto.com/specs/lexicon
- ATProto NSID spec: https://atproto.com/specs/nsid

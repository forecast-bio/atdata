# atdata Lexicon Reference — `ac.foundation.dataset.*`

Developer-facing reference for the ATProto Lexicon definitions that power atdata's distributed dataset federation. These lexicons define the record types, query endpoints, and extensibility mechanisms used to publish, discover, and transform datasets on the atmosphere.

For ATProto fundamentals (Lexicon syntax, XRPC, NSID format), see:

- [`atproto_lexicon_spec.md`](atproto_lexicon_spec.md) — AT Protocol Lexicon specification
- [`atproto_lexicon_guide.md`](atproto_lexicon_guide.md) — Practical Lexicon concepts guide
- [`python_atproto_sdk.md`](python_atproto_sdk.md) — Community Python SDK reference

Lexicon source files: [`src/atdata/lexicons/`](../src/atdata/lexicons/)

---

## Namespace

All lexicons live under the `ac.foundation.dataset` namespace:

- `ac.foundation` — Organization (foundation.ac)
- `dataset` — Domain (distributed datasets)

| NSID | Lexicon Type | Purpose |
|------|-------------|---------|
| `ac.foundation.dataset.schema` | record | Sample type definitions (JSON Schema + NDArray shim) |
| `ac.foundation.dataset.record` | record | Dataset index entries with storage references |
| `ac.foundation.dataset.lens` | record | Bidirectional transformations between sample types |
| `ac.foundation.dataset.resolveSchema` | query | Fetch latest schema version by NSID |
| `ac.foundation.dataset.schemaType` | token | Extensible registry of schema format identifiers |
| `ac.foundation.dataset.arrayFormat` | token | Extensible registry of array serialization formats |
| `ac.foundation.dataset.storageExternal` | object | External URL-based storage (union member) |
| `ac.foundation.dataset.storageBlobs` | object | ATProto PDS blob storage (union member) |
| NDArray shim | JSON Schema | Numpy NDArray type definition for use in sample schemas |

Any PDS can host records in this namespace. The PDS is lexicon-agnostic — records are created with `validate=False` since the PDS does not know custom lexicons.

---

## Record Relationships

```
                         ┌─────────────────────────────┐
                         │  ac.foundation.dataset.lens  │
                         │                               │
                         │  sourceSchema ──┐             │
                         │  targetSchema ──┼─ AT-URIs    │
                         │  getterCode     │             │
                         │  putterCode     │             │
                         └────────────────┬┘             │
                                          │              │
    ┌─────────────────────────────┐       │              │
    │ ac.foundation.dataset.schema│◄──────┘              │
    │                              │◄─────────────────────┘
    │  schemaType ─► schemaType   │
    │  schema (union):            │
    │    └─ #jsonSchemaFormat     │
    │         └─ arrayFormatVersions ─► arrayFormat
    └──────────────┬──────────────┘
                   │
                   │ AT-URI (schemaRef)
                   ▼
    ┌─────────────────────────────┐
    │ ac.foundation.dataset.record│
    │                              │
    │  storage (union):           │
    │    ├─ storageExternal       │
    │    └─ storageBlobs          │
    └─────────────────────────────┘
```

- **Schema** defines sample types. Lens records reference schemas via AT-URIs.
- **Record** is a dataset index entry. It references a schema via `schemaRef`.
- **Lens** connects two schemas (source → target) with external code references.
- **Token registries** (`schemaType`, `arrayFormat`) govern extensibility.
- **Storage objects** (`storageExternal`, `storageBlobs`) are union members within record.

---

## Core Record Types

### `ac.foundation.dataset.schema`

Defines a PackableSample-compatible sample type using JSON Schema.

- **Lexicon type:** `record`
- **Key type:** `any` (custom rkey format)
- **Record key convention:** `{NSID}@{semver}`

The `"key": "any"` declaration permits freeform record keys. atdata uses the compound format `{NSID}@{semver}` where the NSID portion is a permanent identifier for the schema type and the semver portion pins a specific version. This enables immutable version history — each version is a separate, permanent record.

Example rkey: `com.example.imagesample@1.0.0`

#### Fields

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `name` | `string` | yes | maxLength: 100 | Human-readable display name. The NSID in the record URI provides unique identification; name collisions across NSIDs are acceptable. |
| `version` | `string` | yes | maxLength: 100, pattern: semver | Semantic version (e.g., `1.0.0`). Must match the version in the rkey. |
| `schemaType` | `ref` → `ac.foundation.dataset.schemaType` | yes | — | Indicates which union member is present in the `schema` field. Currently: `"jsonSchema"`. |
| `schema` | `union` | yes | open (`closed: false`) | Schema definition. Currently supports `#jsonSchemaFormat`. Open union allows future formats without breaking changes. |
| `description` | `string` | no | maxLength: 5000 | Human-readable description of the sample type. |
| `metadata` | `object` | no | maxProperties: 50 | Optional metadata. Known sub-fields: `license` (string, SPDX, maxLength: 200), `tags` (array of strings, max 30 items). |
| `createdAt` | `string` (datetime) | yes | — | ISO 8601 timestamp. Immutable once set. |

#### `#jsonSchemaFormat` sub-definition

The currently supported schema format. This is a JSON Schema Draft 7 object with atdata extensions.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `$type` | `string` | yes | Must be `"ac.foundation.dataset.schema#jsonSchemaFormat"`. |
| `$schema` | `string` | yes | Must be `"http://json-schema.org/draft-07/schema#"`. |
| `type` | `string` | yes | Must be `"object"` (sample types are always objects). |
| `properties` | `object` | yes | Field definitions for the sample type. Must have at least 1 property. |
| `arrayFormatVersions` | `object` | no | Mapping from `arrayFormat` identifiers to semver strings. Keys are token values (e.g., `"ndarrayBytes"`), values are version strings (e.g., `"1.0.0"`). Defaults to `{"ndarrayBytes": "1.0.0"}` when omitted. |

NDArray fields within `properties` use `$ref: "#/$defs/ndarray"` and declare constraints via extension fields:

- `x-atdata-dtype` — Numpy dtype string (e.g., `"uint8"`, `"float32"`)
- `x-atdata-shape` — Shape constraints as array (`null` for variable dimensions, e.g., `[null, null, 3]`)
- `x-atdata-notes` — Freeform annotation

The `$defs.ndarray` definition must be included inline (see [NDArray Shim](#ndarray-shim)).

#### Example record

Stored at rkey `imagesample@1.0.0`:

```json
{
  "$type": "ac.foundation.dataset.schema",
  "name": "ImageSample",
  "version": "1.0.0",
  "schemaType": "jsonSchema",
  "schema": {
    "$type": "ac.foundation.dataset.schema#jsonSchemaFormat",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ImageSample",
    "type": "object",
    "arrayFormatVersions": {
      "ndarrayBytes": "1.0.0"
    },
    "required": ["image", "label"],
    "properties": {
      "image": {
        "$ref": "#/$defs/ndarray",
        "description": "RGB image with variable height/width",
        "x-atdata-dtype": "uint8",
        "x-atdata-shape": [null, null, 3],
        "x-atdata-notes": "Images must have 3 color channels (RGB)"
      },
      "label": {
        "type": "string",
        "description": "Human-readable label for the image"
      },
      "confidence": {
        "type": "number",
        "description": "Optional confidence score",
        "minimum": 0,
        "maximum": 1
      }
    },
    "$defs": {
      "ndarray": {
        "type": "string",
        "format": "byte",
        "description": "Numpy array serialized using numpy .npy format",
        "contentEncoding": "base64",
        "contentMediaType": "application/octet-stream"
      }
    }
  },
  "description": "Sample type for images with labels",
  "metadata": {
    "license": "MIT",
    "tags": ["computer-vision", "image-classification"]
  },
  "createdAt": "2025-01-06T12:00:00Z"
}
```

---

### `ac.foundation.dataset.record`

Index record for a WebDataset-backed dataset.

- **Lexicon type:** `record`
- **Key type:** `tid` (timestamp-based, ATProto default)

#### Fields

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `name` | `string` | yes | maxLength: 200 | Human-readable dataset name. |
| `schemaRef` | `string` (at-uri) | yes | maxLength: 500 | AT-URI of the schema record for this dataset's samples. |
| `storage` | `union` | yes | — | Storage location. Members: `storageExternal`, `storageBlobs`. |
| `description` | `string` | no | maxLength: 5000 | Human-readable description. |
| `metadata` | `bytes` | no | maxLength: 100000 | Msgpack-encoded dict for extended key-value metadata. Bytes rather than JSON to stay within ATProto record size limits. |
| `tags` | `array` of `string` | no | max 30 items, each maxLength: 150 | Searchable tags for discovery. |
| `size` | `ref` → `#datasetSize` | no | — | Dataset size information. |
| `license` | `string` | no | maxLength: 200 | SPDX license identifier (e.g., `MIT`, `Apache-2.0`, `CC-BY-4.0`). |
| `createdAt` | `string` (datetime) | yes | — | ISO 8601 timestamp. |

#### `#datasetSize` sub-definition

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `samples` | `integer` | no | minimum: 0 | Total number of samples. |
| `bytes` | `integer` | no | minimum: 0 | Total size in bytes. |
| `shards` | `integer` | no | minimum: 1 | Number of WebDataset tar shards. |

#### Example: external storage

```json
{
  "$type": "ac.foundation.dataset.record",
  "name": "CIFAR-10 Training Set",
  "schemaRef": "at://did:plc:abc123/ac.foundation.dataset.schema/imageclassification@1.0.0",
  "storage": {
    "$type": "ac.foundation.dataset.storageExternal",
    "urls": [
      "s3://my-bucket/cifar10-train-{000000..000049}.tar"
    ]
  },
  "description": "CIFAR-10 training images (50,000 samples) stored as WebDataset shards on S3",
  "tags": ["computer-vision", "classification", "cifar10", "training"],
  "size": {
    "samples": 50000,
    "bytes": 178456789,
    "shards": 50
  },
  "license": "MIT",
  "createdAt": "2025-01-06T12:00:00Z"
}
```

#### Example: blob storage

```json
{
  "$type": "ac.foundation.dataset.record",
  "name": "Small Sample Dataset",
  "schemaRef": "at://did:plc:def456/ac.foundation.dataset.schema/textsample@2.1.0",
  "storage": {
    "$type": "ac.foundation.dataset.storageBlobs",
    "blobs": [
      {
        "$type": "blob",
        "ref": { "$link": "bafyreig4rvsqx3vfzdchq2qx7xr2nq2y4vjvd4w5pqtjwkqiw7h5e6vf7e" },
        "mimeType": "application/x-tar",
        "size": 1234567
      },
      {
        "$type": "blob",
        "ref": { "$link": "bafyreig5saabc3defghijklmnopqrstuvwxyz123456789abcdefghijk" },
        "mimeType": "application/x-tar",
        "size": 2345678
      }
    ]
  },
  "description": "Small text dataset stored directly on PDS for maximum decentralization",
  "tags": ["nlp", "text", "small-dataset"],
  "size": {
    "samples": 1000,
    "bytes": 3580245,
    "shards": 2
  },
  "license": "CC-BY-4.0",
  "createdAt": "2025-01-07T10:30:00Z"
}
```

---

### `ac.foundation.dataset.lens`

Bidirectional transformation between two sample types, with code stored in external repositories.

- **Lexicon type:** `record`
- **Key type:** `tid` (timestamp-based)

#### Fields

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `name` | `string` | yes | maxLength: 100 | Human-readable lens name. |
| `sourceSchema` | `string` | yes | maxLength: 500 | AT-URI of the source schema. |
| `targetSchema` | `string` | yes | maxLength: 500 | AT-URI of the target schema. |
| `description` | `string` | no | maxLength: 1000 | What this transformation does. |
| `getterCode` | `ref` → `#codeReference` | yes | — | Code reference for getter function (Source → Target). |
| `putterCode` | `ref` → `#codeReference` | yes | — | Code reference for putter function (Target, Source → Source). |
| `language` | `string` | no | maxLength: 50 | Programming language (e.g., `"python"`, `"typescript"`). |
| `metadata` | `object` | no | — | Arbitrary metadata (author, performance notes, etc.). |
| `createdAt` | `string` (datetime) | yes | — | ISO 8601 timestamp. |

#### `#codeReference` sub-definition

References a function in an external git repository. Code is referenced rather than stored inline for security — users can review code before executing it, and commit hashes ensure immutability.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `repository` | `string` | yes | maxLength: 500 | Repository URL (e.g., `https://github.com/user/repo` or `at://did/tangled.repo/...`). |
| `commit` | `string` | yes | maxLength: 40 | Git commit hash. Ensures immutability. |
| `path` | `string` | yes | maxLength: 500 | Path to function (e.g., `lenses/color.py:rgb_to_grayscale`). |
| `branch` | `string` | no | maxLength: 100 | Branch name (informational only — commit hash is authoritative). |

#### Example record

```json
{
  "$type": "ac.foundation.dataset.lens",
  "name": "RGB to Grayscale Conversion",
  "sourceSchema": "at://did:plc:abc123/ac.foundation.dataset.schema/rgbimage@1.0.0",
  "targetSchema": "at://did:plc:abc123/ac.foundation.dataset.schema/grayscaleimage@1.0.0",
  "description": "Converts RGB images to grayscale using standard luminosity formula",
  "getterCode": {
    "repository": "https://github.com/alice/vision-lenses",
    "commit": "a1b2c3d4e5f6789abcdef0123456789abcdef012",
    "path": "lenses/color.py:rgb_to_grayscale",
    "branch": "main"
  },
  "putterCode": {
    "repository": "https://github.com/alice/vision-lenses",
    "commit": "a1b2c3d4e5f6789abcdef0123456789abcdef012",
    "path": "lenses/color.py:grayscale_to_rgb",
    "branch": "main"
  },
  "language": "python",
  "metadata": {
    "author": "alice.bsky.social",
    "performance": "O(n) where n is number of pixels",
    "reversible": false,
    "notes": "Putter creates approximate RGB by duplicating grayscale channel"
  },
  "createdAt": "2025-01-07T14:00:00Z"
}
```

---

## Query Endpoint

### `ac.foundation.dataset.resolveSchema`

Resolve a schema by its permanent NSID identifier. When version is omitted, resolves to the most recently created schema with the given NSID. Follows the same pattern as `resolveLabel`.

- **Lexicon type:** `query`

#### Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `handle` | `string` | yes | DID or handle of the schema owner. |
| `schemaId` | `string` | yes | The permanent NSID identifier (the `{NSID}` portion of the rkey, without the `@{semver}` suffix). maxLength: 500 |
| `version` | `string` | no | Specific version to resolve. If omitted, resolves to latest. maxLength: 20 |

#### Response

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `uri` | `string` (at-uri) | yes | AT-URI of the resolved schema record. |
| `cid` | `string` | yes | CID of the resolved schema record. |
| `record` | `ref` → `ac.foundation.dataset.schema` | yes | The full schema record. |

#### Errors

| Name | Description |
|------|-------------|
| `SchemaNotFound` | No schema found with the given NSID. |

---

## Storage Union Types

The `storage` field on `ac.foundation.dataset.record` is a union with two members.

### `ac.foundation.dataset.storageExternal`

External URL-based storage for WebDataset tar archives.

- **Lexicon type:** `object` (union member)

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `urls` | `array` of `string` (uri) | yes | minLength: 1, each maxLength: 1000 | WebDataset URLs. Supports brace notation for sharding (e.g., `data-{000000..000099}.tar`). |

Use cases: large datasets on S3, HTTP servers, IPFS gateways. No size limits imposed by ATProto.

### `ac.foundation.dataset.storageBlobs`

ATProto PDS blob-based storage for WebDataset tar archives.

- **Lexicon type:** `object` (union member)

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `blobs` | `array` of `blob` | yes | minLength: 1 | ATProto blob references. Each blob is a tar archive. |

Each blob reference has the standard ATProto blob format:

```json
{
  "$type": "blob",
  "ref": { "$link": "<CID>" },
  "mimeType": "application/x-tar",
  "size": 1234567
}
```

Use cases: small datasets, maximum decentralization, no external infrastructure dependency. Subject to PDS blob size limits.

---

## Extensibility Mechanisms

### Token Registries

Both `schemaType` and `arrayFormat` use the same extensibility pattern:

1. A **main def** of type `string` with a `knownValues` array listing recognized values.
2. Each known value has a corresponding **token def** providing canonical documentation.
3. New values are added to `knownValues` and a new token def is added — no breaking changes.
4. Consumers should tolerate unknown values (forward compatibility).

### `ac.foundation.dataset.schemaType`

Registry of schema format identifiers. The `schemaType` field on a schema record declares which union member appears in the `schema` field.

| Token | knownValue | Corresponding Format Def | Description |
|-------|------------|--------------------------|-------------|
| `#jsonSchema` | `"jsonSchema"` | `schema#jsonSchemaFormat` | JSON Schema Draft 7 with NDArray shim |

Extension path: to add e.g. Avro support, add `"avro"` to `knownValues`, create an `#avro` token def, and add a `schema#avroFormat` object def to the schema union.

### `ac.foundation.dataset.arrayFormat`

Registry of array serialization format identifiers. Schema records reference these via the `arrayFormatVersions` mapping in `#jsonSchemaFormat`.

| Token | knownValue | Current Version | Shim URL Pattern | Description |
|-------|------------|-----------------|------------------|-------------|
| `#ndarrayBytes` | `"ndarrayBytes"` | `1.0.0` | `https://foundation.ac/schemas/atdata-ndarray-bytes/{version}/` | Numpy `.npy` binary format |

Each format has versioned specifications maintained at canonical URLs by foundation.ac.

### Open Unions

The `schema` field on schema records uses `"closed": false`, meaning new union members can appear without updating the lexicon definition. Consumers must handle unknown `$type` values gracefully (e.g., by reporting "unsupported schema format" rather than crashing).

The `storage` field on dataset records is a standard union (implicitly open in ATProto semantics).

---

## NDArray Shim

The NDArray shim (`ndarray_shim.json`) bridges ATProto/JSON Schema and numpy. It defines how NDArray fields are represented in JSON Schema while remaining compatible with atdata's msgpack/WebDataset serialization.

- **`$id`:** `https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0`
- **Version:** `1.0.0`

### `$defs.ndarray` definition

```json
{
  "type": "string",
  "format": "byte",
  "description": "Numpy array serialized using numpy .npy format via np.save",
  "contentEncoding": "base64",
  "contentMediaType": "application/octet-stream"
}
```

In **JSON** contexts, the array data is base64-encoded. In **msgpack** contexts (the primary serialization path in atdata), it is raw bytes.

### Extension fields

When a schema property references `$defs.ndarray`, it can declare additional constraints:

| Extension Field | Type | Description |
|----------------|------|-------------|
| `x-atdata-dtype` | `string` | Numpy dtype (e.g., `"float32"`, `"uint8"`, `"int64"`). |
| `x-atdata-shape` | `array` | Shape constraints. `null` entries mean variable dimensions (e.g., `[null, null, 3]` for H×W×3). |
| `x-atdata-notes` | `string` | Freeform annotation about the array. |

atdata's codegen reads these fields to produce Python dataclasses with proper `NDArray` type annotations and dtype/shape metadata.

---

## AT-URI Patterns

Concrete URI templates for each collection:

```
at://{did}/ac.foundation.dataset.schema/{nsid}@{semver}
at://{did}/ac.foundation.dataset.record/{tid}
at://{did}/ac.foundation.dataset.lens/{tid}
```

- **`{did}`** — Decentralized Identifier of the repository owner (e.g., `did:plc:abc123`).
- **`{nsid}@{semver}`** — Schema rkey: permanent identifier + version (e.g., `com.example.imagesample@1.0.0`).
- **`{tid}`** — Timestamp Identifier, ATProto's default auto-generated key for records and lenses.

---

## Implementation Cross-Reference

| Lexicon NSID | Python Type | Source File |
|-------------|-------------|-------------|
| `ac.foundation.dataset.schema` | `SchemaRecord` | `src/atdata/atmosphere/schema.py` |
| `ac.foundation.dataset.record` | `DatasetRecord` | `src/atdata/atmosphere/records.py` |
| `ac.foundation.dataset.lens` | `LensRecord` | `src/atdata/atmosphere/lens.py` |
| (all types) | `AtUri`, `FieldType`, `FieldDef`, `StorageLocation`, `CodeReference` | `src/atdata/atmosphere/_types.py` |
| (blob storage) | `PDSBlobStore` | `src/atdata/atmosphere/store.py` |
| (lexicon loader) | `load_lexicon()`, `list_lexicons()` | `src/atdata/lexicons/__init__.py` |

The namespace constant `LEXICON_NAMESPACE = "ac.foundation.dataset"` is defined in `src/atdata/atmosphere/_types.py`.

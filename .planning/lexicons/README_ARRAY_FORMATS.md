# Array Format Registry

This document explains the token-based registry pattern for atdata array serialization formats.

## Overview

Array formats define how numpy NDArray fields are serialized in atdata sample types. The system provides:

1. **Token-based registry**: `ac.foundation.dataset.arrayFormat` Lexicon
2. **Version tracking**: Each schema declares which format versions it uses
3. **Canonical shim schemas**: Foundation.ac maintains standard JSON Schema shims at predictable URLs

## Pattern

### arrayFormat Lexicon Structure

```json
{
  "lexicon": 1,
  "id": "ac.foundation.dataset.arrayFormat",
  "defs": {
    "main": {
      "type": "string",
      "knownValues": ["ndarrayBytes"],
      "maxLength": 50
    },
    "ndarrayBytes": {
      "type": "token",
      "description": "Numpy .npy binary format..."
    }
  }
}
```

### Usage in sampleSchema

Schema records declare format versions in `arrayFormatVersions` field:

```json
{
  "$type": "ac.foundation.dataset.sampleSchema",
  "schemaType": "jsonSchema",
  "schema": {
    "$type": "ac.foundation.dataset.sampleSchema#jsonSchemaFormat",
    "arrayFormatVersions": {
      "ndarrayBytes": "1.0.0"
    },
    "properties": {
      "image": {
        "$ref": "#/$defs/ndarray",
        "x-atdata-dtype": "uint8"
      }
    },
    "$defs": {
      "ndarray": {
        "type": "string",
        "format": "byte",
        ...
      }
    }
  }
}
```

## Canonical Shim Schema URLs

Foundation.ac maintains JSON Schema shims at canonical URLs:

```
https://foundation.ac/schemas/atdata-{format}-bytes/{version}/
```

Examples:
- `https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0/`
- `https://foundation.ac/schemas/atdata-arrow-bytes/1.0.0/` (future)

These shim schemas define the JSON Schema representation (base64-encoded bytes) for each format.

## Default Behavior

If `arrayFormatVersions` is omitted, the system defaults to:

```json
{
  "ndarrayBytes": "1.0.0"
}
```

This ensures backward compatibility and simplifies common cases.

## Current Array Formats

| Token Def | knownValue | Current Version | Description |
|-----------|------------|-----------------|-------------|
| `#ndarrayBytes` | `"ndarrayBytes"` | `1.0.0` | Numpy .npy binary format with dtype/shape header |

## Adding New Array Formats

To add support for a new array format (e.g., Apache Arrow):

### 1. Add token def to arrayFormat Lexicon

Edit `ac.foundation.dataset.arrayFormat.json`:

```json
{
  "defs": {
    "main": {
      "knownValues": ["ndarrayBytes", "arrowBytes"]
    },
    "arrowBytes": {
      "type": "token",
      "description": "Apache Arrow IPC format for array serialization..."
    }
  }
}
```

### 2. Publish shim schema at canonical URL

Create and publish JSON Schema shim at:
```
https://foundation.ac/schemas/atdata-arrow-bytes/1.0.0/
```

### 3. Use in sample schemas

Declare format version in schema records:

```json
{
  "arrayFormatVersions": {
    "arrowBytes": "1.0.0"
  }
}
```

## Version Evolution

### Minor/Patch Updates

For backward-compatible changes:
- Publish new version at new URL (e.g., `1.1.0`)
- Update `arrayFormatVersions` in schema records
- Old versions remain accessible

### Major Updates

For breaking changes:
- Consider new format name (e.g., `ndarrayBytes2`)
- Or use major version in URL structure
- Schemas can migrate via Lens transformations

## Design Rationale

This pattern provides:

1. **Centralized Discovery**: Query `ac.foundation.dataset.arrayFormat` to see all supported formats
2. **Explicit Versioning**: Each schema declares exactly which format versions it uses
3. **Canonical References**: Predictable URLs for shim schemas maintained by foundation.ac
4. **Extensibility**: New formats added via tokens without breaking existing schemas
5. **Flexibility**: Schemas can use multiple formats simultaneously (if needed)

## Relationship to Codegen

When atdata codegen processes a sampleSchema:

1. Reads `arrayFormatVersions` to know which formats are used
2. Fetches canonical shim schemas from foundation.ac URLs
3. Generates Python dataclasses with proper NDArray type hints
4. Implements serialization using appropriate format (currently `.npy` via `_helpers.py`)

## References

- [ac.foundation.dataset.arrayFormat Lexicon](./ac.foundation.dataset.arrayFormat.json)
- [ac.foundation.dataset.sampleSchema Lexicon](./ac.foundation.dataset.sampleSchema.json)
- [NDArray Shim Specification](../.planning/ndarray_shim_spec.md)
- [ATProto Lexicon Token Type](https://atproto.com/guides/lexicon)

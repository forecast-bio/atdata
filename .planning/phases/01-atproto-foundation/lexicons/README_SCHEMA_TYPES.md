# Schema Type Registry

This document explains the token-based registry pattern for atdata schema types.

## Pattern

Schema types in atdata are managed through the `ac.foundation.dataset.schemaType` Lexicon:

1. **Single Lexicon file**: `ac.foundation.dataset.schemaType.json`
2. **Main def**: String type with `knownValues` listing supported schema types
3. **Token defs**: Each schema type has a corresponding token def (e.g., `#jsonSchema`)
4. **Reference in sampleSchema**: The `schemaType` field refs to `ac.foundation.dataset.schemaType`

## Structure

```json
{
  "lexicon": 1,
  "id": "ac.foundation.dataset.schemaType",
  "defs": {
    "main": {
      "type": "string",
      "knownValues": ["jsonSchema"],
      "maxLength": 50
    },
    "jsonSchema": {
      "type": "token",
      "description": "JSON Schema Draft 7 format..."
    }
  }
}
```

## Usage in sampleSchema

The `schemaType` field references the schemaType Lexicon:

```json
{
  "$type": "ac.foundation.dataset.sampleSchema",
  "name": "ImageSample",
  "version": "1.0.0",
  "schemaType": "jsonSchema",
  "schema": {
    "$type": "ac.foundation.dataset.sampleSchema#jsonSchemaFormat",
    ...
  }
}
```

In the Lexicon definition:

```json
{
  "schemaType": {
    "type": "ref",
    "ref": "ac.foundation.dataset.schemaType"
  }
}
```

## Adding New Schema Types

To add support for a new schema format (e.g., Avro, Protobuf):

### 1. Add token def to schemaType Lexicon

Edit `ac.foundation.dataset.schemaType.json`:

```json
{
  "defs": {
    "main": {
      "type": "string",
      "knownValues": ["jsonSchema", "avro"],
      "maxLength": 50
    },
    "avro": {
      "type": "token",
      "description": "Apache Avro schema format..."
    }
  }
}
```

### 2. Add format def to sampleSchema Lexicon

Edit `ac.foundation.dataset.sampleSchema.json`:

```json
{
  "defs": {
    "avroFormat": {
      "type": "object",
      "description": "Apache Avro schema format...",
      "required": ["$type", "type"],
      "properties": {
        "$type": {
          "type": "string",
          "const": "ac.foundation.dataset.sampleSchema#avroFormat"
        },
        "type": {
          "type": "string"
        },
        "fields": {
          "type": "array"
        }
      }
    }
  }
}
```

### 3. Update schema union refs

In sampleSchema main record:

```json
{
  "schema": {
    "type": "union",
    "refs": [
      "ac.foundation.dataset.sampleSchema#jsonSchemaFormat",
      "ac.foundation.dataset.sampleSchema#avroFormat"
    ],
    "closed": false
  }
}
```

## Current Schema Types

| Token Def | knownValue | Format Def | Description |
|-----------|------------|------------|-------------|
| `#jsonSchema` | `"jsonSchema"` | `#jsonSchemaFormat` | JSON Schema Draft 7 |

## Design Rationale

This pattern provides:

1. **Centralized Registry**: Single Lexicon (`schemaType`) lists all supported types
2. **Type Safety**: Token defs provide canonical documentation for each schema type
3. **Extensibility**: New types added to `knownValues` + token defs without breaking changes
4. **Validation**: Refs ensure schemaType values are validated against known types
5. **Discoverability**: Query `ac.foundation.dataset.schemaType` to see all supported types

## References

- [ATProto Lexicon Token Type](https://atproto.com/guides/lexicon)
- [ATProto Lexicon Spec](.reference/atproto_lexicon_spec.md)

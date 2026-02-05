# AT Protocol Lexicon Guide

> **Source**: [AT Protocol Lexicon Documentation](https://atproto.com/guides/lexicon)

## Overview

Lexicon is a JSON-based schema language that defines RPC methods and record types for AT Protocol. It enables interoperability by establishing agreed-upon behaviors and semantics across the open network.

## Key Concepts

### NSIDs (Namespaced Identifiers)

Schemas use reverse-DNS format identifiers indicating ownership:

```
com.atproto.repo.createRecord   # Core ATProto API
app.bsky.feed.post              # Bluesky app record type
ac.foundation.dataset.sampleSchema  # Our custom namespace
```

Format: `authority.name` where authority is reverse-DNS

### Why Not RDF?

Lexicon prioritizes:
- Schema enforcement (not optional metadata)
- Code generation with types and validation
- Practical developer experience

## Schema Types

### 1. Record Types

Define the structure of data stored in repositories:

```json
{
  "lexicon": 1,
  "id": "com.example.follow",
  "defs": {
    "main": {
      "type": "record",
      "key": "tid",
      "record": {
        "type": "object",
        "required": ["subject", "createdAt"],
        "properties": {
          "subject": { "type": "string", "format": "did" },
          "createdAt": { "type": "string", "format": "datetime" }
        }
      }
    }
  }
}
```

Records stored in repos have a `$type` field mapping to their schema.

### 2. Query Methods

Define HTTP GET endpoints:

```json
{
  "lexicon": 1,
  "id": "com.example.getProfile",
  "defs": {
    "main": {
      "type": "query",
      "parameters": {
        "type": "params",
        "required": ["actor"],
        "properties": {
          "actor": { "type": "string", "format": "at-identifier" }
        }
      },
      "output": {
        "encoding": "application/json",
        "schema": { "$ref": "#/defs/profileView" }
      }
    }
  }
}
```

Maps to: `GET /xrpc/com.example.getProfile?actor=...`

### 3. Procedure Methods

Define HTTP POST endpoints:

```json
{
  "lexicon": 1,
  "id": "com.example.updateProfile",
  "defs": {
    "main": {
      "type": "procedure",
      "input": {
        "encoding": "application/json",
        "schema": { ... }
      },
      "output": {
        "encoding": "application/json",
        "schema": { ... }
      }
    }
  }
}
```

### 4. Tokens

Declare reusable global identifiers for extensible enums:

```json
{
  "lexicon": 1,
  "id": "com.example.status.active",
  "defs": {
    "main": {
      "type": "token",
      "description": "User is active"
    }
  }
}
```

Instead of hardcoding enum values, use tokens. Teams can add values without collisions.

## Field Types

### Primitives

| Type | Description |
|------|-------------|
| `string` | Text, with optional format/length constraints |
| `integer` | Whole numbers |
| `boolean` | true/false |
| `bytes` | Binary data (base64 encoded in JSON) |
| `cid-link` | Content identifier reference |
| `unknown` | Any JSON value |

### String Formats

| Format | Description |
|--------|-------------|
| `at-uri` | AT Protocol URI |
| `at-identifier` | Handle or DID |
| `did` | Decentralized identifier |
| `handle` | User handle |
| `datetime` | ISO 8601 timestamp |
| `uri` | Generic URI |
| `language` | BCP 47 language tag |

### Complex Types

```json
// Object
{
  "type": "object",
  "required": ["field1"],
  "properties": {
    "field1": { "type": "string" },
    "field2": { "type": "integer" }
  }
}

// Array
{
  "type": "array",
  "items": { "type": "string" },
  "maxLength": 100
}

// Union (discriminated)
{
  "type": "union",
  "refs": [
    "#defs/typeA",
    "#defs/typeB"
  ]
}

// Reference to another schema
{
  "type": "ref",
  "ref": "com.example.otherSchema#defs/thing"
}
```

### Blob References

For binary data stored separately:

```json
{
  "type": "blob",
  "accept": ["image/jpeg", "image/png"],
  "maxSize": 1000000
}
```

## Versioning Rules

**Published schemas are immutable regarding constraints.**

- Loosening constraints breaks old software validation
- Tightening constraints breaks new software validation
- Only **optional** constraints may be added to previously unconstrained fields
- Major changes require **new NSIDs**

## Schema Distribution

Schemas should be published as machine-readable, network-accessible resources:

1. Host at well-known URL: `https://authority.com/.well-known/lexicons/`
2. Or embed in documentation
3. Ensure canonical representation exists for consumers

## Record Keys (rkeys)

Records in collections are identified by keys:

| Key Type | Description |
|----------|-------------|
| `tid` | Timestamp-based ID (sortable, unique) |
| `literal:self` | Singleton record (e.g., profile) |
| `any` | Any valid string |

TID format: 13-character base32-sortable timestamp

## Example: Complete Lexicon

```json
{
  "lexicon": 1,
  "id": "ac.foundation.dataset.sampleSchema",
  "revision": 1,
  "description": "Schema definition for a PackableSample type",
  "defs": {
    "main": {
      "type": "record",
      "key": "tid",
      "description": "A sample schema record",
      "record": {
        "type": "object",
        "required": ["name", "version", "fields"],
        "properties": {
          "name": {
            "type": "string",
            "description": "Human-readable schema name"
          },
          "version": {
            "type": "string",
            "description": "Semantic version"
          },
          "fields": {
            "type": "array",
            "items": { "type": "ref", "ref": "#defs/fieldDef" }
          },
          "createdAt": {
            "type": "string",
            "format": "datetime"
          }
        }
      }
    },
    "fieldDef": {
      "type": "object",
      "required": ["name", "fieldType"],
      "properties": {
        "name": { "type": "string" },
        "fieldType": { "type": "ref", "ref": "#defs/fieldType" },
        "optional": { "type": "boolean", "default": false }
      }
    },
    "fieldType": {
      "type": "union",
      "refs": [
        "#defs/primitiveType",
        "#defs/arrayType"
      ]
    },
    "primitiveType": {
      "type": "object",
      "required": ["kind"],
      "properties": {
        "kind": {
          "type": "string",
          "knownValues": ["string", "int", "float", "bool", "bytes"]
        }
      }
    },
    "arrayType": {
      "type": "object",
      "required": ["kind", "elementType"],
      "properties": {
        "kind": { "type": "string", "const": "ndarray" },
        "elementType": { "type": "string" },
        "shape": {
          "type": "array",
          "items": { "type": "integer" }
        }
      }
    }
  }
}
```

## XRPC (Cross-Server RPC)

Lexicons map to HTTP endpoints:

```
com.example.getProfile()
  → GET /xrpc/com.example.getProfile

com.example.createPost()
  → POST /xrpc/com.example.createPost
```

## Validation Behavior

The PDS can validate records against lexicons, but:

1. PDS is lexicon-agnostic by default
2. Validation can be disabled: `validate: false`
3. Unknown lexicons are stored without validation
4. Rate limits prevent abuse (not schema enforcement)

## Resources

- **Lexicon Specification**: https://atproto.com/specs/lexicon
- **Lexicon Guide**: https://atproto.com/guides/lexicon
- **Bluesky Lexicons**: https://github.com/bluesky-social/atproto/tree/main/lexicons

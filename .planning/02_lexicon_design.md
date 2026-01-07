# Lexicon Design for ATProto Integration

## Overview

This document specifies the three Lexicon schemas needed for `atdata` ATProto integration:

1. **Schema Record** (`app.bsky.atdata.schema`) - Defines PackableSample types
2. **Dataset Record** (`app.bsky.atdata.dataset`) - Index records pointing to WebDataset files
3. **Lens Record** (`app.bsky.atdata.lens`) - Transformation mappings between schemas

## Design Principles

- **Self-describing**: Records contain all necessary metadata
- **Versioned**: Schema evolution supported through versioning
- **Lightweight**: Minimal overhead, fast to parse
- **Extensible**: Future additions don't break existing records
- **Language-agnostic**: Usable from Python, TypeScript, Rust, etc.

## 1. Schema Record Lexicon

**NSID**: `app.bsky.atdata.schema` (tentative namespace)

**Purpose**: Define a reusable PackableSample type that can be instantiated via codegen

### Proposed Structure

```json
{
  "lexicon": 1,
  "id": "app.bsky.atdata.schema",
  "defs": {
    "main": {
      "type": "record",
      "description": "Definition of a PackableSample-compatible sample type",
      "key": "tid",
      "record": {
        "type": "object",
        "required": ["name", "version", "fields", "createdAt"],
        "properties": {
          "name": {
            "type": "string",
            "description": "Human-readable name for this sample type",
            "maxLength": 100
          },
          "version": {
            "type": "string",
            "description": "Semantic version (e.g., '1.0.0')",
            "maxLength": 20
          },
          "description": {
            "type": "string",
            "description": "Human-readable description",
            "maxLength": 1000
          },
          "fields": {
            "type": "array",
            "description": "List of fields in this sample type",
            "items": {
              "type": "ref",
              "ref": "#field"
            }
          },
          "metadata": {
            "type": "object",
            "description": "Arbitrary metadata (author, license, etc.)"
          },
          "createdAt": {
            "type": "string",
            "format": "datetime"
          }
        }
      }
    },
    "field": {
      "type": "object",
      "description": "A field within a sample type",
      "required": ["name", "type"],
      "properties": {
        "name": {
          "type": "string",
          "description": "Field name (Python identifier)",
          "maxLength": 100
        },
        "type": {
          "type": "ref",
          "ref": "#fieldType"
        },
        "optional": {
          "type": "boolean",
          "description": "Whether field can be None",
          "default": false
        },
        "description": {
          "type": "string",
          "description": "Field documentation",
          "maxLength": 500
        }
      }
    },
    "fieldType": {
      "type": "union",
      "refs": [
        "#primitiveType",
        "#arrayType",
        "#nestedType"
      ]
    },
    "primitiveType": {
      "type": "object",
      "required": ["kind", "primitive"],
      "properties": {
        "kind": {
          "type": "string",
          "const": "primitive"
        },
        "primitive": {
          "type": "string",
          "enum": ["str", "int", "float", "bool", "bytes"]
        }
      }
    },
    "arrayType": {
      "type": "object",
      "required": ["kind", "dtype"],
      "properties": {
        "kind": {
          "type": "string",
          "const": "ndarray"
        },
        "dtype": {
          "type": "string",
          "description": "Numpy dtype string (e.g., 'float32', 'uint8')",
          "maxLength": 20
        },
        "shape": {
          "type": "array",
          "description": "Optional shape constraint (null for dynamic dimensions)",
          "items": {
            "type": "integer"
          }
        }
      }
    },
    "nestedType": {
      "type": "object",
      "required": ["kind", "schemaRef"],
      "properties": {
        "kind": {
          "type": "string",
          "const": "nested"
        },
        "schemaRef": {
          "type": "string",
          "description": "AT-URI reference to another schema record"
        }
      }
    }
  }
}
```

### Example Schema Record

```json
{
  "$type": "app.bsky.atdata.schema",
  "name": "ImageSample",
  "version": "1.0.0",
  "description": "Sample containing an image with label",
  "fields": [
    {
      "name": "image",
      "type": {
        "kind": "ndarray",
        "dtype": "uint8",
        "shape": [null, null, 3]
      },
      "description": "RGB image with variable height/width"
    },
    {
      "name": "label",
      "type": {
        "kind": "primitive",
        "primitive": "str"
      },
      "description": "Human-readable label"
    },
    {
      "name": "confidence",
      "type": {
        "kind": "primitive",
        "primitive": "float"
      },
      "optional": true,
      "description": "Optional confidence score"
    }
  ],
  "metadata": {
    "author": "alice.bsky.social",
    "license": "MIT"
  },
  "createdAt": "2025-01-06T12:00:00Z"
}
```

### Design Questions

1. **Shape constraints**: Should we enforce shape constraints, or just document them?
   - Option A: Runtime validation against shape
   - Option B: Documentation only, actual shapes can vary
   - **Recommendation**: Documentation only initially, validation in future versions

2. **Custom types**: Should we support custom serialization hooks?
   - Current approach: Only primitive + NDArray
   - Future: Allow references to custom serialization functions?

3. **Schema inheritance**: Should schemas support inheritance/composition?
   - Could reference parent schema and add fields
   - **Defer to future version**

## 2. Dataset Record Lexicon

**NSID**: `app.bsky.atdata.dataset`

**Purpose**: Index record pointing to WebDataset files with associated metadata

### Proposed Structure

```json
{
  "lexicon": 1,
  "id": "app.bsky.atdata.dataset",
  "defs": {
    "main": {
      "type": "record",
      "description": "Index record for a WebDataset-backed dataset",
      "key": "tid",
      "record": {
        "type": "object",
        "required": ["name", "schemaRef", "urls", "createdAt"],
        "properties": {
          "name": {
            "type": "string",
            "description": "Human-readable dataset name",
            "maxLength": 200
          },
          "schemaRef": {
            "type": "string",
            "description": "AT-URI reference to the schema record for this dataset's samples"
          },
          "urls": {
            "type": "array",
            "description": "WebDataset URLs (supports brace notation)",
            "items": {
              "type": "string",
              "format": "uri",
              "maxLength": 1000
            },
            "minLength": 1
          },
          "description": {
            "type": "string",
            "description": "Human-readable description",
            "maxLength": 5000
          },
          "metadata": {
            "type": "bytes",
            "description": "Msgpack-encoded metadata dict",
            "maxLength": 100000
          },
          "tags": {
            "type": "array",
            "description": "Searchable tags",
            "items": {
              "type": "string",
              "maxLength": 50
            },
            "maxLength": 20
          },
          "size": {
            "type": "object",
            "description": "Dataset size information",
            "properties": {
              "samples": {
                "type": "integer",
                "description": "Total number of samples"
              },
              "bytes": {
                "type": "integer",
                "description": "Total size in bytes"
              }
            }
          },
          "license": {
            "type": "string",
            "description": "License (SPDX identifier preferred)",
            "maxLength": 100
          },
          "createdAt": {
            "type": "string",
            "format": "datetime"
          }
        }
      }
    }
  }
}
```

### Example Dataset Record

```json
{
  "$type": "app.bsky.atdata.dataset",
  "name": "CIFAR-10 Training Set",
  "schemaRef": "at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm",
  "urls": [
    "s3://my-bucket/cifar10-train-{000000..000049}.tar"
  ],
  "description": "CIFAR-10 training images (50,000 samples) stored as WebDataset shards",
  "metadata": "<msgpack bytes>",
  "tags": ["computer-vision", "classification", "cifar10"],
  "size": {
    "samples": 50000,
    "bytes": 178456789
  },
  "license": "MIT",
  "createdAt": "2025-01-06T12:00:00Z"
}
```

### Design Questions

1. **WebDataset storage**: Where are the actual `.tar` files?
   - Phase 1: External storage (S3, HTTP, etc.) - just store URLs
   - Future: Could use ATProto blob storage for smaller datasets
   - **Recommendation**: External only for now

2. **Metadata size limit**: What's reasonable for msgpack metadata?
   - Could store large metadata as separate blob
   - **Recommendation**: 100KB limit, use blob for larger

3. **Versioning**: Should we support dataset versioning?
   - Could link to previous version
   - **Defer to future version**

## 3. Lens Record Lexicon

**NSID**: `app.bsky.atdata.lens`

**Purpose**: Define bidirectional transformations between sample types

### Proposed Structure

```json
{
  "lexicon": 1,
  "id": "app.bsky.atdata.lens",
  "defs": {
    "main": {
      "type": "record",
      "description": "Bidirectional transformation between two sample types",
      "key": "tid",
      "record": {
        "type": "object",
        "required": ["name", "sourceSchema", "targetSchema", "createdAt"],
        "properties": {
          "name": {
            "type": "string",
            "description": "Human-readable lens name",
            "maxLength": 100
          },
          "sourceSchema": {
            "type": "string",
            "description": "AT-URI reference to source schema"
          },
          "targetSchema": {
            "type": "string",
            "description": "AT-URI reference to target schema"
          },
          "description": {
            "type": "string",
            "description": "What this transformation does",
            "maxLength": 1000
          },
          "getterCode": {
            "type": "ref",
            "ref": "#transformCode"
          },
          "putterCode": {
            "type": "ref",
            "ref": "#transformCode"
          },
          "metadata": {
            "type": "object",
            "description": "Arbitrary metadata"
          },
          "createdAt": {
            "type": "string",
            "format": "datetime"
          }
        }
      }
    },
    "transformCode": {
      "type": "union",
      "refs": [
        "#pythonCode",
        "#codeReference"
      ]
    },
    "pythonCode": {
      "type": "object",
      "required": ["kind", "source"],
      "properties": {
        "kind": {
          "type": "string",
          "const": "python"
        },
        "source": {
          "type": "string",
          "description": "Python function source code",
          "maxLength": 50000
        }
      }
    },
    "codeReference": {
      "type": "object",
      "required": ["kind", "repository", "path"],
      "properties": {
        "kind": {
          "type": "string",
          "const": "reference"
        },
        "repository": {
          "type": "string",
          "description": "Git repository URL",
          "maxLength": 500
        },
        "commit": {
          "type": "string",
          "description": "Git commit hash",
          "maxLength": 40
        },
        "path": {
          "type": "string",
          "description": "Path to function within repo",
          "maxLength": 500
        }
      }
    }
  }
}
```

### Example Lens Record

```json
{
  "$type": "app.bsky.atdata.lens",
  "name": "image_to_grayscale",
  "sourceSchema": "at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm",
  "targetSchema": "at://did:plc:def456/app.bsky.atdata.schema/7mn8op56pqr",
  "description": "Convert RGB images to grayscale",
  "getterCode": {
    "kind": "reference",
    "repository": "https://github.com/alice/lenses",
    "commit": "a1b2c3d4e5f6",
    "path": "lenses/vision.py:image_to_grayscale"
  },
  "putterCode": {
    "kind": "reference",
    "repository": "https://github.com/alice/lenses",
    "commit": "a1b2c3d4e5f6",
    "path": "lenses/vision.py:grayscale_to_image"
  },
  "metadata": {
    "author": "alice.bsky.social"
  },
  "createdAt": "2025-01-06T12:00:00Z"
}
```

### Design Questions - CRITICAL

1. **Code storage security**: Storing executable code is dangerous!
   - **Option A**: Code reference only (GitHub + commit hash) - safer
   - **Option B**: Allow inline code but require manual approval - flexible
   - **Option C**: AST/bytecode representation - complex
   - **Recommendation**: Start with references only (Option A), defer inline code

2. **Lens verification**: How to verify well-behavedness?
   - Could store test cases
   - Could require proof of GetPut/PutGet laws
   - **Defer to future**

3. **Lens composition**: Should lenses be composable?
   - Network could auto-compose transformations
   - **Defer to future**

## Schema Representation Format Decision

**Question**: What format should we use to represent field types internally?

### Option 1: JSON Schema
**Pros**:
- Standard, widely supported
- Validation tooling exists
- Human-readable

**Cons**:
- Not designed for codegen
- NDArray representation awkward
- Overly complex for our needs

### Option 2: Protobuf
**Pros**:
- Designed for codegen
- Compact binary format
- Cross-language support excellent

**Cons**:
- Not ATProto-native
- Requires compilation step
- Less human-readable

### Option 3: Custom Format (as shown above)
**Pros**:
- Tailored exactly to PackableSample needs
- Native ATProto Lexicon
- Clean NDArray representation
- Easy to extend

**Cons**:
- Need to write our own codegen
- Less ecosystem tooling

### Recommendation: Option 3 (Custom Format)

**Rationale**:
1. PackableSample has specific needs (NDArray, msgpack serialization)
2. ATProto Lexicon provides all the structure we need
3. Writing our own codegen gives us full control
4. Can still use JSON Schema for validation if needed

The proposed Lexicon structure above uses this approach.

## Implementation Checklist (Phase 1)

- [ ] Finalize Lexicon JSON definitions for all three record types
- [ ] Create reference documentation with examples
- [ ] Decide on schema representation format (recommendation: custom)
- [ ] Resolve open questions (code storage, versioning, etc.)
- [ ] Validate Lexicons against ATProto spec
- [ ] Create example records for testing

## Future Extensions

### Schema Evolution
- Support schema versioning with migration paths
- Compatibility checking (backward/forward compatible)

### Advanced Types
- Generic/parameterized types
- Union types for polymorphic samples
- Schema composition/inheritance

### Lens Network
- Automatic lens composition
- Lens verification and testing
- Performance metadata (transformation cost)

### Dataset Features
- Dataset splitting (train/val/test) references
- Dataset versioning and diffs
- Access control and permissions

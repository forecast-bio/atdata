# NDArray JSON Schema Shim Specification

**Issue**: #52
**Version**: 1.0
**Status**: Draft

## Problem Statement

We need a standard way to represent numpy NDArray types in JSON Schema that:
1. Works with existing atdata msgpack serialization (numpy `.npy` format)
2. Can be validated (where practical)
3. Can be used for code generation
4. Is compatible with JSON Schema tooling
5. Preserves dtype and shape information

## Current Serialization Format

atdata uses `_helpers.array_to_bytes()` which serializes arrays using numpy's native `.npy` format:

```python
def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()
```

**Result**: A bytes object containing:
- Magic bytes (`\x93NUMPY`)
- Version info
- Header with dtype and shape
- Array data

**Key insight**: The .npy format is self-describing - dtype and shape are already in the bytes!

## Design Approach

### Option 1: Pure Metadata (REJECTED)

Describe the semantic array only:
```json
{
  "type": "object",
  "x-atdata-ndarray": true,
  "x-dtype": "uint8",
  "x-shape": [null, null, 3]
}
```

**Problem**: Doesn't match actual msgpack structure (which stores bytes, not objects)

### Option 2: Bytes with Extension Properties (REJECTED)

Describe the bytes with metadata:
```json
{
  "type": "string",
  "format": "byte",
  "x-dtype": "uint8",
  "x-shape": [null, null, 3]
}
```

**Problem**:
- Non-standard use of extension properties
- JSON Schema doesn't know how to validate these
- Codegen tools won't understand x- properties

### Option 3: Reusable Schema Definition (RECOMMENDED)

Create a standard NDArray schema definition that can be $ref'd, with controlled vocabulary for metadata.

## Recommended Specification

### Base NDArray Schema Definition

This should be included in every JSON Schema that uses NDArray types:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$defs": {
    "ndarray": {
      "type": "string",
      "format": "byte",
      "description": "Numpy array serialized using numpy .npy format (includes dtype and shape in binary header)",
      "contentEncoding": "base64",
      "contentMediaType": "application/octet-stream"
    }
  }
}
```

### Using NDArray in Properties

Properties that are NDArray types reference the base definition and add metadata as **sibling properties**:

```json
{
  "properties": {
    "image": {
      "$ref": "#/$defs/ndarray",
      "description": "RGB image with variable height/width",
      "x-atdata-dtype": "uint8",
      "x-atdata-shape": [null, null, 3]
    }
  }
}
```

### Metadata Convention

**Extension properties** (prefixed with `x-atdata-`):
- `x-atdata-dtype`: Numpy dtype string (e.g., "uint8", "float32", "int64")
- `x-atdata-shape`: Array of integers and null (null = dynamic dimension)
- `x-atdata-notes`: Optional human-readable notes about the array

**Standard JSON Schema properties** (used normally):
- `description`: Human-readable description of what the array represents
- `title`: Short name for the field

## Complete Example

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ImageSample",
  "type": "object",
  "required": ["image", "label"],
  "properties": {
    "image": {
      "$ref": "#/$defs/ndarray",
      "description": "RGB image with variable height/width",
      "x-atdata-dtype": "uint8",
      "x-atdata-shape": [null, null, 3],
      "x-atdata-notes": "Images must have 3 color channels (RGB)"
    },
    "depth_map": {
      "$ref": "#/$defs/ndarray",
      "description": "Depth map corresponding to the image",
      "x-atdata-dtype": "float32",
      "x-atdata-shape": [null, null],
      "x-atdata-notes": "Same height and width as image, but single channel"
    },
    "label": {
      "type": "string",
      "description": "Human-readable label"
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
}
```

## Rationale

### Why `type: "string", format: "byte"`?

In msgpack serialization:
- The NDArray field is stored as raw bytes (the .npy format)
- When represented in JSON (for validation/transport), bytes become base64 strings
- JSON Schema's `type: "string", format: "byte"` is the standard way to represent binary data

### Why extension properties (`x-atdata-*`)?

JSON Schema allows custom properties starting with `x-`. Benefits:
1. **Standard**: Well-established convention in JSON Schema ecosystem
2. **Ignored by validators**: Won't cause validation errors
3. **Accessible to codegen**: Tools can parse these for type generation
4. **Self-documenting**: Clear what they mean

### Why not validate dtype/shape at JSON Schema level?

**Technical limitation**: JSON Schema can't validate binary .npy format internals.

**Solution**: Validation happens at **deserialization time**:
1. JSON Schema validates overall structure (field is present, is bytes)
2. When bytes are deserialized to NDArray, check dtype/shape match expectations

## Usage in atdata

### Publishing Schemas

When publishing a PackableSample with NDArray fields:

```python
@atdata.packable
class ImageSample:
    image: NDArray  # Will be annotated with dtype/shape hints
    label: str

# SDK extracts type hints and generates JSON Schema
schema_json = {
    "properties": {
        "image": {
            "$ref": "#/$defs/ndarray",
            "x-atdata-dtype": "uint8",  # From annotation or default
            "x-atdata-shape": [null, null, 3]  # From annotation or None
        }
    }
}
```

### Type Annotations for NDArray

Python type hints to specify dtype/shape:

```python
from typing import Annotated
from numpy.typing import NDArray

# Option 1: Generic NDArray (dtype/shape inferred or not specified)
image: NDArray

# Option 2: With dtype (using numpy typing)
image: NDArray[np.uint8]

# Option 3: With full metadata (using Annotated)
image: Annotated[NDArray[np.uint8], {"shape": [None, None, 3]}]
```

### Code Generation

Codegen reads JSON Schema and produces:

```python
@atdata.packable
class ImageSample:
    image: NDArray  # uint8, shape: [*, *, 3]
    label: str
```

Comment indicates dtype/shape from `x-atdata-*` properties.

## Validation Strategy

### JSON Schema Level (Structural)
✅ Validate field is present (if required)
✅ Validate field is bytes/string (in JSON)
✅ Validate base64 encoding (if in JSON)

### Deserialization Level (Semantic)
✅ Validate .npy format is valid
✅ Validate dtype matches expected (if specified)
✅ Validate shape matches expected (if specified)
✅ Validate shape constraints (e.g., must be 3D)

### Implementation

```python
from atdata.validation import validate_ndarray

def validate_ndarray(
    array: np.ndarray,
    expected_dtype: Optional[str] = None,
    expected_shape: Optional[List[Optional[int]]] = None
) -> tuple[bool, list[str]]:
    """Validate array against expectations."""
    errors = []

    # Check dtype
    if expected_dtype and str(array.dtype) != expected_dtype:
        errors.append(f"Expected dtype {expected_dtype}, got {array.dtype}")

    # Check shape
    if expected_shape:
        if len(array.shape) != len(expected_shape):
            errors.append(f"Expected {len(expected_shape)}D array, got {len(array.shape)}D")
        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
            if expected is not None and actual != expected:
                errors.append(f"Dimension {i}: expected {expected}, got {actual}")

    return len(errors) == 0, errors
```

## Standard NDArray Shim URI

The NDArray shim definition should be published at a canonical URI:

**Proposed**: `at://did:plc:foundation/ac.foundation.dataset.ndarray-shim/1.0.0`

This allows schemas to reference a standard definition:

```json
{
  "properties": {
    "image": {
      "$ref": "at://did:plc:foundation/ac.foundation.dataset.ndarray-shim/1.0.0#/$defs/ndarray",
      "x-atdata-dtype": "uint8"
    }
  }
}
```

Or schemas can inline the definition (recommended for Phase 1):

```json
{
  "$defs": {
    "ndarray": { /* inline definition */ }
  }
}
```

## Alternative: Describe Deserialized Structure

For reference, an alternative approach that describes the "unpacked" structure:

```json
{
  "$defs": {
    "ndarray": {
      "type": "object",
      "description": "Numpy array (deserialized representation)",
      "required": ["dtype", "shape", "data"],
      "properties": {
        "dtype": {"type": "string"},
        "shape": {"type": "array", "items": {"type": "integer"}},
        "data": {"type": "string", "format": "byte"}
      }
    }
  }
}
```

**Problem**: This doesn't match the actual msgpack structure (which is just bytes, not an object with dtype/shape/data fields). The .npy format is opaque bytes, not a structured object.

**Conclusion**: Stick with the recommended approach (bytes with metadata).

## Implementation Checklist

- [ ] Update sampleSchema Lexicon to reference this spec
- [ ] Create standard NDArray shim definition
- [ ] Update schema examples to use the shim correctly
- [ ] Implement validation helpers in Python SDK
- [ ] Add type annotation support for dtype/shape hints
- [ ] Update codegen to read x-atdata-* properties
- [ ] Document in user-facing docs

## Open Questions

1. **Should we support other array libraries?** (PyTorch tensors, JAX arrays, etc.)
   - Could use `x-atdata-array-type: "numpy"|"torch"|"jax"`
   - Recommendation: NumPy only for Phase 1

2. **Should shape constraints be enforced at runtime?**
   - Pro: Catch errors early
   - Con: Performance overhead, flexibility lost
   - Recommendation: Optional validation, off by default

3. **Should we support sparse arrays?**
   - scipy.sparse has different serialization format
   - Recommendation: Defer to future

4. **What about array of arrays?** (ragged arrays)
   - Can be represented as Python lists of NDArrays
   - Recommendation: Not a priority for Phase 1

## Summary

**Recommended Approach**:
- NDArray fields represented as `{"$ref": "#/$defs/ndarray"}` (bytes)
- Dtype and shape specified via `x-atdata-dtype` and `x-atdata-shape`
- Standard `ndarray` definition inlined in every schema
- Validation happens at deserialization, not JSON Schema level
- Codegen reads extension properties to generate proper types

**Benefits**:
- ✅ Compatible with existing msgpack serialization
- ✅ Works with JSON Schema tooling
- ✅ Clear metadata for codegen
- ✅ Flexible (dtype/shape optional)
- ✅ Extensible (can add more x-atdata-* properties)

**Trade-offs**:
- ⚠️ Leaky abstraction (JSON Schema describes bytes, not semantic array)
- ⚠️ Validation split across two layers
- ⚠️ Extension properties not universally understood

**Grade**: B+ (Good practical solution)

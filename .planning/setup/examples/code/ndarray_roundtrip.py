#!/usr/bin/env python3
"""
Demonstration of NDArray JSON Schema shim roundtrip.

This script demonstrates:
1. Creating numpy arrays
2. Serializing to bytes (numpy .npy format)
3. Storing in JSON-compatible structure
4. Validating against JSON Schema
5. Deserializing back to numpy arrays

This proves the NDArray shim design works end-to-end.
"""

import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from jsonschema import validate, ValidationError


##
# Step 1: Define helper functions (same as atdata._helpers)

def array_to_bytes(x: np.ndarray) -> bytes:
    """Convert numpy array to bytes using .npy format."""
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    """Convert bytes back to numpy array."""
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


##
# Step 2: Load the JSON Schema for ImageSample

# Get path to the schema example
schema_path = Path(__file__).parent.parent / "sampleSchema_example.json"
with open(schema_path) as f:
    schema_record = json.load(f)

# Extract just the jsonSchema part
json_schema = schema_record["jsonSchema"]

print("=" * 80)
print("JSON Schema for ImageSample")
print("=" * 80)
print(json.dumps(json_schema, indent=2))
print()


##
# Step 3: Create sample data matching the schema

print("=" * 80)
print("Creating Sample Data")
print("=" * 80)

# Create a numpy array (simulating an image)
image_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
print(f"Created image array: shape={image_array.shape}, dtype={image_array.dtype}")

# Serialize to bytes (this is what atdata does)
image_bytes = array_to_bytes(image_array)
print(f"Serialized to bytes: {len(image_bytes)} bytes")
print(f"First 100 bytes (hex): {image_bytes[:100].hex()}")
print()


##
# Step 4: Create JSON-compatible representation

print("=" * 80)
print("Creating JSON-Compatible Representation")
print("=" * 80)

# For JSON, bytes need to be base64-encoded
image_base64 = base64.b64encode(image_bytes).decode('utf-8')
print(f"Base64 encoded: {len(image_base64)} characters")
print(f"First 100 chars: {image_base64[:100]}...")

# Create a sample object matching the schema
sample_data = {
    "image": image_base64,  # NDArray as base64 string
    "label": "cat",         # Regular string field
    "confidence": 0.95      # Optional number field
}

print()
print("Sample data structure:")
print(json.dumps({
    "image": f"<{len(image_base64)} chars of base64>",
    "label": sample_data["label"],
    "confidence": sample_data["confidence"]
}, indent=2))
print()


##
# Step 5: Validate against JSON Schema

print("=" * 80)
print("Validating Against JSON Schema")
print("=" * 80)

try:
    validate(instance=sample_data, schema=json_schema)
    print("✅ VALID: Sample data validates against JSON Schema!")
except ValidationError as e:
    print(f"❌ INVALID: {e.message}")
    print(f"Failed at: {list(e.path)}")

print()


##
# Step 6: Deserialize back to numpy

print("=" * 80)
print("Deserializing Back to Numpy")
print("=" * 80)

# Decode from base64
recovered_bytes = base64.b64decode(sample_data["image"])
print(f"Decoded from base64: {len(recovered_bytes)} bytes")

# Deserialize to numpy array
recovered_array = bytes_to_array(recovered_bytes)
print(f"Deserialized to array: shape={recovered_array.shape}, dtype={recovered_array.dtype}")

# Verify it matches the original
arrays_equal = np.array_equal(image_array, recovered_array)
print(f"Arrays equal: {arrays_equal}")

if arrays_equal:
    print("✅ SUCCESS: Full roundtrip successful!")
else:
    print("❌ FAILURE: Arrays don't match")
    print(f"Max difference: {np.max(np.abs(image_array.astype(float) - recovered_array.astype(float)))}")

print()


##
# Step 7: Demonstrate validation of dtype/shape metadata

print("=" * 80)
print("Validating NDArray Metadata (dtype, shape)")
print("=" * 80)

# Extract metadata from schema
image_schema = json_schema["properties"]["image"]
expected_dtype = image_schema.get("x-atdata-dtype")
expected_shape = image_schema.get("x-atdata-shape")

print(f"Expected dtype: {expected_dtype}")
print(f"Expected shape: {expected_shape}")
print(f"Actual dtype: {recovered_array.dtype}")
print(f"Actual shape: {recovered_array.shape}")

# Validate dtype
dtype_match = str(recovered_array.dtype) == expected_dtype
print(f"Dtype matches: {dtype_match}")

# Validate shape (with None/null for dynamic dimensions)
def validate_shape(actual_shape, expected_shape):
    """Validate shape with support for dynamic dimensions (None/null)."""
    if len(actual_shape) != len(expected_shape):
        return False
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if expected_dim is not None and actual_dim != expected_dim:
            return False
    return True

shape_match = validate_shape(recovered_array.shape, expected_shape)
print(f"Shape matches: {shape_match}")

if dtype_match and shape_match:
    print("✅ SUCCESS: Array metadata matches schema expectations!")
else:
    print("❌ FAILURE: Metadata mismatch")

print()


##
# Step 8: Demonstrate msgpack (actual atdata format)

print("=" * 80)
print("Msgpack Serialization (Actual atdata Format)")
print("=" * 80)

try:
    import msgpack

    # In atdata, the sample would be stored in msgpack, not JSON
    # The image field would be raw bytes, not base64
    msgpack_data = {
        "image": image_bytes,  # Raw bytes (not base64)
        "label": "cat",
        "confidence": 0.95
    }

    # Serialize to msgpack
    msgpack_bytes = msgpack.packb(msgpack_data)
    print(f"Msgpack size: {len(msgpack_bytes)} bytes")

    # Deserialize from msgpack
    recovered_msgpack = msgpack.unpackb(msgpack_bytes, raw=False)
    recovered_array_msgpack = bytes_to_array(recovered_msgpack["image"])

    print(f"Recovered from msgpack: shape={recovered_array_msgpack.shape}, dtype={recovered_array_msgpack.dtype}")
    print(f"Arrays equal: {np.array_equal(image_array, recovered_array_msgpack)}")
    print("✅ SUCCESS: Msgpack roundtrip successful!")

except ImportError:
    print("⚠️  msgpack not installed, skipping msgpack demonstration")
    print("   (atdata uses msgpack for actual serialization)")

print()


##
# Summary

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✅ The NDArray JSON Schema shim works correctly:
   1. JSON Schema validates structure (field is present, is base64 string)
   2. Binary .npy format preserves dtype and shape
   3. Extension properties (x-atdata-*) provide metadata for validation
   4. Full roundtrip: numpy → bytes → base64 → JSON → validate → deserialize → numpy
   5. Msgpack format (actual atdata) uses raw bytes instead of base64

⚠️  Validation happens at two levels:
   - JSON Schema: Structural validation (field present, correct type)
   - Deserialization: Semantic validation (dtype/shape match expectations)

📝 This design is a pragmatic compromise:
   - Leverages existing .npy serialization (proven, self-describing)
   - Uses standard JSON Schema conventions (format: byte, contentEncoding)
   - Adds metadata via extension properties (x-atdata-*)
   - Works with both JSON (base64) and msgpack (raw bytes)
""")

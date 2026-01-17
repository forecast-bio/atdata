# Packable Samples

Packable samples are typed dataclasses that can be serialized with msgpack for storage in WebDataset tar files.

## The `@packable` Decorator

The recommended way to define a sample type is with the `@packable` decorator:

```python
import numpy as np
from numpy.typing import NDArray
import atdata

@atdata.packable
class ImageSample:
    image: NDArray
    label: str
    confidence: float
```

This creates a dataclass that:
- Inherits from `PackableSample`
- Has automatic msgpack serialization
- Handles NDArray conversion to/from bytes

## Supported Field Types

### Primitives

```python
@atdata.packable
class PrimitiveSample:
    name: str
    count: int
    score: float
    active: bool
    data: bytes
```

### NumPy Arrays

Fields annotated as `NDArray` are automatically converted:

```python
@atdata.packable
class ArraySample:
    features: NDArray          # Required array
    embeddings: NDArray | None  # Optional array
```

**Note**: Bytes in NDArray-typed fields are always interpreted as serialized arrays. Don't use `NDArray` for raw binary data.

### Lists

```python
@atdata.packable
class ListSample:
    tags: list[str]
    scores: list[float]
```

## Serialization

### Packing to Bytes

```python
sample = ImageSample(
    image=np.random.rand(224, 224, 3).astype(np.float32),
    label="cat",
    confidence=0.95,
)

# Serialize to msgpack bytes
packed_bytes = sample.packed
print(f"Size: {len(packed_bytes)} bytes")
```

### Unpacking from Bytes

```python
# Deserialize from bytes
restored = ImageSample.from_bytes(packed_bytes)

# Arrays are automatically restored
assert np.array_equal(sample.image, restored.image)
assert sample.label == restored.label
```

### WebDataset Format

The `as_wds` property returns a dict ready for WebDataset:

```python
wds_dict = sample.as_wds
# {'__key__': '1234...', 'msgpack': b'...'}
```

Write samples to a tar file:

```python
import webdataset as wds

with wds.writer.TarWriter("data-000000.tar") as sink:
    for i, sample in enumerate(samples):
        # Use custom key or let as_wds generate one
        sink.write({**sample.as_wds, "__key__": f"sample_{i:06d}"})
```

## Direct Inheritance (Alternative)

You can also inherit directly from `PackableSample`:

```python
from dataclasses import dataclass

@dataclass
class DirectSample(atdata.PackableSample):
    name: str
    values: NDArray
```

This is equivalent to using `@packable` but more verbose.

## How It Works

### Serialization Flow

1. **Packing** (`sample.packed`):
   - NDArray fields → converted to bytes via `array_to_bytes()`
   - Other fields → passed through
   - All fields → packed with msgpack

2. **Unpacking** (`Sample.from_bytes()`):
   - Bytes → unpacked with ormsgpack
   - Dict → passed to `__init__`
   - `__post_init__` → calls `_ensure_good()`
   - NDArray fields → bytes converted back to arrays

### The `_ensure_good()` Method

This method runs automatically after construction and handles NDArray conversion:

```python
def _ensure_good(self):
    for field in dataclasses.fields(self):
        if _is_possibly_ndarray_type(field.type):
            value = getattr(self, field.name)
            if isinstance(value, bytes):
                setattr(self, field.name, bytes_to_array(value))
```

## Best Practices

### Do

```python
@atdata.packable
class GoodSample:
    features: NDArray           # Clear type annotation
    label: str                  # Simple primitives
    metadata: dict              # Msgpack-compatible dicts
    scores: list[float]         # Typed lists
```

### Don't

```python
@atdata.packable
class BadSample:
    # DON'T: Nested dataclasses not supported
    nested: OtherSample

    # DON'T: Complex objects that aren't msgpack-serializable
    callback: Callable

    # DON'T: Use NDArray for raw bytes
    raw_data: NDArray  # Use 'bytes' type instead
```

## Related

- [Datasets](datasets.md) - Loading and iterating samples
- [Lenses](lenses.md) - Transforming between sample types

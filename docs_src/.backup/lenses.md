# Lenses

Lenses provide bidirectional transformations between sample types, enabling datasets to be viewed through different schemas without duplicating data.

## Overview

A lens consists of:
- **Getter**: Transforms source type `S` to view type `V`
- **Putter**: Updates source based on a modified view (optional)

## Creating a Lens

Use the `@lens` decorator to define a getter:

```python
import atdata
from numpy.typing import NDArray

@atdata.packable
class FullSample:
    image: NDArray
    label: str
    confidence: float
    metadata: dict

@atdata.packable
class SimpleSample:
    label: str
    confidence: float

@atdata.lens
def simplify(src: FullSample) -> SimpleSample:
    return SimpleSample(label=src.label, confidence=src.confidence)
```

The decorator:
1. Creates a `Lens` object from the getter function
2. Registers it in the global `LensNetwork` registry
3. Extracts source/view types from annotations

## Adding a Putter

To enable bidirectional updates, add a putter:

```python
@simplify.putter
def simplify_put(view: SimpleSample, source: FullSample) -> FullSample:
    return FullSample(
        image=source.image,
        label=view.label,
        confidence=view.confidence,
        metadata=source.metadata,
    )
```

The putter receives:
- `view`: The modified view value
- `source`: The original source value

It returns an updated source that reflects changes from the view.

## Using Lenses with Datasets

Lenses integrate with `Dataset.as_type()`:

```python
dataset = atdata.Dataset[FullSample]("data-{000000..000009}.tar")

# View through a different type
simple_ds = dataset.as_type(SimpleSample)

for batch in simple_ds.ordered(batch_size=32):
    # Only SimpleSample fields available
    labels = batch.label
    scores = batch.confidence
```

## Direct Lens Usage

Lenses can also be called directly:

```python
full = FullSample(
    image=np.zeros((224, 224, 3)),
    label="cat",
    confidence=0.95,
    metadata={"source": "training"}
)

# Apply getter
simple = simplify(full)
# Or: simple = simplify.get(full)

# Apply putter
modified_simple = SimpleSample(label="dog", confidence=0.87)
updated_full = simplify.put(modified_simple, full)
# updated_full has label="dog", confidence=0.87, but retains
# original image and metadata
```

## Lens Laws

Well-behaved lenses should satisfy these properties:

### GetPut Law
If you get a view and immediately put it back, the source is unchanged:
```python
view = lens.get(source)
assert lens.put(view, source) == source
```

### PutGet Law
If you put a view, getting it back yields that view:
```python
updated = lens.put(view, source)
assert lens.get(updated) == view
```

### PutPut Law
Putting twice is equivalent to putting once with the final value:
```python
result1 = lens.put(v2, lens.put(v1, source))
result2 = lens.put(v2, source)
assert result1 == result2
```

## Trivial Putter

If no putter is defined, a trivial putter is used that ignores view updates:

```python
@atdata.lens
def extract_label(src: FullSample) -> SimpleSample:
    return SimpleSample(label=src.label, confidence=src.confidence)

# Without a putter, put() returns the original source unchanged
view = SimpleSample(label="modified", confidence=0.5)
updated = extract_label.put(view, original)
assert updated == original  # No changes applied
```

## LensNetwork Registry

The `LensNetwork` is a singleton that stores all registered lenses:

```python
from atdata.lens import LensNetwork

network = LensNetwork()

# Look up a specific lens
lens = network.transform(FullSample, SimpleSample)

# Raises ValueError if no lens exists
try:
    lens = network.transform(TypeA, TypeB)
except ValueError:
    print("No lens registered for TypeA -> TypeB")
```

## Example: Feature Extraction

```python
@atdata.packable
class RawSample:
    audio: NDArray
    text: str
    speaker_id: int

@atdata.packable
class TextFeatures:
    text: str
    word_count: int

@atdata.lens
def extract_text(src: RawSample) -> TextFeatures:
    return TextFeatures(
        text=src.text,
        word_count=len(src.text.split())
    )

@extract_text.putter
def extract_text_put(view: TextFeatures, source: RawSample) -> RawSample:
    return RawSample(
        audio=source.audio,
        text=view.text,
        speaker_id=source.speaker_id
    )
```

## Related

- [Datasets](datasets.md) - Using lenses with Dataset.as_type()
- [Packable Samples](packable-samples.md) - Defining sample types
- [Atmosphere](atmosphere.md) - Publishing lenses to ATProto federation

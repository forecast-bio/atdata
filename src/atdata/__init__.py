"""A loose federation of distributed, typed datasets.

``atdata`` provides a typed dataset abstraction built on WebDataset, with support
for:

- **Typed samples** with automatic msgpack serialization
- **NDArray handling** with transparent bytes conversion
- **Lens transformations** for viewing datasets through different type schemas
- **Batch aggregation** with automatic numpy array stacking
- **WebDataset integration** for efficient large-scale dataset storage

Quick Start:
    >>> import atdata
    >>> import numpy as np
    >>>
    >>> @atdata.packable
    ... class MyData:
    ...     features: np.ndarray
    ...     label: str
    >>>
    >>> # Create dataset from WebDataset tar files
    >>> ds = atdata.Dataset[MyData]("path/to/data-{000000..000009}.tar")
    >>>
    >>> # Iterate with automatic batching
    >>> for batch in ds.shuffled(batch_size=32):
    ...     features = batch.features  # numpy array (32, ...)
    ...     labels = batch.label  # list of 32 strings

Main Components:
    - ``PackableSample``: Base class for msgpack-serializable samples
    - ``Dataset``: Typed dataset wrapper for WebDataset
    - ``SampleBatch``: Automatic batch aggregation
    - ``Lens``: Bidirectional type transformations
    - ``@packable``: Decorator for creating PackableSample classes
    - ``@lens``: Decorator for creating lens transformations
"""

##
# Expose components

from .dataset import (
    PackableSample,
    SampleBatch,
    Dataset,
    packable,
)

from .lens import (
    Lens,
    LensNetwork,
    lens,
)

# ATProto integration (lazy import to avoid requiring atproto package)
from . import atmosphere


#
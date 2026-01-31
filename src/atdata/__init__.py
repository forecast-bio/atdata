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
    DictSample as DictSample,
    PackableSample as PackableSample,
    SampleBatch as SampleBatch,
    Dataset as Dataset,
    packable as packable,
)

from .lens import (
    Lens as Lens,
    LensNetwork as LensNetwork,
    lens as lens,
)

from ._hf_api import (
    load_dataset as load_dataset,
    DatasetDict as DatasetDict,
    get_default_index as get_default_index,
    set_default_index as set_default_index,
)

from ._protocols import (
    Packable as Packable,
    IndexEntry as IndexEntry,
    AbstractIndex as AbstractIndex,
    AbstractDataStore as AbstractDataStore,
    DataSource as DataSource,
)

from ._sources import (
    URLSource as URLSource,
    S3Source as S3Source,
    BlobSource as BlobSource,
)

from ._exceptions import (
    AtdataError as AtdataError,
    LensNotFoundError as LensNotFoundError,
    SchemaError as SchemaError,
    SampleKeyError as SampleKeyError,
    ShardError as ShardError,
    PartialFailureError as PartialFailureError,
)

from ._schema_codec import (
    schema_to_type as schema_to_type,
)

from ._logging import (
    configure_logging as configure_logging,
    get_logger as get_logger,
)

from .repository import (
    Repository as Repository,
    create_repository as create_repository,
)

from ._cid import (
    generate_cid as generate_cid,
    verify_cid as verify_cid,
)

from .promote import (
    promote_to_atmosphere as promote_to_atmosphere,
)

from .manifest import (
    ManifestField as ManifestField,
    ManifestBuilder as ManifestBuilder,
    ShardManifest as ShardManifest,
    ManifestWriter as ManifestWriter,
    QueryExecutor as QueryExecutor,
    SampleLocation as SampleLocation,
)

# ATProto integration (lazy import to avoid requiring atproto package)
from . import atmosphere as atmosphere

# CLI entry point
from .cli import main as main

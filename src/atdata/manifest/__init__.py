"""Per-shard manifest and query system.

Provides manifest generation during shard writes and efficient
query execution over large datasets without full scans.

Components:

- ``ManifestField``: Annotation marker for manifest-included fields
- ``ManifestBuilder``: Accumulates sample metadata during writes
- ``ShardManifest``: Loaded manifest representation
- ``ManifestWriter``: Serializes manifests to JSON + parquet
- ``QueryExecutor``: Two-phase query over manifest metadata
- ``SampleLocation``: Address of a sample within a shard
"""

from ._fields import ManifestField as ManifestField
from ._fields import AggregateKind as AggregateKind
from ._fields import resolve_manifest_fields as resolve_manifest_fields
from ._aggregates import CategoricalAggregate as CategoricalAggregate
from ._aggregates import NumericAggregate as NumericAggregate
from ._aggregates import SetAggregate as SetAggregate
from ._aggregates import create_aggregate as create_aggregate
from ._builder import ManifestBuilder as ManifestBuilder
from ._manifest import ShardManifest as ShardManifest
from ._manifest import MANIFEST_FORMAT_VERSION as MANIFEST_FORMAT_VERSION
from ._writer import ManifestWriter as ManifestWriter
from ._query import QueryExecutor as QueryExecutor
from ._query import SampleLocation as SampleLocation

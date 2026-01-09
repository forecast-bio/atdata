"""ATProto integration for distributed dataset federation.

This module provides ATProto publishing and discovery capabilities for atdata,
enabling a loose federation of distributed, typed datasets on the AT Protocol
network.

Key components:

- ``AtmosphereClient``: Authentication and session management for ATProto
- ``SchemaPublisher``: Publish PackableSample schemas as ATProto records
- ``DatasetPublisher``: Publish dataset index records with WebDataset URLs
- ``LensPublisher``: Publish lens transformation records

The ATProto integration is additive - existing atdata functionality continues
to work unchanged. These features are opt-in for users who want to publish
or discover datasets on the ATProto network.

Example:
    >>> from atdata.atmosphere import AtmosphereClient, SchemaPublisher
    >>>
    >>> client = AtmosphereClient()
    >>> client.login("handle.bsky.social", "app-password")
    >>>
    >>> publisher = SchemaPublisher(client)
    >>> schema_uri = publisher.publish(MySampleType, version="1.0.0")

Note:
    This module requires the ``atproto`` package to be installed::

        pip install atproto
"""

from .client import AtmosphereClient
from .schema import SchemaPublisher, SchemaLoader
from .records import DatasetPublisher, DatasetLoader
from .lens import LensPublisher, LensLoader
from ._types import (
    AtUri,
    SchemaRecord,
    DatasetRecord,
    LensRecord,
)

__all__ = [
    # Client
    "AtmosphereClient",
    # Schema operations
    "SchemaPublisher",
    "SchemaLoader",
    # Dataset operations
    "DatasetPublisher",
    "DatasetLoader",
    # Lens operations
    "LensPublisher",
    "LensLoader",
    # Types
    "AtUri",
    "SchemaRecord",
    "DatasetRecord",
    "LensRecord",
]

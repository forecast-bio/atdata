"""Coverage tests for atdata.repository._AtmosphereBackend.

Uses mocked AtmosphereClient and patched atmosphere sub-modules to exercise
the backend without requiring a live ATProto connection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from atdata.atmosphere.client import AtmosphereClient
from atdata.repository import _AtmosphereBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_atmo_client():
    """Return a MagicMock that passes isinstance(x, AtmosphereClient)."""
    return MagicMock(spec=AtmosphereClient)


@pytest.fixture
def backend(mock_atmo_client):
    """Return an _AtmosphereBackend wired to the mock client."""
    return _AtmosphereBackend(mock_atmo_client)


@pytest.fixture
def backend_with_store(mock_atmo_client):
    """Return an _AtmosphereBackend with a mock data_store."""
    store = MagicMock()
    return _AtmosphereBackend(mock_atmo_client, data_store=store)


# ---------------------------------------------------------------------------
# Constructor / properties
# ---------------------------------------------------------------------------


def test_atmosphere_backend_constructor_validates_type() -> None:
    """Passing a non-AtmosphereClient object raises TypeError."""
    with pytest.raises(TypeError, match="Expected AtmosphereClient"):
        _AtmosphereBackend("not-a-client")


def test_atmosphere_backend_data_store_property(backend_with_store) -> None:
    """data_store property returns the store passed at construction."""
    assert backend_with_store.data_store is not None


def test_atmosphere_backend_data_store_none(backend) -> None:
    """data_store defaults to None."""
    assert backend.data_store is None


# ---------------------------------------------------------------------------
# Lazy initialisation
# ---------------------------------------------------------------------------


def test_ensure_loaders_lazy_init(backend) -> None:
    """Loaders/publishers are None until _ensure_loaders is called."""
    assert backend._schema_loader is None
    assert backend._dataset_loader is None

    with (
        patch("atdata.atmosphere.schema.SchemaPublisher") as MockSP,
        patch("atdata.atmosphere.schema.SchemaLoader") as MockSL,
        patch("atdata.atmosphere.records.DatasetPublisher") as MockDP,
        patch("atdata.atmosphere.records.DatasetLoader") as MockDL,
    ):
        backend._ensure_loaders()

        MockSP.assert_called_once_with(backend.client)
        MockSL.assert_called_once_with(backend.client)
        MockDP.assert_called_once_with(backend.client)
        MockDL.assert_called_once_with(backend.client)

    # Second call is a no-op (already initialised)
    backend._ensure_loaders()


# ---------------------------------------------------------------------------
# Dataset operations
# ---------------------------------------------------------------------------


def _patch_loaders(backend):
    """Patch _ensure_loaders to inject mocks directly."""
    backend._schema_publisher = MagicMock()
    backend._schema_loader = MagicMock()
    backend._dataset_publisher = MagicMock()
    backend._dataset_loader = MagicMock()


def test_get_dataset(backend) -> None:
    """get_dataset delegates to dataset_loader.get and wraps result."""
    _patch_loaders(backend)
    backend._dataset_loader.get.return_value = {"name": "ds1", "schemaRef": "at://..."}

    entry = backend.get_dataset("my-ds")

    backend._dataset_loader.get.assert_called_once_with("my-ds")
    assert entry.name == "ds1"


def test_list_datasets(backend) -> None:
    """list_datasets returns a list of AtmosphereIndexEntry."""
    _patch_loaders(backend)
    backend._dataset_loader.list_all.return_value = [
        {"uri": "at://did/collection/1", "value": {"name": "a"}},
        {"uri": "at://did/collection/2", "value": {"name": "b"}},
    ]

    entries = backend.list_datasets(repo="did:plc:test")

    backend._dataset_loader.list_all.assert_called_once_with(repo="did:plc:test")
    assert len(entries) == 2
    assert entries[0].name == "a"
    assert entries[1].name == "b"


def test_iter_datasets(backend) -> None:
    """iter_datasets yields AtmosphereIndexEntry lazily."""
    _patch_loaders(backend)
    backend._dataset_loader.list_all.return_value = [
        {"uri": "at://x/y/1", "value": {"name": "first"}},
        {"uri": "at://x/y/2", "value": {"name": "second"}},
    ]

    entries = list(backend.iter_datasets(repo="did:plc:iter"))

    assert len(entries) == 2
    assert entries[0].uri == "at://x/y/1"
    assert entries[1].name == "second"


def test_insert_dataset(backend) -> None:
    """insert_dataset calls publisher.publish then loader.get."""
    _patch_loaders(backend)
    backend._dataset_publisher.publish.return_value = "at://did/col/new"
    backend._dataset_loader.get.return_value = {"name": "new-ds"}

    ds_mock = MagicMock()
    entry = backend.insert_dataset(
        ds_mock,
        name="new-ds",
        schema_ref="at://did/schema/1",
        description="test dataset",
        tags=["a"],
        license="MIT",
    )

    backend._dataset_publisher.publish.assert_called_once_with(
        ds_mock,
        name="new-ds",
        schema_uri="at://did/schema/1",
        description="test dataset",
        tags=["a"],
        license="MIT",
        auto_publish_schema=False,
    )
    backend._dataset_loader.get.assert_called_once_with("at://did/col/new")
    assert entry.name == "new-ds"


def test_insert_dataset_auto_schema(backend) -> None:
    """insert_dataset with schema_ref=None sets auto_publish_schema=True."""
    _patch_loaders(backend)
    backend._dataset_publisher.publish.return_value = "at://did/col/auto"
    backend._dataset_loader.get.return_value = {"name": "auto-ds"}

    backend.insert_dataset(MagicMock(), name="auto-ds")

    call_kwargs = backend._dataset_publisher.publish.call_args[1]
    assert call_kwargs["auto_publish_schema"] is True


# ---------------------------------------------------------------------------
# Schema operations
# ---------------------------------------------------------------------------


def test_publish_schema(backend) -> None:
    """publish_schema delegates to schema_publisher.publish."""
    _patch_loaders(backend)
    backend._schema_publisher.publish.return_value = "at://did/schema/v1"

    uri = backend.publish_schema(
        MagicMock(),
        version="2.0.0",
        description="desc",
        metadata={"k": "v"},
    )

    backend._schema_publisher.publish.assert_called_once_with(
        backend._schema_publisher.publish.call_args[0][0],
        version="2.0.0",
        description="desc",
        metadata={"k": "v"},
    )
    assert uri == "at://did/schema/v1"


def test_get_schema(backend) -> None:
    """get_schema returns the dict from schema_loader.get."""
    _patch_loaders(backend)
    backend._schema_loader.get.return_value = {"type": "object", "fields": []}

    result = backend.get_schema("at://did/schema/abc")

    backend._schema_loader.get.assert_called_once_with("at://did/schema/abc")
    assert result == {"type": "object", "fields": []}


def test_list_schemas(backend) -> None:
    """list_schemas extracts 'value' from each record."""
    _patch_loaders(backend)
    backend._schema_loader.list_all.return_value = [
        {"uri": "at://a", "value": {"name": "s1"}},
        {"uri": "at://b", "value": {"name": "s2"}},
    ]

    schemas = backend.list_schemas(repo="did:plc:schemas")

    backend._schema_loader.list_all.assert_called_once_with(repo="did:plc:schemas")
    assert len(schemas) == 2
    assert schemas[0] == {"name": "s1"}


def test_iter_schemas(backend) -> None:
    """iter_schemas yields schema value dicts."""
    _patch_loaders(backend)
    backend._schema_loader.list_all.return_value = [
        {"value": {"name": "schema_a"}},
        {"value": {"name": "schema_b"}},
    ]

    schemas = list(backend.iter_schemas())

    assert len(schemas) == 2
    assert schemas[0] == {"name": "schema_a"}
    assert schemas[1] == {"name": "schema_b"}


def test_decode_schema(backend) -> None:
    """decode_schema calls get_schema then schema_to_type."""
    _patch_loaders(backend)
    backend._schema_loader.get.return_value = {"name": "Decoded", "fields": []}

    fake_type = type("Decoded", (), {})

    with patch(
        "atdata._schema_codec.schema_to_type", return_value=fake_type
    ) as mock_s2t:
        result = backend.decode_schema("at://did/schema/decode")

    mock_s2t.assert_called_once_with({"name": "Decoded", "fields": []})
    assert result is fake_type

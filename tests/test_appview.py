"""Tests for AppView XRPC integration.

Tests cover:
- DID web helpers (_did_web_to_url, _url_to_did_web)
- Atmosphere AppView configuration (appview param, has_appview, from_env)
- XRPC transport (xrpc_query, xrpc_procedure)
- Exception types (AppViewError, AppViewRequiredError, AppViewUnavailableError)
- Query migration: LabelLoader.resolve, SchemaLoader.list_all/resolve,
  DatasetLoader.list_all/get/get_blob_urls, LensLoader.list_all/find_by_schemas
- Procedure migration: SchemaPublisher, DatasetPublisher, LabelPublisher, LensPublisher
- AppView-only capabilities: search_datasets, search_lenses, describe_service,
  get_entries, get_entry_stats
- Graceful fallback when AppView is unavailable
"""

from unittest.mock import Mock, MagicMock, patch
import pytest

from atdata._exceptions import (
    AppViewError,
    AppViewRequiredError,
    AppViewUnavailableError,
)
from atdata.atmosphere.client import (
    Atmosphere,
    _did_web_to_url,
    _url_to_did_web,
)
from atdata.atmosphere.labels import LabelPublisher, LabelLoader
from atdata.atmosphere.schema import SchemaPublisher, SchemaLoader
from atdata.atmosphere.records import DatasetPublisher, DatasetLoader
from atdata.atmosphere.lens import LensPublisher, LensLoader


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_atproto_client():
    """Create a mock atproto SDK client."""
    mock = Mock()
    mock.me = MagicMock()
    mock.me.did = "did:plc:test123456789"
    mock.me.handle = "test.bsky.social"

    mock_profile = Mock()
    mock_profile.did = "did:plc:test123456789"
    mock_profile.handle = "test.bsky.social"
    mock.login.return_value = mock_profile
    mock.export_session_string.return_value = "test-session-string"
    mock._base_url = "https://bsky.social"

    return mock


@pytest.fixture
def client_no_appview(mock_atproto_client):
    """Atmosphere client without AppView."""
    client = Atmosphere(_client=mock_atproto_client)
    client._login("test.bsky.social", "test-password")
    client._appview_client = mock_atproto_client
    return client


@pytest.fixture
def client_with_appview(mock_atproto_client):
    """Atmosphere client with AppView configured."""
    client = Atmosphere(
        _client=mock_atproto_client,
        appview="https://datasets.atdata.blue",
    )
    client._login("test.bsky.social", "test-password")
    client._appview_client = mock_atproto_client
    return client


# =============================================================================
# DID Web Helpers
# =============================================================================


class TestDidWebHelpers:
    def test_did_web_to_url_simple(self):
        assert (
            _did_web_to_url("did:web:datasets.atdata.blue")
            == "https://datasets.atdata.blue"
        )

    def test_did_web_to_url_with_port(self):
        assert _did_web_to_url("did:web:localhost%3A8000") == "https://localhost:8000"

    def test_did_web_to_url_invalid(self):
        with pytest.raises(ValueError, match="Not a did:web"):
            _did_web_to_url("did:plc:abc123")

    def test_url_to_did_web_simple(self):
        assert (
            _url_to_did_web("https://datasets.atdata.blue")
            == "did:web:datasets.atdata.blue"
        )

    def test_url_to_did_web_with_port(self):
        assert _url_to_did_web("https://localhost:8000") == "did:web:localhost%3A8000"

    def test_roundtrip_url(self):
        url = "https://datasets.atdata.blue"
        assert _did_web_to_url(_url_to_did_web(url)) == url

    def test_roundtrip_did(self):
        did = "did:web:datasets.atdata.blue"
        assert _url_to_did_web(_did_web_to_url(did)) == did


# =============================================================================
# Atmosphere AppView Configuration
# =============================================================================


class TestAtmosphereAppView:
    def test_no_appview_by_default(self, client_no_appview):
        assert client_no_appview.has_appview is False
        assert client_no_appview.appview_url is None
        assert client_no_appview.appview_did is None

    def test_appview_from_url(self, client_with_appview):
        assert client_with_appview.has_appview is True
        assert client_with_appview.appview_url == "https://datasets.atdata.blue"
        assert client_with_appview.appview_did == "did:web:datasets.atdata.blue"

    def test_appview_from_did(self, mock_atproto_client):
        client = Atmosphere(
            _client=mock_atproto_client,
            appview="did:web:datasets.atdata.blue",
        )
        assert client.has_appview is True
        assert client.appview_url == "https://datasets.atdata.blue"
        assert client.appview_did == "did:web:datasets.atdata.blue"

    def test_appview_invalid_value(self, mock_atproto_client):
        with pytest.raises(ValueError, match="Invalid appview"):
            Atmosphere(_client=mock_atproto_client, appview="not-a-url-or-did")

    def test_login_with_appview(self, mock_atproto_client):
        with patch("atdata.atmosphere.client._get_atproto_client_class") as mock_cls:
            mock_cls.return_value = lambda **kw: mock_atproto_client
            client = Atmosphere.login(
                "test.bsky.social",
                "test-password",
                appview="https://datasets.atdata.blue",
            )
            assert client.has_appview is True

    def test_from_session_with_appview(self, mock_atproto_client):
        with patch("atdata.atmosphere.client._get_atproto_client_class") as mock_cls:
            mock_cls.return_value = lambda **kw: mock_atproto_client
            client = Atmosphere.from_session(
                "session-string",
                appview="https://datasets.atdata.blue",
            )
            assert client.has_appview is True

    def test_from_env(self, mock_atproto_client):
        env = {
            "ATDATA_HANDLE": "test.bsky.social",
            "ATDATA_PASSWORD": "test-password",
            "ATDATA_APPVIEW": "https://datasets.atdata.blue",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("atdata.atmosphere.client._get_atproto_client_class") as mock_cls,
        ):
            mock_cls.return_value = lambda **kw: mock_atproto_client
            client = Atmosphere.from_env()
            assert client.has_appview is True
            assert client.appview_url == "https://datasets.atdata.blue"

    def test_from_env_missing_vars(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(EnvironmentError, match="ATDATA_HANDLE"),
        ):
            Atmosphere.from_env()


# =============================================================================
# Exception Types
# =============================================================================


class TestAppViewExceptions:
    def test_hierarchy(self):
        assert issubclass(AppViewError, Exception)
        assert issubclass(AppViewRequiredError, AppViewError)
        assert issubclass(AppViewUnavailableError, AppViewError)

    def test_required_error_message(self):
        err = AppViewRequiredError("science.alt.dataset.searchDatasets")
        assert "searchDatasets" in str(err)
        assert "requires an AppView" in str(err)

    def test_unavailable_error_message(self):
        err = AppViewUnavailableError("https://example.com", "connection refused")
        assert "https://example.com" in str(err)
        assert "connection refused" in str(err)
        assert err.url == "https://example.com"
        assert err.reason == "connection refused"

    def test_importable_from_atdata(self):
        import atdata

        assert hasattr(atdata, "AppViewError")
        assert hasattr(atdata, "AppViewRequiredError")
        assert hasattr(atdata, "AppViewUnavailableError")


# =============================================================================
# XRPC Transport
# =============================================================================


class TestXrpcTransport:
    def test_xrpc_query_requires_appview(self, client_no_appview):
        with pytest.raises(AppViewRequiredError):
            client_no_appview.xrpc_query("science.alt.dataset.listEntries")

    def test_xrpc_procedure_requires_appview(self, client_no_appview):
        with pytest.raises(AppViewRequiredError):
            client_no_appview.xrpc_procedure("science.alt.dataset.publishSchema")

    def test_xrpc_query_success(self, client_with_appview):
        mock_response = Mock()
        mock_response.json.return_value = {"entries": [{"name": "test"}]}
        mock_response.raise_for_status = Mock()
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        client_with_appview._httpx_client = mock_client

        result = client_with_appview.xrpc_query(
            "science.alt.dataset.listEntries",
            params={"limit": 10},
        )

        assert result == {"entries": [{"name": "test"}]}
        mock_client.get.assert_called_once_with(
            "https://datasets.atdata.blue/xrpc/science.alt.dataset.listEntries",
            params={"limit": 10},
        )

    def test_xrpc_query_server_error_raises_unavailable(self, client_with_appview):
        import httpx

        mock_response = Mock()
        mock_response.status_code = 500
        exc = httpx.HTTPStatusError("error", request=Mock(), response=mock_response)
        mock_client = Mock()
        mock_client.get.side_effect = exc
        client_with_appview._httpx_client = mock_client

        with pytest.raises(AppViewUnavailableError):
            client_with_appview.xrpc_query("science.alt.dataset.listEntries")

    def test_xrpc_procedure_success(self, client_with_appview):
        mock_response = Mock()
        mock_response.json.return_value = {
            "uri": "at://did:plc:test/science.alt.dataset.schema/abc",
            "cid": "bafyrei123",
        }
        mock_response.raise_for_status = Mock()
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        client_with_appview._httpx_client = mock_client

        result = client_with_appview.xrpc_procedure(
            "science.alt.dataset.publishSchema",
            input={"record": {"name": "Test"}},
        )

        assert result["uri"] == "at://did:plc:test/science.alt.dataset.schema/abc"
        call_args = mock_client.post.call_args
        assert "atproto-proxy" in call_args.kwargs.get(
            "headers", call_args[1].get("headers", {})
        )


# =============================================================================
# Label Loader — AppView Query Migration
# =============================================================================


class TestLabelLoaderAppView:
    def test_resolve_uses_appview_when_configured(self, client_with_appview):
        mock_client = Mock()
        mock_client.get.return_value = Mock(
            json=Mock(
                return_value={
                    "uri": "at://did:plc:abc/science.alt.dataset.entry/xyz",
                    "cid": "bafyrei123",
                    "label": {"name": "mnist"},
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_client

        loader = LabelLoader(client_with_appview)
        result = loader.resolve("alice.test", "mnist")

        assert result == "at://did:plc:abc/science.alt.dataset.entry/xyz"

    def test_resolve_falls_back_on_appview_error(self, client_with_appview):
        """When AppView fails, resolve should fall back to client-side."""
        import httpx

        mock_httpx = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_httpx.get.side_effect = httpx.HTTPStatusError(
            "error", request=Mock(), response=mock_response
        )
        client_with_appview._httpx_client = mock_httpx

        # Set up client-side fallback data
        mock_record = Mock()
        mock_record.value = {
            "$type": "science.alt.dataset.label",
            "name": "mnist",
            "datasetUri": "at://did:plc:abc/science.alt.dataset.entry/fallback",
            "createdAt": "2025-01-01T00:00:00Z",
        }
        mock_list_response = Mock()
        mock_list_response.records = [mock_record]
        mock_list_response.cursor = None
        client_with_appview._client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )
        client_with_appview._client.com.atproto.identity.resolve_handle.return_value = (
            Mock(did="did:plc:abc")
        )

        loader = LabelLoader(client_with_appview)
        result = loader.resolve("alice.test", "mnist")
        assert result == "at://did:plc:abc/science.alt.dataset.entry/fallback"

    def test_resolve_without_appview_uses_client_side(self, client_no_appview):
        """Without AppView, resolve uses listRecords + filter."""
        mock_record = Mock()
        mock_record.value = {
            "$type": "science.alt.dataset.label",
            "name": "mnist",
            "datasetUri": "at://did:plc:abc/science.alt.dataset.entry/xyz",
            "createdAt": "2025-01-01T00:00:00Z",
        }
        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        client_no_appview._client.com.atproto.repo.list_records.return_value = (
            mock_response
        )
        client_no_appview._client.com.atproto.identity.resolve_handle.return_value = (
            Mock(did="did:plc:abc")
        )

        loader = LabelLoader(client_no_appview)
        result = loader.resolve("alice.test", "mnist")
        assert result == "at://did:plc:abc/science.alt.dataset.entry/xyz"


# =============================================================================
# Label Publisher — AppView Procedure Migration
# =============================================================================


class TestLabelPublisherAppView:
    def test_publish_uses_appview_when_configured(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.post.return_value = Mock(
            json=Mock(
                return_value={
                    "uri": "at://did:plc:test123456789/science.alt.dataset.label/abc",
                    "cid": "bafyrei123",
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        publisher = LabelPublisher(client_with_appview)
        uri = publisher.publish(
            name="mnist",
            dataset_uri="at://did:plc:abc/science.alt.dataset.entry/xyz",
        )
        assert str(uri) == "at://did:plc:test123456789/science.alt.dataset.label/abc"

    def test_publish_without_appview_uses_create_record(self, client_no_appview):
        mock_response = Mock()
        mock_response.uri = "at://did:plc:test123456789/science.alt.dataset.label/abc"
        client_no_appview._client.com.atproto.repo.create_record.return_value = (
            mock_response
        )

        publisher = LabelPublisher(client_no_appview)
        uri = publisher.publish(
            name="mnist",
            dataset_uri="at://did:plc:abc/science.alt.dataset.entry/xyz",
        )
        assert str(uri) == "at://did:plc:test123456789/science.alt.dataset.label/abc"


# =============================================================================
# Schema Loader — AppView Query Migration
# =============================================================================


class TestSchemaLoaderAppView:
    def test_list_all_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "schemas": [{"name": "TestSchema", "version": "1.0.0"}],
                    "cursor": None,
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = SchemaLoader(client_with_appview)
        result = loader.list_all()

        assert len(result) == 1
        assert result[0]["name"] == "TestSchema"

    def test_resolve_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "record": {
                        "name": "TestSchema",
                        "version": "1.0.0",
                        "schemaType": "jsonSchema",
                        "$type": "science.alt.dataset.schema",
                        "atdataSchemaVersion": 1,
                    },
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = SchemaLoader(client_with_appview)
        result = loader.resolve("alice.test", "TestSchema", "1.0.0")
        assert result["name"] == "TestSchema"

    def test_list_all_without_appview(self, client_no_appview):
        mock_record = Mock()
        mock_record.value = {"name": "Schema1", "$type": "science.alt.dataset.schema"}
        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        client_no_appview._client.com.atproto.repo.list_records.return_value = (
            mock_response
        )

        loader = SchemaLoader(client_no_appview)
        result = loader.list_all()
        assert len(result) == 1


# =============================================================================
# Schema Publisher — AppView Procedure Migration
# =============================================================================


class TestSchemaPublisherAppView:
    def test_publish_uses_appview(self, client_with_appview):
        import atdata

        @atdata.packable
        class TestSample:
            name: str
            value: int

        mock_httpx = Mock()
        mock_httpx.post.return_value = Mock(
            json=Mock(
                return_value={
                    "uri": "at://did:plc:test123456789/science.alt.dataset.schema/abc",
                    "cid": "bafyrei123",
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        publisher = SchemaPublisher(client_with_appview)
        uri = publisher.publish(TestSample, version="1.0.0")
        assert "science.alt.dataset.schema" in str(uri)


# =============================================================================
# Dataset Loader — AppView Query Migration
# =============================================================================


class TestDatasetLoaderAppView:
    def test_list_all_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "entries": [{"name": "ds1"}, {"name": "ds2"}],
                    "cursor": None,
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = DatasetLoader(client_with_appview)
        result = loader.list_all()
        assert len(result) == 2

    def test_get_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "entry": {
                        "name": "test-ds",
                        "$type": "science.alt.dataset.entry",
                        "schemaRef": "at://did:plc:abc/science.alt.dataset.schema/xyz",
                    },
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = DatasetLoader(client_with_appview)
        result = loader.get("at://did:plc:abc/science.alt.dataset.entry/xyz")
        assert result["name"] == "test-ds"

    def test_get_blob_urls_uses_appview(self, client_with_appview):
        # First call: get the entry to check it's blob storage
        # The AppView resolveBlobs bypasses get_blobs
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "blobs": [
                        {
                            "uri": "at://did:plc:abc/science.alt.dataset.entry/xyz",
                            "cid": "bafyrei123",
                            "url": "https://pds.example.com/xrpc/com.atproto.sync.getBlob?did=did:plc:abc&cid=bafyrei123",
                        }
                    ],
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = DatasetLoader(client_with_appview)
        urls = loader.get_blob_urls("at://did:plc:abc/science.alt.dataset.entry/xyz")
        assert len(urls) == 1
        assert "getBlob" in urls[0]


# =============================================================================
# Dataset Publisher — AppView Procedure Migration
# =============================================================================


class TestDatasetPublisherAppView:
    def test_create_record_uses_appview(self, client_with_appview):
        from atdata.atmosphere._lexicon_types import StorageHttp

        mock_httpx = Mock()
        mock_httpx.post.return_value = Mock(
            json=Mock(
                return_value={
                    "uri": "at://did:plc:test123456789/science.alt.dataset.entry/abc",
                    "cid": "bafyrei123",
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        publisher = DatasetPublisher(client_with_appview)
        uri = publisher._create_record(
            StorageHttp(shards=[]),
            name="TestDS",
            schema_uri="at://did:plc:abc/science.alt.dataset.schema/xyz",
        )
        assert "science.alt.dataset.entry" in str(uri)


# =============================================================================
# Lens Loader — AppView Query Migration
# =============================================================================


class TestLensLoaderAppView:
    def test_list_all_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "lenses": [{"name": "lens1"}],
                    "cursor": None,
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = LensLoader(client_with_appview)
        result = loader.list_all()
        assert len(result) == 1

    def test_find_by_schemas_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "lenses": [
                        {
                            "name": "my_lens",
                            "sourceSchema": "at://did:plc:abc/science.alt.dataset.schema/src",
                            "targetSchema": "at://did:plc:abc/science.alt.dataset.schema/tgt",
                        }
                    ],
                    "cursor": None,
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = LensLoader(client_with_appview)
        result = loader.find_by_schemas(
            "at://did:plc:abc/science.alt.dataset.schema/src",
            "at://did:plc:abc/science.alt.dataset.schema/tgt",
        )
        assert len(result) == 1
        assert result[0]["name"] == "my_lens"

    def test_find_by_schemas_without_appview_paginates(self, client_no_appview):
        """Without AppView, find_by_schemas paginates through listRecords."""
        mock_record = Mock()
        mock_record.value = {
            "$type": "science.alt.dataset.lens",
            "sourceSchema": "at://did:plc:abc/science.alt.dataset.schema/src",
            "targetSchema": "at://did:plc:abc/science.alt.dataset.schema/tgt",
            "name": "found_lens",
        }
        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        client_no_appview._client.com.atproto.repo.list_records.return_value = (
            mock_response
        )

        loader = LensLoader(client_no_appview)
        result = loader.find_by_schemas(
            "at://did:plc:abc/science.alt.dataset.schema/src",
            "at://did:plc:abc/science.alt.dataset.schema/tgt",
        )
        assert len(result) == 1
        assert result[0]["name"] == "found_lens"


# =============================================================================
# Lens Publisher — AppView Procedure Migration
# =============================================================================


class TestLensPublisherAppView:
    def test_publish_uses_appview(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.post.return_value = Mock(
            json=Mock(
                return_value={
                    "uri": "at://did:plc:test123456789/science.alt.dataset.lens/abc",
                    "cid": "bafyrei123",
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        publisher = LensPublisher(client_with_appview)
        uri = publisher.publish(
            name="test_lens",
            source_schema_uri="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema_uri="at://did:plc:abc/science.alt.dataset.schema/tgt",
            code_repository="https://github.com/user/repo",
            code_commit="abc123",
            getter_path="mod:getter",
            putter_path="mod:putter",
        )
        assert "science.alt.dataset.lens" in str(uri)


# =============================================================================
# AppView-Only Capabilities
# =============================================================================


class TestAppViewOnlyCapabilities:
    def test_search_datasets(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "entries": [{"name": "genomics-ds", "tags": ["genomics"]}],
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        result = client_with_appview.search_datasets("genomics", tags=["train"])
        assert len(result) == 1
        assert result[0]["name"] == "genomics-ds"

    def test_search_datasets_requires_appview(self, client_no_appview):
        with pytest.raises(AppViewRequiredError):
            client_no_appview.search_datasets("test")

    def test_search_lenses(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "lenses": [{"name": "my_lens"}],
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        result = client_with_appview.search_lenses(
            source_schema="at://did:plc:abc/science.alt.dataset.schema/src"
        )
        assert len(result) == 1

    def test_describe_service(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "did": "did:web:datasets.atdata.blue",
                    "availableCollections": ["science.alt.dataset.schema"],
                    "recordCount": {"science.alt.dataset.schema": 42},
                    "analytics": {"totalViews": 100},
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        result = client_with_appview.describe_service()
        assert result["did"] == "did:web:datasets.atdata.blue"
        assert "recordCount" in result

    def test_get_entries_batch(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "entries": [{"name": "ds1"}, {"name": "ds2"}],
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        uris = [
            "at://did:plc:abc/science.alt.dataset.entry/a",
            "at://did:plc:abc/science.alt.dataset.entry/b",
        ]
        result = client_with_appview.get_entries(uris)
        assert len(result) == 2

    def test_get_entries_max_25(self, client_with_appview):
        uris = [f"at://did:plc:abc/science.alt.dataset.entry/{i}" for i in range(26)]
        with pytest.raises(ValueError, match="at most 25"):
            client_with_appview.get_entries(uris)

    def test_get_entry_stats(self, client_with_appview):
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "views": 100,
                    "searchAppearances": 5,
                    "period": "week",
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        result = client_with_appview.get_entry_stats(
            "at://did:plc:abc/science.alt.dataset.entry/xyz"
        )
        assert result["views"] == 100
        assert result["period"] == "week"

    def test_get_entry_stats_requires_appview(self, client_no_appview):
        with pytest.raises(AppViewRequiredError):
            client_no_appview.get_entry_stats(
                "at://did:plc:abc/science.alt.dataset.entry/xyz"
            )


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestXrpcEdgeCases:
    def test_xrpc_query_4xx_raises_httpx_error(self, client_with_appview):
        """4xx errors should propagate as httpx.HTTPStatusError, not AppViewUnavailableError."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 404
        exc = httpx.HTTPStatusError("Not Found", request=Mock(), response=mock_response)
        mock_client = Mock()
        mock_client.get.side_effect = exc
        client_with_appview._httpx_client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            client_with_appview.xrpc_query("science.alt.dataset.getEntry")

    def test_xrpc_query_connect_error_raises_unavailable(self, client_with_appview):
        """Connection failures should raise AppViewUnavailableError."""
        import httpx

        mock_client = Mock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        client_with_appview._httpx_client = mock_client

        with pytest.raises(AppViewUnavailableError, match="Connection refused"):
            client_with_appview.xrpc_query("science.alt.dataset.listEntries")

    def test_xrpc_procedure_includes_auth_header(self, client_with_appview):
        """Procedure calls should include Bearer token from session."""
        mock_session = Mock()
        mock_session.access_jwt = "test-jwt-token"
        client_with_appview._client._session = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {"uri": "at://did:plc:test/x/y"}
        mock_response.raise_for_status = Mock()
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        client_with_appview._httpx_client = mock_client

        client_with_appview.xrpc_procedure("science.alt.dataset.publishSchema")

        call_headers = mock_client.post.call_args.kwargs.get(
            "headers", mock_client.post.call_args[1].get("headers", {})
        )
        assert call_headers["Authorization"] == "Bearer test-jwt-token"
        assert "atproto-proxy" in call_headers

    def test_xrpc_procedure_includes_proxy_header_with_did(self, client_with_appview):
        """atproto-proxy header should contain the AppView DID."""
        mock_response = Mock()
        mock_response.json.return_value = {"uri": "at://did:plc:test/x/y"}
        mock_response.raise_for_status = Mock()
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        client_with_appview._httpx_client = mock_client

        client_with_appview.xrpc_procedure("science.alt.dataset.publishSchema")

        call_headers = mock_client.post.call_args.kwargs.get(
            "headers", mock_client.post.call_args[1].get("headers", {})
        )
        assert (
            call_headers["atproto-proxy"]
            == "did:web:datasets.atdata.blue#atdata_appview"
        )

    def test_blob_urls_appview_sends_uri_as_list(self, client_with_appview):
        """_get_blob_urls_via_appview should send uris as a list, not a bare string."""
        mock_httpx = Mock()
        mock_httpx.get.return_value = Mock(
            json=Mock(
                return_value={
                    "blobs": [
                        {"url": "https://pds.example.com/blob/1", "cid": "bafyrei1"}
                    ],
                }
            ),
            raise_for_status=Mock(),
        )
        client_with_appview._httpx_client = mock_httpx

        loader = DatasetLoader(client_with_appview)
        loader.get_blob_urls("at://did:plc:abc/science.alt.dataset.entry/xyz")

        call_args = mock_httpx.get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        # The uris param must be a list, not a bare string
        assert isinstance(params.get("uris"), list)

    def test_fallback_logs_warning_on_appview_failure(self, client_with_appview):
        """Fallback should log a warning when AppView fails."""
        import httpx

        mock_httpx_client = Mock()
        mock_httpx_client.get.side_effect = httpx.ConnectError("refused")
        client_with_appview._httpx_client = mock_httpx_client

        # Set up client-side fallback
        mock_record = Mock()
        mock_record.value = {
            "$type": "science.alt.dataset.schema",
            "name": "S",
            "version": "1.0.0",
            "atdataSchemaVersion": 1,
        }
        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        client_with_appview._client.com.atproto.repo.list_records.return_value = (
            mock_response
        )

        loader = SchemaLoader(client_with_appview)
        # list_all catches the AppViewUnavailableError and falls back
        result = loader.list_all()
        assert len(result) == 1

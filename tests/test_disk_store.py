"""Tests for atdata.LocalDiskStore."""

from pathlib import Path

import atdata
from conftest import (
    SharedBasicSample,
    SharedNumpySample,
    create_basic_dataset,
    create_numpy_dataset,
)


class TestLocalDiskStoreInit:
    """Tests for LocalDiskStore initialization."""

    def test_default_root(self):
        store = atdata.LocalDiskStore()
        assert store.root == Path.home() / ".atdata" / "data"

    def test_custom_root(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path / "custom")
        assert store.root == (tmp_path / "custom").resolve()
        assert store.root.exists()

    def test_creates_root_directory(self, tmp_path: Path):
        root = tmp_path / "deep" / "nested" / "store"
        assert not root.exists()
        store = atdata.LocalDiskStore(root=root)
        assert store.root.exists()

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        store = atdata.LocalDiskStore(root="~/my-data")
        assert store.root == (tmp_path / "my-data").resolve()


class TestLocalDiskStoreWriteShards:
    """Tests for LocalDiskStore.write_shards()."""

    def test_write_basic_dataset(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path / "store")
        ds = create_basic_dataset(tmp_path, num_samples=5)

        urls = store.write_shards(ds, prefix="test-ds")

        assert len(urls) >= 1
        for url in urls:
            assert Path(url).exists()
            assert url.endswith(".tar")

    def test_write_numpy_dataset(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path / "store")
        ds = create_numpy_dataset(tmp_path, num_samples=3, array_shape=(4, 4))

        urls = store.write_shards(ds, prefix="numpy-ds")

        assert len(urls) >= 1
        # Read back and verify
        result_ds = atdata.Dataset[SharedNumpySample](url=urls[0])
        result = list(result_ds.ordered())
        assert len(result) == 3
        for s in result:
            assert s.data.shape == (4, 4)

    def test_prefix_creates_subdirectory(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path / "store")
        ds = create_basic_dataset(tmp_path, num_samples=3)

        store.write_shards(ds, prefix="datasets/mnist/v1")

        shard_dir = tmp_path / "store" / "datasets" / "mnist" / "v1"
        assert shard_dir.exists()
        assert any(shard_dir.iterdir())

    def test_maxcount_kwarg(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path / "store")
        ds = create_basic_dataset(tmp_path, num_samples=10)

        urls = store.write_shards(ds, prefix="sharded", maxcount=3)

        # With 10 samples and maxcount=3, should get at least 4 shards
        assert len(urls) >= 4

    def test_roundtrip_through_store(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path / "store")
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]

        # Write using conftest helper, then store
        from conftest import create_tar_with_samples

        tar_path = tmp_path / "orig-000000.tar"
        create_tar_with_samples(tar_path, samples)
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))

        urls = store.write_shards(ds, prefix="roundtrip")

        # Read back from stored location
        result_ds = atdata.Dataset[SharedBasicSample](url=urls[0])
        result = list(result_ds.ordered())
        assert len(result) == 5
        for i, s in enumerate(result):
            assert s.name == f"s{i}"
            assert s.value == i


class TestLocalDiskStoreProtocol:
    """Tests for AbstractDataStore protocol compliance."""

    def test_read_url_passthrough(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path)
        assert store.read_url("/some/path.tar") == "/some/path.tar"

    def test_supports_streaming(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path)
        assert store.supports_streaming() is True

    def test_satisfies_protocol(self, tmp_path: Path):
        store = atdata.LocalDiskStore(root=tmp_path)
        # Should satisfy AbstractDataStore protocol structurally
        assert hasattr(store, "write_shards")
        assert hasattr(store, "read_url")
        assert hasattr(store, "supports_streaming")

"""Tests for DatasetMeta dataclass and _resolve_meta helper."""

import dataclasses

import pytest

from atdata.dataset_meta import DatasetMeta, _resolve_meta


class TestDatasetMeta:
    """Tests for the DatasetMeta dataclass."""

    def test_create_with_name_only(self):
        meta = DatasetMeta(name="mnist")
        assert meta.name == "mnist"
        assert meta.schema_ref is None
        assert meta.description is None
        assert meta.tags is None
        assert meta.license is None
        assert meta.metadata is None

    def test_create_with_all_fields(self):
        meta = DatasetMeta(
            name="mnist",
            schema_ref="local://schemas/Mnist@1.0.0",
            description="Handwritten digits",
            tags=["vision", "classification"],
            license="MIT",
            metadata={"source": "yann.lecun.com"},
        )
        assert meta.name == "mnist"
        assert meta.schema_ref == "local://schemas/Mnist@1.0.0"
        assert meta.description == "Handwritten digits"
        assert meta.tags == ["vision", "classification"]
        assert meta.license == "MIT"
        assert meta.metadata == {"source": "yann.lecun.com"}

    def test_is_dataclass(self):
        meta = DatasetMeta(name="test")
        assert dataclasses.is_dataclass(meta)

    def test_replace(self):
        meta = DatasetMeta(name="mnist", description="original")
        updated = dataclasses.replace(meta, description="updated")
        assert updated.name == "mnist"
        assert updated.description == "updated"
        assert meta.description == "original"

    def test_equality(self):
        a = DatasetMeta(name="mnist", tags=["vision"])
        b = DatasetMeta(name="mnist", tags=["vision"])
        assert a == b

    def test_importable_from_atdata(self):
        import atdata

        assert hasattr(atdata, "DatasetMeta")
        assert atdata.DatasetMeta is DatasetMeta


class TestResolveMeta:
    """Tests for the _resolve_meta helper."""

    def test_from_flat_kwargs(self):
        meta = _resolve_meta(
            name="mnist",
            description="digits",
            tags=["vision"],
        )
        assert meta.name == "mnist"
        assert meta.description == "digits"
        assert meta.tags == ["vision"]

    def test_from_meta_object(self):
        original = DatasetMeta(
            name="mnist",
            description="digits",
            tags=["vision"],
        )
        meta = _resolve_meta(original)
        assert meta is original

    def test_raises_when_neither_name_nor_meta(self):
        with pytest.raises(TypeError, match="Either 'meta' or 'name' must be provided"):
            _resolve_meta()

    def test_explicit_kwargs_override_meta(self):
        original = DatasetMeta(
            name="mnist",
            description="original",
            tags=["old"],
            license="MIT",
        )
        meta = _resolve_meta(
            original,
            description="overridden",
            tags=["new"],
        )
        assert meta.name == "mnist"
        assert meta.description == "overridden"
        assert meta.tags == ["new"]
        assert meta.license == "MIT"

    def test_name_override(self):
        original = DatasetMeta(name="original")
        meta = _resolve_meta(original, name="overridden")
        assert meta.name == "overridden"

    def test_none_kwargs_do_not_override(self):
        original = DatasetMeta(
            name="mnist",
            description="digits",
            tags=["vision"],
        )
        meta = _resolve_meta(
            original,
            description=None,
            tags=None,
        )
        assert meta.description == "digits"
        assert meta.tags == ["vision"]

    def test_override_returns_new_instance(self):
        original = DatasetMeta(name="mnist", description="old")
        meta = _resolve_meta(original, description="new")
        assert meta is not original
        assert original.description == "old"

    def test_no_override_returns_same_instance(self):
        original = DatasetMeta(name="mnist")
        meta = _resolve_meta(original)
        assert meta is original

    def test_all_fields_from_kwargs(self):
        meta = _resolve_meta(
            name="ds",
            schema_ref="ref",
            description="desc",
            tags=["t"],
            license="Apache-2.0",
            metadata={"k": "v"},
        )
        assert meta == DatasetMeta(
            name="ds",
            schema_ref="ref",
            description="desc",
            tags=["t"],
            license="Apache-2.0",
            metadata={"k": "v"},
        )

    def test_metadata_dict_override(self):
        original = DatasetMeta(name="ds", metadata={"a": 1})
        meta = _resolve_meta(original, metadata={"b": 2})
        assert meta.metadata == {"b": 2}


class TestDatasetMetaWithIndex:
    """Tests for DatasetMeta integration with Index methods."""

    def test_write_samples_with_meta(self, tmp_path):
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from atdata.stores import LocalDiskStore
        from conftest import SharedBasicSample

        provider = SqliteProvider(path=tmp_path / "test.db")
        store = LocalDiskStore(root=tmp_path / "store")
        index = Index(provider=provider, data_store=store, atmosphere=None)

        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        meta = DatasetMeta(name="meta-ds", metadata={"source": "test"})

        entry = index.write_samples(samples, meta=meta)
        assert entry.name == "meta-ds"

    def test_write_samples_meta_with_kwargs_override(self, tmp_path):
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from atdata.stores import LocalDiskStore
        from conftest import SharedBasicSample

        provider = SqliteProvider(path=tmp_path / "test.db")
        store = LocalDiskStore(root=tmp_path / "store")
        index = Index(provider=provider, data_store=store, atmosphere=None)

        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        meta = DatasetMeta(name="original-name")

        entry = index.write_samples(samples, meta=meta, name="overridden-name")
        assert entry.name == "overridden-name"

    def test_write_samples_flat_kwargs_still_work(self, tmp_path):
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from atdata.stores import LocalDiskStore
        from conftest import SharedBasicSample

        provider = SqliteProvider(path=tmp_path / "test.db")
        store = LocalDiskStore(root=tmp_path / "store")
        index = Index(provider=provider, data_store=store, atmosphere=None)

        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]

        entry = index.write_samples(samples, name="flat-ds")
        assert entry.name == "flat-ds"

    def test_insert_dataset_with_meta(self, tmp_path):
        import atdata
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from conftest import SharedBasicSample, create_tar_with_samples

        provider = SqliteProvider(path=tmp_path / "test.db")
        index = Index(provider=provider, atmosphere=None)

        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[SharedBasicSample](str(tar_path))
        meta = DatasetMeta(name="insert-meta-ds")

        entry = index.insert_dataset(ds, meta=meta)
        assert entry.name == "insert-meta-ds"

    def test_insert_dataset_meta_with_kwargs_override(self, tmp_path):
        import atdata
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from conftest import SharedBasicSample, create_tar_with_samples

        provider = SqliteProvider(path=tmp_path / "test.db")
        index = Index(provider=provider, atmosphere=None)

        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[SharedBasicSample](str(tar_path))
        meta = DatasetMeta(name="original-name", description="from meta")

        entry = index.insert_dataset(ds, meta=meta, name="overridden-name")
        assert entry.name == "overridden-name"

    def test_write_samples_requires_name_or_meta(self, tmp_path):
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from conftest import SharedBasicSample

        provider = SqliteProvider(path=tmp_path / "test.db")
        index = Index(provider=provider, atmosphere=None)

        samples = [SharedBasicSample(name="s0", value=0)]

        with pytest.raises(TypeError, match="Either 'meta' or 'name'"):
            index.write_samples(samples)

    def test_insert_dataset_requires_name_or_meta(self, tmp_path):
        import atdata
        from atdata.index import Index
        from atdata.providers._sqlite import SqliteProvider
        from conftest import SharedBasicSample, create_tar_with_samples

        provider = SqliteProvider(path=tmp_path / "test.db")
        index = Index(provider=provider, atmosphere=None)

        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name="s0", value=0)]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[SharedBasicSample](str(tar_path))

        with pytest.raises(TypeError, match="Either 'meta' or 'name'"):
            index.insert_dataset(ds)

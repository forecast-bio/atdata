"""Tests for atdata._stub_manager module."""

from pathlib import Path

from atdata._stub_manager import (
    StubManager,
    _extract_authority,
    DEFAULT_STUB_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(
    name: str = "StubTestSample",
    version: str = "1.0.0",
    ref: str | None = "atdata://local/sampleSchema/StubTestSample@1.0.0",
    fields: list[dict] | None = None,
) -> dict:
    """Build a minimal schema dict for testing."""
    if fields is None:
        fields = [
            {
                "name": "text",
                "fieldType": {
                    "$type": "ac.foundation.dataset.schemaType#primitive",
                    "primitive": "str",
                },
                "optional": False,
            },
            {
                "name": "value",
                "fieldType": {
                    "$type": "ac.foundation.dataset.schemaType#primitive",
                    "primitive": "int",
                },
                "optional": False,
            },
        ]
    schema: dict = {
        "name": name,
        "version": version,
        "fields": fields,
    }
    if ref is not None:
        schema["$ref"] = ref
    return schema


# ---------------------------------------------------------------------------
# _extract_authority()
# ---------------------------------------------------------------------------


class TestExtractAuthority:
    def test_none_returns_local(self):
        assert _extract_authority(None) == "local"

    def test_empty_string_returns_local(self):
        assert _extract_authority("") == "local"

    def test_local_ref(self):
        assert _extract_authority("atdata://local/sampleSchema/Name@1.0.0") == "local"

    def test_domain_authority(self):
        assert (
            _extract_authority("atdata://alice.bsky.social/sampleSchema/Name@1.0.0")
            == "alice.bsky.social"
        )

    def test_did_authority_replaces_colons(self):
        assert (
            _extract_authority("atdata://did:plc:abc/sampleSchema/Name@1.0.0")
            == "did_plc_abc"
        )

    def test_no_match_returns_local(self):
        assert _extract_authority("http://example.com/foo") == "local"


# ---------------------------------------------------------------------------
# StubManager construction
# ---------------------------------------------------------------------------


class TestStubManagerInit:
    def test_default_stub_dir(self):
        mgr = StubManager()
        assert mgr.stub_dir == DEFAULT_STUB_DIR

    def test_custom_stub_dir_path(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path / "stubs")
        assert mgr.stub_dir == tmp_path / "stubs"

    def test_custom_stub_dir_string(self, tmp_path: Path):
        mgr = StubManager(stub_dir=str(tmp_path / "stubs"))
        assert mgr.stub_dir == tmp_path / "stubs"


# ---------------------------------------------------------------------------
# _module_filename()
# ---------------------------------------------------------------------------


class TestModuleFilename:
    def test_basic(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        assert mgr._module_filename("MySample", "1.0.0") == "MySample_1_0_0.py"

    def test_version_with_dots(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        assert mgr._module_filename("Foo", "2.10.3") == "Foo_2_10_3.py"


# ---------------------------------------------------------------------------
# _module_path()
# ---------------------------------------------------------------------------


class TestModulePath:
    def test_default_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        result = mgr._module_path("MySample", "1.0.0")
        assert result == tmp_path / "local" / "MySample_1_0_0.py"

    def test_custom_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        result = mgr._module_path("MySample", "1.0.0", authority="alice.bsky.social")
        assert result == tmp_path / "alice.bsky.social" / "MySample_1_0_0.py"


# ---------------------------------------------------------------------------
# _module_is_current()
# ---------------------------------------------------------------------------


class TestModuleIsCurrent:
    def test_file_does_not_exist(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        assert mgr._module_is_current(tmp_path / "nonexistent.py", "1.0.0") is False

    def test_file_with_matching_version(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        path = tmp_path / "Sample_1_0_0.py"
        path.write_text('"""Auto-generated module.\n\nSchema: Sample@1.0.0\n"""\n')
        assert mgr._module_is_current(path, "1.0.0") is True

    def test_file_with_wrong_version(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        path = tmp_path / "Sample_1_0_0.py"
        path.write_text('"""Auto-generated module.\n\nSchema: Sample@0.9.0\n"""\n')
        assert mgr._module_is_current(path, "1.0.0") is False

    def test_file_with_no_version_pattern(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        path = tmp_path / "Sample_1_0_0.py"
        path.write_text("# no version line here\n")
        assert mgr._module_is_current(path, "1.0.0") is False

    def test_unreadable_file(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        path = tmp_path / "Sample_1_0_0.py"
        path.write_text('"""Schema: Sample@1.0.0"""')
        path.chmod(0o000)
        try:
            assert mgr._module_is_current(path, "1.0.0") is False
        finally:
            path.chmod(0o644)


# ---------------------------------------------------------------------------
# _ensure_authority_package()
# ---------------------------------------------------------------------------


class TestEnsureAuthorityPackage:
    def test_creates_directory_and_init(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path / "stubs")
        mgr._ensure_authority_package("myauthority")

        authority_dir = tmp_path / "stubs" / "myauthority"
        assert authority_dir.is_dir()
        init = authority_dir / "__init__.py"
        assert init.exists()
        assert "myauthority" in init.read_text()

    def test_also_creates_root_init(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path / "stubs")
        mgr._ensure_authority_package("local")

        root_init = tmp_path / "stubs" / "__init__.py"
        assert root_init.exists()

    def test_idempotent(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path / "stubs")
        mgr._ensure_authority_package("local")
        mgr._ensure_authority_package("local")  # second call should not fail
        assert (tmp_path / "stubs" / "local" / "__init__.py").exists()


# ---------------------------------------------------------------------------
# _write_module_atomic()
# ---------------------------------------------------------------------------


class TestWriteModuleAtomic:
    def test_writes_file(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        target = tmp_path / "local" / "Foo_1_0_0.py"
        content = "# hello world\n"
        mgr._write_module_atomic(target, content, "local")

        assert target.exists()
        assert target.read_text() == content

    def test_overwrites_existing(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        target = tmp_path / "local" / "Foo_1_0_0.py"
        mgr._write_module_atomic(target, "old content\n", "local")
        mgr._write_module_atomic(target, "new content\n", "local")
        assert target.read_text() == "new content\n"


# ---------------------------------------------------------------------------
# ensure_stub()
# ---------------------------------------------------------------------------


class TestEnsureStub:
    def test_generates_module_file(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        path = mgr.ensure_stub(schema)

        assert path is not None
        assert path.exists()
        assert path.name == "StubTestSample_1_0_0.py"
        content = path.read_text()
        assert "class StubTestSample" in content
        assert "Schema: StubTestSample@1.0.0" in content

    def test_missing_name_returns_none(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        del schema["name"]
        assert mgr.ensure_stub(schema) is None

    def test_already_current_does_not_rewrite(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        path1 = mgr.ensure_stub(schema)
        assert path1 is not None
        mtime1 = path1.stat().st_mtime_ns

        path2 = mgr.ensure_stub(schema)
        assert path2 is not None
        assert path2.stat().st_mtime_ns == mtime1  # file not rewritten

    def test_schema_with_ref_uses_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema(
            ref="atdata://alice.bsky.social/sampleSchema/StubTestSample@1.0.0"
        )
        path = mgr.ensure_stub(schema)

        assert path is not None
        assert "alice.bsky.social" in str(path)

    def test_schema_without_ref_uses_local(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema(ref=None)
        path = mgr.ensure_stub(schema)

        assert path is not None
        assert "local" in str(path)

    def test_prints_ide_hint_first_time(self, tmp_path: Path, capsys):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        mgr.ensure_stub(schema)
        captured = capsys.readouterr()
        assert "[atdata]" in captured.err

    def test_ide_hint_only_once(self, tmp_path: Path, capsys):
        mgr = StubManager(stub_dir=tmp_path)
        schema_a = _make_schema(
            name="SampleA", ref="atdata://local/sampleSchema/SampleA@1.0.0"
        )
        schema_b = _make_schema(
            name="SampleB",
            version="2.0.0",
            ref="atdata://local/sampleSchema/SampleB@2.0.0",
        )
        mgr.ensure_stub(schema_a)
        mgr.ensure_stub(schema_b)
        captured = capsys.readouterr()
        assert captured.err.count("[atdata] Generated schema module in:") == 1


# ---------------------------------------------------------------------------
# ensure_module()
# ---------------------------------------------------------------------------


class TestEnsureModule:
    def test_returns_class(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        cls = mgr.ensure_module(schema)

        assert cls is not None
        assert cls.__name__ == "StubTestSample"

    def test_returns_none_for_no_name(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        del schema["name"]
        assert mgr.ensure_module(schema) is None

    def test_caches_result(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        cls1 = mgr.ensure_module(schema)
        cls2 = mgr.ensure_module(schema)
        assert cls1 is cls2

    def test_cache_returns_same_object(self, tmp_path: Path):
        """Verify the cache is hit by checking id() equality."""
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        cls1 = mgr.ensure_module(schema)
        cls2 = mgr.ensure_module(schema)
        assert id(cls1) == id(cls2)


# ---------------------------------------------------------------------------
# list_stubs()
# ---------------------------------------------------------------------------


class TestListStubs:
    def test_empty_directory(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path / "empty")
        assert mgr.list_stubs() == []

    def test_lists_generated_files(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(
            _make_schema(name="Alpha", ref="atdata://local/sampleSchema/Alpha@1.0.0")
        )
        mgr.ensure_stub(
            _make_schema(name="Beta", ref="atdata://local/sampleSchema/Beta@1.0.0")
        )

        stubs = mgr.list_stubs()
        names = {p.name for p in stubs}
        assert "Alpha_1_0_0.py" in names
        assert "Beta_1_0_0.py" in names

    def test_excludes_init_py(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(_make_schema())

        stubs = mgr.list_stubs()
        assert all(p.name != "__init__.py" for p in stubs)

    def test_filter_by_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(
            _make_schema(
                name="LocalSample", ref="atdata://local/sampleSchema/LocalSample@1.0.0"
            )
        )
        mgr.ensure_stub(
            _make_schema(
                name="RemoteSample",
                ref="atdata://alice.bsky.social/sampleSchema/RemoteSample@1.0.0",
            )
        )

        local_stubs = mgr.list_stubs(authority="local")
        remote_stubs = mgr.list_stubs(authority="alice.bsky.social")

        assert len(local_stubs) == 1
        assert local_stubs[0].name == "LocalSample_1_0_0.py"
        assert len(remote_stubs) == 1
        assert remote_stubs[0].name == "RemoteSample_1_0_0.py"

    def test_filter_nonexistent_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(_make_schema())
        assert mgr.list_stubs(authority="nonexistent") == []


# ---------------------------------------------------------------------------
# clear_stubs()
# ---------------------------------------------------------------------------


class TestClearStubs:
    def test_removes_files_and_returns_count(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(
            _make_schema(name="A", ref="atdata://local/sampleSchema/A@1.0.0")
        )
        mgr.ensure_stub(
            _make_schema(name="B", ref="atdata://local/sampleSchema/B@1.0.0")
        )

        removed = mgr.clear_stubs()
        assert removed == 2

    def test_clears_class_cache(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        mgr.ensure_module(schema)
        assert len(mgr._class_cache) > 0

        mgr.clear_stubs()
        assert len(mgr._class_cache) == 0

    def test_removes_empty_authority_dirs(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(_make_schema())
        mgr.clear_stubs()

        # The "local" authority dir should be removed (only had __init__.py left)
        assert not (tmp_path / "local").exists()

    def test_clear_stubs_by_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(
            _make_schema(
                name="LocalSample", ref="atdata://local/sampleSchema/LocalSample@1.0.0"
            )
        )
        mgr.ensure_stub(
            _make_schema(
                name="RemoteSample",
                ref="atdata://alice.bsky.social/sampleSchema/RemoteSample@1.0.0",
            )
        )

        removed = mgr.clear_stubs(authority="local")
        assert removed == 1

        # Remote stubs should still exist
        remaining = mgr.list_stubs()
        assert len(remaining) == 1
        assert remaining[0].name == "RemoteSample_1_0_0.py"

    def test_clear_stubs_authority_clears_only_matching_cache(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        local_schema = _make_schema(
            name="LocalSample", ref="atdata://local/sampleSchema/LocalSample@1.0.0"
        )
        remote_schema = _make_schema(
            name="RemoteSample",
            ref="atdata://alice.bsky.social/sampleSchema/RemoteSample@1.0.0",
        )
        mgr.ensure_module(local_schema)
        mgr.ensure_module(remote_schema)
        assert len(mgr._class_cache) == 2

        mgr.clear_stubs(authority="local")
        assert len(mgr._class_cache) == 1
        assert ("alice.bsky.social", "RemoteSample", "1.0.0") in mgr._class_cache

    def test_clear_on_empty_dir(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path / "empty")
        assert mgr.clear_stubs() == 0


# ---------------------------------------------------------------------------
# clear_stub()
# ---------------------------------------------------------------------------


class TestClearStub:
    def test_removes_specific_file(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(_make_schema())
        assert mgr.clear_stub("StubTestSample", "1.0.0") is True
        assert not mgr._module_path("StubTestSample", "1.0.0").exists()

    def test_returns_false_if_not_exists(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        # directory doesn't even exist yet
        assert mgr.clear_stub("NonExistent", "1.0.0") is False

    def test_clears_cache_entry(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        schema = _make_schema()
        mgr.ensure_module(schema)
        assert ("local", "StubTestSample", "1.0.0") in mgr._class_cache

        mgr.clear_stub("StubTestSample", "1.0.0")
        assert ("local", "StubTestSample", "1.0.0") not in mgr._class_cache


# ---------------------------------------------------------------------------
# get_stub_path()
# ---------------------------------------------------------------------------


class TestGetStubPath:
    def test_returns_path_when_exists(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(_make_schema())
        result = mgr.get_stub_path("StubTestSample", "1.0.0")
        assert result is not None
        assert result.exists()

    def test_returns_none_when_missing(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        assert mgr.get_stub_path("DoesNotExist", "1.0.0") is None

    def test_respects_authority(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        mgr.ensure_stub(
            _make_schema(
                ref="atdata://alice.bsky.social/sampleSchema/StubTestSample@1.0.0"
            )
        )
        # Should not find it under default "local" authority
        assert mgr.get_stub_path("StubTestSample", "1.0.0") is None
        # Should find it under correct authority
        assert (
            mgr.get_stub_path("StubTestSample", "1.0.0", authority="alice.bsky.social")
            is not None
        )


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------


class TestAliases:
    def test_stub_filename_alias(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        assert mgr._stub_filename("X", "1.0.0") == mgr._module_filename("X", "1.0.0")

    def test_stub_path_alias(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        assert mgr._stub_path("X", "1.0.0") == mgr._module_path("X", "1.0.0")

    def test_stub_is_current_alias(self, tmp_path: Path):
        mgr = StubManager(stub_dir=tmp_path)
        path = tmp_path / "fake.py"
        assert mgr._stub_is_current(path, "1.0.0") == mgr._module_is_current(
            path, "1.0.0"
        )

"""Module manager for automatic Python module generation.

This module provides automatic generation and management of Python modules
for dynamically decoded schema types. When enabled, modules are generated
on schema access to provide IDE autocomplete and type checking support.

Unlike simple .pyi stubs, the generated modules are actual Python code that
can be imported at runtime. This allows ``get_schema_type`` to return properly
typed classes that work with both static type checkers and runtime.

Examples:
    >>> from atdata.local import Index
    >>>
    >>> # Enable auto-stub generation
    >>> index = Index(auto_stubs=True)
    >>>
    >>> # Modules are generated automatically on get_schema_type
    >>> MyType = index.get_schema_type("atdata://local/schema/MySample@1.0.0")
    >>> # MyType is now properly typed for IDE autocomplete!
    >>>
    >>> # Get the stub directory path for IDE configuration
    >>> print(f"Add to IDE: {index.stub_dir}")
"""

from pathlib import Path
from typing import Optional, Union, Type
import os
import re
import sys
import tempfile
import fcntl
import importlib.util

from ._schema_codec import generate_module
from ._lens_codec import generate_lens_stub


# Default stub directory location
DEFAULT_STUB_DIR = Path.home() / ".atdata" / "stubs"

# Pattern to extract version from module docstring
_VERSION_PATTERN = re.compile(r"^Schema: .+@(\d+\.\d+\.\d+)", re.MULTILINE)

# Pattern to extract authority from atdata:// URI
_AUTHORITY_PATTERN = re.compile(r"^atdata://([^/]+)/")

# Default authority for schemas without a ref
DEFAULT_AUTHORITY = "local"


def _extract_authority(schema_ref: Optional[str]) -> str:
    """Extract authority from a schema reference URI.

    Args:
        schema_ref: Schema ref like "atdata://local/schema/Name@1.0.0"
            or "atdata://alice.bsky.social/schema/Name@1.0.0"

    Returns:
        Authority string (e.g., "local", "alice.bsky.social", "did_plc_xxx").
        Special characters like ':' are replaced with '_' for filesystem safety.
    """
    if not schema_ref:
        return DEFAULT_AUTHORITY

    match = _AUTHORITY_PATTERN.match(schema_ref)
    if match:
        authority = match.group(1)
        # Make filesystem-safe: replace : with _
        return authority.replace(":", "_")

    return DEFAULT_AUTHORITY


class _StubWriter:
    """Shared infrastructure for writing stub modules atomically.

    Handles directory creation, ``__init__.py`` maintenance, and
    atomic file writes via temp-file-then-rename.  Subclasses provide
    the filename convention and content generation.
    """

    def __init__(self, stub_dir: Path) -> None:
        self._stub_dir = stub_dir
        self._initialized = False

    @property
    def stub_dir(self) -> Path:
        """The directory where module files are written."""
        return self._stub_dir

    def _ensure_dir_exists(self) -> None:
        """Create stub directory with __init__.py if it doesn't exist."""
        if not self._initialized:
            self._stub_dir.mkdir(parents=True, exist_ok=True)
            init_path = self._stub_dir / "__init__.py"
            if not init_path.exists():
                init_path.write_text('"""Auto-generated atdata schema modules."""\n')
            self._initialized = True

    def _ensure_authority_package(self, authority: str) -> None:
        """Ensure authority subdirectory exists with __init__.py."""
        self._ensure_dir_exists()
        authority_dir = self._stub_dir / authority
        authority_dir.mkdir(parents=True, exist_ok=True)
        init_path = authority_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                f'"""Auto-generated schema modules for {authority}."""\n'
            )

    def _write_module_atomic(self, path: Path, content: str, authority: str) -> None:
        """Write module file atomically using temp file and rename.

        This ensures that concurrent processes won't see partial files.
        Uses file locking for additional safety on systems that support it.

        Args:
            path: Destination path for the module file
            content: Module file content to write
            authority: Authority namespace (for creating __init__.py)
        """
        self._ensure_authority_package(authority)

        # Create temp file in same directory for atomic rename
        fd, temp_path = tempfile.mkstemp(
            suffix=".py.tmp",
            dir=path.parent,
        )
        temp_path = Path(temp_path)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, IOError):
                    pass

                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            temp_path.rename(path)

        except Exception:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    def _clean_empty_dirs(self) -> None:
        """Remove empty authority directories (including lone __init__.py)."""
        if not self._stub_dir.exists():
            return
        for subdir in self._stub_dir.iterdir():
            if subdir.is_dir():
                contents = list(subdir.iterdir())
                if len(contents) == 0:
                    try:
                        subdir.rmdir()
                    except OSError:
                        continue
                elif len(contents) == 1 and contents[0].name == "__init__.py":
                    try:
                        contents[0].unlink()
                        subdir.rmdir()
                    except OSError:
                        continue


class SchemaStubManager(_StubWriter):
    """Manages automatic generation of Python modules for decoded schemas.

    Modules are organised by authority (from the schema ref URI) to avoid
    collisions between schemas with the same name from different sources::

        ~/.atdata/stubs/
            __init__.py
            local/
                __init__.py
                MySample_1_0_0.py
            alice.bsky.social/
                __init__.py
                MySample_1_0_0.py

    Args:
        stub_dir: Directory to write module files. Defaults to ``~/.atdata/stubs/``.

    Examples:
        >>> manager = SchemaStubManager()
        >>> schema_dict = {"name": "MySample", "version": "1.0.0", "fields": [...]}
        >>> SampleClass = manager.ensure_module(schema_dict)
    """

    def __init__(self, stub_dir: Optional[Union[str, Path]] = None):
        super().__init__(Path(stub_dir) if stub_dir else DEFAULT_STUB_DIR)
        self._first_generation = True
        self._class_cache: dict[tuple[str, str, str], Type] = {}

    def _module_filename(self, name: str, version: str) -> str:
        """Generate module filename from schema name and version.

        Args:
            name: Schema name (e.g., "MySample")
            version: Schema version (e.g., "1.0.0")

        Returns:
            Filename like "MySample_1_0_0.py"
        """
        safe_version = version.replace(".", "_")
        return f"{name}_{safe_version}.py"

    def _module_path(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> Path:
        """Get full path to module file for a schema."""
        return self._stub_dir / authority / self._module_filename(name, version)

    def _module_is_current(self, path: Path, version: str) -> bool:
        """Check if an existing module file matches the expected version."""
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(500)
                match = _VERSION_PATTERN.search(content)
                if match:
                    return match.group(1) == version
            return False
        except (OSError, IOError):
            return False

    def ensure_stub(self, schema: dict) -> Optional[Path]:
        """Ensure a module file exists for the given schema.

        Args:
            schema: Schema dict with 'name', 'version', and 'fields' keys.

        Returns:
            Path to the module file, or None if schema is missing required fields.
        """
        name = schema.get("name") if hasattr(schema, "get") else None
        version = schema.get("version", "1.0.0") if hasattr(schema, "get") else "1.0.0"
        schema_ref = schema.get("$ref") if hasattr(schema, "get") else None

        if not name:
            return None

        authority = _extract_authority(schema_ref)
        path = self._module_path(name, version, authority)

        if self._module_is_current(path, version):
            return path

        if hasattr(schema, "to_dict"):
            schema_dict = schema.to_dict()
        else:
            schema_dict = schema

        content = generate_module(schema_dict)
        self._write_module_atomic(path, content, authority)

        if self._first_generation:
            self._first_generation = False
            self._print_ide_hint()

        return path

    def ensure_module(self, schema: dict) -> Optional[Type]:
        """Ensure a module exists and return the class from it.

        Args:
            schema: Schema dict with 'name', 'version', and 'fields' keys.

        Returns:
            The PackableSample subclass from the generated module, or None.
        """
        name = schema.get("name") if hasattr(schema, "get") else None
        version = schema.get("version", "1.0.0") if hasattr(schema, "get") else "1.0.0"
        schema_ref = schema.get("$ref") if hasattr(schema, "get") else None

        if not name:
            return None

        authority = _extract_authority(schema_ref)

        cache_key = (authority, name, version)
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        path = self.ensure_stub(schema)
        if path is None:
            return None

        cls = self._import_class_from_module(path, name)
        if cls is not None:
            self._class_cache[cache_key] = cls

        return cls

    def _import_class_from_module(
        self, module_path: Path, class_name: str
    ) -> Optional[Type]:
        """Import a class from a generated module file."""
        if not module_path.exists():
            return None

        try:
            module_name = f"_atdata_generated_{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return getattr(module, class_name, None)
        except (ModuleNotFoundError, AttributeError, ImportError, OSError):
            return None

    def _print_ide_hint(self) -> None:
        """Print a one-time hint about IDE configuration."""
        import sys as _sys

        print(
            f"\n[atdata] Generated schema module in: {self._stub_dir}\n"
            f"[atdata] For IDE support, add this path to your type checker:\n"
            f"[atdata]   VS Code/Pylance: Add to python.analysis.extraPaths\n"
            f"[atdata]   PyCharm: Mark as Sources Root\n"
            f"[atdata]   mypy: Add to mypy_path in mypy.ini\n",
            file=_sys.stderr,
        )

    def get_stub_path(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> Optional[Path]:
        """Get the path to an existing stub file."""
        path = self._module_path(name, version, authority)
        return path if path.exists() else None

    def list_stubs(self, authority: Optional[str] = None) -> list[Path]:
        """List all schema module files in the stub directory."""
        if not self._stub_dir.exists():
            return []

        if authority:
            authority_dir = self._stub_dir / authority
            if not authority_dir.exists():
                return []
            return [p for p in authority_dir.glob("*.py") if p.name != "__init__.py"]

        return [p for p in self._stub_dir.glob("**/*.py") if p.name != "__init__.py"]

    def clear_stubs(self, authority: Optional[str] = None) -> int:
        """Remove schema module files from the stub directory."""
        stubs = self.list_stubs(authority)
        removed = 0
        for path in stubs:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue

        if authority:
            keys_to_remove = [k for k in self._class_cache if k[0] == authority]
        else:
            keys_to_remove = list(self._class_cache.keys())
        for key in keys_to_remove:
            del self._class_cache[key]

        self._clean_empty_dirs()
        return removed

    def clear_stub(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> bool:
        """Remove a specific schema module file."""
        path = self._module_path(name, version, authority)
        if path.exists():
            try:
                path.unlink()
                cache_key = (authority, name, version)
                if cache_key in self._class_cache:
                    del self._class_cache[cache_key]
                return True
            except OSError:
                return False
        return False


class LensStubManager(_StubWriter):
    """Manages automatic generation of Python modules for decoded lenses.

    Lens stubs follow the same directory layout as schema stubs but use
    a ``lens_`` filename prefix to avoid collisions.

    Args:
        stub_dir: Directory to write module files. Defaults to ``~/.atdata/stubs/``.

    Examples:
        >>> manager = LensStubManager()
        >>> path = manager.ensure_lens_stub({"name": "my_lens", "version": "1.0.0"})
    """

    def __init__(self, stub_dir: Optional[Union[str, Path]] = None):
        super().__init__(Path(stub_dir) if stub_dir else DEFAULT_STUB_DIR)

    def _lens_module_filename(self, name: str, version: str) -> str:
        """Generate lens module filename.

        Args:
            name: Lens name (e.g., "image_to_grayscale")
            version: Lens version (e.g., "1.0.0")

        Returns:
            Filename like "lens_image_to_grayscale_1_0_0.py"
        """
        safe_version = version.replace(".", "_")
        return f"lens_{name}_{safe_version}.py"

    def _lens_module_path(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> Path:
        """Get full path to lens module file."""
        return self._stub_dir / authority / self._lens_module_filename(name, version)

    def ensure_lens_stub(self, record: dict) -> Optional[Path]:
        """Ensure a lens stub file exists for the given lens record.

        Args:
            record: Lens record dict with 'name', 'version', etc.

        Returns:
            Path to the lens stub file, or None if record is missing required fields.
        """
        name = record.get("name")
        version = record.get("version", "1.0.0")

        if not name:
            return None

        authority = DEFAULT_AUTHORITY
        path = self._lens_module_path(name, version, authority)

        if path.exists():
            return path

        content = generate_lens_stub(record)
        self._write_module_atomic(path, content, authority)

        return path

    def list_lens_stubs(self, authority: Optional[str] = None) -> list[Path]:
        """List all lens stub files in the stub directory."""
        if not self._stub_dir.exists():
            return []

        pattern = "lens_*.py"
        if authority:
            authority_dir = self._stub_dir / authority
            if not authority_dir.exists():
                return []
            return list(authority_dir.glob(pattern))

        return [
            p for p in self._stub_dir.glob(f"**/{pattern}") if p.name != "__init__.py"
        ]


class StubManager:
    """Backward-compatible facade composing SchemaStubManager and LensStubManager.

    New code should use ``SchemaStubManager`` or ``LensStubManager`` directly.

    Args:
        stub_dir: Directory to write module files. Defaults to ``~/.atdata/stubs/``.

    Examples:
        >>> manager = StubManager()
        >>> schema_dict = {"name": "MySample", "version": "1.0.0", "fields": [...]}
        >>> SampleClass = manager.ensure_module(schema_dict)
        >>> print(manager.stub_dir)
        /Users/you/.atdata/stubs
    """

    def __init__(self, stub_dir: Optional[Union[str, Path]] = None):
        self._schemas = SchemaStubManager(stub_dir)
        self._lenses = LensStubManager(stub_dir)

    def __getattr__(self, name: str):
        """Delegate attribute access to the schema sub-manager for backward compat."""
        return getattr(self._schemas, name)

    @property
    def stub_dir(self) -> Path:
        """The directory where module files are written."""
        return self._schemas.stub_dir

    # Schema delegation
    def ensure_stub(self, schema: dict) -> Optional[Path]:
        """Ensure a schema module file exists. See :meth:`SchemaStubManager.ensure_stub`."""
        return self._schemas.ensure_stub(schema)

    def ensure_module(self, schema: dict) -> Optional[Type]:
        """Ensure a schema module exists and return the class. See :meth:`SchemaStubManager.ensure_module`."""
        return self._schemas.ensure_module(schema)

    def get_stub_path(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> Optional[Path]:
        """Get path to an existing schema stub."""
        return self._schemas.get_stub_path(name, version, authority)

    def list_stubs(self, authority: Optional[str] = None) -> list[Path]:
        """List all schema module files."""
        return self._schemas.list_stubs(authority)

    def clear_stubs(self, authority: Optional[str] = None) -> int:
        """Remove schema module files."""
        return self._schemas.clear_stubs(authority)

    def clear_stub(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> bool:
        """Remove a specific schema module file."""
        return self._schemas.clear_stub(name, version, authority)

    # Lens delegation
    def ensure_lens_stub(self, record: dict) -> Optional[Path]:
        """Ensure a lens stub file exists. See :meth:`LensStubManager.ensure_lens_stub`."""
        return self._lenses.ensure_lens_stub(record)

    def list_lens_stubs(self, authority: Optional[str] = None) -> list[Path]:
        """List all lens stub files."""
        return self._lenses.list_lens_stubs(authority)


__all__ = [
    "StubManager",
    "SchemaStubManager",
    "LensStubManager",
    "DEFAULT_STUB_DIR",
    "DEFAULT_AUTHORITY",
]

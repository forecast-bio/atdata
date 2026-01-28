"""Module manager for automatic Python module generation.

This module provides automatic generation and management of Python modules
for dynamically decoded schema types. When enabled, modules are generated
on schema access to provide IDE autocomplete and type checking support.

Unlike simple .pyi stubs, the generated modules are actual Python code that
can be imported at runtime. This allows ``decode_schema`` to return properly
typed classes that work with both static type checkers and runtime.

Examples:
    >>> from atdata.local import Index
    >>>
    >>> # Enable auto-stub generation
    >>> index = Index(auto_stubs=True)
    >>>
    >>> # Modules are generated automatically on decode_schema
    >>> MyType = index.decode_schema("atdata://local/sampleSchema/MySample@1.0.0")
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
        schema_ref: Schema ref like "atdata://local/sampleSchema/Name@1.0.0"
            or "atdata://alice.bsky.social/sampleSchema/Name@1.0.0"

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


class StubManager:
    """Manages automatic generation of Python modules for decoded schemas.

    The StubManager handles:
    - Determining module file paths from schema metadata
    - Checking if modules exist and are current
    - Generating modules atomically (write to temp, rename)
    - Creating __init__.py files for proper package structure
    - Importing classes from generated modules
    - Cleaning up old modules

    Modules are organized by authority (from the schema ref URI) to avoid
    collisions between schemas with the same name from different sources::

        ~/.atdata/stubs/
            __init__.py
            local/
                __init__.py
                MySample_1_0_0.py
            alice.bsky.social/
                __init__.py
                MySample_1_0_0.py
            did_plc_abc123/
                __init__.py
                OtherSample_2_0_0.py

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
        if stub_dir is None:
            self._stub_dir = DEFAULT_STUB_DIR
        else:
            self._stub_dir = Path(stub_dir)

        self._initialized = False
        self._first_generation = True
        # Cache of imported classes: (authority, name, version) -> class
        self._class_cache: dict[tuple[str, str, str], Type] = {}

    @property
    def stub_dir(self) -> Path:
        """The directory where module files are written."""
        return self._stub_dir

    def _ensure_dir_exists(self) -> None:
        """Create stub directory with __init__.py if it doesn't exist."""
        if not self._initialized:
            self._stub_dir.mkdir(parents=True, exist_ok=True)
            # Create root __init__.py
            init_path = self._stub_dir / "__init__.py"
            if not init_path.exists():
                init_path.write_text('"""Auto-generated atdata schema modules."""\n')
            self._initialized = True

    def _module_filename(self, name: str, version: str) -> str:
        """Generate module filename from schema name and version.

        Replaces dots in version with underscores to avoid confusion
        with file extensions.

        Args:
            name: Schema name (e.g., "MySample")
            version: Schema version (e.g., "1.0.0")

        Returns:
            Filename like "MySample_1_0_0.py"
        """
        safe_version = version.replace(".", "_")
        return f"{name}_{safe_version}.py"

    def _stub_filename(self, name: str, version: str) -> str:
        """Alias for _module_filename for backwards compatibility."""
        return self._module_filename(name, version)

    def _module_path(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> Path:
        """Get full path to module file for a schema.

        Args:
            name: Schema name
            version: Schema version
            authority: Authority from schema ref (e.g., "local", "alice.bsky.social")

        Returns:
            Path like ~/.atdata/stubs/local/MySample_1_0_0.py
        """
        return self._stub_dir / authority / self._module_filename(name, version)

    def _stub_path(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> Path:
        """Alias for _module_path for backwards compatibility."""
        return self._module_path(name, version, authority)

    def _module_is_current(self, path: Path, version: str) -> bool:
        """Check if an existing module file matches the expected version.

        Reads the module docstring to extract the version and compares
        it to the expected version.

        Args:
            path: Path to the module file
            version: Expected schema version

        Returns:
            True if module exists and version matches
        """
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(500)  # Read first 500 chars for docstring
                match = _VERSION_PATTERN.search(content)
                if match:
                    return match.group(1) == version
            return False
        except (OSError, IOError):
            return False

    def _stub_is_current(self, path: Path, version: str) -> bool:
        """Alias for _module_is_current for backwards compatibility."""
        return self._module_is_current(path, version)

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
            dir=path.parent,  # Use parent dir (authority subdir) for atomic rename
        )
        temp_path = Path(temp_path)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                # Try to get exclusive lock (non-blocking, ignore if unavailable)
                # File locking is best-effort - not all filesystems support it
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, IOError):
                    # Lock unavailable (NFS, Windows, etc.) - proceed without lock
                    # Atomic rename provides the real protection
                    pass

                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (on POSIX systems)
            temp_path.rename(path)

        except Exception:
            # Clean up temp file on error - best effort, ignore failures
            try:
                temp_path.unlink()
            except OSError:
                pass  # Temp file cleanup failed, re-raising original error
            raise

    def _write_stub_atomic(self, path: Path, content: str) -> None:
        """Legacy method - extracts authority from path and calls _write_module_atomic."""
        # Extract authority from path (parent directory name)
        authority = path.parent.name
        self._write_module_atomic(path, content, authority)

    def ensure_stub(self, schema: dict) -> Optional[Path]:
        """Ensure a module file exists for the given schema.

        If a current module already exists, returns its path without
        regenerating. Otherwise, generates the module and writes it.

        Modules are namespaced by the authority from the schema's $ref URI
        to avoid collisions between schemas with the same name from
        different sources.

        Args:
            schema: Schema dict with 'name', 'version', and 'fields' keys.
                Can also be a LocalSchemaRecord (supports dict-style access).
                Should include '$ref' for proper namespacing.

        Returns:
            Path to the module file, or None if schema is missing required fields.
        """
        # Extract schema metadata (works with dict or LocalSchemaRecord)
        name = schema.get("name") if hasattr(schema, "get") else None
        version = schema.get("version", "1.0.0") if hasattr(schema, "get") else "1.0.0"
        schema_ref = schema.get("$ref") if hasattr(schema, "get") else None

        if not name:
            return None

        # Extract authority from schema ref for namespacing
        authority = _extract_authority(schema_ref)
        path = self._module_path(name, version, authority)

        # Skip if current module exists
        if self._module_is_current(path, version):
            return path

        # Generate and write module
        # Convert to dict if needed for generate_module
        if hasattr(schema, "to_dict"):
            schema_dict = schema.to_dict()
        else:
            schema_dict = schema

        content = generate_module(schema_dict)
        self._write_module_atomic(path, content, authority)

        # Print helpful message on first generation
        if self._first_generation:
            self._first_generation = False
            self._print_ide_hint()

        return path

    def ensure_module(self, schema: dict) -> Optional[Type]:
        """Ensure a module exists and return the class from it.

        This is the primary method for getting a properly-typed class from
        a schema. It generates the module if needed, imports the class,
        and returns it with proper type information.

        Args:
            schema: Schema dict with 'name', 'version', and 'fields' keys.
                Can also be a LocalSchemaRecord (supports dict-style access).
                Should include '$ref' for proper namespacing.

        Returns:
            The PackableSample subclass from the generated module, or None
            if schema is missing required fields.
        """
        # Extract schema metadata
        name = schema.get("name") if hasattr(schema, "get") else None
        version = schema.get("version", "1.0.0") if hasattr(schema, "get") else "1.0.0"
        schema_ref = schema.get("$ref") if hasattr(schema, "get") else None

        if not name:
            return None

        authority = _extract_authority(schema_ref)

        # Check cache first
        cache_key = (authority, name, version)
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        # Ensure module exists
        path = self.ensure_stub(schema)
        if path is None:
            return None

        # Import and cache the class
        cls = self._import_class_from_module(path, name)
        if cls is not None:
            self._class_cache[cache_key] = cls

        return cls

    def _import_class_from_module(
        self, module_path: Path, class_name: str
    ) -> Optional[Type]:
        """Import a class from a generated module file.

        Uses importlib to dynamically load the module and extract the class.

        Args:
            module_path: Path to the .py module file
            class_name: Name of the class to import

        Returns:
            The imported class, or None if import fails
        """
        if not module_path.exists():
            return None

        try:
            # Create a unique module name based on the path
            module_name = f"_atdata_generated_{module_path.stem}"

            # Load the module spec
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                return None

            # Create and execute the module
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get the class from the module
            cls = getattr(module, class_name, None)
            return cls

        except (ModuleNotFoundError, AttributeError, ImportError, OSError):
            # Import failed - return None and let caller fall back to dynamic generation
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
        """Get the path to an existing stub file.

        Args:
            name: Schema name
            version: Schema version
            authority: Authority namespace (default: "local")

        Returns:
            Path if stub exists, None otherwise
        """
        path = self._stub_path(name, version, authority)
        return path if path.exists() else None

    def list_stubs(self, authority: Optional[str] = None) -> list[Path]:
        """List all module files in the stub directory.

        Args:
            authority: If provided, only list modules for this authority.
                If None, lists all modules across all authorities.

        Returns:
            List of paths to existing module files (excludes __init__.py)
        """
        if not self._stub_dir.exists():
            return []

        if authority:
            # List modules for specific authority
            authority_dir = self._stub_dir / authority
            if not authority_dir.exists():
                return []
            return [p for p in authority_dir.glob("*.py") if p.name != "__init__.py"]

        # List all modules across all authorities (recursive, excluding __init__.py)
        return [p for p in self._stub_dir.glob("**/*.py") if p.name != "__init__.py"]

    def clear_stubs(self, authority: Optional[str] = None) -> int:
        """Remove module files from the stub directory.

        Args:
            authority: If provided, only clear modules for this authority.
                If None, clears all modules across all authorities.

        Returns:
            Number of files removed
        """
        stubs = self.list_stubs(authority)
        removed = 0
        for path in stubs:
            try:
                path.unlink()
                removed += 1
            except OSError:
                # File already removed or permission denied - skip and continue
                continue

        # Clear the class cache for removed modules
        if authority:
            keys_to_remove = [k for k in self._class_cache if k[0] == authority]
        else:
            keys_to_remove = list(self._class_cache.keys())
        for key in keys_to_remove:
            del self._class_cache[key]

        # Clean up empty authority directories (including __init__.py)
        if self._stub_dir.exists():
            for subdir in self._stub_dir.iterdir():
                if subdir.is_dir():
                    # Check if only __init__.py remains
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

        return removed

    def clear_stub(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> bool:
        """Remove a specific module file.

        Args:
            name: Schema name
            version: Schema version
            authority: Authority namespace (default: "local")

        Returns:
            True if file was removed, False if it didn't exist
        """
        path = self._stub_path(name, version, authority)
        if path.exists():
            try:
                path.unlink()
                # Clear from class cache
                cache_key = (authority, name, version)
                if cache_key in self._class_cache:
                    del self._class_cache[cache_key]
                return True
            except OSError:
                return False
        return False


__all__ = [
    "StubManager",
    "DEFAULT_STUB_DIR",
    "DEFAULT_AUTHORITY",
]

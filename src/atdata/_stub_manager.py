"""Stub file manager for automatic .pyi generation.

This module provides automatic generation and management of .pyi stub files
for dynamically decoded schema types. When enabled, stubs are generated
on schema access to provide IDE autocomplete and type checking support.

Example:
    >>> from atdata.local import Index
    >>>
    >>> # Enable auto-stub generation
    >>> index = Index(auto_stubs=True)
    >>>
    >>> # Stubs are generated automatically on get_schema/decode_schema
    >>> schema = index.get_schema("atdata://local/sampleSchema/MySample@1.0.0")
    >>>
    >>> # Get the stub directory path for IDE configuration
    >>> print(f"Add to IDE: {index.stub_dir}")
"""

from pathlib import Path
from typing import Optional, Union
import os
import re
import tempfile
import fcntl

from ._schema_codec import generate_stub


# Default stub directory location
DEFAULT_STUB_DIR = Path.home() / ".atdata" / "stubs"

# Pattern to extract version from stub file header
_VERSION_PATTERN = re.compile(r"^# Schema: .+@(\d+\.\d+\.\d+)")

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
    """Manages automatic generation of .pyi stub files for decoded schemas.

    The StubManager handles:
    - Determining stub file paths from schema metadata
    - Checking if stubs exist and are current
    - Generating stubs atomically (write to temp, rename)
    - Cleaning up old stubs

    Stubs are organized by authority (from the schema ref URI) to avoid
    collisions between schemas with the same name from different sources::

        ~/.atdata/stubs/
            local/
                MySample_1_0_0.pyi
            alice.bsky.social/
                MySample_1_0_0.pyi
            did_plc_abc123/
                OtherSample_2_0_0.pyi

    Args:
        stub_dir: Directory to write stub files. Defaults to ``~/.atdata/stubs/``.

    Example:
        >>> manager = StubManager()
        >>> schema_dict = {"name": "MySample", "version": "1.0.0", "fields": [...]}
        >>> manager.ensure_stub(schema_dict)
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

    @property
    def stub_dir(self) -> Path:
        """The directory where stub files are written."""
        return self._stub_dir

    def _ensure_dir_exists(self) -> None:
        """Create stub directory if it doesn't exist."""
        if not self._initialized:
            self._stub_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def _stub_filename(self, name: str, version: str) -> str:
        """Generate stub filename from schema name and version.

        Replaces dots in version with underscores to avoid confusion
        with file extensions.

        Args:
            name: Schema name (e.g., "MySample")
            version: Schema version (e.g., "1.0.0")

        Returns:
            Filename like "MySample_1_0_0.pyi"
        """
        safe_version = version.replace(".", "_")
        return f"{name}_{safe_version}.pyi"

    def _stub_path(self, name: str, version: str, authority: str = DEFAULT_AUTHORITY) -> Path:
        """Get full path to stub file for a schema.

        Args:
            name: Schema name
            version: Schema version
            authority: Authority from schema ref (e.g., "local", "alice.bsky.social")

        Returns:
            Path like ~/.atdata/stubs/local/MySample_1_0_0.pyi
        """
        return self._stub_dir / authority / self._stub_filename(name, version)

    def _stub_is_current(self, path: Path, version: str) -> bool:
        """Check if an existing stub file matches the expected version.

        Reads the first few lines of the stub to extract the version
        comment and compares it to the expected version.

        Args:
            path: Path to the stub file
            version: Expected schema version

        Returns:
            True if stub exists and version matches
        """
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                # Read first 3 lines to find version comment
                for _ in range(3):
                    line = f.readline()
                    match = _VERSION_PATTERN.match(line)
                    if match:
                        return match.group(1) == version
            return False
        except (OSError, IOError):
            return False

    def _write_stub_atomic(self, path: Path, content: str) -> None:
        """Write stub file atomically using temp file and rename.

        This ensures that concurrent processes won't see partial files.
        Uses file locking for additional safety on systems that support it.

        Args:
            path: Destination path for the stub file
            content: Stub file content to write
        """
        self._ensure_dir_exists()

        # Ensure authority subdirectory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in same directory for atomic rename
        fd, temp_path = tempfile.mkstemp(
            suffix=".pyi.tmp",
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
                    # This is safe because atomic rename provides the real protection
                    _ = None  # Explicit no-op

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
                _ = None  # Temp file cleanup failed, but we're re-raising anyway
            raise

    def ensure_stub(self, schema: dict) -> Optional[Path]:
        """Ensure a stub file exists for the given schema.

        If a current stub already exists, returns its path without
        regenerating. Otherwise, generates the stub and writes it.

        Stubs are namespaced by the authority from the schema's $ref URI
        to avoid collisions between schemas with the same name from
        different sources.

        Args:
            schema: Schema dict with 'name', 'version', and 'fields' keys.
                Can also be a LocalSchemaRecord (supports dict-style access).
                Should include '$ref' for proper namespacing.

        Returns:
            Path to the stub file, or None if schema is missing required fields.
        """
        # Extract schema metadata (works with dict or LocalSchemaRecord)
        name = schema.get("name") if hasattr(schema, "get") else None
        version = schema.get("version", "1.0.0") if hasattr(schema, "get") else "1.0.0"
        schema_ref = schema.get("$ref") if hasattr(schema, "get") else None

        if not name:
            return None

        # Extract authority from schema ref for namespacing
        authority = _extract_authority(schema_ref)
        path = self._stub_path(name, version, authority)

        # Skip if current stub exists
        if self._stub_is_current(path, version):
            return path

        # Generate and write stub
        # Convert to dict if needed for generate_stub
        if hasattr(schema, "to_dict"):
            schema_dict = schema.to_dict()
        else:
            schema_dict = schema

        content = generate_stub(schema_dict)
        self._write_stub_atomic(path, content)

        # Print helpful message on first generation
        if self._first_generation:
            self._first_generation = False
            self._print_ide_hint()

        return path

    def _print_ide_hint(self) -> None:
        """Print a one-time hint about IDE configuration."""
        import sys
        print(
            f"\n[atdata] Generated stub file in: {self._stub_dir}\n"
            f"[atdata] For IDE support, add this path to your type checker:\n"
            f"[atdata]   VS Code/Pylance: Add to python.analysis.extraPaths\n"
            f"[atdata]   PyCharm: Mark as Sources Root\n"
            f"[atdata]   mypy: Add to mypy_path in mypy.ini\n",
            file=sys.stderr,
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
        """List all stub files in the stub directory.

        Args:
            authority: If provided, only list stubs for this authority.
                If None, lists all stubs across all authorities.

        Returns:
            List of paths to existing stub files
        """
        if not self._stub_dir.exists():
            return []

        if authority:
            # List stubs for specific authority
            authority_dir = self._stub_dir / authority
            if not authority_dir.exists():
                return []
            return list(authority_dir.glob("*.pyi"))

        # List all stubs across all authorities (recursive)
        return list(self._stub_dir.glob("**/*.pyi"))

    def clear_stubs(self, authority: Optional[str] = None) -> int:
        """Remove stub files from the stub directory.

        Args:
            authority: If provided, only clear stubs for this authority.
                If None, clears all stubs across all authorities.

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

        # Clean up empty authority directories
        if self._stub_dir.exists():
            for subdir in self._stub_dir.iterdir():
                if subdir.is_dir() and not any(subdir.iterdir()):
                    try:
                        subdir.rmdir()
                    except OSError:
                        continue

        return removed

    def clear_stub(
        self, name: str, version: str, authority: str = DEFAULT_AUTHORITY
    ) -> bool:
        """Remove a specific stub file.

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
                return True
            except OSError:
                return False
        return False


__all__ = [
    "StubManager",
    "DEFAULT_STUB_DIR",
    "DEFAULT_AUTHORITY",
]

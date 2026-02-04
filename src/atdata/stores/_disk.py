"""Local filesystem data store for WebDataset shards.

Writes and reads WebDataset tar archives on the local filesystem,
implementing the ``AbstractDataStore`` protocol.

Examples:
    >>> store = LocalDiskStore(root="~/.atdata/data")
    >>> urls = store.write_shards(dataset, prefix="mnist/v1")
    >>> print(urls[0])
    /home/user/.atdata/data/mnist/v1/data--a1b2c3--000000.tar
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import webdataset as wds

if TYPE_CHECKING:
    from atdata.dataset import Dataset


class LocalDiskStore:
    """Local filesystem data store.

    Writes WebDataset shards to a directory on disk. Implements the
    ``AbstractDataStore`` protocol for use with ``Index``.

    Args:
        root: Root directory for shard storage. Defaults to
            ``~/.atdata/data/``. Created automatically if it does
            not exist.

    Examples:
        >>> store = LocalDiskStore()
        >>> urls = store.write_shards(dataset, prefix="my-dataset")
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = Path.home() / ".atdata" / "data"
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Root directory for shard storage."""
        return self._root

    def write_shards(
        self,
        ds: "Dataset",
        *,
        prefix: str,
        **kwargs: Any,
    ) -> list[str]:
        """Write dataset shards to the local filesystem.

        Args:
            ds: The Dataset to write.
            prefix: Path prefix within root (e.g., ``'datasets/mnist/v1'``).
            **kwargs: Additional args passed to ``wds.writer.ShardWriter``
                (e.g., ``maxcount``, ``maxsize``).

        Returns:
            List of absolute file paths for the written shards.

        Raises:
            RuntimeError: If no shards were written.
        """
        from atdata._logging import get_logger, log_operation

        log = get_logger()
        shard_dir = self._root / prefix
        shard_dir.mkdir(parents=True, exist_ok=True)

        new_uuid = str(uuid4())[:8]
        shard_pattern = str(shard_dir / f"data--{new_uuid}--%06d.tar")

        from atdata._helpers import ShardWriteResult, sha256_file

        written_shards: list[str] = []
        checksums: dict[str, str] = {}

        def _track_shard(path: str) -> None:
            resolved = str(Path(path).resolve())
            written_shards.append(resolved)
            checksums[resolved] = sha256_file(resolved)

        # Filter out kwargs that are specific to other stores (e.g. S3)
        # and not understood by wds.writer.ShardWriter / TarWriter.
        writer_kwargs = {k: v for k, v in kwargs.items() if k not in ("cache_local",)}

        with log_operation("LocalDiskStore.write_shards", prefix=prefix):
            with wds.writer.ShardWriter(
                shard_pattern,
                post=_track_shard,
                **writer_kwargs,
            ) as sink:
                for sample in ds.ordered(batch_size=None):
                    sink.write(sample.as_wds)

            if not written_shards:
                raise RuntimeError(
                    f"No shards written for prefix {prefix!r} in {self._root}"
                )

            log.info(
                "LocalDiskStore.write_shards: wrote %d shard(s)", len(written_shards)
            )

        return ShardWriteResult(written_shards, checksums)

    def read_url(self, url: str) -> str:
        """Resolve a storage URL for reading.

        Local filesystem paths are returned as-is since WebDataset
        can read them directly.

        Args:
            url: Absolute file path to a shard.

        Returns:
            The same path, unchanged.
        """
        return url

    def supports_streaming(self) -> bool:
        """Whether this store supports streaming reads.

        Returns:
            ``True`` — local filesystem supports streaming.
        """
        return True

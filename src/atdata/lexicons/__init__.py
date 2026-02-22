"""ATProto Lexicon definitions for the atdata federation.

This package contains the canonical Lexicon JSON files for the
``science.alt.dataset`` namespace. These define the ATProto record
types used by atdata for publishing schemas, datasets, and lenses
to the AT Protocol network.

Lexicons:
    science.alt.dataset.schema
        Versioned sample type definitions (PackableSample schemas).
    science.alt.dataset.entry
        Dataset index records pointing to WebDataset storage.
    science.alt.dataset.lens
        Bidirectional transformations between schemas.
    science.alt.dataset.label
        Named labels pointing to dataset entries with optional versioning.
    science.alt.dataset.schemaType
        Extensible token for schema format identifiers.
    science.alt.dataset.arrayFormat
        Extensible token for array serialization formats.
    science.alt.dataset.storageHttp
        HTTP/HTTPS URL-based storage with per-shard checksums.
    science.alt.dataset.storageS3
        S3/S3-compatible object storage with per-shard checksums.
    science.alt.dataset.storageBlobs
        ATProto PDS blob-based storage.
    science.alt.dataset.storageExternal
        (Deprecated) External URL-based storage.
    science.alt.dataset.resolveSchema
        XRPC query for resolving a schema by NSID, optionally pinned to a version.
    science.alt.dataset.resolveLabel
        XRPC query for resolving a named dataset label.

The ``ndarray_shim.json`` file defines the standard NDArray type
for use within JSON Schema definitions.

Examples:
    >>> from atdata.lexicons import load_lexicon
    >>> schema_lex = load_lexicon("science.alt.dataset.schema")
    >>> schema_lex["id"]
    'science.alt.dataset.schema'
"""

import json
from importlib import resources
from functools import lru_cache
from typing import Any


NAMESPACE = "science.alt.dataset"

LEXICON_IDS = (
    f"{NAMESPACE}.schema",
    f"{NAMESPACE}.entry",
    f"{NAMESPACE}.lens",
    f"{NAMESPACE}.label",
    f"{NAMESPACE}.schemaType",
    f"{NAMESPACE}.arrayFormat",
    f"{NAMESPACE}.storageHttp",
    f"{NAMESPACE}.storageS3",
    f"{NAMESPACE}.storageBlobs",
    f"{NAMESPACE}.storageExternal",  # deprecated
    f"{NAMESPACE}.resolveSchema",
    f"{NAMESPACE}.resolveLabel",
)


@lru_cache(maxsize=16)
def load_lexicon(lexicon_id: str) -> dict[str, Any]:
    """Load a lexicon definition by its NSID.

    Uses the NSID-to-path convention: dots become directory separators,
    with the final segment as the filename.  For example,
    ``science.alt.dataset.schema`` resolves to
    ``science/alt/dataset/schema.json``.

    Args:
        lexicon_id: The lexicon NSID, e.g. ``"science.alt.dataset.schema"``.

    Returns:
        Parsed JSON dictionary containing the lexicon definition.

    Raises:
        FileNotFoundError: If no lexicon file exists for the given ID.

    Examples:
        >>> lex = load_lexicon("science.alt.dataset.schema")
        >>> lex["defs"]["main"]["type"]
        'record'
    """
    # Convert NSID to path: science.alt.dataset.schema → science/alt/dataset/schema.json
    parts = lexicon_id.split(".")
    path_parts = parts[:-1]  # directory components
    filename = f"{parts[-1]}.json"  # final segment + .json

    ref = resources.files(__package__)
    for part in path_parts:
        ref = ref.joinpath(part)
    ref = ref.joinpath(filename)

    try:
        text = ref.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No lexicon file found for '{lexicon_id}'. "
            f"Expected {'/'.join(path_parts)}/{filename} in {__package__}."
        ) from None
    return json.loads(text)


@lru_cache(maxsize=1)
def load_ndarray_shim() -> dict[str, Any]:
    """Load the NDArray JSON Schema shim definition.

    Returns:
        Parsed JSON dictionary containing the NDArray shim schema.

    Examples:
        >>> shim = load_ndarray_shim()
        >>> shim["$defs"]["ndarray"]["type"]
        'string'
    """
    ref = resources.files(__package__).joinpath("ndarray_shim.json")
    return json.loads(ref.read_text(encoding="utf-8"))


def list_lexicons() -> tuple[str, ...]:
    """Return the tuple of all known lexicon NSIDs.

    Returns:
        Tuple of lexicon ID strings.

    Examples:
        >>> "science.alt.dataset.schema" in list_lexicons()
        True
    """
    return LEXICON_IDS


__all__ = [
    "NAMESPACE",
    "LEXICON_IDS",
    "load_lexicon",
    "load_ndarray_shim",
    "list_lexicons",
]

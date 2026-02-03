"""ATProto Lexicon definitions for the atdata federation.

This package contains the canonical Lexicon JSON files for the
``ac.foundation.dataset`` namespace. These define the ATProto record
types used by atdata for publishing schemas, datasets, and lenses
to the AT Protocol network.

Lexicons:
    ac.foundation.dataset.schema
        Versioned sample type definitions (PackableSample schemas).
    ac.foundation.dataset.record
        Dataset index records pointing to WebDataset storage.
    ac.foundation.dataset.lens
        Bidirectional transformations between schemas.
    ac.foundation.dataset.schemaType
        Extensible token for schema format identifiers.
    ac.foundation.dataset.arrayFormat
        Extensible token for array serialization formats.
    ac.foundation.dataset.storageExternal
        External URL-based storage (S3, HTTP, IPFS).
    ac.foundation.dataset.storageBlobs
        ATProto PDS blob-based storage.
    ac.foundation.dataset.getLatestSchema
        XRPC query for fetching the latest schema version.

The ``ndarray_shim.json`` file defines the standard NDArray type
for use within JSON Schema definitions.

Examples:
    >>> from atdata.lexicons import load_lexicon
    >>> schema_lex = load_lexicon("ac.foundation.dataset.schema")
    >>> schema_lex["id"]
    'ac.foundation.dataset.schema'
"""

import json
from importlib import resources
from functools import lru_cache
from typing import Any


NAMESPACE = "ac.foundation.dataset"

LEXICON_IDS = (
    f"{NAMESPACE}.schema",
    f"{NAMESPACE}.record",
    f"{NAMESPACE}.lens",
    f"{NAMESPACE}.schemaType",
    f"{NAMESPACE}.arrayFormat",
    f"{NAMESPACE}.storageExternal",
    f"{NAMESPACE}.storageBlobs",
    f"{NAMESPACE}.getLatestSchema",
)


@lru_cache(maxsize=16)
def load_lexicon(lexicon_id: str) -> dict[str, Any]:
    """Load a lexicon definition by its NSID.

    Args:
        lexicon_id: The lexicon NSID, e.g. ``"ac.foundation.dataset.schema"``.

    Returns:
        Parsed JSON dictionary containing the lexicon definition.

    Raises:
        FileNotFoundError: If no lexicon file exists for the given ID.

    Examples:
        >>> lex = load_lexicon("ac.foundation.dataset.schema")
        >>> lex["defs"]["main"]["type"]
        'record'
    """
    filename = f"{lexicon_id}.json"
    ref = resources.files(__package__).joinpath(filename)
    try:
        text = ref.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No lexicon file found for '{lexicon_id}'. "
            f"Expected {filename} in {__package__}."
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
        >>> "ac.foundation.dataset.schema" in list_lexicons()
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

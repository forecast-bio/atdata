---
title: "Open Work Stream Context and Technical Notes"
tags: ["backlog", "technical"]
sources: []
contributors: ["maxine--futura"]
created: 2026-03-31
updated: 2026-03-31
---

# Open Work Stream Context

Technical context for the open issues migrated from chainlink. This page preserves investigation notes and implementation plans from the original issues.

## @handle/TypeName@version Schema Resolution (crosslink #2)

The schema resolution system currently supports three ref formats:
- \`atdata://local/schema/Name@version\` (local SQLite)
- \`local://schemas/module.Class@version\` (Python type)
- \`at://did:plc:.../ac.foundation.dataset.schema/rkey\` (ATProto record)

The proposed @handle format (e.g., \`@foundation.ac/MnistSample@1.0.0\`) mirrors the existing \`@handle/dataset-name\` pattern in load_dataset. 

**Implementation path:**
1. Detect \`@handle/\` prefix in \`_parse_schema_ref()\` (src/atdata/index/_schema.py:272)
2. Resolve handle → DID via atmosphere backend
3. Query \`list_records\` on \`ac.foundation.dataset.schema\` filtered by name+version
4. Return matching schema record

**Key files:** \`_schema.py\`, \`_index.py\` (get_schema at L1378, get_schema_type at L1459), \`repository.py\` (get_schema at L373), \`atmosphere/__init__.py\` (get_schema at L336)

## Cross-Account Record Resolution Bug (crosslink #3)

**Root cause:** \`Atmosphere.get_record()\` always sends requests to the authenticated user's PDS. When fetching records from a different DID, the request goes to the wrong PDS. Additionally, the anonymous client defaults to bsky.social AppView, which doesn't index custom lexicon records.

**Fix options (from investigation):**
1. Resolve target DID's PDS via plc.directory and query directly
2. Always route read-only getRecord through AppView
3. Detect foreign DID and route through AppView

Option 1 is most correct. Option 2 depends on AppView indexing custom lexicons (currently doesn't). Option 3 has the same AppView limitation.

## Manifest Property in Dataset Lexicon (crosslink #4)

GH Issue #62. Add optional \`manifests\` array to \`ac.foundation.dataset.record\` lexicon.

**Implementation:**
1. Update \`ac.foundation.dataset.record.json\` with \`shardManifestRef\` definition
2. Add \`ShardManifestRef\` to \`_lexicon_types.py\` (LexDatasetRecord)
3. Round-trip serialization tests
4. Update atmosphere publishers/loaders

## Lexicon Namespace Migration (crosslink #6)

Moving from \`ac.foundation.dataset\` → \`science.alt.dataset\`. This affects:
- All lexicon JSON files
- Record type references in atmosphere module
- NSID constants throughout codebase
- Test fixtures and mocks

The lexicons will be consumed from the separate \`atdata-lexicon\` repo rather than bundled.

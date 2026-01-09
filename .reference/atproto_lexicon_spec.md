# Lexicon Specification - AT Protocol

## Overview

"Lexicon is a schema definition language used to describe atproto records, HTTP endpoints (XRPC), and event stream messages."

The language builds on the atproto Data Model and incorporates concepts similar to JSON Schema and OpenAPI, while adding protocol-specific features. This specification covers version 1 of the Lexicon language.

## Type Categories

Lexicon types fall into several categories:

**Concrete Types:** boolean, integer, string, bytes, cid-link, blob

**Container Types:** array, object

**Sub-types:** params, permission

**Meta Types:** token, ref, union, unknown

**Primary Types:** record, query, procedure, subscription, permission-set

## Lexicon Files

Lexicon schemas are JSON files associated with a single NSID containing one or more definitions. Required file fields:

- `lexicon` (integer): Language version, currently fixed at 1
- `id` (string): The NSID identifier
- `defs` (object): Named definitions with distinct keys
- `description` (string, optional): Overview text

"References to specific definitions within a Lexicon use fragment syntax, like `com.example.defs#someView`."

## Primary Type Definitions

### Record Type

Specifies data objects stored in repositories. Type-specific fields:

- `key` (string): Record key type specification
- `record` (object): Schema with type object describing the record structure

### Query and Procedure (HTTP API)

Describes XRPC endpoints. Fields:

- `parameters`: Optional params schema for query parameters
- `output`: Response body with encoding (MIME type) and optional schema
- `input`: Request body (procedures only)
- `errors`: Array of possible error codes with descriptions

### Subscription (Event Stream)

Defines WebSocket endpoint messages. Fields:

- `parameters`: Optional HTTP parameters
- `message`: Required specification with union schema
- `errors`: Optional error definitions

"Subscription schemas must be a `union` of refs, not an `object` type."

### Permission Set

Bundles permissions for OAuth scopes. Fields:

- `title` / `title:langs`: Display name with localization
- `detail` / `detail:langs`: Human-readable scope description
- `permissions`: Array of permission definitions

## Field Type Definitions

### Primitive Types

**boolean:** Optional `default` and `const` fields

**integer:** Supports `minimum`, `maximum`, `enum`, `default`, `const`

**string:** Supports `format`, `maxLength`, `minLength`, `maxGraphemes`, `minGraphemes`, `knownValues`, `enum`, `default`, `const`

"Strings are Unicode. For non-Unicode encodings, use `bytes` instead."

**bytes:** Raw binary data with optional `minLength` and `maxLength`

**cid-link:** Content identifier links with no type-specific fields

### Container Types

**array:** Contains `items` (required schema) and optional `minLength`/`maxLength`

**object:**
- `properties`: Named field schemas
- `required`: Array of required field names
- `nullable`: Array of fields accepting null values

"There is a semantic difference in data between omitting a field; including the field with value `null`; and including the field with a falsy value."

**blob:** Binary large objects with:
- `accept`: MIME type restrictions (glob patterns supported)
- `maxSize`: Maximum bytes

### Specialized Types

**params:** Limited to HTTP query parameters, supporting only boolean, integer, string, or arrays of these types. Cannot be top-level named definitions.

**permission:** Defines access permissions with `resource` field. Current resources:

- `repo`: Repository write permissions with collection and optional action fields
- `rpc`: Remote API calls with lxm (endpoints), aud (audience), and inheritAud fields

"Permission declarations with unsupported resource types must be ignored by services implementing access control."

**token:** Empty values referenced by name, used for symbolic enumerations. Cannot be used in refs, unions, or as object fields.

### Reference and Union Types

**ref:** References another schema definition globally (by NSID) or locally (by fragment). Reduces schema duplication for reusable definitions.

**union:** Declares multiple possible types at a location. Fields:

- `refs`: Array of schema references
- `closed`: Boolean indicating if type list is fixed (default: false)

"Unions represent that multiple possible types could be present at this location in the schema."

**unknown:** Accepts any data object with no specific validation, but must be a CBOR map. Data may contain optional `$type` field.

## String Formats

Lexicon supports format-constrained strings:

- `at-identifier`: Handle or DID
- `at-uri`: AT-URI reference
- `at-uri-regex`: "Lenient" version accepting unresolved at-identifier
- `cid`: Content identifier
- `datetime`: RFC 3339 timestamp
- `did`: Decentralized identifier
- `handle`: Handle identifier
- `nsid`: Namespaced identifier
- `tid`: Timestamp identifier
- `record-key`: Record key syntax
- `uri`: Generic URI (RFC 3986)
- `language`: IETF language tag (BCP 47)

### Datetime Format

Required elements:
- Intersection of RFC 3339, ISO 8601, and WHATWG HTML standards
- Uppercase T separator between date and time
- Timezone specification (preferably Z for UTC)
- Whole seconds precision (millisecond precision recommended)

Valid example: `1985-04-12T23:20:50.123Z`

Invalid: Missing timezone, lowercase t, insufficient precision, or invalid day/month values

### AT-URI Format

"at-uri": Represents an AT-URI following the AT-URI scheme specification. Examples:
- `at://did:plc:abc123/com.example.record/rkey123`
- `at://alice.bsky.social/app.bsky.feed.post/3k4i5j6k`

"at-uri-regex": "Lenient" version that accepts AT-URIs with unresolved at-identifiers.

### URI Format

"uri": "Flexible to any URI schema, following the generic RFC-3986 on URIs." Supports did, https, wss, ipfs, dns, and at schemes. Maximum length is 8 KBytes.

### Language Format

"language": "An IETF Language Tag string, compliant with BCP 47, defined in RFC 5646." Examples include `ja` (Japanese) and `pt-BR` (Brazilian Portuguese).

## Validation Approach

"For the various identifier formats, when doing Lexicon schema validation the most expansive identifier syntax format should be permitted." Application-level validation of specific identifier methods occurs separately from schema validation.

## When to Use `$type`

Data objects sometimes require a `$type` field for disambiguation:

- `record` objects: Always include `$type`
- `union` variants: Always include `$type` (except top-level subscription messages)
- `blob` objects: Always include `$type`

"Main types must be referenced in `$type` fields as just the NSID, not including a `#main` suffix."

## Validation Options

Three PDS validation approaches:

1. **Explicit validation:** Record must validate against known Lexicon; fails if unavailable
2. **No validation:** Record bypasses Lexicon validation (still validates data model rules)
3. **Optimistic validation (default):** Validates if Lexicon known; allows creation if unavailable

## Lexicon Evolution

Compatibility rules for schema updates:

- New fields must be optional
- Non-optional fields cannot be removed
- Field types cannot change
- Fields cannot be renamed

"If larger breaking changes are necessary, a new Lexicon name must be used."

Lexicon publication occurs through atproto repositories using `com.atproto.lexicon.schema` record types, linked via DNS TXT records for authority resolution.

## Authority and Control

NSID authority derives from DNS domain control. Domain authorities maintain Lexicon definitions with ultimate responsibility for maintenance and distribution. Protocol implementations should treat data failing Lexicon validation as entirely invalid.

"Unexpected fields in data which otherwise conforms to the Lexicon should be ignored."

## Usage Guidelines

Implementations should support translation to JSON Schema and OpenAPI formats for cross-ecosystem compatibility. Care must be taken when deserializing/reserializing to avoid losing unexpected fields that may represent newer schema versions.

## Record Key Types

The `key` field in record definitions specifies the format of record keys (rkeys). Options:

- `"any"`: Any string matching general record-key syntax
- `"tid"`: Must be a valid timestamp identifier
- `"literal:{value}"`: Fixed literal string (e.g., `"literal:self"` for profile records)

## Notes on Implementation

- String grapheme counting should follow Unicode extended grapheme cluster boundaries
- Unknown fields should be preserved during serialization/deserialization when possible
- Services should be permissive with format validation but strict with structural requirements
- Breaking schema changes require new NSIDs rather than version updates

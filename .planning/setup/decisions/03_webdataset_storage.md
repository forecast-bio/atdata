# Decision: WebDataset Storage Strategy

**Issue**: #47
**Status**: Needs decision
**Blocks**: #50 (Lexicon validation)
**Priority**: Critical for Phase 1

## DECISION

Let's build the hybrid approach in from the beginning. Critically:

* We'll keep track of whether dataset index records are referencing an external storage (S3, R2, etc) by URL or a PDS blob using an open union to define the data location
* In the AppView implementation, we can proxy WDS urls for datasets across individual stored blobs, which streamlines some of the design.

This will help us be robust from the start -- particularly for those self-hosting.

---

## Problem Statement

We need to decide where the actual WebDataset `.tar` files are stored and how dataset records reference them. This affects decentralization, reliability, and scalability.

## Context

WebDataset files are:
- **Large**: Typically gigabytes to terabytes
- **Immutable**: Once created, datasets rarely change
- **Sharded**: Split across multiple `.tar` files (e.g., `data-{000000..000099}.tar`)
- **Binary**: Contain msgpack-serialized samples with images/arrays

Current `atdata` usage:
```python
# External storage (S3, HTTP, etc.)
dataset = Dataset[MySample](url="s3://bucket/data-{000000..000009}.tar")
```

## Options

### Option 1: External Storage with URL References ⭐ RECOMMENDED (Phase 1)

**Description**: Store WebDataset files on existing storage (S3, HTTP, IPFS, etc.), record only contains URLs

**Record Format**:
```json
{
  "$type": "app.bsky.atdata.dataset",
  "name": "CIFAR-10 Training Set",
  "urls": [
    "s3://my-bucket/cifar10-train-{000000..000049}.tar"
  ],
  "schemaRef": "at://alice/schema/image",
  ...
}
```

**Supported URL Schemes**:
- `s3://` - AWS S3 and compatible (MinIO, DigitalOcean Spaces)
- `https://` - HTTP/HTTPS servers
- `gs://` - Google Cloud Storage
- `ipfs://` - IPFS (decentralized, content-addressed)
- `file://` - Local files (for development)

**Pros**:
- ✅ **No size limits**: Store datasets of any size
- ✅ **Existing infrastructure**: Leverage proven storage solutions
- ✅ **No ATProto storage costs**: Publishers pay for their own storage
- ✅ **Performance**: Use CDNs, regional endpoints, etc.
- ✅ **Compatibility**: Works with current `atdata` code
- ✅ **Flexibility**: Different storage for different use cases

**Cons**:
- ❌ **Centralization risk**: If storage provider goes down, dataset unavailable
- ❌ **URL rot**: Links can break over time
- ❌ **No permanence guarantee**: Publisher can delete files
- ❌ **Access control complexity**: Need to handle auth for private datasets

**Decentralization**: ⭐⭐ Fair (better with IPFS)
**Reliability**: ⭐⭐⭐ Good (depends on storage provider)
**Cost**: ⭐⭐⭐⭐ Excellent (publishers pay storage costs)
**Implementation Effort**: ⭐⭐⭐⭐⭐ Very Low (already supported)

---

### Option 2: ATProto Blob Storage

**Description**: Store WebDataset files as ATProto blobs, record contains blob CIDs

**Record Format**:
```json
{
  "$type": "app.bsky.atdata.dataset",
  "name": "Small Dataset",
  "blobs": [
    {"$type": "blob", "ref": {"$link": "bafyrei..."}},
    {"$type": "blob", "ref": {"$link": "bafyrei..."}}
  ],
  "schemaRef": "at://alice/schema/image",
  ...
}
```

**Pros**:
- ✅ **True decentralization**: Data lives on ATProto network
- ✅ **Content-addressed**: CIDs guarantee immutability
- ✅ **Permanence**: As permanent as ATProto itself
- ✅ **No external dependencies**: Self-contained

**Cons**:
- ❌ **Size limits**: ATProto may have blob size restrictions (need to verify)
- ❌ **Storage costs**: Who pays for storing large datasets?
- ❌ **Performance**: May be slower than specialized data storage
- ❌ **Scalability**: Not designed for TB-scale datasets
- ❌ **Unknown limitations**: ATProto blob storage is less proven for this use case

**Decentralization**: ⭐⭐⭐⭐⭐ Excellent
**Reliability**: ⭐⭐⭐⭐ Very Good (ATProto network)
**Cost**: ⭐ Poor (storage costs for large datasets)
**Implementation Effort**: ⭐⭐⭐ Medium (need to implement blob upload/download)

---

### Option 3: Hybrid Approach

**Description**: Support both external URLs and ATProto blobs

**Record Format**:
```json
{
  "$type": "app.bsky.atdata.dataset",
  "name": "Hybrid Dataset",
  "storage": {
    "kind": "external",
    "urls": ["s3://bucket/data-{000000..000009}.tar"]
  },
  // OR
  "storage": {
    "kind": "blobs",
    "blobs": [{"$type": "blob", "ref": {"$link": "bafyrei..."}}]
  },
  ...
}
```

**Pros**:
- ✅ Best of both worlds
- ✅ Flexibility for different use cases
- ✅ Can migrate between storage types

**Cons**:
- ❌ More complex Lexicon and implementation
- ❌ Confusing for users (which to choose?)
- ❌ Testing burden (need to test both paths)

**Implementation Effort**: ⭐⭐ High (two systems to maintain)

## Recommendation: Option 1 (External URLs) for Phase 1, Option 3 (Hybrid) for Future

**Rationale**:

1. **Pragmatism**: Most ML datasets are huge (10GB-10TB). ATProto blob storage is not designed for this scale.

2. **Existing Infrastructure**: S3, GCS, HTTP are battle-tested for large file storage. Why reinvent the wheel?

3. **Cost Model**: Publishers pay for their own storage. This is sustainable and aligns incentives.

4. **IPFS for Decentralization**: Users who want decentralization can use `ipfs://` URLs, which are content-addressed and distributed.

5. **Future-Proof**: We can add blob storage later for small datasets (<100MB) without breaking existing datasets.

## Implementation Plan

### Phase 1: External URLs Only

**Lexicon Design**:
```json
{
  "urls": {
    "type": "array",
    "description": "WebDataset URLs (supports brace notation)",
    "items": {
      "type": "string",
      "format": "uri",
      "maxLength": 1000
    },
    "minLength": 1
  }
}
```

**Publisher Implementation**:
```python
publisher = DatasetPublisher(client)
dataset_uri = publisher.publish_dataset(
    dataset,
    name="My Dataset",
    description="Training data for my model"
)
# dataset.url is used directly, no upload needed
```

**Loader Implementation**:
```python
loader = DatasetLoader(client)
dataset = loader.load_dataset("at://alice/dataset/123")
# Creates Dataset with URL from record
# Actual data loading happens lazily via WebDataset
```

**Validation**:
- Check URL format (scheme + netloc + path)
- Support brace notation for sharded datasets
- Don't validate URL accessibility (too slow, may be private)

### Future: Add Blob Storage Option

When ATProto blob storage is more mature and we understand limits:

1. **Add blob support to Lexicon**:
   ```json
   "storage": {
     "type": "union",
     "refs": ["#urlStorage", "#blobStorage"]
   }
   ```

2. **Implement blob upload**:
   - Chunk large files
   - Upload shards as separate blobs
   - Update record with blob CIDs

3. **Size recommendations**:
   - Datasets <100MB → Consider blobs
   - Datasets >100MB → Use external URLs
   - Datasets >10GB → Definitely external URLs

## URL Scheme Support

| Scheme | Support | Notes |
|--------|---------|-------|
| `s3://` | ✅ Phase 1 | AWS S3 and compatible services |
| `https://` | ✅ Phase 1 | Public HTTP/HTTPS servers |
| `http://` | ✅ Phase 1 | Upgraded to HTTPS when possible |
| `gs://` | ✅ Phase 1 | Google Cloud Storage |
| `ipfs://` | ✅ Phase 1 | Decentralized storage via IPFS |
| `file://` | ✅ Phase 1 | Local development only |
| `at://` | ⏳ Future | ATProto blob references |

## Decentralization Strategy

For users who want decentralization without ATProto blobs:

**IPFS + Pinning Services**:
1. Upload dataset to IPFS
2. Pin with service (Pinata, Infura, Web3.Storage)
3. Publish dataset with `ipfs://` URL
4. IPFS ensures content-addressed, distributed storage

**Example**:
```python
# Upload to IPFS (using ipfs client)
ipfs_hash = upload_to_ipfs("data-000000.tar")

# Publish dataset
dataset_uri = publisher.publish_dataset(
    dataset,
    name="My Dataset",
    urls=[f"ipfs://{ipfs_hash}"]
)
```

**Benefits**:
- Content-addressed (CID in URL)
- Distributed (IPFS network)
- Permanent (with pinning)
- No ATProto blob limits

## Access Control Considerations

**Public datasets**: URLs point to public storage
- S3 public buckets
- Public HTTP servers
- IPFS (inherently public)

**Private datasets**: URL points to private storage
- S3 with authentication (pre-signed URLs? credentials?)
- Private HTTP servers (auth tokens?)
- Recommendation: Public datasets only for Phase 1

**Future**: Could add access control metadata to records
```json
{
  "access": {
    "kind": "authenticated",
    "requiredRole": "subscriber"
  }
}
```

## Storage Cost Implications

| Storage Type | Cost Responsibility | Pros | Cons |
|-------------|-------------------|------|------|
| S3 | Publisher | Industry standard, reliable | Ongoing costs |
| IPFS + Pinning | Publisher | Decentralized | Need pinning service |
| HTTP Server | Publisher | Full control | Maintenance burden |
| ATProto Blobs | Publisher? ATProto? | Simple | Unknown cost model |

**Recommendation**: Let publishers choose based on their needs and budget.

## Alternative Approaches Considered

**Torrents**: Use BitTorrent protocol
- Pros: Decentralized, efficient for large files
- Cons: Need seeders, not as well integrated
- Could add in future with `torrent://` scheme

**Arweave**: Permanent storage blockchain
- Pros: True permanence, one-time payment
- Cons: Expensive for large datasets
- Could add in future for critical datasets

## Open Questions

1. **Should we validate URL accessibility when publishing?**
   - Pro: Catch broken links early
   - Con: Slow, may fail for private URLs
   - Recommendation: No validation, trust publishers

2. **Should we mirror datasets automatically?**
   - Could create community mirrors for popular datasets
   - Recommendation: Not for Phase 1, community can organize

3. **What about dataset versioning?**
   - New version = new record with new URLs
   - Could link to previous version in metadata
   - Recommendation: Simple versioning via new records

4. **Should we support multi-region URLs?**
   ```json
   "urls": [
     {"region": "us-east-1", "url": "s3://..."},
     {"region": "eu-west-1", "url": "s3://..."}
   ]
   ```
   - Recommendation: Defer to future if needed

## Success Criteria

After implementing this decision:
- ✅ Datasets can reference external URLs (S3, HTTPS, IPFS)
- ✅ WebDataset brace notation is preserved
- ✅ Loading datasets works with existing `Dataset` class
- ✅ No breaking changes to current `atdata` usage
- ✅ Path clear for future blob storage support

## References

- Lexicon design: `../02_lexicon_design.md` (Dataset Record Lexicon)
- Python client: `../03_python_client.md` (DatasetPublisher/Loader)
- WebDataset documentation: https://webdataset.github.io/webdataset/

---

**Decision Needed By**: Before starting Phase 1 Issue #23 (Dataset Lexicon design)
**Decision Maker**: Project maintainer (max)
**Date Created**: 2026-01-07

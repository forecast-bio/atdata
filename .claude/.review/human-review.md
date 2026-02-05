* We had talked previously about potentially moving `PackableSample` to a `Packable` protocol to simplify some type hints / etc. Let's go over the pros and cons of this. This shows up for the linting / typing of `local.Index.publish_schema`, where the way that @packable is working right now doesn't properly get the PackableSample superclass to register for this signature.

* We have an interesting persistent issue with Redis removing old records; can you think through why it seems like Redis resets our index entries somewhat inconsistently over time? Is there a Redis setting that might be responsible for this?

* We want to make sure that we keep to the pattern that `foo.xs` is an @property that gives a (lazy) iterable for `x`, while `foo.list_xs` is a fully evaluated list for all of the `x`s that uses `foo.xs` under the hood. We should go through the full codebase to evaluate for following this convention.

* `load_dataset` has a couple issues:
    * We updated how `Dataset` is initialized to be able to accommodate a number of different underlying sources of `wds`-compatible data, and we should make it so that the overloads for `load_dataset` properly connect up with this; for example:
        * If `load_dataset` is coming from a specified local index with an S3 store, we should use the S3 credentials there as the source for the `Dataset` returned by `load_dataset`
        * If it's using an atproto location (like `'@maxine.science/mnist'`), the returned `Dataset` should use whatever is the storage mechanism referenced in that atproto record from the network (for example, blobs as at-uris that are wrapped in a file-like interface for passing to `webdataset`).

* Calls like
```python
ds = load_dataset( "@local/proto-text-samples-3", TextSample,
    split = 'train',
    #
    index = index,
)
```
are resulting in linting errors because of the way overloading is handled for `load_dataset`; could you take a look at how the overloads are implemented, to make sure that things are functioning as expected? In particular, it seems like the `AbstractIndex` protocol and `local.Index` don't play nicely for linting!
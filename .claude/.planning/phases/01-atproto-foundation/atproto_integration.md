# Planning for full atproto integration

The overall goal for `atdata` is that the index for datasets is actually present on the atproto distributed repository, with one type of Lexicon schema for actually containing information about `PackableSample` schemas that can be reproduced with code gen, and one type of Lexicon schema designed for the main functionality: records holding the links to the WDS dataset for samples and the msgpack metadata (that can be plugged into the `Dataset` class) as well as a reference to the atproto record containing the schema for the appropriate sample type for the dataset.

## Thoughts on functionality

* Lexicons
    * Definition of a `PackableSample`-compatible sample type schema, that can be used to reconstitute the code in appropriate languages using code gen toolilng
    * Index records that contain links to the actual WebDataset data, as well as to the records with the corresponding sample schema.
    * `Lenses` between defined sample type schemas across the network.
* Python library functionality
    * Logging in with the atproto sdk
    * Posting sample schemas and dataset index records to the appropriate lexicons for the user
* AppView functionality
    * Aggregating index records, making an index of those that is quick to query on 

## Questions for implementation

* What is the best way to store the sample type schemas within atproto Lexicons? I've thought about using JSON schema or protobuf, but want to think through possibilities.
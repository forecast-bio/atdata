"""TODO"""

##
# Imports

from atdata.dataset import (
    Dataset,
)

from uuid import uuid4

from redis_om import (
    EmbeddedJsonModel,
    JsonModel,
    Field,
    Migrator,
)

from typing import (
    Any,
    Optional,
    Dict,
)


##
# Redis object model

class SampleSchema( EmbeddedJsonModel ):
    """TODO"""
    identifier: str = Field( index = True )
    json_schema: Dict[str, Any]

class MetadataSchema( EmbeddedJsonModel ):
    """TODO"""
    identifier: str = Field( index = True )
    json_schema: Dict[str, Any]

class IndexEntry( JsonModel ):
    """TODO"""
    wds_url: str
    sample_schema: SampleSchema
    metadata_schema: Optional[MetadataSchema] = None


##
# Classes

class Index:
    """TODO"""

    ##

    def __init__(self) -> None:
        """TODO"""
        ##
        
        # ...
        
        # Needed before we can do anything with redis-om queries
        Migrator().run()
    
    def list( self ):
        """TODO"""
        ##
        all_entries = IndexEntry.find().all()
        return all_entries

    def add( self, ds: Dataset ) -> IndexEntry:
        """TODO"""
        ##
        test_schema = {
            "$id": "https://schema.dev/fake-schema.schema.json",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "fake-schema",
            "description": "Blue Blah",
            "type": "object",
            "properties": {
                "property_a": {
                "default": 5,
                "type": "integer"
                },
                "property_b": {
                "type": "string"
                }
            }
        }

        test_schema_entry = SampleSchema(
            identifier = str( uuid4() ),
            json_schema = test_schema
        )

        new_index_entry = IndexEntry(
            wds_url = ds.url,
            sample_schema = test_schema_entry,
        )

        return new_index_entry.save()

#
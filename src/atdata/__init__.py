"""A loose federation of distributed, typed datasets"""

##
# Expose components

from .dataset import (
    PackableSample,
    SampleBatch,
    Dataset,
    packable,
)

from .lens import (
    Lens,
    LensNetwork,
    lens,
)


#
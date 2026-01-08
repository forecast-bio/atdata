"""Test lens functionality."""

##
# Imports

import pytest

from dataclasses import dataclass
import webdataset as wds
import atdata

import numpy as np
from numpy.typing import NDArray


##
# Tests

def test_lens():
    """Test a lens between sample types"""

    # Set up the lens scenario

    @atdata.packable
    class Source:
        name: str
        age: int
        height: float

    @atdata.packable
    class View:
        name: str
        height: float
    
    @atdata.lens
    def polite( s: Source ) -> View:
        return View(
            name = s.name,
            height = s.height,
        )
    
    @polite.putter
    def polite_update( v: View, s: Source ) -> Source:
        return Source(
            name = v.name,
            height = v.height,
            #
            age = s.age,
        )
    
    # Test with an example sample

    test_source = Source(
        name = 'Hello World',
        age = 42,
        height = 182.9,
    )
    correct_view = View(
        name = test_source.name,
        height = test_source.height,
    )

    test_view = polite( test_source )
    assert test_view == correct_view, \
        f'Incorrect lens behavior: {test_view}, and not {correct_view}'

    # This lens should be well-behaved

    update_view = View(
        name = 'Now Taller',
        height = 192.9,
    )

    x = polite( polite.put( update_view, test_source ) )
    assert x == update_view, \
        f'Violation of GetPut: {x} =/= {update_view}'
    
    y = polite.put( polite( test_source ), test_source )
    assert y == test_source, \
        f'Violation of PutGet: {y} =/= {test_source}'

    # PutPut law: put(v2, put(v1, s)) = put(v2, s)
    another_view = View(
        name = 'Different Name',
        height = 165.0,
    )
    z1 = polite.put( another_view, polite.put( update_view, test_source ) )
    z2 = polite.put( another_view, test_source )
    assert z1 == z2, \
        f'Violation of PutPut: {z1} =/= {z2}'

def test_conversion( tmp_path ):
    """Test automatic interconversion between sample types"""

    @dataclass
    class Source( atdata.PackableSample ):
        name: str
        height: float
        favorite_pizza: str
        favorite_image: NDArray
    
    @dataclass
    class View( atdata.PackableSample ):
        name: str
        favorite_pizza: str
        favorite_image: NDArray
    
    @atdata.lens
    def polite( s: Source ) -> View:
        return View(
            name = s.name,
            favorite_pizza = s.favorite_pizza,
            favorite_image = s.favorite_image,
        )
    
    # Map a test sample through the view
    test_source = Source(
        name = 'Larry',
        height = 42.,
        favorite_pizza = 'pineapple',
        favorite_image = np.random.randn( 224, 224 )
    )
    test_view = polite( test_source )

    # Create a test dataset

    k_test = 100
    test_filename = (
        tmp_path
        / 'test-source.tar'
    ).as_posix()

    with wds.writer.TarWriter( test_filename ) as dest:
        for i in range( k_test ):
            # Create a new copied sample
            cur_sample = Source(
                name = test_source.name,
                height = test_source.height,
                favorite_pizza = test_source.favorite_pizza,
                favorite_image = test_source.favorite_image,
            )
            dest.write( cur_sample.as_wds )
    
    # Try reading the test dataset

    ds = (
        atdata.Dataset[Source]( test_filename )
            .as_type( View )
    )

    assert ds.sample_type == View, \
        'Auto-mapped'

    sample: View | None = None
    for sample in ds.ordered( batch_size = None ):
        # Load only the first sample
        break

    assert sample is not None, \
        'Did not load any samples from `Source` dataset'

    assert sample.name == test_view.name, \
        f'Divergence on auto-mapped dataset: `name` should be {test_view.name}, but is {sample.name}'
    assert sample.favorite_pizza == test_view.favorite_pizza, \
        f'Divergence on auto-mapped dataset: `favorite_pizza` should be {test_view.favorite_pizza}, but is {sample.favorite_pizza}'
    assert np.all( sample.favorite_image == test_view.favorite_image ), \
        f'Divergence on auto-mapped dataset: `favorite_image`'


##
# Edge case tests for coverage


def test_lens_get_method():
    """Test calling lens.get() explicitly instead of lens()."""
    @atdata.packable
    class GetSource:
        value: int

    @atdata.packable
    class GetView:
        doubled: int

    @atdata.lens
    def doubler(s: GetSource) -> GetView:
        return GetView(doubled=s.value * 2)

    source = GetSource(value=5)

    # Test both calling conventions
    result_call = doubler(source)
    result_get = doubler.get(source)

    assert result_call == result_get
    assert result_get.doubled == 10


def test_lens_trivial_putter():
    """Test lens without explicit putter uses trivial putter."""
    @atdata.packable
    class TrivialSource:
        a: int
        b: str

    @atdata.packable
    class TrivialView:
        a: int

    # Create lens without putter
    @atdata.lens
    def extract_a(s: TrivialSource) -> TrivialView:
        return TrivialView(a=s.a)

    source = TrivialSource(a=10, b="hello")
    view = TrivialView(a=99)

    # Trivial putter should return source unchanged
    result = extract_a.put(view, source)
    assert result == source, "Trivial putter should return source unchanged"


def test_lens_network_missing_lens():
    """Test LensNetwork raises ValueError for unregistered lens."""
    from atdata.lens import LensNetwork

    @atdata.packable
    class UnregisteredSource:
        x: int

    @atdata.packable
    class UnregisteredView:
        y: int

    network = LensNetwork()

    with pytest.raises(ValueError, match="No registered lens"):
        network.transform(UnregisteredSource, UnregisteredView)


##
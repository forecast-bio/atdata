"""Test dataaset functionality."""

##
# Imports

# Tests
import pytest

# System
from dataclasses import dataclass

# External
import numpy as np
import webdataset as wds

# Local
import atdata
import atdata.dataset as atds

# Typing
from numpy.typing import NDArray
from typing import (
    Type,
    Any,
)


##
# Sample test cases

@dataclass
class BasicTestSample( atdata.PackableSample ):
    name: str
    position: int
    value: float

@dataclass
class NumpyTestSample( atdata.PackableSample ):
    label: int
    image: NDArray

@atdata.packable
class BasicTestSampleDecorated:
    name: str
    position: int
    value: float

@atdata.packable
class NumpyTestSampleDecorated:
    label: int
    image: NDArray

test_cases = [
    {
        'SampleType': BasicTestSample,
        'sample_data': {
            'name': 'Hello, world!',
            'position': 42,
            'value': 1024.768,
        },
        'sample_wds_stem': 'basic_test',
        'test_parquet': True,
    },
    {
        'SampleType': NumpyTestSample,
        'sample_data':
        {
            'label': 9_001,
            'image': np.random.randn( 1024, 1024 ),
        },
        'sample_wds_stem': 'numpy_test',
        'test_parquet': False,
    },
    {
        'SampleType': BasicTestSampleDecorated,
        'sample_data': {
            'name': 'Hello, world!',
            'position': 42,
            'value': 1024.768,
        },
        'sample_wds_stem': 'basic_test_decorated',
        'test_parquet': True,
    },
    {
        'SampleType': NumpyTestSampleDecorated,
        'sample_data':
        {
            'label': 9_001,
            'image': np.random.randn( 1024, 1024 ),
        },
        'sample_wds_stem': 'numpy_test_decorated',
        'test_parquet': False,
    },
]


## Tests

@pytest.mark.parametrize(
    ('SampleType', 'sample_data'),
    [ (case['SampleType'], case['sample_data'])
      for case in test_cases ]
)
def test_create_sample(
            SampleType: Type[atdata.PackableSample],
            sample_data: atds.MsgpackRawSample,
        ):
    """Test our ability to create samples from semi-structured data"""

    sample = SampleType.from_data( sample_data )
    assert isinstance( sample, SampleType ), \
        f'Did not properly form sample for test type {SampleType}'

    for k, v in sample_data.items():
        cur_assertion: bool
        if isinstance( v, np.ndarray ):
            cur_assertion = np.all( getattr( sample, k ) == v ) == True
        else:
            cur_assertion = getattr( sample, k ) == v
        assert cur_assertion, \
            f'Did not properly incorporate property {k} of test type {SampleType}'

#

# def test_decorator_syntax():
#     """Test use of decorator syntax for sample types"""
    
#     @atdata.packable
#     class BasicTestSampleDecorated:
#         name: str
#         position: int
#         value: float

#     @atdata.packable
#     class NumpyTestSampleDecorated:
#         label: int
#         image: NDArray
    
#     ##
    
#     test_create_sample( BasicTestSampleDecorated, {
#         'name': 'Hello, world!',
#         'position': 42,
#         'value': 1024.768,
#     } )

#     test_create_sample( NumpyTestSampleDecorated, {
#         'label': 9_001,
#         'image': np.random.randn( 1024, 1024 ),
#     } )

#

@pytest.mark.parametrize(
    ('SampleType', 'sample_data', 'sample_wds_stem'),
    [ (case['SampleType'], case['sample_data'], case['sample_wds_stem'])
      for case in test_cases ]
)
def test_wds(
            SampleType: Type[atdata.PackableSample],
            sample_data: atds.MsgpackRawSample,
            sample_wds_stem: str,
            tmp_path
        ):
    """Test our ability to write samples as `WebDatasets` to disk"""

    ## Testing hyperparameters

    n_copies = 100
    shard_maxcount = 10
    batch_size = 4
    n_iterate = 10

    ## Write sharded dataset

    file_pattern = (
        tmp_path
        / (f'{sample_wds_stem}' + '-{shard_id}.tar')
    ).as_posix()
    file_wds_pattern = file_pattern.format( shard_id = '%06d' )

    with wds.ShardWriter(
        pattern = file_wds_pattern,
        maxcount = shard_maxcount,
    ) as sink:
        
        for i_sample in range( n_copies ):
            new_sample = SampleType.from_data( sample_data )
            assert isinstance( new_sample, SampleType ), \
                f'Did not properly form sample for test type {SampleType}'

            sink.write( new_sample.as_wds )
    

    ## Ordered

    # Read first shard, no batches

    first_filename = file_pattern.format( shard_id = f'{0:06d}' )
    dataset = atdata.Dataset[SampleType]( first_filename )

    iterations_run = 0
    for i_iterate, cur_sample in enumerate( dataset.ordered( batch_size = None ) ):

        assert isinstance( cur_sample, SampleType ), \
            f'Single sample for {SampleType} written to `wds` is of wrong type'
        
        # Check sample values
        
        for k, v in sample_data.items():
            if isinstance( v, np.ndarray ):
                is_correct = np.all( getattr( cur_sample, k ) == v )
            else:
                is_correct = getattr( cur_sample, k ) == v
            assert is_correct, \
                f'{SampleType}: Incorrect sample value found for {k} - {type( getattr( cur_sample, k ) )}'

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, \
        f"Only found {iterations_run} samples, not {n_iterate}"

    # Read all shards, batches

    start_id = f'{0:06d}'
    end_id = f'{9:06d}'
    first_filename = file_pattern.format( shard_id = '{' + start_id + '..' + end_id + '}' )
    dataset = atdata.Dataset[SampleType]( first_filename )

    iterations_run = 0
    for i_iterate, cur_batch in enumerate( dataset.ordered( batch_size = batch_size ) ):
        
        assert isinstance( cur_batch, atdata.SampleBatch ), \
            f'{SampleType}: Batch sample is not correctly a batch'
        
        assert cur_batch.sample_type == SampleType, \
            f'{SampleType}: Batch `sample_type` is incorrect type'
        
        if i_iterate == 0:
            cur_n = len( cur_batch.samples )
            assert cur_n == batch_size, \
                f'{SampleType}: Batch has {cur_n} samples, not {batch_size}'
        
        assert isinstance( cur_batch.samples[0], SampleType ), \
            f'{SampleType}: Batch sample of wrong type ({type( cur_batch.samples[0])})'
        
        # Check batch values
        for k, v in sample_data.items():
            cur_batch_data = getattr( cur_batch, k )

            if isinstance( v, np.ndarray ):
                assert isinstance( cur_batch_data, np.ndarray ), \
                    f'{SampleType}: `NDArray` not carried through to batch'
                
                is_correct = all( 
                    [ np.all( cur_batch_data[i] == v )
                      for i in range( cur_batch_data.shape[0] ) ]
                )

            else:
                is_correct = all( 
                    [ cur_batch_data[i] == v
                      for i in range( len( cur_batch_data ) ) ]
                )

            assert is_correct, \
                f'{SampleType}: Incorrect sample value found for {k}'

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, \
        "Only found {iterations_run} samples, not {n_iterate}"
    

    ## Shuffled

    # Read first shard, no batches

    first_filename = file_pattern.format( shard_id = f'{0:06d}' )
    dataset = atdata.Dataset[SampleType]( first_filename )

    iterations_run = 0
    for i_iterate, cur_sample in enumerate( dataset.shuffled( batch_size = None ) ):
        
        assert isinstance( cur_sample, SampleType ), \
            f'Single sample for {SampleType} written to `wds` is of wrong type'
        
        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, \
        f"Only found {iterations_run} samples, not {n_iterate}"

    # Read all shards, batches

    start_id = f'{0:06d}'
    end_id = f'{9:06d}'
    first_filename = file_pattern.format( shard_id = '{' + start_id + '..' + end_id + '}' )
    dataset = atdata.Dataset[SampleType]( first_filename )

    iterations_run = 0
    for i_iterate, cur_sample in enumerate( dataset.shuffled( batch_size = batch_size ) ):
        
        assert isinstance( cur_sample, atdata.SampleBatch ), \
            f'{SampleType}: Batch sample is not correctly a batch'
        
        assert cur_sample.sample_type == SampleType, \
            f'{SampleType}: Batch `sample_type` is incorrect type'
        
        if i_iterate == 0:
            cur_n = len( cur_sample.samples )
            assert cur_n == batch_size, \
                f'{SampleType}: Batch has {cur_n} samples, not {batch_size}'
        
        assert isinstance( cur_sample.samples[0], SampleType ), \
            f'{SampleType}: Batch sample of wrong type ({type( cur_sample.samples[0])})'
        
        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, \
        "Only found {iterations_run} samples, not {n_iterate}"

#

@pytest.mark.parametrize(
    ('SampleType', 'sample_data', 'sample_wds_stem', 'test_parquet'),
    [ (
        case['SampleType'],
        case['sample_data'],
        case['sample_wds_stem'],
        case['test_parquet']
      )
      for case in test_cases ]
)
def test_create_sample(
            SampleType: Type[atdata.PackableSample],
            sample_data: atds.MsgpackRawSample,
            sample_wds_stem: str,
            test_parquet: bool,
            tmp_path
        ):
    """Test our ability to export a dataset to `parquet` format"""

    # Skip irrelevant test cases
    if not test_parquet:
        return

    ## Testing hyperparameters

    n_copies_dataset = 1_000
    n_per_file = 100

    ## Start out by writing tar dataset

    wds_filename = (tmp_path / f'{sample_wds_stem}.tar').as_posix()
    with wds.TarWriter( wds_filename ) as sink:
        for _ in range( n_copies_dataset ):
            new_sample = SampleType.from_data( sample_data )
            sink.write( new_sample.as_wds )
    
    ## Now export to `parquet`

    dataset = atdata.Dataset[SampleType]( wds_filename )
    parquet_filename = tmp_path / f'{sample_wds_stem}.parquet'
    dataset.to_parquet( parquet_filename )

    ## Double-check our `parquet` export
    
    # TODO

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
    
    # TODO Test PutPut

##
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

test_cases = [
    {
        'SampleType': BasicTestSample,
        'sample_data': {
            'name': 'Hello, world!',
            'position': 42,
            'value': 1024.768,
        },
        'sample_wds_stem': 'basic_test',
    },
    {
        'SampleType': NumpyTestSample,
        'sample_data': 
        {
            'label': 9_001,
            'image': np.random.randn( 1024, 1024 ),
        },
        'sample_wds_stem': 'numpy_test',
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
                f'{SampleType}: Incorrect sample value found for {k}'

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, \
        f"Only found {iterations_run} samples, not {n_iterate}"

    # Read all shards, batches

    start_id = f'{0:06d}'
    end_id = f'{9:06d}'
    first_filename = file_pattern.format( shard_id = '{' + start_id + '..' + end_id + '}' )
    print( first_filename )
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
    print( first_filename )
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


##
#
# Copyright 2020 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import pathlib

import pytest

from dnikit.base import CachedProducer, Producer, Batch, pipeline
from dnikit.processors import Cacher
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t

_BATCH_LENGTH = 64


# Helper functions and classes
# ------------------------------------------------------------------------------
def _get_number_of_pickled_files(storage_path: pathlib.Path) -> int:
    return len(list(storage_path.glob("**/*.pkl")))


def _assert_same_batches(batches_one: t.Iterable[Batch],
                         batches_two: t.Iterable[Batch]) -> None:
    batches_one = list(batches_one)
    batches_two = list(batches_two)
    assert len(batches_one) == len(batches_two)
    for batch_one, batch_two in zip(batches_one, batches_two):
        assert batch_one.fields == batch_two.fields


# Test fixtures
# ------------------------------------------------------------------------------
@pytest.fixture(params=[16, 50, 64])
def batch_size(request: t.Any) -> int:
    return request.param


# Actual tests
# ------------------------------------------------------------------------------
def test_caching(producer_with_num_calls: t.Tuple[Producer, t.Callable[[], int]],
                 batch_size: int) -> None:
    producer, get_num_producer_calls = producer_with_num_calls
    cacher = Cacher()

    # Initial logic checks
    assert not cacher.cached
    assert not (cacher.storage_path / ".cache.done").exists()
    assert (cacher.storage_path / ".dni_cache_dir").exists()
    assert _get_number_of_pickled_files(cacher.storage_path) == 0

    # Create pipeline and consume data (this will cache all results)
    pipelined_producer = pipeline(producer, cacher)
    num_calls = get_num_producer_calls()
    # Check batches produced are the same with/without caching
    _assert_same_batches(producer(batch_size), pipelined_producer(batch_size))
    # Check two calls were made to the root producer
    assert get_num_producer_calls() == num_calls + 2

    # Check cache and pickled files
    assert cacher.cached
    assert (cacher.storage_path / ".cache.done").exists()

    # Check all pickled files were stored
    expected_pickled_files = math.ceil(_BATCH_LENGTH / batch_size)
    assert _get_number_of_pickled_files(cacher.storage_path) == expected_pickled_files

    # check all batches are the same whether coming from the producer or the cached producer
    _assert_same_batches(producer(batch_size), pipelined_producer(batch_size))
    # Assert only one more call was made to the root producer
    assert get_num_producer_calls() == num_calls + 3

    # Check using as_producer() gives the same results
    cached_producer = cacher.as_producer()
    assert cached_producer.storage_path == cacher.storage_path
    _assert_same_batches(producer(batch_size), cached_producer(batch_size))
    assert get_num_producer_calls() == num_calls + 4

    # check that Batch.StdKeys.IDENTIFIER was added
    batch = next(iter(pipelined_producer(batch_size)))
    assert batch.metadata[Batch.StdKeys.IDENTIFIER] is not None


def test_caching_specific_path(producer: Producer,
                               batch_size: int,
                               tmp_path: pathlib.Path) -> None:
    # Instantiate cached producer and logic checks
    cacher = Cacher(tmp_path)
    assert not cacher.cached
    assert cacher.storage_path == tmp_path.resolve()
    assert not (cacher.storage_path / ".cache.done").exists()
    assert (cacher.storage_path / ".dni_cache_dir").exists()

    # Generate once, this pass will cache responses to storage_path
    pipelined_producer = pipeline(producer, cacher)
    _assert_same_batches(producer(batch_size), pipelined_producer(batch_size))
    assert cacher.cached
    assert _get_number_of_pickled_files(cacher.storage_path) != 0
    _assert_same_batches(producer(batch_size), pipelined_producer(batch_size))

    # Create a CachedProducer and test it
    cached_producer = CachedProducer(tmp_path)
    assert cached_producer.storage_path == tmp_path.resolve()
    _assert_same_batches(producer(batch_size), cached_producer(batch_size))

    # Test .as_producer() method as well
    cached_producer2 = cacher.as_producer()
    assert cached_producer2.storage_path == tmp_path.resolve()
    _assert_same_batches(producer(batch_size), cached_producer2(batch_size))

    # Check there are cacher files in the tmp path
    cache_marker = (tmp_path / ".dni_cache_dir")
    assert cache_marker.exists()
    Cacher.clear(tmp_path)
    assert not cache_marker.exists()


def test_invalid_caching_operations(producer: Producer, tmp_path: pathlib.Path) -> None:
    cacher = Cacher(tmp_path)
    _ = pipeline(producer, cacher)

    # Cannot pipeline more than once
    with pytest.raises(DNIKitException):
        pipeline(producer, cacher)

    # Cannot create two cachers with the same directory
    with pytest.raises(DNIKitException):
        Cacher(cacher.storage_path)

    # Check cannot clear non-existent directory
    with pytest.raises(NotADirectoryError):
        Cacher.clear(tmp_path / "nonexistent-ssjsdhr")

    # Cannot get a producer if caching is not done
    with pytest.raises(DNIKitException):
        cacher.as_producer()

    # Cannot instantiate a Producer if storage path does not contain caching files
    with pytest.raises(DNIKitException):
        CachedProducer(cacher.storage_path)


def test_caching_with_different_batch_sizes(producer: Producer,
                                            batch_size: int,
                                            tmp_path: pathlib.Path) -> None:
    _TEST_BATCH_SIZES = (8, 32, 80)
    cacher = Cacher(tmp_path)

    # Create pipeline and consume data (this will cache all results)
    pipelined_producer = pipeline(producer, cacher)
    _assert_same_batches(producer(batch_size), pipelined_producer(batch_size))
    assert cacher.cached
    cached_producer = cacher.as_producer()

    # Now check with smaller and bigger
    for test_size in _TEST_BATCH_SIZES:
        _assert_same_batches(producer(test_size), pipelined_producer(test_size))
        _assert_same_batches(producer(test_size), cached_producer(test_size))


def test_cached_producer_copy(producer: Producer, tmp_path: pathlib.Path) -> None:
    # Prepare first CachedProducer
    # ----------------------------
    _BATCH_SIZE = 42
    storage_path1 = tmp_path / "storage_path1"
    cacher = Cacher(storage_path1)
    pipelined_producer = pipeline(producer, cacher)

    # Trigger caching of data
    list(pipelined_producer(_BATCH_SIZE))
    # Logic checks
    assert cacher.cached
    # Store number of pickled files
    num_pickle_files = _get_number_of_pickled_files(cacher.storage_path)

    # Get a CachedProducer
    cached_producer = cacher.as_producer()

    # Test copying of CachedProducer to a new empty dir
    # -------------------------------------------------
    storage_path2 = tmp_path / "storage_path2"
    # Move to new location
    cached_producer2 = cached_producer.copy_to(storage_path2)

    # Check copy was successful
    assert cached_producer2.storage_path == storage_path2.resolve()
    assert (storage_path2 / ".cache.done").exists()
    assert (storage_path2 / ".dni_cache_dir").exists()
    assert _get_number_of_pickled_files(storage_path2) == num_pickle_files
    _assert_same_batches(producer(_BATCH_SIZE), cached_producer2(_BATCH_SIZE))

    # Test copying of CachedProducer to an existing dir
    # -------------------------------------------------
    storage_path3 = tmp_path / "storage_path3"
    storage_path3.mkdir()
    # Create an empty pickle file
    (storage_path3 / "1.pkl").touch()
    # Move the cached generator to a new folder with already one file,
    # and do not overwrite (default behavior). This should raise an exception.
    with pytest.raises(DNIKitException):
        cached_producer.copy_to(storage_path3)

    # Move the cached generator to a new folder with already one file, and overwrite.
    # Should NOT raise exception.
    cached_producer3 = cached_producer.copy_to(storage_path3, overwrite=True)

    # Check copy was successful
    assert cached_producer3.storage_path == storage_path3.resolve()
    assert (storage_path3 / ".cache.done").exists()
    assert (storage_path3 / ".dni_cache_dir").exists()
    assert _get_number_of_pickled_files(storage_path3) == num_pickle_files
    _assert_same_batches(producer(_BATCH_SIZE), cached_producer3(_BATCH_SIZE))

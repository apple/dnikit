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

from dataclasses import dataclass
import typing as t

import pytest
import numpy as np

from dnikit.base._batch._fields import _Fields
from dnikit.base._batch._metadata_storage import (
    _DictMetaKeyTrait,
    _MetaKeyTrait,
    _MetadataStorage,
    _update_metadata_storage,
    _new_mutable_metadata_storage
)
from dnikit.base._batch._storage import (
    _BatchStorage,
    _concatenate_batches,
    _subset_batch
)


@dataclass(frozen=True)
class _TestMetadata:
    value: int


_BATCH_SIZE = 5
_DICT_META = _DictMetaKeyTrait()
_DICT_META_2 = _DictMetaKeyTrait()
_SIMPLE_META = _MetaKeyTrait()


def _range_data(start_from: int) -> np.ndarray:
    return np.arange(start_from, start_from + _BATCH_SIZE).reshape(_BATCH_SIZE, 1)


def _make_batch_storage(fields: t.Mapping[str, np.ndarray],
                        snapshots: t.Mapping[str, t.Mapping[str, np.ndarray]],
                        metadata: _MetadataStorage) -> _BatchStorage:
    return _BatchStorage(
        fields=_Fields(fields),
        snapshots={
            snapshot: _BatchStorage(_Fields(snapshot_data))
            for snapshot, snapshot_data in snapshots.items()
        },
        metadata=metadata
        )


@pytest.fixture
def first() -> _BatchStorage:
    fields = {
        "digits": _range_data(0),
        "tens": _range_data(10),
    }
    snapshot_data = {"hundreds": _range_data(100)}
    snapshot_data_2 = {"thousands": _range_data(1000)}
    squares_digits = [_TestMetadata(i**2) for i in range(_BATCH_SIZE)]
    squares_tens = [_TestMetadata((10+i)**2) for i in range(_BATCH_SIZE)]
    cubes = [_TestMetadata(i**3) for i in range(_BATCH_SIZE)]
    negatives = [_TestMetadata(-i) for i in range(_BATCH_SIZE)]

    return _make_batch_storage(
        fields=fields,
        snapshots={"origin": snapshot_data, "midpoint": snapshot_data_2},
        metadata={
            _DICT_META: {"digits": squares_digits, "tens": squares_tens},
            _DICT_META_2: {"cubes": cubes},
            _SIMPLE_META: {None: negatives},
        })


@pytest.fixture
def second() -> _BatchStorage:
    fields = {
        "digits": _range_data(5),
        "tens": _range_data(15),
    }
    snapshot_data = {"hundreds": _range_data(105)}
    snapshot_data_2 = {"thousands": _range_data(1005)}
    squares_digits = [_TestMetadata(i**2) for i in range(5, 10)]
    squares_tens = [_TestMetadata(i**2) for i in range(15, 20)]
    cubes = [_TestMetadata(i**3) for i in range(5, 10)]
    negatives = [_TestMetadata(-i) for i in range(5, 10)]

    return _make_batch_storage(
        fields=fields,
        snapshots={"origin": snapshot_data, "midpoint": snapshot_data_2},
        metadata={
            _DICT_META: {"digits": squares_digits, "tens": squares_tens},
            _DICT_META_2: {"cubes": cubes},
            _SIMPLE_META: {None: negatives},
        })


@pytest.fixture(params=[
    "batch_key_missing",
    "wrong_shape",
    "missing_snapshot",
    "missing_metadata",
    "missing_metadata_field"])
def invalid(request: t.Any, second: _BatchStorage) -> _BatchStorage:
    # Shallow copy elements from second_batch
    new_fields = dict(second.fields)
    new_snapshots = {
        snapshot: dict(storage.fields)
        for snapshot, storage in second.snapshots.items()
    }
    new_metadata = _new_mutable_metadata_storage()
    _update_metadata_storage(new_metadata, second.metadata)

    if request.param == "batch_key_missing":
        del new_fields["tens"]
    elif request.param == "wrong_shape":
        new_fields["digits"] = np.array([5, 6, 7, 8, 9])
        new_fields["tens"] = np.array([15, 16, 17, 18, 19])
    elif request.param == "missing_snapshot":
        del new_snapshots["origin"]
    elif request.param == "missing_snapshot_field":
        new_snapshots["origin"] = {"random": second.fields["hundreds"]}
    elif request.param == "missing_metadata":
        del new_metadata[_DICT_META]
    elif request.param == "missing_metadata_field":
        del new_metadata[_DICT_META]["digits"]

    # Return invalid batch storage
    return _make_batch_storage(
        fields=new_fields,
        snapshots=new_snapshots,
        metadata=new_metadata
    )


def test_batch_storage_concatenation(first: _BatchStorage, second: _BatchStorage) -> None:
    result = _concatenate_batches([first, second])

    assert result.batch_size == first.batch_size + second.batch_size
    assert result.snapshots["origin"].batch_size == result.batch_size
    assert result.snapshots["midpoint"].batch_size == result.batch_size
    assert len(result.metadata[_DICT_META]["digits"]) == result.batch_size
    assert len(result.metadata[_DICT_META]["tens"]) == result.batch_size
    assert len(result.metadata[_DICT_META_2]["cubes"]) == result.batch_size

    # Check contents, snapshots and metadata
    for i in range(10):
        assert result.fields["digits"][i, 0] == i
        assert result.fields["tens"][i, 0] == 10 + i
        assert result.snapshots["origin"].fields["hundreds"][i, 0] == 100 + i
        assert result.snapshots["midpoint"].fields["thousands"][i, 0] == 1000 + i
        assert result.metadata[_DICT_META]["digits"][i].value == i ** 2
        assert result.metadata[_DICT_META]["tens"][i].value == (i + 10) ** 2
        assert result.metadata[_DICT_META_2]["cubes"][i].value == i ** 3
        assert result.metadata[_SIMPLE_META][None][i].value == -i

    # re-split
    first_split = _subset_batch(result, slice(None, _BATCH_SIZE))
    second_split = _subset_batch(result, slice(_BATCH_SIZE, None))
    assert first == first_split
    assert second == second_split


def test_batch_storage_multiple_concatenation(first: _BatchStorage) -> None:
    result = _concatenate_batches([first] * 4)
    assert result.batch_size == _BATCH_SIZE * 4

    # Check contents, snapshots and metadata
    for i in range(20):
        j = i % _BATCH_SIZE
        assert result.fields["digits"][i, 0] == j
        assert result.fields["tens"][i, 0] == 10 + j
        assert result.snapshots["origin"].fields["hundreds"][i, 0] == 100 + j
        assert result.snapshots["midpoint"].fields["thousands"][i, 0] == 1000 + j
        assert result.metadata[_DICT_META]["digits"][i].value == j ** 2
        assert result.metadata[_DICT_META]["tens"][i].value == (j + 10) ** 2
        assert result.metadata[_DICT_META_2]["cubes"][i].value == j ** 3
        assert result.metadata[_SIMPLE_META][None][i].value == -j


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_batch_storage_split(first: _BatchStorage, batch_size: int) -> None:
    first_result = _subset_batch(first, slice(None, batch_size))
    second_result = _subset_batch(first, slice(batch_size, None))

    assert first_result.batch_size == batch_size
    assert second_result.batch_size == (first.batch_size - batch_size)

    # Check contents, snapshots and metadata for first split
    for i in range(batch_size):
        assert first_result.fields["digits"][i, 0] == i
        assert first_result.fields["tens"][i, 0] == 10 + i
        assert first_result.snapshots["origin"].fields["hundreds"][i, 0] == 100 + i
        assert first_result.snapshots["midpoint"].fields["thousands"][i, 0] == 1000 + i
        assert first_result.metadata[_DICT_META]["digits"][i].value == i ** 2
        assert first_result.metadata[_DICT_META]["tens"][i].value == (10 + i) ** 2
        assert first_result.metadata[_SIMPLE_META][None][i].value == -i

    # Check contents, snapshots and metadata for second split
    for i in range(batch_size, first.batch_size):
        j = i - batch_size
        assert second_result.fields["digits"][j, 0] == i
        assert second_result.fields["tens"][j, 0] == 10 + i
        assert second_result.snapshots["origin"].fields["hundreds"][j, 0] == 100 + i
        assert second_result.snapshots["midpoint"].fields["thousands"][j, 0] == 1000 + i
        assert second_result.metadata[_DICT_META]["digits"][j].value == i ** 2
        assert second_result.metadata[_DICT_META]["tens"][j].value == (10 + i) ** 2
        assert second_result.metadata[_DICT_META_2]["cubes"][j].value == i ** 3
        assert second_result.metadata[_SIMPLE_META][None][j].value == -i

    # re-concatenate
    result = _concatenate_batches([first_result, second_result])
    assert first == result


@pytest.mark.parametrize("indices", [
    (0,), (1,), (4,),  # single indices: min, middle, max index
    (-1,), (-2,), (-4,),  # negatives
    (0, 3, 4), (3, 2, -3), (4, 1),  # multiple indices, sorted and unsorted, within range
    (1, 1, 2, 3, 4, 4, 0)  # duplicate, but valid indices
])
def test_batch_storage_subset(first: _BatchStorage, indices: t.Tuple[int, ...]) -> None:
    batch_subset = _subset_batch(first, indices)
    assert batch_subset.batch_size == len(indices)

    # Check contents, snapshots and metadata for batch subset
    for i in range(batch_subset.batch_size):
        j = indices[i] if indices[i] >= 0 else _BATCH_SIZE + indices[i]
        assert batch_subset.fields["digits"][i, 0] == j
        assert batch_subset.fields["tens"][i, 0] == 10 + j
        assert batch_subset.snapshots["origin"].fields["hundreds"][i, 0] == 100 + j
        assert batch_subset.snapshots["midpoint"].fields["thousands"][i, 0] == 1000 + j
        assert batch_subset.metadata[_DICT_META]["digits"][i].value == j ** 2
        assert batch_subset.metadata[_DICT_META]["tens"][i].value == (10 + j) ** 2
        assert batch_subset.metadata[_DICT_META_2]["cubes"][i].value == j ** 3
        assert batch_subset.metadata[_SIMPLE_META][None][i].value == -j


def test_invalid_batch_storage_concatenation(first: _BatchStorage, invalid: _BatchStorage) -> None:
    with pytest.raises(ValueError):
        _concatenate_batches([first, invalid])


@pytest.mark.parametrize("selector", [
    slice(0),
    slice(_BATCH_SIZE, None),
    slice(-_BATCH_SIZE, None, -1),
    slice(None, None, -1)
])
def test_batch_storage_subset_out_of_range(first: _BatchStorage, selector: slice) -> None:
    empty_subset = _subset_batch(first, slice(0))
    assert empty_subset.batch_size == 0
    assert empty_subset.fields["digits"].size == 0
    assert empty_subset.fields["tens"].size == 0
    assert empty_subset.snapshots["origin"].fields["hundreds"].size == 0
    assert empty_subset.snapshots["midpoint"].fields["thousands"].size == 0
    assert not empty_subset.metadata[_DICT_META]["digits"]
    assert not empty_subset.metadata[_DICT_META]["tens"]
    assert not empty_subset.metadata[_DICT_META_2]["cubes"]
    assert not empty_subset.metadata[_SIMPLE_META][None]


@pytest.mark.parametrize("indices", [
    (-5,), (7,),  # single indices out of range
    (-5, 3, 4), (3, 4, 5), (0, 1, 2, 3, 4, 5)  # one or many out of range
])
def test_invalid_batch_storage_subset(first: _BatchStorage, indices: t.Tuple[int, ...]) -> None:
    with pytest.raises(IndexError):
        _subset_batch(first, indices)


def test_empty_batch_storage_subset(first: _BatchStorage) -> None:
    assert _subset_batch(first, tuple()).batch_size == 0
    assert "digits" in first.fields
    assert "tens" in first.fields
    assert "hundreds" in first.snapshots["origin"].fields
    assert "thousands" in first.snapshots["midpoint"].fields
    assert _DICT_META in first.metadata and "digits" in first.metadata[_DICT_META]
    assert _DICT_META in first.metadata and "tens" in first.metadata[_DICT_META]
    assert _DICT_META_2 in first.metadata and "cubes" in first.metadata[_DICT_META_2]
    assert _SIMPLE_META in first.metadata

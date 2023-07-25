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
import pickle
import typing as t

import numpy as np
import pytest

from dnikit.base import Batch
from dnikit.base._batch._fields import _Fields
from dnikit.base._batch._storage import _BatchStorage
from dnikit.exceptions import DNIKitException

_BATCH_SIZE = 42


@dataclass(frozen=True)
class _TestMetadata:
    random: int


_DICT_META = Batch.DictMetaKey[_TestMetadata]("_DICT_META")
_SIMPLE_META = Batch.MetaKey[_TestMetadata]("_SIMPLE_META")


def _make_batch_data(batch_size: int) -> t.Dict[str, np.ndarray]:
    return {
        "layer1": np.random.randn(batch_size, 10, 20),
        "layer2": np.random.randn(batch_size, 5),
        "layer3": np.random.randn(batch_size)
    }


def _make_metadata(batch_size: int) -> t.List[_TestMetadata]:
    return [_TestMetadata(np.random.randint(100)) for _ in range(batch_size)]


def _same_data(a: t.Mapping[str, np.ndarray], b: t.Mapping[str, np.ndarray]) -> bool:
    return(
        set(a.keys()) == set(b.keys())
        and all(np.allclose(a[field], b[field]) for field in a)
    )


def test_batch() -> None:
    batch_size = 10
    batch_data = _make_batch_data(batch_size)
    batch = Batch(batch_data)

    # Check all properties of batch
    assert len(batch.fields) == len(batch_data)
    assert batch.fields is not batch_data
    assert _same_data(batch.fields, batch_data)
    assert batch.batch_size == batch_size
    assert not batch.snapshots
    assert not batch.metadata

    # mypy will complain if any element from batch is modified
    # except for the numpy arrays (which will trigger a runtime error)
    with pytest.raises(ValueError):
        batch.fields["layer1"][0, 1, 2] = 42.0


def test_batch_builder() -> None:
    batch_data = _make_batch_data(10)
    snapshot_data = _make_batch_data(10)
    metadata = _make_metadata(10)

    builder = Batch.Builder(batch_data)
    builder.snapshots = {"origin": Batch(snapshot_data)}
    builder.metadata[_DICT_META] = {"layer1": metadata}
    batch = builder.make_batch()

    # Check data
    assert batch.fields is not batch_data
    assert _same_data(batch.fields, batch_data)
    # Check snapshots
    assert batch.snapshots["origin"].fields is not snapshot_data
    assert _same_data(batch.snapshots["origin"].fields, snapshot_data)
    # Check metadata
    assert batch.metadata[_DICT_META] is not builder.metadata[_DICT_META]
    assert batch.metadata[_DICT_META]["layer1"] is not metadata
    assert batch.metadata[_DICT_META]["layer1"] == metadata

    # Test building a batch from another batch
    builder2 = Batch.Builder(base=batch)
    builder2.metadata[_DICT_META]["layer2"] = metadata
    batch2 = builder2.make_batch()

    assert batch2.fields is not batch_data
    assert _same_data(batch2.fields, batch_data)
    assert batch2.snapshots["origin"] is not snapshot_data
    assert _same_data(batch2.snapshots["origin"].fields, snapshot_data)
    assert batch2.metadata[_DICT_META] is not batch.metadata[_DICT_META]
    assert batch2.metadata[_DICT_META]["layer1"] is not metadata
    assert batch2.metadata[_DICT_META]["layer1"] == metadata
    assert batch2.metadata[_DICT_META]["layer2"] is not metadata
    assert batch2.metadata[_DICT_META]["layer2"] == metadata


def test_invalid_batch_creation() -> None:
    # Empty batch
    with pytest.raises(ValueError):
        Batch({})

    # Create a valid batch
    batch_data = _make_batch_data(_BATCH_SIZE)
    batch = Batch(batch_data)
    with pytest.raises(ValueError):
        batch.fields["layer1"][0] = 0.5

    # Different sizes batch
    invalid_batch_data = dict(batch_data)
    invalid_batch_data["layer_4"] = np.random.randn(10, 1)
    with pytest.raises(DNIKitException):
        Batch(invalid_batch_data)

    # Batch size doesn't match batch data
    invalid_snapshot_data = _make_batch_data(10)
    with pytest.raises(DNIKitException):
        Batch(_storage=_BatchStorage(
            fields=_Fields(batch_data),
            snapshots={"snapshot": _BatchStorage(_Fields(invalid_snapshot_data))}
        ))

    # Metadata size must be equal to batch_size
    invalid_metadata = Batch.Builder.MutableMetadataType()
    invalid_metadata[_DICT_META] = {"layer1": _make_metadata(10)}
    with pytest.raises(DNIKitException):
        Batch(_storage=_BatchStorage(
            fields=_Fields(batch_data),
            metadata=invalid_metadata._storage
        ))


def test_batch_builder_fields() -> None:
    NAME = Batch.DictMetaKey[int]('name')

    builder1 = Batch.Builder(fields={"test": np.arange(12)})
    builder1.metadata[NAME] = {"test": list(range(12))}

    # this has the NAME metadata
    batch = builder1.make_batch()
    assert len(batch.fields["test"]) == 12
    assert len(batch.metadata[NAME]["test"]) == 12


def test_batch_equality() -> None:
    builder1 = Batch.Builder(_make_batch_data(_BATCH_SIZE))
    builder1.snapshots = {"origin": Batch(_make_batch_data(_BATCH_SIZE))}
    builder1.metadata[_DICT_META] = {"layer1": _make_metadata(_BATCH_SIZE)}
    batch1a = builder1.make_batch()
    batch1b = builder1.make_batch()

    builder2 = Batch.Builder(_make_batch_data(_BATCH_SIZE))
    builder2.snapshots = {"origin": Batch(_make_batch_data(_BATCH_SIZE))}
    builder2.metadata[_DICT_META] = {"layer1": _make_metadata(_BATCH_SIZE)}
    batch2 = builder2.make_batch()

    assert batch1a is not batch1b
    assert batch1a == batch1b
    assert batch1a != batch2

    assert batch1a.fields is not batch1b.fields
    assert batch1a.fields == batch1b.fields
    assert batch1a.fields != batch2.fields

    assert batch1a.snapshots is not batch1b.snapshots
    assert batch1a.snapshots == batch1b.snapshots
    assert batch1a.snapshots != batch2.fields

    assert batch1a.metadata is not batch1b.metadata
    assert batch1a.metadata == batch1b.metadata
    assert batch1a.metadata != batch2.metadata


def test_batch_pickling() -> None:
    builder = Batch.Builder(_make_batch_data(_BATCH_SIZE))
    builder.snapshots = {"origin": Batch(_make_batch_data(_BATCH_SIZE))}
    builder.metadata[_DICT_META] = {"layer1": _make_metadata(_BATCH_SIZE)}
    original = builder.make_batch()

    data = pickle.dumps(original)
    replica = pickle.loads(data)

    assert original is not replica
    assert original == replica
    assert original.fields is not replica.fields
    assert original.snapshots is not replica.snapshots
    assert original.metadata is not replica.metadata


def test_batch_elements() -> None:
    data = _make_batch_data(_BATCH_SIZE)
    snapshot_data = _make_batch_data(_BATCH_SIZE)
    metadata = _make_metadata(_BATCH_SIZE)
    nofields_data = _make_metadata(_BATCH_SIZE)

    builder = Batch.Builder(data)
    builder.snapshots = {"origin": Batch(snapshot_data)}
    builder.metadata[_DICT_META] = {"layer1":  metadata}
    builder.metadata[_SIMPLE_META] = nofields_data
    batch = builder.make_batch()

    # Test Batch.ElementType
    for i, element in enumerate(batch.elements):
        assert element == batch.elements[i]
        # Compare field data
        assert np.all(element.fields["layer1"] == data["layer1"][i])
        assert np.all(element.fields["layer2"] == data["layer2"][i])
        assert np.all(element.fields["layer3"] == data["layer3"][i])
        # Compare snapshots
        snap_element = element.snapshots["origin"]
        assert element != snap_element
        assert np.all(snap_element.fields["layer1"] == snapshot_data["layer1"][i])
        assert np.all(snap_element.fields["layer2"] == snapshot_data["layer2"][i])
        assert np.all(snap_element.fields["layer3"] == snapshot_data["layer3"][i])
        # Verify metadata
        assert element.metadata[_DICT_META]["layer1"] == metadata[i]
        assert element.metadata[_SIMPLE_META] == nofields_data[i]

    # Test Batch from selected indices
    subset = batch.elements[0, 20, 30]
    assert subset.batch_size == 3
    # Verify field data
    assert np.all(subset.fields["layer1"] == data["layer1"][(0, 20, 30), ...])
    assert np.all(subset.fields["layer2"] == data["layer2"][(0, 20, 30), ...])
    assert np.all(subset.fields["layer3"] == data["layer3"][(0, 20, 30), ...])
    # Verify snapshots
    subset_snap = subset.snapshots["origin"]
    assert subset != subset_snap
    assert np.all(subset_snap.fields["layer1"] == snapshot_data["layer1"][(0, 20, 30), ...])
    assert np.all(subset_snap.fields["layer2"] == snapshot_data["layer2"][(0, 20, 30), ...])
    assert np.all(subset_snap.fields["layer3"] == snapshot_data["layer3"][(0, 20, 30), ...])
    # Verify metadata
    assert subset.metadata[_DICT_META]["layer1"] == [metadata[0], metadata[20], metadata[30]]
    assert subset.metadata[_SIMPLE_META] == [
        nofields_data[0], nofields_data[20], nofields_data[30]
    ]

    # Test Batch slices
    subset = batch.elements[10:14:2]
    assert subset.batch_size == 2
    # Verify field data
    assert np.all(subset.fields["layer1"] == data["layer1"][(10, 12), ...])
    assert np.all(subset.fields["layer2"] == data["layer2"][(10, 12), ...])
    assert np.all(subset.fields["layer3"] == data["layer3"][(10, 12), ...])
    # Verify snapshots
    subset_snap = subset.snapshots["origin"]
    assert subset != subset_snap
    assert np.all(subset_snap.fields["layer1"] == snapshot_data["layer1"][(10, 12), ...])
    assert np.all(subset_snap.fields["layer2"] == snapshot_data["layer2"][(10, 12), ...])
    assert np.all(subset_snap.fields["layer3"] == snapshot_data["layer3"][(10, 12), ...])
    # Verify metadata
    assert subset.metadata[_DICT_META]["layer1"] == [metadata[10], metadata[12]]
    assert subset.metadata[_SIMPLE_META] == [nofields_data[10], nofields_data[12]]

    # Test Batch subset from iterable
    subset = batch.elements[range(5)]
    assert subset.batch_size == 5
    # Verify field data
    assert np.all(subset.fields["layer1"] == data["layer1"][range(5)])
    assert np.all(subset.fields["layer2"] == data["layer2"][range(5)])
    assert np.all(subset.fields["layer3"] == data["layer3"][range(5)])
    # Verify snapshots
    subset_snap = subset.snapshots["origin"]
    assert subset != subset_snap
    assert np.all(subset_snap.fields["layer1"] == snapshot_data["layer1"][range(5)])
    assert np.all(subset_snap.fields["layer2"] == snapshot_data["layer2"][range(5)])
    assert np.all(subset_snap.fields["layer3"] == snapshot_data["layer3"][range(5)])
    # Verify metadata
    assert subset.metadata[_DICT_META]["layer1"] == [metadata[i] for i in range(5)]
    assert subset.metadata[_SIMPLE_META] == [nofields_data[i] for i in range(5)]


def test_batch_metadata_only() -> None:
    b = Batch.Builder()
    meta = Batch.DictMetaKey[str]('labels')
    b.metadata[meta] = {"gender": ["m", "f", "o", "m"]}
    batch = b.make_batch()

    assert batch.batch_size == 4


def test_batch_metadata_only_inconsistent_sizes() -> None:
    b = Batch.Builder()
    meta = Batch.DictMetaKey[str]('labels')

    # sequence lengths must match
    b.metadata[meta] = {"gender": ["m", "f", "o", "m"], "other": ["test"]}

    with pytest.raises(DNIKitException):
        _ = b.make_batch()


def test_builder_errors() -> None:
    batch_data = _make_batch_data(_BATCH_SIZE)

    with pytest.raises(ValueError):
        _ = Batch.Builder(fields=batch_data, base=Batch(fields=batch_data))

    with pytest.raises(ValueError):
        # Not testing type here (it's incorrect). Trying to test `ValueError`.
        _ = Batch.Builder(fields=Batch(fields=batch_data))  # type: ignore

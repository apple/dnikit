#
# Copyright 2021 Apple Inc.
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

import typing as t
from dataclasses import dataclass

import numpy as np

from dnikit.base import Batch, pipeline, peek_first_batch
from dnikit.processors import (
    MetadataRemover,
    MetadataRenamer,
    FieldRenamer
)


@dataclass(frozen=True)
class _TestMetadata:
    name: str


_FIELD_NAMES = ['resp_a', 'resp_b', 'resp_c']
_META_KEY = Batch.DictMetaKey[_TestMetadata]("META_KEY")
_SIMPLE_META_KEY = Batch.MetaKey[_TestMetadata]("SIMPLE_META_KEY")


def stub_dataset(batch_size: int) -> t.Iterable[Batch]:
    rs = np.random.RandomState(1234)
    batch_shape = (batch_size, 20, 20, 6)
    for _ in range(1):
        data = {r: rs.normal(0, 1, batch_shape) for r in _FIELD_NAMES}
        builder = Batch.Builder(data)
        builder.metadata[_META_KEY] = {
            r: [_TestMetadata(r) for _ in range(batch_size)]
            for r in _FIELD_NAMES
        }
        builder.metadata[_SIMPLE_META_KEY] = [
            _TestMetadata(f"simple_{i}")
            for i in range(batch_size)
        ]
        yield builder.make_batch()


def test_metadata_rename_fields() -> None:
    # simple rename
    producer = pipeline(stub_dataset, MetadataRenamer(mapping={_FIELD_NAMES[0]: "new"}))
    batch = peek_first_batch(producer)

    expected_fields = {_FIELD_NAMES[1], _FIELD_NAMES[2], "new"}

    # the field got renamed
    assert set(batch.fields.keys()) == set(_FIELD_NAMES)

    # so did the metadata associated with this field
    assert set(batch.metadata[_META_KEY].keys()) == expected_fields

    # the metadata holds a string that is the original name of the field
    assert batch.metadata[_META_KEY]["new"][0].name == _FIELD_NAMES[0]
    for i in range(1, 3):
        assert batch.metadata[_META_KEY][_FIELD_NAMES[i]][0].name == _FIELD_NAMES[i]


def test_metadata_transitive_rename() -> None:
    # transitive rename 0 = 1, 1 = 2 -- this needs to be done in order
    # or simultaneously.  it also implicitly removes a field as 0 gets replaced.
    mapping = {_FIELD_NAMES[0]: _FIELD_NAMES[1], _FIELD_NAMES[1]: _FIELD_NAMES[2]}
    producer = pipeline(stub_dataset, FieldRenamer(mapping), MetadataRenamer(mapping))
    batch = peek_first_batch(producer)

    # only two fields left
    expected_fields = {_FIELD_NAMES[1], _FIELD_NAMES[2]}

    assert set(batch.fields.keys()) == expected_fields
    assert set(batch.metadata[_META_KEY].keys()) == expected_fields

    # the metadata holds a string that is the original name of the field
    assert batch.metadata[_META_KEY][_FIELD_NAMES[1]][0].name == _FIELD_NAMES[0]
    assert batch.metadata[_META_KEY][_FIELD_NAMES[2]][0].name == _FIELD_NAMES[1]


def test_metadata_swap() -> None:
    # swap two fields, 0 = 1, 1 = 0.  This needs to be done simultaneously.
    mapping = {_FIELD_NAMES[0]: _FIELD_NAMES[1], _FIELD_NAMES[1]: _FIELD_NAMES[0]}
    producer = pipeline(stub_dataset, FieldRenamer(mapping), MetadataRenamer(mapping))
    batch = peek_first_batch(producer)

    # only two fields left
    expected_fields = set(_FIELD_NAMES)

    assert set(batch.fields.keys()) == expected_fields
    assert set(batch.metadata[_META_KEY].keys()) == expected_fields

    # the metadata holds a string that is the original name of the field
    assert batch.metadata[_META_KEY][_FIELD_NAMES[0]][0].name == _FIELD_NAMES[1]
    assert batch.metadata[_META_KEY][_FIELD_NAMES[1]][0].name == _FIELD_NAMES[0]
    assert batch.metadata[_META_KEY][_FIELD_NAMES[2]][0].name == _FIELD_NAMES[2]


def test_metadata_remove_all() -> None:
    producer = pipeline(stub_dataset, MetadataRemover())
    batch = peek_first_batch(producer)
    assert _META_KEY not in batch.metadata
    assert _SIMPLE_META_KEY not in batch.metadata


def test_metadata_keep_all() -> None:
    producer = pipeline(stub_dataset, MetadataRemover(keep=True))
    batch = peek_first_batch(producer)
    assert set(batch.metadata[_META_KEY].keys()) == set(_FIELD_NAMES)
    assert _SIMPLE_META_KEY in batch.metadata


def test_metadata_remove_metakey_and_keys() -> None:
    producer = pipeline(
        stub_dataset,
        MetadataRemover(meta_keys=_META_KEY, keys=(_FIELD_NAMES[1], _FIELD_NAMES[2]))
    )
    batch = peek_first_batch(producer)
    assert set(batch.metadata[_META_KEY].keys()) == {_FIELD_NAMES[0], }
    assert batch.metadata[_META_KEY][_FIELD_NAMES[0]][0].name == _FIELD_NAMES[0]
    assert _SIMPLE_META_KEY in batch.metadata


def test_metadata_keep_metakey_and_keys() -> None:
    producer = pipeline(
        stub_dataset,
        MetadataRemover(meta_keys=_META_KEY, keys=_FIELD_NAMES[0], keep=True)
    )
    batch = peek_first_batch(producer)

    # only one field left
    assert set(batch.metadata[_META_KEY].keys()) == {_FIELD_NAMES[0], }
    assert batch.metadata[_META_KEY][_FIELD_NAMES[0]][0].name == _FIELD_NAMES[0]
    assert _SIMPLE_META_KEY not in batch.metadata


def test_metadata_remove_keys() -> None:
    # remove a single field
    producer = pipeline(stub_dataset, MetadataRemover(keys=_FIELD_NAMES[1]))
    batch = peek_first_batch(producer)

    # only two fields left
    expected_fields = {_FIELD_NAMES[0], _FIELD_NAMES[2]}

    assert set(batch.fields.keys()) == set(_FIELD_NAMES)
    assert set(batch.metadata[_META_KEY].keys()) == expected_fields
    assert _SIMPLE_META_KEY in batch.metadata

    # the metadata holds a string that is the original name of the field
    for i in [0, 2]:
        assert batch.metadata[_META_KEY][_FIELD_NAMES[i]][0].name == _FIELD_NAMES[i]


def test_metadata_keep_keys() -> None:
    # keep a single keys
    producer = pipeline(stub_dataset, MetadataRemover(keys=_FIELD_NAMES[1], keep=True))
    batch = peek_first_batch(producer)

    # Check only selected key is present
    assert set(batch.metadata[_META_KEY].keys()) == {_FIELD_NAMES[1]}
    assert _SIMPLE_META_KEY not in batch.metadata

    # the metadata holds a string that is the original name of the field
    assert batch.metadata[_META_KEY][_FIELD_NAMES[1]][0].name == _FIELD_NAMES[1]


def test_metadata_remove_mixed_keys() -> None:
    producer = pipeline(
        stub_dataset,
        MetadataRemover(meta_keys=[_META_KEY, _SIMPLE_META_KEY], keys=_FIELD_NAMES[0])
    )
    batch = peek_first_batch(producer)
    assert set(batch.metadata[_META_KEY].keys()) == {_FIELD_NAMES[1], _FIELD_NAMES[2]}
    assert _SIMPLE_META_KEY not in batch.metadata


def test_metadata_keep_mixed_metakeys() -> None:
    producer = pipeline(
        stub_dataset,
        MetadataRemover(meta_keys=[_META_KEY, _SIMPLE_META_KEY], keys=_FIELD_NAMES[0], keep=True)
    )
    batch = peek_first_batch(producer)
    assert set(batch.metadata[_META_KEY].keys()) == {_FIELD_NAMES[0]}
    assert _SIMPLE_META_KEY in batch.metadata


def test_metadata_keep_simple_metakey() -> None:
    producer = pipeline(
        stub_dataset,
        MetadataRemover(meta_keys=_SIMPLE_META_KEY, keep=True)
    )
    batch = peek_first_batch(producer)
    assert _META_KEY not in batch.metadata
    assert _SIMPLE_META_KEY in batch.metadata
    assert batch.metadata[_SIMPLE_META_KEY] == [
        _TestMetadata(f"simple_{i}")
        for i in range(len(batch.metadata[_SIMPLE_META_KEY]))
    ]

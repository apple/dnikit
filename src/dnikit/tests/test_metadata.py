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

import pytest

from dnikit.base import Batch
from dnikit.base._batch._metadata_storage import _copy_metadata_storage
import dnikit.typing._types as t


_DICT_KEY_1 = Batch.DictMetaKey[int]("_DICT_KEY_1")
_DICT_KEY_2 = Batch.DictMetaKey[str]("_DICT_KEY_2")
_DICT_KEY_3 = Batch.DictMetaKey[int]("_DICT_KEY_3")
_SIMPLE_KEY_1 = Batch.MetaKey[int]("_SIMPLE_KEY_1")
_SIMPLE_KEY_2 = Batch.MetaKey[int]("_SIMPLE_KEY_2")
_UNION_META = Batch.MetaKey[t.Union[int, str]]("_UNION_META")


def test_metadata() -> None:
    # MutableMetadata
    # ===============

    # Empty initialisation
    mutable = Batch.Builder.MutableMetadataType()
    assert not mutable

    # Setter
    mutable[_DICT_KEY_1] = {"field1": [1, 2, 3]}
    mutable[_DICT_KEY_1]["field2"] = [4, -1, 6]
    mutable[_DICT_KEY_1]["field2"][1] = 5
    mutable[_DICT_KEY_2]["field3"] = ["a", "b", "c"]
    mutable[_DICT_KEY_3] = {}
    mutable[_SIMPLE_KEY_1] = (-1, 2, 3)
    mutable[_SIMPLE_KEY_2] = [-1, -2, -3]
    mutable[_SIMPLE_KEY_2] = [4, 5, 6]
    mutable[_SIMPLE_KEY_1][0] = 1
    assert mutable

    # Getter
    assert mutable[_DICT_KEY_1]["field1"] == [1, 2, 3]
    assert mutable[_DICT_KEY_1]["field2"] == [4, 5, 6]
    assert mutable[_DICT_KEY_2].get("field3") == ["a", "b", "c"]
    assert mutable[_DICT_KEY_3] == {}
    assert mutable[_SIMPLE_KEY_1] == [1, 2, 3]
    assert mutable[_SIMPLE_KEY_2] == [4, 5, 6]

    # Deleter
    del mutable[_DICT_KEY_1]["field2"]
    del mutable[_SIMPLE_KEY_2]

    # Contains
    # Metadata with dict
    assert _DICT_KEY_1 in mutable and "field1" in mutable[_DICT_KEY_1]
    assert _DICT_KEY_2 in mutable
    assert _DICT_KEY_3 in mutable
    assert _DICT_KEY_1 in mutable and "field2" not in mutable[_DICT_KEY_1]
    assert _SIMPLE_KEY_1 in mutable
    assert _SIMPLE_KEY_2 not in mutable

    # Metadata
    # ========

    # Empty initialisation
    assert not bool(Batch.MetadataType())

    # Conversion to metadata
    metadata = Batch.MetadataType(_copy_metadata_storage(mutable._storage))
    assert isinstance(metadata, Batch.MetadataType)
    assert metadata

    # Getters
    assert metadata[_DICT_KEY_1]["field1"] == [1, 2, 3]
    assert metadata[_DICT_KEY_2]["field3"] == ["a", "b", "c"]
    assert metadata[_SIMPLE_KEY_1] == [1, 2, 3]

    # Contains
    assert _DICT_KEY_1 in metadata and "field2" not in metadata[_DICT_KEY_1]
    assert _DICT_KEY_2 in metadata and "field3" in metadata[_DICT_KEY_2]
    assert _DICT_KEY_3 not in metadata
    assert _SIMPLE_KEY_1 in metadata
    assert _SIMPLE_KEY_2 not in metadata

    # Keys
    meta_keys = {_DICT_KEY_1, _DICT_KEY_2, _SIMPLE_KEY_1}
    assert metadata.keys() == meta_keys

    # MetadataElement
    # ===============

    # Empty initialisation
    empty_element = Batch.MetadataType.ElementType(Batch.MetadataType()._storage, 0)
    assert not bool(empty_element)

    # Conversion to MetadataElement
    element = Batch.MetadataType.ElementType(metadata._storage, 1)
    assert bool(element)

    # Getters
    assert element[_DICT_KEY_1]["field1"] == 2
    assert element[_DICT_KEY_2]["field3"] == "b"
    assert element[_SIMPLE_KEY_1] == 2

    # Contains
    assert _DICT_KEY_1 in element and "field2" not in element[_DICT_KEY_1]
    assert _DICT_KEY_2 in element and "field3" in element[_DICT_KEY_2]
    assert _DICT_KEY_3 not in element
    # Metadata without fields
    assert _SIMPLE_KEY_1 in element
    assert _SIMPLE_KEY_2 not in element

    # Description
    assert element.keys() == metadata.keys()


def test_metadata_exceptions() -> None:
    metadata = Batch.MetadataType()
    with pytest.raises(KeyError):
        _ = metadata[_DICT_KEY_1]

    with pytest.raises(KeyError):
        _ = metadata[_SIMPLE_KEY_1]


def test_metadata_union() -> None:
    mutable = Batch.Builder.MutableMetadataType()

    mutable[_UNION_META] = (1, 2, 'a', 'b')
    mutable[_UNION_META][1] = 3
    mutable[_UNION_META][-1] = 'c'

    assert mutable
    assert mutable[_UNION_META] == [1, 3, 'a', 'c']
    assert _UNION_META in mutable

    metadata = Batch.MetadataType(_copy_metadata_storage(mutable._storage))
    assert metadata
    assert _UNION_META in metadata
    assert metadata[_UNION_META] == [1, 3, 'a', 'c']
    assert metadata.keys() == {_UNION_META}

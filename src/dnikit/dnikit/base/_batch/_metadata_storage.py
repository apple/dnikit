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

import abc
from collections import defaultdict

import dnikit.typing._types as t
from dnikit import _dict_utils


# Class traits to differentiate between different metadata layouts
class _MetaKeyTrait(abc.ABC):

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        assert any(x.__qualname__ == "Batch.MetaKey" for x in cls.mro()), (
            f"_MetaKeyProtocol may only be subclassed by Batch.MetaKey, got {cls.__qualname__}"
        )


class _DictMetaKeyTrait(abc.ABC):

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        assert any(x.__qualname__ == "Batch.DictMetaKey" for x in cls.mro()), (
            f"_DictMetaKeyProtocol may only be subclassed by Batch.MetaKey, got {cls.__qualname__}"
        )


# Storage typedefs
_MetadataStorage = t.Mapping[
    t.Union[_MetaKeyTrait, _DictMetaKeyTrait],
    t.Mapping[t.Optional[str], t.Sequence]
]
_MutableMetadataStorage = t.DefaultDict[
    t.Union[_MetaKeyTrait, _DictMetaKeyTrait],
    t.DefaultDict[t.Optional[str], t.MutableSequence]
]
_AnyMetadataStorage = t.Union[_MetadataStorage, _MutableMetadataStorage]


# The following functions are _very_ loose with type annotations, this is because
# to allow reuse between Metadata and MutableMetadata. The overloads have the actual
# strict definitions (mypy will only look at the right overload)
def _new_mutable_metadata_storage() -> _MutableMetadataStorage:
    return defaultdict(lambda: defaultdict(list))


def _get_metadata_item(storage: _AnyMetadataStorage,
                       meta_key: t.Union[_MetaKeyTrait, _DictMetaKeyTrait]) -> t.Any:
    if isinstance(meta_key, _MetaKeyTrait):
        return storage[meta_key][None]
    elif isinstance(meta_key, _DictMetaKeyTrait):
        return storage[meta_key]
    else:
        assert False, "Unknown type of MetaKey, cannot determine metadata layout"


def _get_metadata_element_item(storage: _AnyMetadataStorage,
                               meta_key: t.Union[_MetaKeyTrait, _DictMetaKeyTrait],
                               index: int) -> t.Any:
    if isinstance(meta_key, _MetaKeyTrait):
        return storage[meta_key][None][index]
    elif isinstance(meta_key, _DictMetaKeyTrait):
        return {
            field: value[index]
            for field, value in storage[meta_key].items()
        }
    else:
        assert False, "Unknown type of MetaKey, cannot determine metadata layout"


def _set_metadata_item(storage: _MutableMetadataStorage,
                       meta_key: t.Union[_MetaKeyTrait, _DictMetaKeyTrait],
                       value: t.Any) -> None:
    if isinstance(meta_key, _MetaKeyTrait):
        storage[meta_key][None] = list(value)
    elif isinstance(meta_key, _DictMetaKeyTrait):
        storage[meta_key] = defaultdict(list, {k: list(v) for k, v in value.items()})
    else:
        assert False, "Unknown type of MetaKey, cannot determine metadata layout"


def _copy_metadata_storage(storage: _AnyMetadataStorage) -> _MetadataStorage:
    return {
        meta_key: {
            key: list(value)
            for key, value in meta_values.items()
            if value
        }
        for meta_key, meta_values in storage.items()
        if meta_values
    }


def _update_metadata_storage(result: _MutableMetadataStorage,
                             storage: _AnyMetadataStorage) -> _MutableMetadataStorage:
    result.update({
        meta_key: defaultdict(list, {
            key: list(value)
            for key, value in meta_values.items()
            if value
        })
        for meta_key, meta_values in storage.items()
        if meta_values
    })
    return result


def _rename_dict_metakey_fields(storage: _MutableMetadataStorage,
                                mapping: t.Mapping[str, str],
                                meta_keys: t.AbstractSet[_DictMetaKeyTrait]
                                ) -> _MutableMetadataStorage:
    selected_meta_keys = meta_keys if meta_keys else frozenset(
        meta_key
        for meta_key in storage.keys()
        if isinstance(meta_key, _DictMetaKeyTrait)
    )
    for meta_key in selected_meta_keys:
        cast_mapping = t.cast(t.Mapping[t.Optional[str], t.Optional[str]], mapping)
        storage[meta_key] = defaultdict(
            list,
            _dict_utils.rename_keys(storage[meta_key], cast_mapping)
        )
    return storage


def _remove_meta_keys(storage: _MutableMetadataStorage,
                      meta_keys: t.AbstractSet[_MetaKeyTrait],
                      keep: bool) -> _MutableMetadataStorage:
    all_simple_meta_keys = frozenset(
        metakey
        for metakey in storage.keys()
        if isinstance(metakey, _MetaKeyTrait)
    )
    selected_meta_keys = meta_keys or all_simple_meta_keys
    if not keep:
        for meta_key in selected_meta_keys:
            del storage[meta_key]
    else:
        for meta_key in all_simple_meta_keys - selected_meta_keys:
            del storage[meta_key]
    return storage


def _remove_dict_meta_keys(storage: _MutableMetadataStorage,
                           meta_keys: t.AbstractSet[_DictMetaKeyTrait],
                           keys: t.Optional[t.AbstractSet[str]],
                           keep: bool) -> _MutableMetadataStorage:
    all_dict_meta_keys = frozenset(
        metakey
        for metakey in storage.keys()
        if isinstance(metakey, _DictMetaKeyTrait)
    )
    selected_meta_keys = meta_keys or all_dict_meta_keys
    # Flag to check whether DictMetaKeys should be completely deleted
    # (skip full deletion if keep is true or if fields were specified)
    skip_full_deletion = keep or keys is not None

    for meta_key in selected_meta_keys:
        if skip_full_deletion:
            _dict_utils.delete_keys(storage[meta_key], keys, keep=keep)
        else:
            del storage[meta_key]  # Delete the whole entry

    if keep:
        for meta_key in all_dict_meta_keys - selected_meta_keys:
            del storage[meta_key]

    return storage

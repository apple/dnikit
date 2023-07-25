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

import collections.abc

from dnikit.base import Batch, PipelineStage
import dnikit.typing as dt
import dnikit.typing._types as t

_OneOrManyMetaKeysOrNone = dt.OneManyOrNone[t.Union[Batch.MetaKey, Batch.DictMetaKey]]


def _resolve_meta_keys(value: _OneOrManyMetaKeysOrNone
                       ) -> t.Tuple[t.AbstractSet[Batch.MetaKey], t.AbstractSet[Batch.DictMetaKey]]:
    simple_meta_keys = set()
    dict_meta_keys = set()
    if value is None:
        pass
    elif isinstance(value, Batch.MetaKey):
        simple_meta_keys.add(value)
    elif isinstance(value, Batch.DictMetaKey):
        dict_meta_keys.add(value)
    else:
        for mk in value:
            if isinstance(mk, Batch.MetaKey):
                simple_meta_keys.add(mk)
            elif isinstance(mk, Batch.DictMetaKey):
                dict_meta_keys.add(mk)

    return simple_meta_keys, dict_meta_keys


def _contains_metakey(value: dt.OneManyOrNone[t.Any]) -> bool:
    if isinstance(value, Batch.MetaKey) or isinstance(value, Batch.DictMetaKey):
        return True
    if isinstance(value, collections.abc.Collection):
        return (
            any(isinstance(x, Batch.MetaKey) for x in value)
            or any(isinstance(x, Batch.DictMetaKey) for x in value)
        )
    return False


@t.final
class MetadataRemover(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that removes some
    :attr:`metadata <dnikit.base.Batch.metadata>` from a :class:`Batch <dnikit.base.Batch>`.

    Args:
        meta_keys: **[keyword arg, optional]** either a single instance or an iterable of
            :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` /
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` that
            may be removed. If ``None`` (the default case), this processor will operate on all
            :attr:`metadata <dnikit.base.Batch.metadata>` keys.
        keys: **[keyword arg, optional]** key within metadata to be removed.
            :attr:`metadata <dnikit.base.Batch.metadata>` with metadata key type
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` is a mapping from
            ``str: data-type``. This argument specifies the str ``key-field`` that will be removed
            from the batch's metadata, where the metadata must have metadata key type
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`. If
            ``None`` (the default case), this processor will operate on all ``key-fields`` for
            metadata with type :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`
            metadata key.
        keep: **[keyword arg, optional]** if True, the selected ``meta_keys`` and ``keys`` now
            specify what to **keep**, and all other data will be removed.
    """

    def __init__(self, *,
                 meta_keys: _OneOrManyMetaKeysOrNone = None,
                 keys: t.Any = None,
                 keep: bool = False):
        super().__init__()

        if _contains_metakey(keys):
            raise ValueError("`keys` contains metadata keys. Use `meta_keys` for this instead.")

        self._simple_meta_keys, self._dict_meta_keys = _resolve_meta_keys(meta_keys)
        self._keys = dt.resolve_one_many_or_none(keys, str)

        # Logic to select what to keep and what what to remove
        simple_metakeys_selected = bool(self._simple_meta_keys)
        dict_metakeys_selected = bool(self._dict_meta_keys) or bool(self._keys)
        no_selection = not simple_metakeys_selected and not dict_metakeys_selected

        self._keep_simple_metakeys = (
            keep and (simple_metakeys_selected or no_selection)
            # If it's necessary to remove, but only dict_metakeys were selected,
            # simple_metakeys should be kept
            or (not keep and not simple_metakeys_selected and dict_metakeys_selected)
        )

        self._keep_dict_metakeys = (
            keep and (dict_metakeys_selected or no_selection)
            # If it's necessary to remove, but only simple_metakeys were selected,
            # dict_metakeys should be kept
            or (not keep and not dict_metakeys_selected and simple_metakeys_selected)
        )

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)
            builder.metadata._remove_simple_meta_keys(
                self._simple_meta_keys, keep=self._keep_simple_metakeys
            )
            builder.metadata._remove_dict_meta_keys(
                meta_keys=self._dict_meta_keys, keys=self._keys, keep=self._keep_dict_metakeys
            )
            return builder.make_batch()
        return batch_processor


@t.final
class MetadataRenamer(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that renames some
    :attr:`metadata <dnikit.base.Batch.metadata>` fields in a :class:`Batch <dnikit.base.Batch>`.
    This only works with metadata that has key type
    :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`.

    Args:
        mapping: a dictionary (or similar) whose keys are the old metadata
            field names and values are the new metadata field names.
        meta_keys: **[keyword arg, optional]** either a single instance or an iterable of
            metadata keys of type :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` whose
            ``key-fields`` will be renamed. If ``None`` (the default case), all ``key-fields``
            for all metadata keys will be renamed.

    Note:
        ``MetadataRenamer`` only works with
        class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` (which has entries
        that can be renamed).
    """

    def __init__(self,
                 mapping: t.Mapping[str, str], *,
                 meta_keys: dt.OneManyOrNone[Batch.DictMetaKey] = None) -> None:
        super().__init__()
        self._mapping = mapping
        _, self._meta_keys = _resolve_meta_keys(meta_keys)

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)
            builder.metadata._rename_fields(self._mapping, self._meta_keys)
            return builder.make_batch()
        return batch_processor

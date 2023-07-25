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

import dnikit.typing._types as t

_K = t.TypeVar('_K')
_V = t.TypeVar('_V')


def rename_keys(dictionary: t.Mapping[_K, _V],
                mapping: t.Mapping[_K, _K]) -> t.MutableMapping[_K, _V]:
    """
    Rename keys in a dictionary and return the new mapping.

    Args:
        dictionary: dictionary to rename keys in
        mapping: mapping of old -> new keys
    """

    # if a field is going to be implicitly deleted by renaming something on top of it, suppress it
    omit = frozenset(set(mapping.values()) - set(mapping.keys()))

    return {
        mapping[key] if key in mapping else key: values
        for key, values in dictionary.items()
        if key not in omit
    }


def delete_keys(dictionary: t.MutableMapping[_K, _V],
                keys: t.Optional[t.AbstractSet[_K]],
                keep: bool = False) -> None:
    """
    Delete keys in a mutable dictionary.

    Args:
        dictionary: dictionary to mutate
        keys: keys to keep/remove
        keep: if true, the set of keys are the only keys to keep
    """
    all_keys = frozenset(dictionary.keys())
    selected_keys = keys or frozenset(dictionary.keys())
    if keep:
        for key in all_keys:
            if key not in selected_keys:
                del dictionary[key]
    else:
        for key in selected_keys:
            if key in dictionary:
                del dictionary[key]


def seq_of_dict_to_dict_of_seq(values: t.Sequence[t.Mapping[_K, _V]]
                               ) -> t.Mapping[_K, t.Sequence[_V]]:
    """
    Converts a sequence of mappings to a mapping of sequences.  The structure of
    each of the dictionaries must be identical (same set of keys).

    This is the inverse of `dict_of_seq_to_seq_of_dict()`.

    Example:
        For example:

        .. code-block:: python

            [
                { "a": 1, "b": 2 },
                { "a": 3, "b": 4 },
            ]

        would be converted to:

        .. code-block:: python

            {
                "a": [1, 3],
                "b": [2, 4],
            }

    Args:
        values: sequence of mappings

    Returns:
        mapping of sequences
    """
    if len(values) > 0:
        result = {
            key: [value]
            for key, value in values[0].items()
        }
        for m in values[1:]:
            for key, value in m.items():
                result[key].append(value)
        return result
    else:
        return {}


def dict_of_seq_to_seq_of_dict(values: t.Mapping[_K, t.Sequence[_V]]
                               ) -> t.Sequence[t.Mapping[_K, _V]]:
    """
    Converts a mapping of sequences to a sequence of mappings.  The length of each of the sequences
    must be identical.

    This is the inverse of `seq_of_dict_to_dict_of_seq()`.

    Example:
        For example:

        .. code-block:: python

            {
                "a": [1, 3],
                "b": [2, 4],
            }

        would be converted to:

        .. code-block:: python

            [
                { "a": 1, "b": 2 },
                { "a": 3, "b": 4 },
            ]

    Args:
        values: mapping of sequences

    Returns:
        sequence of mappings
    """
    any_key = next(iter(values.keys()))
    length = len(values[any_key])
    return [
        {
            key: values[key][index]
            for key in values.keys()
        }
        for index in range(length)
    ]


def subscript_dict_of_seq(values: t.Mapping[_K, t.Sequence[_V]], index: int) -> t.Mapping[_K, _V]:
    """
    Return a particular element of from `seq_of_dict_to_dict_of_seq()` applied to the values.
    If the index is out of bounds, an IndexError will be thrown.

    Example:
        For example calling it with this mapping and an index of 1:

        .. code-block:: python

            {
                "a": [1, 3],
                "b": [2, 4],
            }

        would produce:

        .. code-block:: python

            { "a": 3, "b": 4 }

    Args:
        values: mapping of sequences
        index: index to return

    Returns:
        a mapping from the given index

    """
    return {
        key: values[key][index]
        for key in values.keys()
    }


def ordered_values_tuple(values: t.Mapping[_K, _V], keys: t.Iterable[_K]) -> t.Tuple[_V, ...]:
    """
    Given a mapping and an ordered list of keys, produce an ordered tuple with
    values from the mapping.  ``values`` is required to contain all the the ``keys`` --
    a KeyError will result if it does not.

    Args:
        values: the mapping with the values
        keys: the ordered keys

    Returns:
        tuple of values from the mapping in the order specified by the keys
    """
    return tuple(values[key] for key in keys)

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

import pytest
import dnikit.typing._types as t

from dnikit._dict_utils import (
    rename_keys,
    delete_keys,
    seq_of_dict_to_dict_of_seq,
    dict_of_seq_to_seq_of_dict,
    subscript_dict_of_seq,
    ordered_values_tuple,
)


SEQ_OF_DICT = t.Sequence[t.Mapping[str, int]]
DICT_OF_SEQ = t.Mapping[str, t.Sequence[int]]


def test_rename_simple() -> None:
    dictionary = {
        "a": "a",
        "b": "b"
    }
    mapping = {
        "a": "c"
    }
    expected = {
        "c": "a",
        "b": "b"
    }

    assert rename_keys(dictionary, mapping) == expected


def test_rename_swap() -> None:
    # swap values -- the exchange has to be done simultaneously
    dictionary = {
        "a": "a",
        "b": "b"
    }
    mapping = {
        "a": "b",
        "b": "a"
    }
    expected = {
        "b": "a",
        "a": "b"
    }

    assert rename_keys(dictionary, mapping) == expected


def test_rename_transitive() -> None:
    # a = b, b = c -- this needs to be done in order and needs to drop the old c value
    # (implicitly replace it)
    dictionary = {
        "a": "a",
        "b": "b",
        "c": "c"
    }
    mapping = {
        "a": "b",
        "b": "c"
    }
    expected = {
        "b": "a",
        "c": "b"
    }

    assert rename_keys(dictionary, mapping) == expected


def test_delete_keys() -> None:
    # remove given keys
    dictionary = {
        "a": "a",
        "b": "b",
        "c": "c"
    }
    expected = {
        "c": "c"
    }
    delete_keys(dictionary, frozenset(["a", "b"]))
    assert dictionary == expected


def test_delete_keys_keep() -> None:
    # keep given keys
    dictionary = {
        "a": "a",
        "b": "b",
        "c": "c"
    }
    expected = {
        "a": "a"
    }
    delete_keys(dictionary, frozenset("a"), keep=True)
    assert dictionary == expected


def test_delete_all_keys() -> None:
    dictionary = {
        "a": "a",
        "b": "b",
        "c": "c"
    }

    delete_keys(dictionary, keys=None)
    assert not dictionary


def test_delete_keep_all_keys() -> None:
    dictionary = {
        "a": "a",
        "b": "b",
        "c": "c"
    }

    delete_keys(dictionary, keys=None, keep=True)
    assert dictionary == {"a": "a", "b": "b", "c": "c"}


@pytest.fixture
def seq_of_dict() -> SEQ_OF_DICT:
    return [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


@pytest.fixture
def dict_of_seq() -> DICT_OF_SEQ:
    return {
        "a": [1, 3],
        "b": [2, 4],
    }


def test_seq_of_dict_to_dict_of_seq(seq_of_dict: SEQ_OF_DICT, dict_of_seq: DICT_OF_SEQ) -> None:
    result = seq_of_dict_to_dict_of_seq(seq_of_dict)
    assert result == dict_of_seq


def test_dict_of_seq_to_seq_of_dict(dict_of_seq: DICT_OF_SEQ, seq_of_dict: SEQ_OF_DICT) -> None:
    result = dict_of_seq_to_seq_of_dict(dict_of_seq)
    assert result == seq_of_dict


def test_subscript_dict_of_seq(dict_of_seq: DICT_OF_SEQ) -> None:
    expected = {"a": 3, "b": 4}

    result = subscript_dict_of_seq(dict_of_seq, 1)
    assert result == expected


def test_ordered_values_tuple() -> None:
    values = {"a": 3, "b": 4}
    keys = ["b", "a"]
    expected = (4, 3)

    result = ordered_values_tuple(values, keys)
    assert result == expected

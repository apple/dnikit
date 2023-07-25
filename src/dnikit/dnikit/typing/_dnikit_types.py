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

import pathlib
import numpy as np
from typing import (
    Any,
    Optional,
    Type,
    TypeVar,
    Union,
    AbstractSet,
    Collection,
    List,
    cast,
    Tuple,
)


# Add or own type extensions to deal with one or many of something
_T = TypeVar("_T")  # _T can be anything
OneOrMany = Union[_T, Collection[_T]]
OneManyOrNone = Union[None, _T, Collection[_T]]

# Enable usage of either a str or pathlib.Path
PathOrStr = Union[str, pathlib.Path]

# Similar to "array like" -- these are types that can be losslessly converted
# to/from string and they might be used as Identifiers.
StringLike = Any

TrainTestSplitType = Tuple[Tuple[np.ndarray, np.ndarray],
                           Tuple[np.ndarray, np.ndarray]]
""" A datatype for the typical ``(x_train, y_train), (x_test, y_test)`` split. """


def resolve_one_or_many(x: OneOrMany[_T], cls: Type[_T]) -> AbstractSet[_T]:
    """
    Convert a ``OneOrMany[_T]`` to a frozenset of ``_T``.

    Args:
        x: object or collection of objects
        cls: type of each object (or type of individual object)

    Return:
        frozen set of objects
    """
    return frozenset((x,)) if isinstance(x, cls) else frozenset(cast(Collection[_T], x))


def resolve_one_many_or_none(x: OneManyOrNone[_T], cls: Type[_T]) -> Optional[AbstractSet[_T]]:
    """
    Convert a ``OneManyOrNone[_T]`` to a frozenset of ``_T`` or ``None``.

    Args:
        x: object or collection of objects or None
        cls: type of each object (or type of individual object)

    Return:
        frozen set of objects or ``None``
    """
    if x is None:
        return None
    return resolve_one_or_many(x, cls)


def resolve_one_or_many_to_list(x: OneOrMany[_T], cls: Type[_T]) -> List[_T]:
    """
    Convert a ``OneOrMany[_T]`` to a *list* of ``_T``.

    Args:
        x: object or collection of objects
        cls: type of each object (or type of individual object)

    Return:
        list of objects
    """
    return list((x,)) if isinstance(x, cls) else list(cast(Collection[_T], x))


def resolve_path_or_str(x: PathOrStr) -> pathlib.Path:
    """
    Convert a :class:`pathlib.Path` or string path into a pathlib Path

    Args:
        x: :class:`pathlib.Path` or str pathname

    Return:
        :class:`pathlib.Path` instance of the path
    """
    return x if isinstance(x, pathlib.Path) else pathlib.Path(x)

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

import sys

# Import commonly used typing members
# This is a selection of commonly typing Types in dnikit, feel free to add one if needed
# The purpose of including this is to resolve the typing vs. typing_extensions import
# exceptions once.
# See current members of typing: https://github.com/python/cpython/blob/3.8/Lib/typing.py
# Note: remember to export these on __all__
from typing import (
    # Super-special typing primitives.
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    # ABCs (from collections.abc).
    AbstractSet,  # collections.abc.Set.
    Container,
    ContextManager,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Collection,
    Hashable,
    # Concrete collection types.
    Dict,
    List,
    Set,
    FrozenSet,
    Generator,
    DefaultDict,
    # One-off things.
    cast,
    NewType,
    overload,
    TYPE_CHECKING,
)

# Import features from typing extensions that are used throughout the codebase
# https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-new-additions-to-the-typing-module
if sys.version_info >= (3, 8):
    from typing import final, Final, Protocol, Literal, runtime_checkable
else:
    from typing_extensions import final, Final, Protocol, Literal, runtime_checkable


# Declare all public members
__all__ = [
    # Imports from typing
    "Any",
    "Callable",
    "ClassVar",
    "Generic",
    "Optional",
    "Tuple",
    "Type",
    "TypeVar",
    "Union",
    "AbstractSet",
    "Container",
    "ContextManager",
    "Iterable",
    "Iterator",
    "KeysView",
    "Mapping",
    "MutableMapping",
    "MutableSequence",
    "MutableSet",
    "Sequence",
    "Collection",
    "Hashable",
    "Dict",
    "List",
    "Set",
    "FrozenSet",
    "Generator",
    "DefaultDict",
    "cast",
    "NewType",
    "overload",
    "TYPE_CHECKING",
    # Imports from typing extensions
    "final",
    "Final",
    "Protocol",
    "Literal",
    "runtime_checkable",
]

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

import dnikit.typing._types as t


# valid characters for remove_special_characters
_characters_to_keep: t.Final = frozenset(
    'abcdefghijklmnopqrstuvwxyz' +
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    '0123456789' +
    '-_')


def remove_special_characters(value: str, keep: t.Optional[frozenset] = None) -> str:
    """
    Convert a string with spaces, and other characters into something that looks more like a
    path (restricted character set).

    Args:
        value: string to convert
        keep: optional set of characters to keep.  the default is [a-zA-Z0-9]
            plus dash and underscore

    Returns:
        the cleaned version of the string (spaces replaced with _ and special characters removed)
    """
    value = value.replace(" ", "_")
    value = ''.join([c for c in value if c in (keep or _characters_to_keep)])
    return value

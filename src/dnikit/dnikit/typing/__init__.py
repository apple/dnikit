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

from ._dnikit_types import (
    OneOrMany,
    OneManyOrNone,
    PathOrStr,
    StringLike,
    TrainTestSplitType,
    resolve_one_or_many,
    resolve_one_many_or_none,
    resolve_one_or_many_to_list,
    resolve_path_or_str
)


__all__ = [
    "OneOrMany",
    "OneManyOrNone",
    "PathOrStr",
    "StringLike",
    "TrainTestSplitType",
    "resolve_one_or_many",
    "resolve_one_many_or_none",
    "resolve_one_or_many_to_list",
    "resolve_path_or_str",
]

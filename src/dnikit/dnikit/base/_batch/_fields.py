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

import dataclasses

import numpy as np

import dnikit.typing._types as t


@t.final
@dataclasses.dataclass(frozen=True)
class _Fields(t.Mapping[str, np.ndarray]):
    _storage: t.Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __getitem__(self, field: str) -> np.ndarray:
        return self._storage[field]

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._storage)

    def __len__(self) -> int:
        return len(self._storage)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Fields):
            return False
        return (
            frozenset(self.keys()) == frozenset(other.keys())
            and all(np.array_equal(self[k], other[k]) for k in self)
        )

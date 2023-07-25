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

# Use this file to define shared fixtures for tests
# See https://docs.pytest.org/en/stable/fixture.html for the list of built-in fixture (including
# `tmp_path`) as well as more information about conftest.py

import typing as t

import pytest
import numpy as np

from dnikit.base import Batch, Producer

_BATCH_LENGTH = 64


class _TestProducer:
    def __init__(self) -> None:
        self._data = {
            "field_a": np.random.randn(_BATCH_LENGTH, 3, 7, 11),
            "field_b": np.random.randn(_BATCH_LENGTH, 83),
            "field_c": np.random.randn(_BATCH_LENGTH, 11, 127),
            "field_d": np.random.randn(_BATCH_LENGTH, 31, 5)
        }
        self._num_calls = 0

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        # Count number of times batches are produced
        self._num_calls += 1
        for start in range(0, _BATCH_LENGTH, batch_size):
            # Compute start and end indices
            end = start + batch_size
            if end > _BATCH_LENGTH:
                end = _BATCH_LENGTH

            # Yield batch of requested size (or lower)
            yield Batch({
                field: value[start:end, ...]
                for field, value in self._data.items()
            })

    @property
    def num_calls(self) -> int:
        return self._num_calls


@pytest.fixture
def producer() -> Producer:
    return _TestProducer()


@pytest.fixture
def producer_with_num_calls() -> t.Tuple[Producer, t.Callable[[], int]]:
    result = _TestProducer()

    def num_calls_getter() -> int:
        return result._num_calls

    return result, num_calls_getter

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

import typing as t

import numpy as np
import pytest

from dnikit.base import Batch, Producer
from dnikit.base._producer import _accumulate_batches, _resize_batches, _produce_elements
from dnikit.exceptions import DNIKitException


def _stub_producer(batch_size: int) -> t.Iterable[Batch]:
    return tuple()


def test_accumulate_batches(producer: Producer) -> None:
    accumulated = _accumulate_batches(producer)
    i = None
    batch = None
    for i, batch in enumerate(producer(1)):
        assert frozenset(batch.fields.keys()) == frozenset(accumulated.fields.keys())
        for f in batch.fields.keys():
            assert np.array_equal(accumulated.fields[f][i, ...], batch.fields[f][0, ...])

    assert i is not None
    assert batch is not None

    # Check shapes of accumulated batch
    for f in batch.fields.keys():
        expected_shape = (i + 1, *batch.fields[f].shape[1:])
        assert accumulated.fields[f].shape == expected_shape


def test_invalid_accumulate_batches() -> None:
    with pytest.raises(DNIKitException):
        _accumulate_batches(_stub_producer)


@pytest.mark.parametrize("batch_size, new_size", [
    (32, 7), (100, 2), (4, 17)
])
def test_resize_producer(batch_size: int, new_size: int) -> None:
    batches = [
        Batch({"data": np.arange(i, min(i + batch_size, 1000))})
        for i in range(0, 1000, batch_size)
    ]
    producer = _resize_batches(batches)
    seen_last_batch = False
    value = 0
    for i, batch in enumerate(producer(new_size)):
        assert batch.batch_size <= new_size
        assert not seen_last_batch
        # Allow only one batch (the last one) to be smaller than producer_batch_size
        seen_last_batch = batch.batch_size < new_size
        # Check data
        for j, element in enumerate(batch.elements):
            value = element.fields["data"].item()
            assert value == (i * new_size) + j

    assert value == 999


@pytest.mark.parametrize("batch_size", [4, 32, 100])
def test_produce_elements(batch_size: int) -> None:
    def my_producer(batch_size: int) -> t.Iterable[Batch]:
        yield from (
            Batch({"data": np.arange(i, min(i + batch_size, 1000))})
            for i in range(0, 1000, batch_size)
        )

    for i, element in enumerate(_produce_elements(my_producer)):
        assert element.fields["data"].item() == i

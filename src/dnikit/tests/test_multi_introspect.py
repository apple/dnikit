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

import dataclasses
import typing as t

import numpy as np
import pytest

from dnikit.base import Introspector, Producer, Batch, multi_introspect
from dnikit.exceptions import DNIKitException

_BATCH_SIZE = 32


@dataclasses.dataclass(frozen=True)
class Adder(Introspector):
    sum_value: float
    batches_consumed: int

    @staticmethod
    def introspect(producer: Producer) -> "Adder":
        sum_value = 0
        batches_consumed = 0
        for batch in producer(batch_size=_BATCH_SIZE):
            batches_consumed += 1
            sum_value += np.sum(batch.fields["data"]).item()
        return Adder(sum_value, batches_consumed)


@dataclasses.dataclass(frozen=True)
class Maxer(Introspector):
    max_value: float
    batches_consumed: int

    @staticmethod
    def introspect(producer: Producer) -> "Maxer":
        max_value = 0
        batches_consumed = 0
        for batch in producer(batch_size=_BATCH_SIZE):
            batches_consumed += 1
            max_value = max(np.max(batch.fields["data"]).item(), max_value)
        return Maxer(max_value, batches_consumed)


def faulty_introspect(producer: Producer) -> None:
    count = 0
    for _ in producer(batch_size=_BATCH_SIZE):
        count += 1
        if count == 2:
            raise RuntimeError("Faulty introspector is faulty")


@dataclasses.dataclass
class MyProducer(Producer):
    times_called: int = dataclasses.field(default=0)
    batches_produced: int = dataclasses.field(default=0)
    max_value: t.ClassVar[int] = 63

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        self.times_called += 1
        begin = 0
        while begin <= self.max_value:
            end = min(begin + batch_size, self.max_value + 1)
            data = np.arange(begin, end)
            yield Batch({"data": data})
            begin = end
            self.batches_produced += 1


@dataclasses.dataclass
class FaultyProducer(Producer):
    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        count = 0
        while True:
            data = np.random.random(32)
            yield Batch({"data": data})
            count += 1
            if count == 3:
                raise RuntimeError("Faulty producer is faulty")


@pytest.mark.timeout(60, method="thread")
def test_multi_introspect_with_single_introspector() -> None:
    producer = MyProducer()
    results = multi_introspect(Maxer.introspect, producer=producer)
    assert len(results) == 1
    assert producer.times_called == 1
    assert isinstance(results[0], Maxer)
    assert results[0].max_value == producer.max_value
    assert results[0].batches_consumed == producer.batches_produced


@pytest.mark.timeout(60, method="thread")
def test_multi_introspect_with_two_introspectors() -> None:
    producer = MyProducer()
    adder, maxer = multi_introspect(
        Adder.introspect,
        Maxer.introspect,
        producer=producer
    )
    assert producer.times_called == 1

    assert isinstance(adder, Adder)
    assert adder.sum_value == (1 + producer.max_value) * (producer.max_value/2)
    assert adder.batches_consumed == producer.batches_produced

    assert isinstance(maxer, Maxer)
    assert maxer.max_value == producer.max_value
    assert maxer.batches_consumed == producer.batches_produced


@pytest.mark.timeout(60, method="thread")
def test_multi_introspect() -> None:
    producer = MyProducer()
    results = multi_introspect(
        Adder.introspect,
        Maxer.introspect,
        Adder.introspect,
        Maxer.introspect,
        producer=producer
    )
    assert len(results) == 4
    assert producer.times_called == 1

    assert isinstance(results[0], Adder)
    assert results[0].sum_value == (1 + producer.max_value) * (producer.max_value / 2)
    assert results[0].batches_consumed == producer.batches_produced

    assert isinstance(results[1], Maxer)
    assert results[1].max_value == producer.max_value
    assert results[1].batches_consumed == producer.batches_produced

    assert isinstance(results[2], Adder)
    assert results[2].sum_value == results[0].sum_value
    assert results[2].batches_consumed == producer.batches_produced

    assert isinstance(results[3], Maxer)
    assert results[3].max_value == results[1].max_value
    assert results[3].batches_consumed == producer.batches_produced


@pytest.mark.timeout(60, method="thread")
def test_multi_introspect_with_faulty_producer() -> None:
    with pytest.raises(DNIKitException) as exc_info:
        _ = multi_introspect(
            Maxer.introspect,
            Adder.introspect,
            producer=FaultyProducer()
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "Faulty producer is faulty"


@pytest.mark.timeout(60, method="thread")
def test_multi_introspect_with_faulty_introspector() -> None:
    with pytest.raises(DNIKitException) as exc_info:
        _ = multi_introspect(
            Maxer.introspect,
            Adder.introspect,
            faulty_introspect,
            producer=MyProducer()
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "Faulty introspector is faulty"

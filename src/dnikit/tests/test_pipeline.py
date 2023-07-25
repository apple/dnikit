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

from dnikit.base import Batch, PipelineStage, pipeline, peek_first_batch

_BATCH_SIZE = 64


def sample_producer(batch_size: int) -> t.Iterable[Batch]:
    for _ in range(100):
        yield Batch({"input": np.random.random((batch_size, 32, 64))})


class NewFieldAdder(PipelineStage):
    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            new_batch = {}
            for field in batch.fields.keys():
                new_batch[field] = batch.fields[field]
                new_batch[f"{field}_doubled"] = batch.fields[field]
                new_batch[f"{field}_halved"] = batch.fields[field]
            return Batch(new_batch)
        return batch_processor


class Halver(PipelineStage):
    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            new_batch = {}
            for field in batch.fields.keys():
                if "halved" in field:
                    new_batch[field] = batch.fields[field] / 2.0
                else:
                    new_batch[field] = batch.fields[field]
            return Batch(new_batch)
        return batch_processor


class Doubler(PipelineStage):
    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            new_batch = {}
            for field in batch.fields.keys():
                if "doubled" in field:
                    new_batch[field] = batch.fields[field] * 2.0
                else:
                    new_batch[field] = batch.fields[field]
            return Batch(new_batch)
        return batch_processor


def test_pipeline() -> None:
    producer = pipeline(sample_producer, NewFieldAdder(), Halver(), Doubler())
    # Generate a single batch
    batch = peek_first_batch(producer, _BATCH_SIZE)
    # Check batch was generated as expected
    assert batch.batch_size == _BATCH_SIZE
    assert "input" in batch.fields
    assert "input_doubled" in batch.fields
    assert "input_halved" in batch.fields
    assert np.allclose(batch.fields["input_doubled"], batch.fields["input"] * 2.0)
    assert np.allclose(batch.fields["input_halved"], batch.fields["input"] / 2.0)


def test_pipeline_wrong_order() -> None:
    producer = pipeline(sample_producer, Halver(), Doubler(), NewFieldAdder())
    # Generate a single batch
    batch = peek_first_batch(producer, _BATCH_SIZE)
    # Check batch was generated as expected
    assert batch.batch_size == _BATCH_SIZE
    assert "input" in batch.fields
    assert "input_doubled" in batch.fields
    assert "input_halved" in batch.fields
    assert batch.fields["input_doubled"] is batch.fields["input"]
    assert batch.fields["input_halved"] is batch.fields["input"]


def test_pipeline_nested_tuple() -> None:
    num_batches = 32
    data = np.random.random((_BATCH_SIZE*num_batches, 32, 64))

    def consistent_producer(batch_size: int) -> t.Iterable[Batch]:
        # A producer that pulls from the same underlying data
        for k in range(num_batches):
            idx = k * batch_size
            yield Batch({"input": data[idx:idx+batch_size, ...]})

    # Generate three pipelines: one without tuples,
    # two with different tuples but same overall order:
    prod_orig = pipeline(consistent_producer, NewFieldAdder(), Halver(), Doubler())
    prod_a = pipeline(consistent_producer, (NewFieldAdder(), Halver()), Doubler())
    prod_b = pipeline(consistent_producer, NewFieldAdder(), (Halver(), Doubler()))
    # Generate a single batch for each
    batch_orig = peek_first_batch(prod_orig, _BATCH_SIZE)
    batch_a = peek_first_batch(prod_a, _BATCH_SIZE)
    batch_b = peek_first_batch(prod_b, _BATCH_SIZE)

    # Verify batches
    def verify_batch(batch: Batch) -> None:
        assert batch.batch_size == _BATCH_SIZE
        assert "input" in batch.fields
        assert "input_doubled" in batch.fields
        assert "input_halved" in batch.fields
        assert np.allclose(batch.fields["input_doubled"], batch.fields["input"] * 2.0)
        assert np.allclose(batch.fields["input_halved"], batch.fields["input"] / 2.0)
    verify_batch(batch_orig)
    verify_batch(batch_a)
    verify_batch(batch_b)

    # Check that batches are equal
    def batch_equal(a: Batch, b: Batch) -> None:
        assert np.allclose(a.fields["input"], b.fields["input"])
        assert np.allclose(a.fields["input_doubled"], b.fields["input_doubled"])
        assert np.allclose(a.fields["input_halved"], b.fields["input_halved"])
    batch_equal(batch_orig, batch_a)
    batch_equal(batch_orig, batch_b)

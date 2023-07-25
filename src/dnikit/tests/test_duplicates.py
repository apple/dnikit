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

from dataclasses import dataclass
import pytest
import typing as t
import numpy as np

from dnikit.introspectors import Duplicates
from dnikit.base import pipeline, Producer, PipelineStage, Batch
from dnikit.samples import StubProducer


@dataclass(frozen=True)
class AddIndexIdentifier(PipelineStage):
    """
    Simple PipelineStage that will add an identifier to ``Batch.StdKeys.IDENTIFIER``.

    .. code-block:: python

        index_identifier = AddIndexIdentifier()

        producer = pipeline(initial_producer, index_identifier)
    """

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        # an array that can be captured to provide an incrementing counter
        base_index = [0]

        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)

            start = base_index[0]
            end = base_index[0] + batch.batch_size
            builder.metadata[Batch.StdKeys.IDENTIFIER] = list(range(start, end))

            base_index[0] += batch.batch_size
            return builder.make_batch()

        return batch_processor


@pytest.fixture
def duplicate_producer() -> Producer:
    dataset_size = 5000
    random_s = np.random.RandomState(seed=42)
    response_stub_data = {
        "a": np.concatenate((
            random_s.normal(10, .01, (int(dataset_size / 10), 10)),  # 10% duplicates
            random_s.normal(100, 50, (int(9 * dataset_size / 10), 10))
        )),
        "b": np.concatenate((
            random_s.normal(10, .01, (int(dataset_size / 5), 10)),  # 20% duplicates
            random_s.normal(100, 50, (int(4 * dataset_size / 5), 10))
        )),
        "c": np.concatenate((
            random_s.normal(10, .01, (int(dataset_size / 2), 10)),  # 50% duplicates
            random_s.normal(100, 50, (int(dataset_size / 2), 10))
        )),
    }
    return pipeline(StubProducer(response_stub_data), AddIndexIdentifier())


def test_duplicate_introspector(duplicate_producer: Producer) -> None:
    # use percentile to find the threshold -- this is normal distribution data
    # so it isn't shaped like the typical "long tail"
    duplicates = Duplicates.introspect(duplicate_producer,
                                       threshold=Duplicates.ThresholdStrategy.Percentile(98))
    assert duplicates is not None

    # Response C should find the most duplicates, then B, then A, with the fewest
    assert len(duplicates.results['a']) < len(duplicates.results['b'])
    assert len(duplicates.results['b']) < len(duplicates.results['c'])


def test_build_inverse_mapping() -> None:
    duplicate_list = [
        [10, 11, 12, 13],
        [1, 2, 3, 4],
        [6, 8],
        [1, 2, 3, 7]
    ]

    ground_truth = {
        frozenset([10, 11, 12, 13]),
        frozenset([1, 2, 3, 4, 7]),
        frozenset([6, 8]),
    }

    clusters = Duplicates._combine_clusters(duplicate_list)

    assert len(clusters) == len(ground_truth)
    for c in clusters:
        assert set(c) in ground_truth


def test_build_inverse_mapping_join_cluster() -> None:
    duplicate_list = [
        # joins on the 2
        [1, 2, 3, 4],
        [7, 2, 8, 9],
        # joins on the 11
        [10, 11, 12],
        [11, 13],
        # combines both of those clusters
        [1, 13],
    ]

    ground_truth = {
        frozenset([1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]),
    }

    clusters = Duplicates._combine_clusters(duplicate_list)

    assert len(clusters) == len(ground_truth)
    for c in clusters:
        assert set(c) in ground_truth

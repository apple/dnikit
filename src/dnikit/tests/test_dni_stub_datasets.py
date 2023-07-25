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

"""Test cases for components under the dnikit.dataset module."""
import pytest
from _pytest.fixtures import SubRequest
import typing as t

from dnikit.samples import StubImageDataset, StubGatedAdditionDataset


def test_stub_image_dataset_batch_size_divisible() -> None:
    # dataset_size, image_width=640, image_height=480, channel_count=3
    stub_dataset_size = 110
    dataset = StubImageDataset(dataset_size=stub_dataset_size)
    total_data_points = 0
    for batch in dataset(batch_size=10):
        assert batch.batch_size <= 10
        total_data_points += batch.batch_size

    assert total_data_points == stub_dataset_size


def test_stub_gated_addition_dataset() -> None:
    # dataset_size, minimum_sequence_length, maximum_sequence_length, hidden_size
    stub_dataset_size = 108
    dataset = StubGatedAdditionDataset(dataset_size=stub_dataset_size)
    total_data_points = 0
    for batch in dataset(batch_size=10):
        assert batch.batch_size <= 10
        total_data_points += batch.batch_size

    assert total_data_points == stub_dataset_size


@pytest.fixture(params=[(10, 1), (10, 2), (11, 10), (10, 10), (10, 11)])
def dataset_batch_combo(request: SubRequest) -> t.Tuple[int, int]:
    return request.param


def test_stub_image_dataset(dataset_batch_combo: t.Tuple[int, int]) -> None:
    stub_dataset_size = dataset_batch_combo[0]
    dataset = StubImageDataset(dataset_size=stub_dataset_size)
    total_data_points = 0
    for batch in dataset(batch_size=dataset_batch_combo[1]):
        assert batch.batch_size <= dataset_batch_combo[1]
        total_data_points += batch.batch_size

    assert total_data_points == stub_dataset_size, (
        f'Dataset size {dataset_batch_combo[0]}, batch size {dataset_batch_combo[1]}. '
        f'Only returned {total_data_points} data points'
    )


def test_stub_image_dataset_zero_batch() -> None:
    stub_dataset_size = 10
    dataset = StubImageDataset(dataset_size=stub_dataset_size)

    with pytest.raises(ValueError):
        for _ in dataset(batch_size=0):
            continue

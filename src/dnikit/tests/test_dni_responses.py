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

import pytest
import typing as t
import numpy as np

from dnikit.samples import StubProducer

_RESPONSE_LENGTH = 123


@pytest.mark.parametrize(
    'data',
    [
        ({
            'response_a': np.random.randn(_RESPONSE_LENGTH, 3, 7, 11),
            'response_b': np.random.randn(_RESPONSE_LENGTH, 83),
            'response_c': np.random.randn(_RESPONSE_LENGTH, 11, 127),
            'response_d': np.random.randn(_RESPONSE_LENGTH, 31, 5)
        }),
    ])
def test_stub_response_producer(data: t.Mapping[str, np.ndarray]) -> None:
    producer = StubProducer(data)
    response_length: int = data[list(data.keys())[0]].shape[0]
    for batch_size in [1, 7, 11, response_length, response_length + 11]:
        produced_data = {}
        for batch in producer(batch_size):
            for response_field in batch.fields:
                if response_field not in produced_data:
                    produced_data[response_field] = batch.fields[response_field]
                else:
                    produced_data[response_field] = np.concatenate(
                        (produced_data[response_field],
                         batch.fields[response_field]),
                        axis=0
                    )

        for response_field in data:
            assert (produced_data[response_field].shape == data[response_field].shape)
            assert (np.array_equal(produced_data[response_field], data[response_field]))

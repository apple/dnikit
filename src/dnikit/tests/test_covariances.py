#
# Copyright 2019 Apple Inc.
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

import numpy as np

from dnikit.base import pipeline, Producer
from dnikit.samples import StubProducer
from dnikit.processors import Pooler
from dnikit.introspectors._pfa import _covariances_calculator as oncov

RESPONSE_LENGTH = 84

STUB_DATA = {
    'layers_a': np.random.randn(RESPONSE_LENGTH, 5, 7, 3),  # NHWC images
    'layers_b': np.random.randn(RESPONSE_LENGTH, 32, 20, 16),  # NHWC images
    'layers_c': np.random.randn(RESPONSE_LENGTH, 11),  # 1D vectors
}

FULL_RESP = ['layers_a', 'layers_b']


@pytest.fixture
def producer() -> Producer:
    return pipeline(
        StubProducer(STUB_DATA),
        # limit the pooling operation to 4D tensors
        Pooler(dim=(1, 2), method=Pooler.Method.MAX, fields=FULL_RESP)
    )


def test_covariances_result_shapes(producer: Producer) -> None:

    cov_results = oncov._prepare_covariances(7, producer)

    assert(set(cov_results.keys()) == set(STUB_DATA.keys()))

    for resp_name in STUB_DATA.keys():
        n_channels = STUB_DATA[resp_name].shape[-1]

        cov = cov_results[resp_name].get_centered_covariances()

        assert(isinstance(cov, np.ndarray))
        assert(cov.shape == (n_channels, n_channels))


def test_covariances_results(producer: Producer) -> None:

    cov_results = oncov._prepare_covariances(7, producer)

    for resp_name, stub_array in STUB_DATA.items():

        if len(stub_array.shape) > 2:
            # Simulating max pooling
            stub_array_2d = np.max(stub_array, axis=(1, 2))
        else:
            stub_array_2d = stub_array

        reference_covariance = np.cov(stub_array_2d, rowvar=False, bias=False)
        computed_covariance = cov_results[resp_name].get_centered_covariances()

        assert(np.allclose(computed_covariance, reference_covariance))


def test_simple_online_test() -> None:
    # Assume batch size = 20
    # Assume number of filters/channels = 10
    r1 = np.random.random((20, 10))
    r2 = np.random.random((20, 10))
    r3 = np.random.random((20, 10))
    r123 = np.concatenate((r1, r2, r3), axis=0)

    np_cov = np.cov(r123.T)
    s_cov = oncov._CovariancesCalculator()

    s_cov.add_batch(r1)
    s_cov.add_batch(r2)
    s_cov.add_batch(r3)

    s_cov_centered_not_biased = s_cov.get_centered_covariances()

    assert np.allclose(s_cov_centered_not_biased, np_cov)

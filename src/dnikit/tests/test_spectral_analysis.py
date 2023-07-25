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

import pytest
import numpy as np

from dnikit.samples import StubProducer
from dnikit.exceptions import DNIKitException
from dnikit.introspectors import PFA


def test_invalid_responses() -> None:
    response_length = 31
    stub_data_list = [
        {
            'response_a': np.random.randn(response_length, 3, 7, 11),
            'pooled_response_b': np.random.randn(response_length, 83)
        },
        {
            'pooled_response_c': np.random.randn(response_length, 83),
            'response_d': np.random.randn(response_length, 3, 7, 11)
        }
    ]

    for stub_data in stub_data_list:
        response_generator = StubProducer(stub_data)

        with pytest.raises(DNIKitException):
            PFA.introspect(response_generator)


def test_spectral_analysis_result_shapes() -> None:
    response_length = 84
    stub_data = {
        'pooled_response_a': np.random.randn(response_length, 3),
        'pooled_response_b': np.random.randn(response_length, 83),
        'pooled_response_c': np.random.randn(response_length, 11),
        'pooled_response_d': np.random.randn(response_length, 7)
    }

    response_generator = StubProducer(stub_data)
    spectral_analysis_results = PFA.introspect(response_generator, batch_size=7)._internal_result

    assert(set(spectral_analysis_results.keys())
           == set(stub_data.keys()))

    for response_name in stub_data.keys():
        assert spectral_analysis_results[response_name].covariances.shape == (
            stub_data[response_name].shape[1], stub_data[response_name].shape[1]
        )

        assert spectral_analysis_results[response_name].eigenvalues.shape == (
            stub_data[response_name].shape[1],
        )

        assert spectral_analysis_results[response_name].eigenvectors.shape == (
            stub_data[response_name].shape[1], stub_data[response_name].shape[1]
        )


def ensure_equal_covariance_rank(stub_data: t.Mapping[str, np.ndarray]) -> None:
    """
    Checks that the covariance of all responses contained in stub_data have the same rank
    by verifying that such covariance matrices have the same number of non-zero eigenvalues
    """
    response_generator = StubProducer(stub_data)
    spectral_analysis_results = PFA.introspect(response_generator)._internal_result

    non_zero_eigenvalues_by_response = {
        response_name: np.sum(1 - np.isclose(0,
                                             spectral_analysis_results[response_name].eigenvalues))
        for response_name in spectral_analysis_results.keys()
    }

    non_zero_eigenvalues_set = set(
        non_zero_eigenvalues_by_response[response_name]
        for response_name in non_zero_eigenvalues_by_response
    )
    assert(len(non_zero_eigenvalues_set) == 1)


def test_data_rank_conservation_wrt_feature_repetition() -> None:
    """
    Test that adding copies of existing features doesn't change the number of non-zero
    eigenvalues of the covariance matrix of the feature matrix
    """
    response_length = 31
    full_rank_data = np.random.randn(response_length, 3)
    stub_data = {
        'pooled_response_a': full_rank_data,
        'pooled_response_b': np.concatenate(
            (full_rank_data, full_rank_data),
            axis=1),
        'pooled_response_c': np.concatenate(
            (full_rank_data, full_rank_data, full_rank_data),
            axis=1),
        'pooled_response_d': np.concatenate(
            (full_rank_data, full_rank_data, full_rank_data, full_rank_data),
            axis=1),
    }

    ensure_equal_covariance_rank(stub_data)


def test_data_rank_conservation_wrt_feature_linear_combination() -> None:
    """
    Test that adding copies of existing features doesn't change the number of non-zero
    eigenvalues of the covariance matrix of the feature matrix
    """

    def make_random_linear_combinations(x: np.ndarray, output_count: int) -> np.ndarray:
        # Creates `output_count` features that are linear combinations of features in x"""
        random_state = np.random.RandomState()
        results = np.zeros((x.shape[0], output_count))

        for i in range(output_count):
            linear_comb_coeefs = random_state.randn(1, x.shape[1])
            results[:, i] = np.sum(np.multiply(x, linear_comb_coeefs), axis=1)

        return results

    response_length = 31
    full_rank_data = np.random.randn(response_length, 3)
    stub_data = {
        'pooled_response_a': full_rank_data,
        'pooled_response_b': np.concatenate(
            (full_rank_data, make_random_linear_combinations(full_rank_data, 3)),
            axis=1),
        'pooled_response_c': np.concatenate(
            (full_rank_data, make_random_linear_combinations(full_rank_data, 7)),
            axis=1),
        'pooled_response_d': np.concatenate(
            (full_rank_data, make_random_linear_combinations(full_rank_data, 11)),
            axis=1),
    }

    ensure_equal_covariance_rank(stub_data)

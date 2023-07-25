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

import dnikit.typing._types as t
import numpy as np
import pytest

from dnikit.introspectors._familiarity._gaussian_familiarity import (
    _MixtureOfMultivariateGaussianDistributions
)
from dnikit.base import Producer, pipeline
from dnikit.introspectors import Familiarity, GMMCovarianceType

from dnikit.samples import StubProducer
from sklearn.mixture import GaussianMixture


def create_familiarity_processor(gaussian_count: int,
                                 covariance_type: GMMCovarianceType = GMMCovarianceType.FULL,
                                 ) -> t.Tuple[Producer, t.Mapping[str, np.ndarray], Familiarity]:
    random_state = np.random.RandomState(seed=42)

    dataset_size = 500
    stub_data = {
        'response_a': random_state.randn(dataset_size, 3),
        'response_b': random_state.randn(dataset_size, 83),
        'response_c': random_state.randn(dataset_size, 11),
        'response_d': random_state.randn(dataset_size, 31),
        'response_e': random_state.randn(dataset_size, 1)
    }
    source = StubProducer(stub_data)

    # provide a random state for GMMFamiliarityAlgorithm to produce consistent output
    strategy = Familiarity.Strategy.GMM(gaussian_count=gaussian_count,
                                        convergence_threshold=1e-3,
                                        max_iterations=200,
                                        covariance_type=covariance_type,
                                        _random_state=np.random.RandomState(seed=1))
    familiarity_processor = Familiarity.introspect(source, strategy=strategy)
    producer = pipeline(source, familiarity_processor)

    return producer, stub_data, familiarity_processor


def test_gmm_familiarity_introspector() -> None:
    producer, stub_data, fam_processor = create_familiarity_processor(gaussian_count=5)

    expected_responses = set(stub_data.keys())
    for postprocessed_familiarity_batch in producer(11):
        assert set(postprocessed_familiarity_batch.fields.keys()) == expected_responses


def test_gmm_familiarity_introspector_result() -> None:
    producer, stub_data, fam_processor = create_familiarity_processor(gaussian_count=5)

    # because the random state is controlled for the fit, it's possible to verify that the
    # results stay the same over time
    for batch in producer(5):
        batch_results = batch.metadata[fam_processor.meta_key]['response_a']

        assert np.isclose(batch_results[0].score, -2.9240012833835696, atol=0.3)
        assert np.isclose(batch_results[1].score, -3.875730430163283, atol=0.3)

        break


@pytest.mark.parametrize("gaussian_count", [1, 5])
@pytest.mark.parametrize("number_new_samples", [1, 4])
@pytest.mark.parametrize("cov_type", [GMMCovarianceType.FULL,
                                      GMMCovarianceType.DIAG])
def test_gmm_familiarity_scores(gaussian_count: int, number_new_samples: int,
                                cov_type: GMMCovarianceType) -> None:
    """
    Test that the scores familiarity and check that results are the same as returned by the sklearn
    GaussianMixture
    """
    # Fit familiarity
    producer, stub_data, familiarity_processor = create_familiarity_processor(
        gaussian_count=gaussian_count,
        covariance_type=cov_type
    )

    # Fit sklearn GMM and compare with DNIKit fitted GMM
    sklearn_gm = {}
    random_state_instance = np.random.RandomState(seed=1)
    for response_name, response_data in stub_data.items():
        sklearn_gm[response_name] = GaussianMixture(n_components=gaussian_count,
                                                    max_iter=200,
                                                    tol=1e-3,
                                                    covariance_type=cov_type.value,
                                                    random_state=random_state_instance)
        sklearn_gm[response_name].fit(response_data)

        def extract_covs(c: np.ndarray) -> np.ndarray:
            """Returns squared covariances regardless of fitting type."""
            if cov_type is GMMCovarianceType.FULL:
                return c
            elif cov_type is GMMCovarianceType.DIAG:
                return np.array([np.diag(ci) for ci in c])
            else:
                raise NotImplementedError(f'Unknown covariance type {cov_type}')

        # compare fitted gaussians
        distribution = familiarity_processor._distributions[response_name]
        assert isinstance(distribution, _MixtureOfMultivariateGaussianDistributions)
        assert np.allclose(
            sklearn_gm[response_name].means_,
            [np.atleast_1d(np.squeeze(g.mean)) for g in distribution.gaussians]
        )
        assert np.allclose(
            extract_covs(sklearn_gm[response_name].covariances_),
            [g.covariance for g in distribution.gaussians]
        )
        assert np.allclose(
            sklearn_gm[response_name].weights_,
            distribution.weights
        )

    # Generate new data to be evaluated and compare evaluation
    random_state = np.random.RandomState(seed=12345)
    new_data = {}
    new_data_sk_score_sample = {}
    for response_name, response_data in stub_data.items():
        # generate test samples for each layer
        new_data[response_name] = random_state.randn(number_new_samples, response_data.shape[1])

        # compute weighted log-likelihood
        new_data_sk_score_sample[response_name] = (
            sklearn_gm[response_name].score_samples(new_data[response_name])
        )

    # Evaluate familiarity of new data
    new_data_producer = StubProducer(new_data)
    familiarity_producer = pipeline(new_data_producer, familiarity_processor)

    # Compare sk and dnikit familiarity evaluation on new data
    # Note: the current implementation achieves a slightly different results when evaluating the
    # log-pdf. This seems to be a numerical precision problem. For this reason, results are
    # evaluated in terms of pdf and not log-pdf.
    for i, batch in enumerate(familiarity_producer(1)):
        for response_name in batch.fields.keys():
            assert np.isclose(
                np.exp(new_data_sk_score_sample[response_name][i]),
                np.exp(batch.metadata[familiarity_processor.meta_key][response_name][0].score),
                atol=1e-2
            )

    # Compare sk and dnikit ranking
    # Note: due to the issue of numerical stability it can happen if the seed is changed that the
    # ranking becomes slightly different. See issue #427
    for response_name in stub_data.keys():
        dnikit_sorted_scores = sorted([
            (index, batch.metadata[familiarity_processor.meta_key][response_name][index].score)
            for batch in familiarity_producer(number_new_samples)
            for index in range(batch.batch_size)
        ], reverse=True, key=lambda x: x[1])

        sk_sorted_scores = sorted([
            (index, new_data_sk_score_sample[response_name][index])
            for index in range(new_data_sk_score_sample[response_name].shape[0])
        ], reverse=True, key=lambda x: x[1])

        # Note: if the number of samples is increased, it's possible to get a pair of samples where
        # order is swapped
        # Solution: would be to achieve same result as sklearn by improving numerical precision
        assert [v[0] for v in dnikit_sorted_scores] == [v[0] for v in sk_sorted_scores]

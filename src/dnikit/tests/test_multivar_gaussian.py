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

import pytest
import numpy as np
import scipy
import typing as t

from dnikit.introspectors._familiarity._gaussian_familiarity import (
    _MultivariateGaussianDistribution,
    _MixtureOfMultivariateGaussianDistributions,
)


@pytest.fixture
def feature_dimensions() -> t.Sequence[int]:
    return [1, 2, 11, 20]


def test_multivariate_gaussian_distribution(feature_dimensions: t.Sequence[int]) -> None:
    """
    Test that dnikit MultivariateGaussian's log-pdfs are similar to scipy's.
    """

    random_state = np.random.RandomState(42)

    for feature_dimension in feature_dimensions:
        data_point_count = random_state.randint(low=10, high=100)

        x = random_state.randn(data_point_count, feature_dimension)
        mean = np.zeros((1, feature_dimension))
        covariance = np.eye(feature_dimension)

        dni_log_pdfs = _MultivariateGaussianDistribution._create(
            mean,
            covariance
        ).evaluate_log_pdf(x)

        scipy_log_pdfs = scipy.stats.multivariate_normal.logpdf(x, mean.flatten(), covariance)

        assert np.isclose(
            np.sum(np.abs(dni_log_pdfs - scipy_log_pdfs)),
            0
        )


def test_mixture_of_multivariate_gaussian_distributions(feature_dimensions: t.Sequence[int]
                                                        ) -> None:
    """
    Given a gaussian mixture model with identical and equally weighted gaussians,
    verify that the mixture scores are identical to the single gaussian probability density.
    """

    random_state = np.random.RandomState(42)

    for feature_dimension in feature_dimensions:
        data_point_count = random_state.randint(low=10, high=100)

        x = random_state.randn(data_point_count, feature_dimension)
        mean = np.zeros((1, feature_dimension))
        covariance = np.eye(feature_dimension)

        # compute single gaussian scores
        single_multivariate_gaussian_distribution = _MultivariateGaussianDistribution._create(
            mean,
            covariance
        )
        single_gaussian_scores = single_multivariate_gaussian_distribution.evaluate_log_pdf(x)

        # compute mixture scores (with identical equally weighted gaussians)
        mixture = _MixtureOfMultivariateGaussianDistributions._create(
            mean_list=np.array([mean, mean]),
            covariance_list=np.array([covariance, covariance]),
            weights=np.array([0.5, 0.5])
        )
        result = mixture.compute_familiarity_score(x)

        # assert mixture scores identical to single gaussian scores
        assert np.isclose(
            np.sum(np.abs(single_gaussian_scores - [x.score for x in result])),
            0
        )

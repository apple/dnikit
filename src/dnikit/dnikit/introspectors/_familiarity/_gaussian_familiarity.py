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

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from ._protocols import FamiliarityDistribution, FamiliarityResult
import dnikit.typing._types as t


@t.final
@dataclass(frozen=True)
class _MultivariateGaussianDistribution:
    """
    A Multivariate Gaussian Distribution parameterized by a mean vector and a covariance matrix.
    """

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        if len(self.mean.shape) != 2 or self.mean.shape[0] != 1:
            raise ValueError(
                f"The mean vector had an unexpected shape: {self.mean.shape}. Expected shape (1, ?)"
            )

        if self.covariance.shape != (self.mean.shape[1], self.mean.shape[1]):
            raise ValueError(
                f"Specified covariance matrix has an unexpected shape: {self.covariance.shape}. "
                f"Given the mean vector shape {self.mean.shape}, the expected covariance matrix "
                f"shape is ({self.mean.shape[1]}, {self.mean.shape[1]})"
                )

    @staticmethod
    def _create(mean: np.ndarray, covariance: np.ndarray) -> "_MultivariateGaussianDistribution":
        if len(mean.shape) == 1:
            mean = np.expand_dims(mean, axis=0)

        return _MultivariateGaussianDistribution(mean, covariance)

    def evaluate_log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Return the log of the pdf evaluated for each point in x.
        """
        if len(x.shape) != 2 or x.shape[1] != self.mean.shape[1]:
            raise ValueError(
                f"The shape of input data matrix, {x.shape}, doesn't match this distribution."
                f"The expected shape of data matrix is (n_observations, {self.mean.shape[1]})"
            )

        log_pdfs = multivariate_normal.logpdf(x, mean=np.squeeze(self.mean), cov=self.covariance)

        return np.atleast_1d(log_pdfs)


@t.final
@dataclass(frozen=True)
class _MixtureOfMultivariateGaussianDistributions(FamiliarityDistribution):
    """
    A Mixture of Gaussian Distributions.
    """

    weights: np.ndarray
    """shape is (gaussian_count)"""
    log_weights: np.ndarray
    """ shape is (gaussian_count)"""
    gaussians: t.List[_MultivariateGaussianDistribution]
    """List of MultivariateGaussianDistribution, length is gaussian_count"""

    @t.final
    @dataclass(frozen=True)
    class GMMFamiliarityResult(FamiliarityResult):
        """Result of applying a _MixtureOfMultivariateGaussianDistributions to a response."""

        score: float
        """Familiarity score. Note: This will actually be the log score."""

    @staticmethod
    def _create(mean_list: np.ndarray, covariance_list: np.ndarray,
                weights: np.ndarray) -> "_MixtureOfMultivariateGaussianDistributions":

        log_weights = np.log(weights)

        if len(mean_list) != len(covariance_list) or len(covariance_list) != len(weights):
            raise ValueError(
                f"The cardinalities of mean_list ({len(mean_list)}), "
                f"covariance_list ({len(covariance_list)}) "
                f"and weights ({len(weights)}) must be equal."
            )

        gaussians = [
            _MultivariateGaussianDistribution._create(mean, covariance)
            for mean, covariance in zip(mean_list, covariance_list)
        ]

        return _MixtureOfMultivariateGaussianDistributions(weights, log_weights, gaussians)

    def compute_familiarity_score(self, x: np.ndarray) -> t.Sequence[FamiliarityResult]:
        """
        Compute and return the density of this mixture of distributions at :param:x.
        """

        total_log_pdf = []

        for index, gaussian in enumerate(self.gaussians):
            log_pdf = gaussian.evaluate_log_pdf(x)
            total_log_pdf.append(log_pdf + self.log_weights[index])
        log_scores = logsumexp(total_log_pdf, axis=0, keepdims=False, return_sign=False)

        return [
            _MixtureOfMultivariateGaussianDistributions.GMMFamiliarityResult(score=log_score)
            for log_score in log_scores
        ]

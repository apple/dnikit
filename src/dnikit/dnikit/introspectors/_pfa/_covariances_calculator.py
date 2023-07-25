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

from dataclasses import dataclass

import numpy as np
from collections import defaultdict

from dnikit.exceptions import DNIKitException
from dnikit.base import Producer
import dnikit.typing._types as t


def _compute_eigens(input_matrix: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Compute and return the eigenvalues and eigenvectors of `input_matrix`
    in descending order of the eigenvalues
    """
    eigenvalues, eigenvectors = np.linalg.eigh(input_matrix)
    eigenvalues = np.maximum(0.0, eigenvalues)
    return eigenvalues[::-1], eigenvectors[::-1]


@t.final
@dataclass(frozen=True)
class PFACovariancesResult:
    """
    Encapsulates the results of the covariance calculation

    Args:
        covariances: see :attr:`covariances`
        eigenvalues: see :attr:`eigenvalues`
        eigenvectors: see :attr:`eigenvectors`
        original_output_count: see :attr:`original_output_count`
        inactive_units: see :attr:`inactive_units`
    """

    covariances: np.ndarray
    """
    The covariances matrix.
    This is a two dimensional square array of size :attr:`original_output_count`.
    """

    eigenvalues: np.ndarray
    """
    The eigenvalues of the covariances.
    This is a one dimensional array of size :attr:`original_output_count`.
    """

    eigenvectors: np.ndarray
    """
    The eigenvectors of the covariances.
    This is a two dimensional square array of size :attr:`original_output_count`.
    """

    original_output_count: int
    """The number of features in the response data"""

    inactive_units: np.ndarray
    """The indices of the inactive units"""

    @staticmethod
    def make_covariance_result(*,
                               covariances: np.ndarray,
                               epsilon_inactive: float = 1e-8) -> "PFACovariancesResult":
        """
         Create an instance of ``PFACovariancesResult``.

         Args:
             covariances: **[keyword arg]** the covariances
             epsilon_inactive: **[keyword arg, optional]** factor used to identify inactive units
                (whose var < epsilon_inactive * np.max(var)).
         """
        original_output_count = covariances.shape[0]
        eigenvalues, eigenvectors = _compute_eigens(covariances)

        # Count number of inactive units
        var = np.abs(np.diag(covariances))
        zeros = (var < epsilon_inactive * np.max(var))
        inactive_units = np.where(zeros)[0].astype(dtype=int)

        return PFACovariancesResult(
            covariances,
            eigenvalues,
            eigenvectors,
            original_output_count,
            inactive_units
        )


@t.final
class _CovariancesCalculator:
    """
    Compute the covariances of a set of data.

    Main methods:
        add_batch                   -- register a new batch of `N` data samples.
        get_count                   -- get the number of data samples previously added.
        get_centered_covariances    -- get the covariance of the data.
    """

    def __init__(self) -> None:
        self._count = 0
        self._sum_x: t.Optional[np.ndarray] = None
        self._sum_xxt: t.Optional[np.ndarray] = None

    def _check_or_init_size(self, c: int) -> None:
        """Raise an error if data dimension `C` mismatch."""
        if self._sum_x is None:
            self._sum_x = np.zeros(c)
            self._sum_xxt = np.zeros([c, c])
        elif self._sum_x.size != c:
            raise ValueError('Invalid dimensions: expected {}, got {}'.format(
                self._sum_x.shape, c))

    def add_batch(self, x_batch: np.ndarray) -> None:
        """Register a batch of data of shape `(Batch, C)`."""
        n, c = x_batch.shape
        self._check_or_init_size(c)
        self._count += n
        self._sum_x += np.sum(x_batch, axis=0)
        self._sum_xxt += x_batch.T.dot(x_batch)

    def get_count(self) -> int:
        """Number of data samples added so far."""
        return self._count

    def get_original_output_counts(self) -> int:
        """Number of features in the input."""
        assert self._sum_x is not None
        return self._sum_x.shape[0]

    def get_centered_covariances(self) -> np.ndarray:
        """Covariances of the data samples added so far, of shape `(C, C)`."""
        if self._sum_xxt is None or self._sum_x is None:
            raise ValueError('No data provided, mean is undefined')
        sum_xxt = self._sum_xxt
        sum_x = self._sum_x
        count = self.get_count()
        mean = sum_x / count
        if count <= 1:
            return sum_xxt / count - np.outer(mean, mean)
        else:
            return sum_xxt / (count - 1) - np.outer(mean, sum_x / (count - 1))

    def _get_result(self, epsilon_inactive: float = 1e-8) -> PFACovariancesResult:
        """Return a `_CovarianceResult` representing the result of the covariance calculation"""
        return PFACovariancesResult.make_covariance_result(
            covariances=self.get_centered_covariances(),
            epsilon_inactive=epsilon_inactive
        )


def _prepare_covariances(batch_size: int,
                         producer: Producer) -> t.Mapping[str, _CovariancesCalculator]:
    """
    Prepare the per-response `_CovariancesCalculator` -- these can compute the covariance for
    the accumulated data.
    """
    covariances: t.Mapping[str, _CovariancesCalculator] = defaultdict(_CovariancesCalculator)
    for resp_batch in producer(batch_size):
        for response_name, response in resp_batch.fields.items():

            if len(response.shape) > 2:
                raise DNIKitException(
                        f'Unable to introspect response {response_name}, of shape {response.shape},'
                        f'which has more than two dimensions.')

            covariances[response_name].add_batch(response)
    return covariances

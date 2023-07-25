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

import enum
from dataclasses import dataclass

import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.mixture import GaussianMixture as SKGMM

from dnikit.base import Batch, Producer
# private function to gather all batches at once, see Caution below
from dnikit.base._producer import _accumulate_batches
import dnikit.typing._types as t

from ._gaussian_familiarity import (
    _MixtureOfMultivariateGaussianDistributions
)
from ._protocols import (
    FamiliarityDistribution,
    FamiliarityResult,
    FamiliarityStrategyType,
)


@t.final
class GMMCovarianceType(enum.Enum):
    """
    Covariance type to be learnt from data.
    Typically, use ``FULL`` for low dimensional data and ``DIAG`` for high dimensional data.

    The main problem with ``FULL`` in high dimensions is that the algorithm learns ``dim x dim``
    parameters for each gaussian, and so overfitting or degenerate solutions may be a problem.

    The boundary between low and high dimensional data is fuzzy, and the choice of covariance
    type also depends on the application, data distribution or amount of data available.

    A general rule is:

    - If there are concerns about overfitting due to a lack of data, dimensions are high wrt.
      the data available, etc. Then use ``DIAG``. This is typically the case when working with
      DNN embeddings.
    - Else, use ``FULL``. For example, if fitting 2D data.

    For more information about covariance types, refer to the
    `sklearn GMM covariances page
    <https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html>`_.
    """
    FULL = 'full'
    """Full covariance type, all ``dim x dim`` parameters will be learnt from data."""

    DIAG = 'diag'
    """Diagonal covariance type, only the diagonal parameters will be learnt from data."""


@t.final
@dataclass(frozen=True)
class GMM(FamiliarityStrategyType):
    """
    A :class:`FamiliarityStrategyType <dnikit.introspectors.FamiliarityStrategyType>` that fits a
    mixture of multivariate gaussian distributions on the introspected responses using
    :class:`sklearn.mixture.GaussianMixture`.

    Args:
        gaussian_count: **[keyword arg, optional]** Number of gaussian distributions to be fitted
            in the mixture model.
        convergence_threshold: **[keyword arg, optional]** Convergence threshold used when fitting
            the mixture model.
        max_iterations: **[keyword arg, optional]** Maximum number of iterations to use when
            fitting the mixture model.
        covariance_type: **[keyword arg, optional]** Covariance type, usually
            :attr:`GMMCovarianceType.DIAG <dnikit.introspectors.GMMCovarianceType.DIAG>` or
            :attr:`GMMCovarianceType.FULL <dnikit.introspectors.GMMCovarianceType.FULL>`.
            See `sklearn's GaussianMixture docs
            <https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html>`_
            for extra information.
    """

    gaussian_count: int = 5
    """Number of gaussian distributions to be fitted in the mixture model."""

    convergence_threshold: float = 1e-3
    """Convergence threshold used when fitting the mixture model."""

    max_iterations: int = 200
    """Maximum number of iterations to use when fitting the mixture model."""

    covariance_type: GMMCovarianceType = GMMCovarianceType.DIAG
    """
    Covariance type, usually
    :attr:`GMMCovarianceType.DIAG <dnikit.introspectors.GMMCovarianceType.DIAG>` or
    :attr:`GMMCovarianceType.FULL <dnikit.introspectors.GMMCovarianceType.FULL>`.
    See `sklearn's GaussianMixture docs
    <https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html>`_
    for extra information.
    """

    _random_state: t.Optional[RandomState] = None
    """
    An optional :class:`RandomState` to pass to :class:`sklearn.mixture.GaussianMixture`
    to ensure consistent responses.
    """

    metadata_key: t.ClassVar = Batch.DictMetaKey[FamiliarityResult]("GMM")
    """Key used to view the GMM Familiarity results."""

    # Note: the explicit init is here so that editors see the parameters to the init.  Ideally
    # the dataclass init would be enough, but the indirection through Familiarity.Strategy
    # seems to throw it.

    def __init__(self, *, gaussian_count: int = 5,
                 convergence_threshold: float = 1e-3,
                 max_iterations: int = 200,
                 covariance_type: GMMCovarianceType = GMMCovarianceType.DIAG,
                 _random_state: t.Optional[RandomState] = None) -> None:
        object.__setattr__(self, "gaussian_count", gaussian_count)
        object.__setattr__(self, "convergence_threshold", convergence_threshold)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "covariance_type", covariance_type)
        object.__setattr__(self, "_random_state", _random_state)

    def __call__(self, producer: Producer,
                 batch_size: int = 1024) -> t.Mapping[str, FamiliarityDistribution]:
        accumulated_responses = _accumulate_batches(producer, batch_size=batch_size)
        mixture_model_per_response = {}

        for response_name in accumulated_responses.fields:
            model = SKGMM(
                n_components=self.gaussian_count,
                max_iter=self.max_iterations,
                tol=self.convergence_threshold,
                random_state=self._random_state,
                covariance_type=self.covariance_type.value,
            )

            model.fit(accumulated_responses.fields[response_name])

            # model.covariances_ has shape:
            #  (n_components, n_features) if 'diag',
            #  (n_components, n_features, n_features) if 'full'
            # Convert to a 'full' covariance for _MixtureOfMultivariateGaussianDistributions.
            covariances = (np.array([np.diag(vals) for vals in model.covariances_])
                           if self.covariance_type is GMMCovarianceType.DIAG
                           else model.covariances_)

            # Logic check
            _, dims = accumulated_responses.fields[response_name].shape
            assert covariances.shape == (self.gaussian_count, dims, dims), (
                f'Covariances must have dims {(self.gaussian_count, dims, dims)}, '
                f'found {covariances.shape}.'
            )

            mixture_model_per_response[response_name] = (
                _MixtureOfMultivariateGaussianDistributions._create(
                    mean_list=model.means_,
                    covariance_list=covariances,
                    weights=model.weights_
                )
            )

        return mixture_model_per_response

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
import logging

import numpy as np
from scipy.stats import entropy

from ._recommendation import (
    PFAKLDiagnostics,
    PFAEnergyDiagnostics,
    PFARecipe
)
from ._covariances_calculator import PFACovariancesResult
import dnikit.typing._types as t

_logger = logging.getLogger("dnikit.introspectors.pfa")


class PFAStrategyType(t.Protocol):
    """
    Protocol for PFA strategies (:class:`PFA.Strategy`).  These examine per-layer
    :class:`PFACovariancesResult` and
    produces per-layer :class:`PFARecipe`.

    Note:
        This takes all layers and produces a result for each of the layers, but the algorithm
        operates on each layer independently.
    """

    def __call__(self, covariances: t.Mapping[str, PFACovariancesResult]
                 ) -> t.Mapping[str, PFARecipe]:
        """
        Args:
            covariances: mapping from layer name (:attr:`field <dnikit.base.Batch.fields>` name)
                to :class:`PFACovariancesResult` for that layer
        """
        ...


@t.final
@dataclass(frozen=True)
class KL(PFAStrategyType):
    """
    KL strategy for generating PFA recipes.

    Args:
        interpolation_function: **[optional]** the interpolation function to use,
            see :class:`KLInterpolationFunction`.
    """

    class KLInterpolationFunction(t.Protocol):
        """
        A protocol to map a KL divergence to the ratio of the number of units in the layer.
        The KL divergence is that between the distribution of
        eigenvalues of the covariance matrix of model responses and the uniform distribution.
        """

        def __call__(self, kl_divergence: float, max_kl_divergence: float) -> float: ...

    @t.final
    class LinearInterpolation(KLInterpolationFunction):
        """
        A concrete :class:`KLInterpolationFunction
        <dnikit.introspectors.PFA.Strategy.KL.KLInterpolationFunction>` function that performs its
        intended mapping by linearly interpolating [kl_divergence, max_kl_divergence] to [0, 1]
        """

        def __call__(self, kl_divergence: float, max_kl_divergence: float) -> float:
            return 1 - np.divide(kl_divergence, max_kl_divergence)

    interpolation_function: t.Optional[KLInterpolationFunction] = None

    # Note: the explicit init is here so that editors see the parameters to the init.  Ideally
    # the dataclass init would be enough, but the indirection through PFA.Strategy
    # seems to throw it.
    def __init__(self, interpolation_function: t.Optional[KLInterpolationFunction] = None
                 ) -> None:
        if interpolation_function is None:
            interpolation_function = KL.LinearInterpolation()
        object.__setattr__(self, "interpolation_function", interpolation_function)

    def __call__(self, covariances: t.Mapping[str, PFACovariancesResult]
                 ) -> t.Mapping[str, PFARecipe]:
        """
        Generate a KL recipe.

        :ref:`_pfa`

        Note: this takes all layers and produces a result for each of the layers but the algorithm
        is per-layer (independent).

        Args:
            covariances: The information about the response.

        Returns:
            The :class:`PFARecipe` for KL.
        """

        def sum_norm(x: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(x, ord=1)
            if norm == 0:
                return np.zeros_like(x)
            return x / norm

        result = {}

        for response_name, cov in covariances.items():
            original_output_count = cov.original_output_count
            eigenvalue_distribution = sum_norm(cov.eigenvalues) + np.finfo(float).eps
            flat_distribution = np.ones_like(eigenvalue_distribution) / original_output_count

            kl_divergence = entropy(
                pk=eigenvalue_distribution,
                qk=flat_distribution
            )
            max_kl_divergence = np.log(original_output_count)

            assert self.interpolation_function
            units_ratio = self.interpolation_function(
                kl_divergence,
                max_kl_divergence
            )

            recommended_output_count = int(np.ceil(
                original_output_count * units_ratio
            ))

            diagnostics = PFAKLDiagnostics(
                kl_divergence=kl_divergence,
                units_ratio=units_ratio
            )

            result[response_name] = PFARecipe._make_recipe(
                covariances=cov,
                recommended_output_count=recommended_output_count,
                diagnostics=diagnostics,
            )

        return result


@t.final
@dataclass(frozen=True)
class Energy(PFAStrategyType):
    """
    Energy strategy for generating PFA recipes -- this targets a given :attr:`energy_threshold`
    to keep.

    Args:
        energy_threshold: The spectral energy to keep
        min_kept_count: **[optional]** The minimum number of outputs to keep per response
    """

    energy_threshold: float
    """The spectral energy to keep"""

    min_kept_count: int = 0
    """The minimum number of output to keep per response"""

    # Note: the explicit init is here so that editors can see the parameters to the init.  Ideally
    # the dataclass init would be enough, but the indirection through PFA.Strategy
    # seems to throw it.
    def __init__(self, energy_threshold: float, min_kept_count: int = 0) -> None:
        """
        Initialize the Energy algorithm.

        Args:
            energy_threshold: The spectral energy to keep
            min_kept_count: The minimum number of output to keep per response
        """
        object.__setattr__(self, "energy_threshold", energy_threshold)
        object.__setattr__(self, "min_kept_count", min_kept_count)

        if self.energy_threshold > 1.0 or self.energy_threshold < 0.0:
            raise ValueError(
                f'energy_threshold should be between 0.0 and 1.0, but it is {self.energy_threshold}'
            )

    def __call__(self, covariances: t.Mapping[str, PFACovariancesResult]
                 ) -> t.Mapping[str, PFARecipe]:
        """
        Generate a Energy recipe.

        Note: this takes all layers and produces a result for each of the layers but the algorithm
        is per-layer (independent).

        Args:
            covariances: The information about the response.

        Returns:
            A :class:`PFARecipe` for Energy.
        """

        result = {}

        for response_name, cov in covariances.items():
            eigenvalues = cov.eigenvalues
            original_output_count = len(eigenvalues)

            max_energy = float(np.sum(eigenvalues))
            total_kept_count = 0
            total_kept_energy = 0

            for eigenvalue in eigenvalues:
                total_kept_energy += eigenvalue
                total_kept_count += 1

                kept_energy_ratio = np.divide(
                    total_kept_energy, max_energy
                )
                if kept_energy_ratio >= self.energy_threshold:
                    break

            if total_kept_count < self.min_kept_count:
                _logger.warning('In order to satisfy the request to preserve at least'
                                f' {self.min_kept_count} units, the constraint to preserve '
                                f'not more than {self.energy_threshold} energy will be violated.'
                                ' If this is not desirable please set '
                                f'min_kept_count={total_kept_count}')

            total_kept_count = min(
                max(total_kept_count, self.min_kept_count),
                original_output_count
            )

            diagnostics = PFAEnergyDiagnostics(total_kept_energy=total_kept_energy)

            result[response_name] = PFARecipe._make_recipe(
                covariances=cov,
                recommended_output_count=total_kept_count,
                diagnostics=diagnostics,
            )

        return result


@t.final
@dataclass(frozen=True)
class Size(PFAStrategyType):
    """
    Size strategy for generating PFA recipes -- this targets a given ``relative_size`` to produce
    a cross-layer energy threshold that will produce that result.

    Args:
        relative_size: The relative amount of channels to keep (in 0..1)
        min_kept_count: **[optional]** The minimum number of output to keep per response
        epsilon_energy: **[optional]** Minimum level of energy
    """

    relative_size: float
    """The relative amount of channels to keep (in 0..1)"""
    min_kept_count: int = 0
    """The minimum number of output to keep per response"""
    epsilon_energy: float = 1e-8
    """Minimum level of energy"""

    # Note: the explicit init is here so that editors see the parameters to the init.  Ideally
    # the dataclass init would be enough, but the indirection through PFA.Strategy
    # seems to throw it.
    def __init__(self,
                 relative_size: float,
                 min_kept_count: int = 0,
                 epsilon_energy: float = 1e-8) -> None:
        """
        Initialize the Size algorithm.

        Args:
            relative_size: The relative amount of channels to keep (in 0..1)
            min_kept_count: The minimum number of output to keep per response
            epsilon_energy: Minimum level of energy
        """
        object.__setattr__(self, "relative_size", relative_size)
        object.__setattr__(self, "min_kept_count", min_kept_count)
        object.__setattr__(self, "epsilon_energy", epsilon_energy)

        if self.relative_size > 1.0 or self.relative_size < 0.0:
            raise ValueError(
                f'relative_size should be between 0.0 and 1.0, but it is {self.relative_size}'
            )

    def __call__(self, covariances: t.Mapping[str, PFACovariancesResult]
                 ) -> t.Mapping[str, PFARecipe]:
        """
        Generate a Size recipe.

        :ref:`_pfa`

        Note: this takes all layers and produces a result that is optimized across all the layers.

        Args:
            covariances: The information about the response.

        Returns:
            A Size recipe.
        """

        # For each layer, store the energy in the first `k-1` layers for all `k`
        layers_cum_energies = {}

        for response_name, cov in covariances.items():
            # Get eigenvalues and ensure they are valid
            eigenvalues = cov.eigenvalues
            assert np.all(np.isfinite(eigenvalues))

            # Get the cum. sum of energy, starting with highest eigenvalues
            cum_energy = -np.cumsum(np.sort(-np.abs(eigenvalues)))
            total_energy = cum_energy[-1]

            if total_energy > 0:
                # Normalize the cumsum and make it "exclusive"
                cum_energy = np.hstack(([0], cum_energy[:-1]))
                cum_energy = cum_energy / total_energy
            else:
                # All zero => discard all
                # Since the threshold is forced to be in [eps, 1 - eps], units with
                # cumulated energy of 0 (resp. 1) are always included (resp. excluded).
                # So all are set to 1 but the first min_kept to 0
                cum_energy = np.ones_like(cum_energy)

            # Currently disabled so that the percentile is not affected. Ensure that
            # min_kept_count is picked at the moment of application of the threshold
            # # Ensure at least `min_kept_count` is picked
            # cum_energy[:min_kept_count] = 0

            # Store the resulting energy
            layers_cum_energies[response_name] = cum_energy

        # Find the energy threshold that yields the desired number of channels
        energy_values = np.hstack(list(layers_cum_energies.values()))
        energy_t = np.percentile(energy_values, 100 * self.relative_size)
        # ensure 0 < t < 1
        energy_t = min(max(energy_t, self.epsilon_energy), 1 - self.epsilon_energy)

        # Get the recipe for that threshold
        return Energy(energy_threshold=energy_t, min_kept_count=self.min_kept_count)(covariances)

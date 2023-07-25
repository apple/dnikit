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

from ._covariances_calculator import PFACovariancesResult
import dnikit.typing._types as t


@t.final
@dataclass(frozen=True)
class PFAKLDiagnostics:
    """
    Diagnostic information for :class:`PFA.Strategy.KL`

    Args:
        kl_divergence: see :attr:`kl_divergence`
        units_ratio: see :attr:`units_ratio`
    """

    kl_divergence: float
    """Computed Kullback-Leibler (KL) divergence"""

    units_ratio: float
    """Indicates how much of a layer is deemed uncorrelated based on the KL divergence found."""


@t.final
@dataclass(frozen=True)
class PFAEnergyDiagnostics:
    """
    Diagnostic information for :class:`PFA.Strategy.Energy`

    Args:
        total_kept_energy: see :attr:`total_kept_energy`
    """

    total_kept_energy: float
    """Energy remaining after recommended compression"""


_OptionalDiagnostic = t.Union[PFAKLDiagnostics, PFAEnergyDiagnostics, None]


@t.final
@dataclass(frozen=True)
class PFARecipe:
    """
    Recommendation about a specific model response. This will likely never be instantiated
    directly, and instead an instance will be returned from :func:`pfa.get_recipe <PFA.get_recipe>`.
    """

    original_output_count: int
    """Original length of the response."""

    recommended_output_count: int
    """Recommended length of the response."""

    maximally_correlated_units: t.Sequence[int]
    """Maximally correlated units found with this recommendation."""

    number_inactive_units: int
    """Number of inactive units. If ``maximally_correlated_units`` exists then the first
    ``number_inactive_units`` are the units selected due to inactivity. """

    diagnostics: _OptionalDiagnostic
    """Per algorithm diagnostic information"""

    @staticmethod
    def _make_recipe(*, covariances: PFACovariancesResult,
                     recommended_output_count: int,
                     diagnostics: _OptionalDiagnostic) -> "PFARecipe":
        """
        Construct a new PFARecipe -- used internally by PFA to produce the results.

        Note: this produces a recipe with an empty `maximally_correlated_units`.  The unit methods
        (see :class:`PFA.UnitSelectionStrategy`) will produce a new recipe with the list provided.
        See `_apply_unit_result()`.  This intermediate recipe will never be seen by users.

        Args:
            covariances: **[keyword arg]** a PFACovariancesResult
            recommended_output_count: **[keyword arg]** the number of units to keep
            diagnostics: **[keyword arg]** optional diagnostic value

        Returns:
            a new ``PFARecipe``
        """
        return PFARecipe(
            original_output_count=covariances.original_output_count,
            recommended_output_count=recommended_output_count,
            maximally_correlated_units=[],
            number_inactive_units=covariances.inactive_units.shape[0],
            diagnostics=diagnostics
        )

    def _apply_unit_result(self, maximally_correlated_units: t.Sequence[int]) -> "PFARecipe":
        """Return a new PFARecipe augmented with `maximally_correlated_units`."""
        return PFARecipe(
            original_output_count=self.original_output_count,
            recommended_output_count=self.recommended_output_count,
            maximally_correlated_units=maximally_correlated_units,
            number_inactive_units=self.number_inactive_units,
            diagnostics=self.diagnostics
        )

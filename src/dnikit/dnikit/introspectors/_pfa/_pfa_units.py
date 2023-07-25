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

from dnikit.exceptions import DNIKitException
from ._covariances_calculator import PFACovariancesResult
import dnikit.typing._types as t

_logger = logging.getLogger("dnikit.introspectors.pfa")


class PFAUnitSelectionStrategyType(t.Protocol):
    """
    Given a correlation matrix and a number of units to keep, choose which units are
    maximally correlated.
    """

    def __call__(self, covariances: PFACovariancesResult, *,
                 num_units_to_keep: int) -> np.ndarray:
        """
        Args:
            covariances: the covariance data for the layer
            num_units_to_keep: **[keyword arg, optional]** number of recommended units to be kept

        Returns:
            :class:`numpy.ndarray` with the list of indexes that corresponds to the unit that is
            maximally correlated (the first part of the list contains the indices of the inactive
            units). The number of inactive units can be found in
            ``covariances.inactive_units.shape[0]``
        """
        ...


def _get_corr_and_inactive_units(covariances: PFACovariancesResult
                                 ) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Given the name of a layer compute and return the absolute value of the correlation matrix
    and the indices of previously computed inactive units. In addition to return the
    absolute value of the correlation matrix, the correlation coefficients on the main
    diagonal and those of dead units are set to NaN.

    Args:
        covariances: the covariance data for the layer

    Returns:
        A tuple with two ndarrays. The first is the absolute value of the correlation matrix,
        the second is the list of inactive units.
    """
    # Compute the correlation matrix
    corr = np.abs(_get_non_diagonal_correlation_mat(covariances))
    found_indices = covariances.inactive_units

    # Ensure diagonal won't be chosen
    np.fill_diagonal(corr, np.nan)

    # Also mark inactive units as maximally correlated
    corr[found_indices, :] = np.nan
    corr[:, found_indices] = np.nan

    return corr, found_indices


def _validate(num_units_to_keep: int) -> None:
    if num_units_to_keep <= 0:
        raise ValueError(
            f'Number of units to keep should be greater than zero but found {num_units_to_keep}.'
        )


def _compute_num_active_units(corr: np.ndarray,
                              num_units_to_keep: int,
                              found_indices: np.ndarray) -> int:
    num_active_units = corr.shape[0] - found_indices.shape[0]
    if num_active_units < 0:
        raise DNIKitException(
            'Requested to mark all units as correlated but no units are available'
        )
    if num_active_units < num_units_to_keep:
        raise DNIKitException(f'The request to keep {num_units_to_keep} cannot be satisfied '
                              f'since there are only {num_active_units} active units')
    return num_active_units


def _get_non_diagonal_correlation_mat(covariances: PFACovariancesResult,
                                      epsilon: float = 1e-8) -> np.ndarray:
    """
    Given the name of a layer compute the non-diagonal correlation matrix (starting from the
    previously computed covariance).

    Args:
        covariances: the covariance data for the layer
        epsilon: small value used to avoid division by zero

    Returns:
         2D ndarray that contains the covariance matrix (note that the diagonal contains NaNs)
    """
    covar = covariances.covariances
    var = np.abs(np.diag(covar))
    normalizer = np.sqrt(var[:, None] * var[None, :])
    corr = covar / np.maximum(normalizer, epsilon)

    return corr


def _select_units_l1_given_corr(corr: np.ndarray,
                                found_indices: np.ndarray,
                                num_units_to_keep: int,
                                direction_function: t.Callable[[np.ndarray], np.ndarray]
                                ) -> np.ndarray:
    """
    Given a correlation matrix, the list of already found correlated indices (which includes masked
    and inactive units), a `dir` that specifies whether to find correlated units with  "max" or
    "min" L1 correlation, and the minimum number of units to keep, compute which units are
    correlated (according to the criteria specified by the dir), and return the new list with
    the indices of all found correlated units. Note that `found_indices` is also changed in-place.
    If two or more units have same L1 norm then it chooses the one with the highest/lowest
    coefficient (same as max/min_abs).

    Args:
        corr: 2D correlation matrix
        found_indices: list of already found correlated indices (typically masked and inactive
            units).
        num_units_to_keep: number of units to preserve
        direction_function: numpy function for direction to select coefficients in

    Returns:
        ndarray with the list of indices that corresponds to the found correlated units.
    """
    _validate(num_units_to_keep)
    num_active_units = _compute_num_active_units(corr, num_units_to_keep, found_indices)

    corr = np.abs(corr)
    for it in range(num_active_units - num_units_to_keep):
        # compute l1 norm
        l1n = np.nansum(corr, axis=0)

        # mark already chosen units from the t.List
        l1n[found_indices] = np.nan

        # find highest/lowest l1 norm
        extreme_val = direction_function(l1n)
        if np.isnan(extreme_val):
            raise DNIKitException(f'All the L1 values are NaN. This is the |correlation matrix| '
                                  f'after {it} iterations: {corr}')

        # get highest/lowest l1 units
        extreme_cols = np.where(np.isclose(l1n, extreme_val))[0]

        # if multiple candidates found, then get the one with highest/lowest
        # abs correlation coeff.
        if len(extreme_cols) > 1:
            # if more than one unit has the same highest/lowest coefficient then chose the
            # first
            selected_coeff = direction_function(corr[:, extreme_cols])
            selected_col = np.where(np.isclose(corr[:, extreme_cols],
                                               selected_coeff))[0][0]
        else:
            selected_col = extreme_cols[0]

        _logger.debug(f'Identified unit {selected_col} with '
                      f'total magnitude {l1n[selected_col]}')
        found_indices = np.append(found_indices, selected_col)

        # ignore the found unit dimension
        corr[selected_col, :] = np.nan
        corr[:, selected_col] = np.nan

    assert np.unique(found_indices).shape[0] == len(found_indices), \
        f'Found correlated indexes should be unique {found_indices}'
    return found_indices


def _select_units_abs_given_corr(corr: np.ndarray,
                                 found_indices: np.ndarray,
                                 num_units_to_keep: int,
                                 direction_function: t.Callable[[np.ndarray], np.ndarray]
                                 ) -> np.ndarray:
    """
    Given a correlation matrix, the list of already found indices (which includes masked and
    inactive units), a `dir`` that specifies whether to find correlated units with  "max" or "min"
    correlation coefficient, and the minimum number of units to keep, compute which units
    are correlated (according to the criteria specified by the dir),
    and return the new list with the indices of all found correlated units.

    Args:
        corr: 2D correlation matrix
        found_indices: list of already found correlated indices (typically masked and
            inactive units).
        num_units_to_keep: number of units to preserve
        direction_function: numpy function for direction to select coefficients in

    Returns:
        ndarray with the list of indices that corresponds to the units that are correlated.
    """
    _validate(num_units_to_keep)
    num_active_units = _compute_num_active_units(corr, num_units_to_keep, found_indices)

    result = found_indices
    corr = np.abs(corr)
    for it in range(num_active_units - num_units_to_keep):

        selected_coeff = direction_function(corr)
        selected_cels = np.where(np.isclose(corr, selected_coeff))
        assert not np.isnan(selected_coeff)
        assert len(selected_cels[0]) >= 2

        # check the second highest/lowest coefficient
        tmp_corr = np.copy(corr)
        # mark one value per each row
        selected_rows = []
        for i, r in enumerate(selected_cels[0]):
            if r in selected_rows:
                # mark only one element per row
                continue
            selected_rows.append(r)
            tmp_corr[r][selected_cels[1][i]] = np.nan
        other_selected_coeff = direction_function(tmp_corr[selected_rows, :])
        other_selected_cels = np.where(np.isclose(tmp_corr[selected_rows, :],
                                                  other_selected_coeff))
        if np.isnan(other_selected_coeff):
            # No more choices, choose the first one
            selected_unit = selected_cels[0][0]
        else:
            # choose the first in case there are more than one
            selected_unit = selected_cels[0][sorted(other_selected_cels[0])[0]]
        result = np.append(result, selected_unit)

        _logger.debug(f'Identified unit {selected_unit} '
                      f'with coefficient {selected_coeff} and other '
                      f'coefficient {other_selected_coeff}')

        # ignore the found correlated unit dimension
        corr[selected_unit, :] = np.nan
        corr[:, selected_unit] = np.nan

    _logger.debug(f'Removing following units: {result}')
    assert np.unique(result).shape[0] == len(result), (
        f'Found correlated indexes should be unique {result}'
    )
    return result


class _DirectionalDistanceCalculation(t.Protocol):

    def __call__(self, corr: np.ndarray, found_indices: np.ndarray, num_units_to_keep: int,
                 direction_function: t.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Given a correlation matrix, the list of already found indices (which includes masked and
        inactive units), a `dir`` that specifies how to find correlated units with
        correlation coefficient, and the minimum number of units to keep, compute which units
        are correlated (according to the criteria specified by the dir),
        and return the new list with the indices of all found correlated units.

        Args:
            corr: 2D correlation matrix
            found_indices: list of already found correlated indices (typically masked and inactive
                units).
            num_units_to_keep: number of units to preserve
            direction_function: numpy function for direction to select coefficients in

        Returns:
            ndarray with the list of indices that corresponds to the units that are correlated.
        """
        ...


@dataclass(frozen=True)
class _DirectionalStrategy(PFAUnitSelectionStrategyType):
    """
    Compute which units are correlated according to a specified direction,
    using a given distance metric.
    """
    @t.final
    class Distance:
        ABS: t.Final[_DirectionalDistanceCalculation] = _select_units_abs_given_corr
        L1: t.Final[_DirectionalDistanceCalculation] = _select_units_l1_given_corr

    @t.final
    class UnitDirection:
        MIN: t.Final[t.Callable[[np.ndarray], np.ndarray]] = np.nanmin
        MAX: t.Final[t.Callable[[np.ndarray], np.ndarray]] = np.nanmax

    distance: _DirectionalDistanceCalculation
    """Distance function"""

    direction: t.Callable[[np.ndarray], np.ndarray]
    """Direction of selection"""

    def _select_units_given_corr(self, corr: np.ndarray, found_indices: np.ndarray,
                                 num_units_to_keep: int) -> np.ndarray:
        return self.distance(
            corr=corr,
            found_indices=found_indices,
            num_units_to_keep=num_units_to_keep,
            direction_function=self.direction
        )

    def __call__(self, covariances: PFACovariancesResult, *, num_units_to_keep: int) -> np.ndarray:
        """
        Given a correlation matrix and a number of units to keep, choose which units
        are maximally correlated. If two or more units have same L1 norm
        then it chooses the one with the highest/lowest coefficient (same as min_abs)

        Args:
            covariances: the covariance data for the layer
            num_units_to_keep: number of recommended units to be kept

        Returns:
            ndarray with the list of indexes that corresponds to the unit that is maximally
            correlated (the first part of the list contains the indices of the inactive units). The
            number of inactive units can be found in covariances.inactive_units.shape[0].
         """

        _validate(num_units_to_keep)

        corr, found_indices = _get_corr_and_inactive_units(covariances)
        if corr.shape[0] - len(found_indices) < num_units_to_keep:
            _logger.warning(f'Requested to keep {num_units_to_keep}, however, found '
                            f'{len(found_indices)} inactive units. All of them are'
                            f'highly correlated')
            return found_indices

        return self._select_units_given_corr(
            corr=corr,
            found_indices=found_indices,
            num_units_to_keep=num_units_to_keep,
        )


@t.final
@dataclass(frozen=True)
class AbsMax(_DirectionalStrategy):
    """Given a correlation matrix, choose units based on the one with the greatest coefficient"""

    def __init__(self) -> None:
        super().__init__(
            distance=_DirectionalStrategy.Distance.ABS,
            direction=_DirectionalStrategy.UnitDirection.MAX
        )


@t.final
@dataclass(frozen=True)
class AbsMin(_DirectionalStrategy):
    """Given a correlation matrix, choose units based on the one with the lowest coefficient"""

    def __init__(self) -> None:
        super().__init__(
            distance=_DirectionalStrategy.Distance.ABS,
            direction=_DirectionalStrategy.UnitDirection.MIN
        )


@t.final
@dataclass(frozen=True)
class L1Max(_DirectionalStrategy):
    """Given a correlation matrix, choose units based on the one with the greatest L1 norm"""

    def __init__(self) -> None:
        super().__init__(
            distance=_DirectionalStrategy.Distance.L1,
            direction=_DirectionalStrategy.UnitDirection.MAX
        )


@t.final
@dataclass(frozen=True)
class L1Min(_DirectionalStrategy):
    """Given a correlation matrix, choose units based on the one with the lowest L1 norm"""

    def __init__(self) -> None:
        super().__init__(
            distance=_DirectionalStrategy.Distance.L1,
            direction=_DirectionalStrategy.UnitDirection.MIN
        )

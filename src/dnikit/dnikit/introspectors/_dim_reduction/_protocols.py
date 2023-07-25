#
# Copyright 2021 Apple Inc.
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

import numpy as np
import dnikit.typing._types as t


@t.runtime_checkable
class DimensionReductionStrategyType(t.Protocol):
    """
    Strategy for performing dimension reduction on a single layer.  This is initialized
    with the target dimensions.

    The :func:`fit_incremental()` method is called repeatedly for each batch that is
    processed.  When all data has been visited, the :func:`fit_complete()` method is
    called.  Algorithms that require the full data set in memory may collect values
    with the first call and then combine and process in :func:`fit_complete()`.

    :func:`transform()` is used to transform high dimensional data into the target
    dimensions.
    """

    def default_batch_size(self) -> int:
        """Compute the default batch size."""
        ...

    def check_batch_size(self, batch_size: int) -> None:
        """
        Validate the batch_size and throw an error if there is an issue.

        Args:
            batch_size: batch size to validate
        """
        ...

    @property
    def target_dimensions(self) -> int:
        """How many dimensions this is reducing to."""
        ...

    def fit_incremental(self, data: np.ndarray) -> None:
        """
        Fit the reducer to the incremental ``data``

        Args:
            data: data to fit the reducer to
        """
        ...

    def fit_complete(self) -> None:
        """Called when all `fit` data has been passed."""
        ...

    @property
    def is_one_shot(self) -> bool:
        """
        Returns True if this can transform input data via :func:`transform()`,
        or if the entire input data set is transformed at once via
        :func:`transform_one_shot()`.
        """
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the given high dimensional ``data`` into the target dimensions.
        See :func:`is_one_shot`.

        Args:
            data: data to transform
        """
        ...

    def transform_one_shot(self) -> np.ndarray:
        """
        Returns the input data transformed per the reducer.
        See :func:`is_one_shot`.
        """
        ...

    def _clone(self) -> 'DimensionReductionStrategyType': ...

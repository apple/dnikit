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

import numpy as np

from dnikit.base import Producer, Batch
import dnikit.typing._types as t


class FamiliarityResult(t.Protocol):
    """Protocol for the result of applying a :class:`FamiliarityDistribution` to a response."""

    score: float
    """Familiarity score."""


class FamiliarityDistribution(t.Protocol):
    """
    The per-response result of :class:`FamiliarityStrategyType`.  An instance
    of this represents the distribution for a single layer and can evaluate the contents
    of a response.
    """

    def compute_familiarity_score(self, x: np.ndarray) -> t.Sequence[FamiliarityResult]:
        """
        Compute and return the :class:`Familiarity score <FamiliarityResult>` for
        each data point in ``x``.

        Args:
            x: input data samples to score according to the built distribution

        Returns:
            :class:`Familiarity score <FamiliarityResult>` for each data sample
        """
        ...


class FamiliarityStrategyType(t.Protocol):
    """
    Protocol for a class/function that takes a :class:`Producer <dnikit.base.Producer>` and produces
    a per-layer mapping of :class:`FamiliarityDistribution`.
    """

    metadata_key: t.ClassVar[Batch.DictMetaKey[FamiliarityResult]]
    """
    Key that will be used to view the metadata for a particular strategy.
    """

    def __call__(self, producer: Producer,
                 batch_size: int = 1024) -> t.Mapping[str, FamiliarityDistribution]:
        """
        Args:
            producer: producer of model responses
            batch_size: **[optional]** how many data samples to pull through the ``producer``
                at a time

        Returns:
            mapping of layer name (:attr:`fields <dnikit.base.Batch.fields>` of input ``producer``.
        """
        ...

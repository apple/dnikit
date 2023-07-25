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

from dataclasses import dataclass
import functools

from dnikit.base import (
    Batch,
    Introspector,
    pipeline,
    Producer,
)
from dnikit.introspectors._familiarity._familiarity import Familiarity
from dnikit.introspectors._familiarity._protocols import FamiliarityStrategyType
from dnikit.processors import Composer
import dnikit.typing._types as t
from ._dataframe_formatting import _DataframeFamiliarityTitle, _DataframeColumnPrefixes


@t.final
@dataclass(frozen=True)
class _NamedFamiliarity:
    """
    Wrapper around familiarity to keep track of model name
    """

    model: Familiarity
    """:class:`Familiarity` model"""

    title: _DataframeFamiliarityTitle
    """Title object to produce column titles from responses"""

    metadata_key: t.Optional[Batch.DictMetaKey] = None
    """Metadata key associated with data used to fit this familiarity model"""

    @classmethod
    def from_overall_familiarity(cls, producer: Producer, *, batch_size: int,
                                 strategy: t.Optional[FamiliarityStrategyType] = None,
                                 ) -> '_NamedFamiliarity':
        """
        Create a ``_NamedFamiliarity`` object from overall familiarity introspection.

        Args:
            producer: producer of responses to fit model to
            batch_size: batch size for running Familiarity
            strategy: :class:`Familiarity` strategy to use for fitting model

        Returns:
            instance of ``_NamedFamiliarity`` with title and metadata key for model fit to all data
        """
        if strategy is None:
            strategy = Familiarity.Strategy.GMM()

        return cls(
            model=Familiarity.introspect(producer, strategy=strategy, batch_size=batch_size),
            title=_DataframeFamiliarityTitle.from_overall_familiarity(),
            metadata_key=None
        )


@t.final
@dataclass(frozen=True)
class _SplitFamiliarity(Introspector):
    """
    Utils for running per-label Familiarity.
    """

    @staticmethod
    def get_label_introspectors(*, label_mapping: t.Mapping[str, t.Sequence[str]], batch_size: int,
                                strategy: t.Optional[FamiliarityStrategyType] = None
                                ) -> t.Sequence[t.Callable[[Producer], _NamedFamiliarity]]:
        """
        Get the per-label :class:`Familiarity` introspector calls for given set of dict-type labels

        Args:
            label_mapping: mapping from label dimension to a list of labels for that dimension
            strategy: :class:`Familiariity` strategy to use for all model fitting
            batch_size: batch size for running Familiarity

        Return:
            List of partial ``_SplitFamiliarity`` introspect methods with the label dimensions and
            labels already set, so a :class:`Familiarity` model can be fit per dimension-label
            combination
        """
        if strategy is None:
            strategy = Familiarity.Strategy.GMM()

        return [
            functools.partial(_SplitFamiliarity.introspect, strategy=strategy,
                              label_type=label_type, label=label, batch_size=batch_size)
            for label_type, labels in label_mapping.items()
            for label in labels
        ]

    @staticmethod
    def introspect(producer: Producer, *,
                   batch_size: int, label_type: str, label: str,
                   strategy: t.Optional[FamiliarityStrategyType] = None,
                   metadata_key: Batch.DictMetaKey = Batch.StdKeys.LABELS) -> _NamedFamiliarity:
        """
        Build a per-label :class:`Familiarity` model by chaining a :class:`Composer` filter
        onto the :class:`Producer`

        Args:
            producer: :class:`Producer` of responses to filter data for :class:`Familiarity` model
            batch_size: batch size for running Familiarity
            label_type: label dimension to use for filtering within ``metadata_key``
            label: label to filter by, for ``label_type`` of ``metadata_key``
            strategy: :class:`Familiarity` strategy to use for model fitting
            metadata_key: :class:`Batch.DictMetaKey` to use for filtering

        Return:
            :class:`_Named_Familiarity` populated with :class:`Familiarity` introspect result,
                :class:`_DataframeFamiliarityTitle`, and ``metadata_key``
        """
        if strategy is None:
            strategy = Familiarity.Strategy.GMM()

        filtered_producer = pipeline(
            producer,
            Composer.from_dict_metadata(
                metadata_key=metadata_key,
                label_dimension=label_type,
                label=label
            ),
        )

        return _NamedFamiliarity(
            model=Familiarity.introspect(
                filtered_producer, strategy=strategy, batch_size=batch_size),
            title=_DataframeFamiliarityTitle(
                type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
                label_dimension=label_type,
                label=label
            ),
            metadata_key=metadata_key,
        )

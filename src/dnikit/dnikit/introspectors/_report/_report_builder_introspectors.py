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

from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:
    class PandasStub:
        DataFrame = None
    pd = PandasStub()

import dnikit.typing._types as t

from dnikit.base import (
    Introspector,
    pipeline,
    Producer,
    Batch,
)
from dnikit.introspectors._dim_reduction._dimension_reduction import DimensionReduction
from dnikit.introspectors._duplicates import (
    Duplicates,
    DuplicatesThresholdStrategyType
)
from ._dataframe_formatting import (
    _DataframeColumnPrefixes,
    _DataframeFamiliarityTitle,
    make_projection_titles,
    make_duplicates_title,
)
from ._familiarity_wrappers import _NamedFamiliarity


@t.final
@dataclass(frozen=True)
class _SummaryBuilder(Introspector):
    """
    Parse data overall to extract summary of identifiers and metadata labels.
    """

    data: pd.DataFrame
    """Dataframe of ids and metadata labels for all data samples."""

    unique_labels: t.Mapping[str, t.Mapping[str, int]]
    """Set of unique labels, per label dimension."""

    def filtered_labels(self, minimum_count: int) -> t.Mapping[str, t.Sequence[str]]:
        """
        Return :property:`unique_labels` where any labels with fewer than
        ``minimum_count`` are removed.
        """
        return {
            label_type: [
                str(label)
                for label, c in label_counts.items()
                if c >= minimum_count
            ]
            for label_type, label_counts in self.unique_labels.items()
        }

    @classmethod
    def introspect(cls, producer: Producer, batch_size: int) -> '_SummaryBuilder':
        """
        Create data frame of the ids and metadata labels of all data samples from the producer.

        Note: this code assumes that Batch.StdKeys.LABELS metadata values are all strings,
        since the DatasetReport's main introspect function does this conversion.
        """

        # Go through all data and extract ids and labels (per label_dimension)
        temp_dict_data: t.Mapping[str, t.List[t.Sequence[t.Hashable]]] = defaultdict(list)
        for batch in producer(batch_size):
            temp_dict_data[_DataframeColumnPrefixes.ID.value].append(
                [str(b_identifier) for b_identifier in batch.metadata[Batch.StdKeys.IDENTIFIER]])

            if Batch.StdKeys.LABELS in batch.metadata:
                for label_dimension, labels in batch.metadata[Batch.StdKeys.LABELS].items():
                    temp_dict_data[label_dimension].append(labels)

        dict_data: t.Mapping[str, t.List[str]] = {
            key: list(np.concatenate([np.array(x) for x in list_of_lists]))
            for key, list_of_lists in temp_dict_data.items()
        }

        # Now get unique labels from existing dict_data
        unique_labels = {}
        for label_dimension, labels in dict_data.items():
            if label_dimension == _DataframeColumnPrefixes.ID.value:
                continue
            assert len(labels) == len(dict_data[_DataframeColumnPrefixes.ID.value])
            unique_labels[label_dimension] = Counter(labels)

        return cls(data=pd.DataFrame(dict_data), unique_labels=unique_labels)


@t.final
@dataclass(frozen=True)
class _DuplicatesBuilder(Introspector):
    """
    Builder of duplicates table.
    """

    data: pd.DataFrame
    """
    DataFrame of duplicate clusters.  Each column (series) in the dataframe
    represents a different response_name.  Each series has one row per element
    in the source Producer.  If the value is ``-1``, it doesn't belong to any
    cluster, otherwise the value is the cluster number it belongs to.

    For example this result would indicate that index 3, 5, and 6 belong
    to a cluster of duplicates in the ``result`` response:

    .. code-block:: python

        >>> pd.DataFrame({"duplicates_result": pd.Series(a)})
           duplicates_result
        0                 -1
        1                 -1
        2                 -1
        3                  0
        4                 -1
        5                  0
        6                  0
        7                 -1
        8                 -1
        9                 -1
    """

    @classmethod
    def introspect(cls, producer: Producer, *,
                   batch_size: int,
                   threshold: t.Optional[DuplicatesThresholdStrategyType] = None
                   ) -> '_DuplicatesBuilder':
        """
        Create a ``DuplicatesBuilder`` that holds a pandas DataFrame describing
        any duplicate clusters.  See :field:`data`.

        Args:
            producer: the producer of the data
            batch_size: the batch size to use in reading
            threshold: :class:`Duplicates` threshold strategy

        Returns:
            a :class:`_DuplicatesBuilder` with the duplicate clusters
        """
        if threshold is None:
            threshold = Duplicates.ThresholdStrategy.Slope()

        duplicates = Duplicates.introspect(producer, batch_size=batch_size, threshold=threshold)

        columns = {}
        for response_name, clusters in duplicates.results.items():

            # build the data for the series, see :field:`data`
            result = np.full((duplicates.count, ), -1, dtype="i")
            for cluster_number, cluster in enumerate(clusters):
                for i in cluster.indices:
                    result[i] = cluster_number

            columns[make_duplicates_title(response_name)] = pd.Series(result)

        return cls(data=pd.DataFrame(columns))


@t.final
@dataclass(frozen=True)
class _FamiliarityBuilder(Introspector):
    """
    Score data based on familiarity model (for all layers) and extract into pandas dataframe.
    """

    data: pd.DataFrame
    """DataFrame of ids and metadata labels for all data samples."""

    @staticmethod
    def _get_split_responses_from_frames(frames: t.Sequence[pd.DataFrame]) -> t.Set[str]:

        return set([
            _DataframeFamiliarityTitle.get_response_name_from_title(title=title)
            for frame in frames
            for title in frame.keys()
            # If any overall familiarity slipped in, filter it out
            if _DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY.value in title
        ])

    @staticmethod
    def condense_dataframes_by_label_dim(*, frames: t.Sequence[pd.DataFrame],
                                         label_dimension: str,
                                         labels: t.Sequence[str]) -> pd.DataFrame:
        """
        Given a list of columns of scores for a particular label_dimension, condense into one column
        This will only parse splitFamiliarity columns and ignore overall familiarity.

        Args:
            frames: List of Familiarity to consider for parsing
            label_dimension: Dimension to condense for the given frames
            labels: labels for the ``label_dimension`` that will be condensed

        Return:
            pandas dataframe where every column is the familiarity for all labels of that response
            and dimension
        """
        assert len(frames) > 0, f"No frames in condense dataframes by label: {label_dimension}"

        response_names = _FamiliarityBuilder._get_split_responses_from_frames(frames)

        # combine dataframes
        all_data = pd.concat(frames, axis=1)

        # base of new dataframe that will be built below
        combined_frame_data = {}

        # There will only be as many columns as there are responses
        for response in response_names:
            new_title = _DataframeFamiliarityTitle(
                type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
                label_dimension=label_dimension
            ).make_title(response=response)

            combined_frame_data[new_title] = [{
                # Exactly the columns desired by re-building the desired column title
                label: all_data[
                    _DataframeFamiliarityTitle.add_label_to_title(title=new_title, label=label)
                ][index]
                for label in labels
            } for index in range(len(all_data))]

        return pd.DataFrame.from_dict(combined_frame_data)

    @staticmethod
    def condense_dataframes_by_labels(*,
                                      results: t.Sequence['_FamiliarityBuilder'],
                                      label_mapping: t.Mapping[str, t.Sequence[str]]
                                      ) -> pd.DataFrame:
        """
        Given a list of :class:`_FamiliarityBuilder` results that has columns of scores for a set of
        label_dimensions and labels, condense into one column per label dimension per response.
        This will only parse splitFamiliarity columns and ignore overall familiarity columns.

        Args:
            results: List of :class:`_FamiliarityBuilder` results to consider for parsing
            label_mapping: mapping from label dimension to a list of labels for that dimension

        Return:
            pandas dataframe where every column is the familiarity for all labels of that response
            and dimension
        """
        frames = [result.data for result in results]

        combined = [
            _FamiliarityBuilder.condense_dataframes_by_label_dim(
                frames=frames,
                label_dimension=lbl_dim,
                labels=lbls
            )
            for lbl_dim, lbls in label_mapping.items()
        ]
        return pd.concat(combined, axis=1)

    @classmethod
    def introspect(cls, producer: Producer, *, batch_size: int,
                   named_fam: _NamedFamiliarity) -> '_FamiliarityBuilder':
        """
        Create a ``_FamiliarityBuilder`` that scores all responses of the producer according
        to the familiarity model of the :class:`_NamedFamiliarity` input, and holds a pandas
        DataFrame of the scores for each response. See :field:`data`.

        Prefix, suffix, and label are combined into single title with response for the builder
        output. These match Symphony formatting standards: https://apple.github.io/ml-symphony/

        Default (prefix, suffix, label) args are for scoring overall :class:`Familiarity`.

        The input familiarity model scores data from the producer for all responses present in the
        producer. An example of ``_FamiliarityBuilder.data`` for both overall familiarity and split
        Familiarity are below.

        .. code-block:: python

               familiarity_layer1   familiarity_layer2
            0                 -15                  -54
            1                 -12                  -76
            2                 -13                  -94
            3                 -17                  -85
            4                 -11                  -59

               splitFamiliarity_layer1_byAttr_color_red   splitFamiliarity_layer2_byAttr_color_red
            0                                       -15                                        -54
            1                                       -12                                        -76
            2                                       -13                                        -94
            3                                       -17                                        -85
            4                                       -11                                        -59


        Args:
            producer: the :class:`Producer` of the data
            batch_size: the batch size to use in reading
            familiarity: built :class:`Familiarity` model to use to score the data
            prefix: prefix for resulting pandas columns (default for overall familiarity column)
            suffix: suffix for resulting pandas columns (default for overall familiarity column)
            label: label for resulting pandas columns (default for overall familiarity column)

        Returns:
            a :class:`_FamiliarityBuilder` with the columns of familiarity scores for each response
        """

        # To avoid exponential list concatenation, add entire batches to list with extra dimension
        #    (hence t.List[t.Sequence[float]]) and then concatenate later into correct data format
        temp_familiarity_data: t.Mapping[str, t.List[t.Sequence[float]]] = defaultdict(list)

        # Extract familiarity scores
        meta_key = named_fam.model.meta_key
        scored_producer = pipeline(producer, named_fam.model)
        for batch in scored_producer(batch_size):
            for response_name in batch.metadata[meta_key].keys():
                temp_familiarity_data[response_name].append(
                    [r.score for r in batch.metadata[meta_key][response_name]]
                )

        # Remap dictionary keys to correct title format and concatenate scores into one list
        fam_data: t.Mapping[str, t.List[str]] = {
            named_fam.title.make_title(response=response_name): list(np.concatenate(list_of_lists))
            for response_name, list_of_lists in temp_familiarity_data.items()
        }

        return cls(data=pd.DataFrame(fam_data))


@t.final
@dataclass(frozen=True)
class _ProjectionBuilder(Introspector):
    """
    Builder of 2d projection data.
    """

    data: pd.DataFrame
    """
    DataFrame of projections.  This will have two columns per response_name:

    - projection_{response_name}_x
    - projection_{response_name}_y
    """

    @classmethod
    def introspect(cls, producer: Producer, *,
                   batch_size: int,
                   projection_model: DimensionReduction,
                   ) -> '_ProjectionBuilder':
        """
        Create a ``_ProjectionBuilder`` that holds a pandas DataFrame describing
        the 2d projection of the data.  See :field:`data`.

        Args:
            producer: the producer of the data
            batch_size: the batch size to use in reading
            projection_model: prepared dimension_reduction (to 2 dimensions)

        Returns:
            a :class:`_ProjectionBuilder` with the projected data
        """

        producer = pipeline(producer, projection_model)

        results = defaultdict(list)
        for batch in producer(batch_size):
            for response_name, data in batch.fields.items():
                assert data.shape[1] == 2
                x, y = make_projection_titles(response_name)
                results[x].append(data[:, 0])
                results[y].append(data[:, 1])

        return cls(
            data=pd.DataFrame(
                {
                    key: pd.Series(np.concatenate(values))
                    for key, values in results.items()
                }
            )
        )

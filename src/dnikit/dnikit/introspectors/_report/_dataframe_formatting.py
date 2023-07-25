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

from collections import defaultdict
from dataclasses import dataclass
import enum

import dnikit.typing._types as t
from ._string_util import remove_special_characters


class _FamiliarityPrefixes(enum.Enum):
    OVERALL = 'familiarity'
    SPLIT_FAMILIARITY = 'splitFamiliarity'
    SPLIT_MIDDLE_KEYWORD = 'byAttr'


class _DataframeColumnPrefixes(enum.Enum):
    """Prefixes for various columns of datatable input."""
    ID = 'id'
    DUPLICATES = 'duplicates'
    PROJECTION = 'projection'
    FAMILIARITY = _FamiliarityPrefixes


########################################################################
# Utils for split familiarity title formatting (familiarity per label) #
########################################################################


@t.final
@dataclass(frozen=True)
class _DataframeFamiliarityTitle:
    type: _FamiliarityPrefixes
    """
    Either _DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_MIDDLE_KEYWORD for split familiarity or
    _DataframeColumnPrefixes.FAMILIARITY.value.OVERALL for overall familiarity.
    """

    label_dimension: t.Optional[str]
    """Label dimension associated with this title."""

    label: t.Optional[str] = None
    """Label associated with this title, defaults to None for a generic label dimension title"""

    def __post_init__(self) -> None:
        if self.label_dimension is None:
            assert self.label is None, "if label_dimension is None, label must also be None"

    @staticmethod
    def get_response_name_from_title(*, title: str) -> str:
        """From title, extract response name"""
        split_keyword = _DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_MIDDLE_KEYWORD.value
        if split_keyword in title:
            # split familiarity format (response is between prefix and SPLIT_MIDDLE_KEYWORD)
            parts = title.split('_')
            return '_'.join(parts[1:parts.index(split_keyword)])
        else:
            # overall familiarity format (everything after prefix)
            return '_'.join(title.split('_')[1:])

    @staticmethod
    def _format_split_suffix(label_dimension: str) -> str:
        """Given label type, format suffix of title (added after prefix and response)"""
        split_keyword = _DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_MIDDLE_KEYWORD.value
        return f'_{split_keyword}_{label_dimension}'

    @staticmethod
    def add_label_to_title(title: str, label: str) -> str:
        assert label is not None
        return f"{title}_{label}"

    @classmethod
    def from_overall_familiarity(cls) -> '_DataframeFamiliarityTitle':
        return _DataframeFamiliarityTitle(
            type=_DataframeColumnPrefixes.FAMILIARITY.value.OVERALL,
            label_dimension=None,
            label=None
        )

    @classmethod
    def from_dimension_and_label(cls, label_dimension: t.Optional[str] = None,
                                 label: t.Optional[str] = None) -> '_DataframeFamiliarityTitle':
        if label_dimension is None:
            assert label is None, "Pass both label and label dimension, else neither"
            # Overall familiarity
            return cls.from_overall_familiarity()

        return cls(
            type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
            label_dimension=label_dimension,
            label=label if label is not None else None
        )

    def make_title(self, response: str) -> str:
        """
        Return title
        """
        cleaned_resp = remove_special_characters(response)
        title = f"{self.type.value}_{cleaned_resp}"

        if self.type == _DataframeColumnPrefixes.FAMILIARITY.value.OVERALL:
            # Overall familiarity
            return title

        # Split familiarity
        middle = _DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_MIDDLE_KEYWORD.value
        title = f"{title}_{middle}_{self.label_dimension}"
        if self.label is None:
            # No label
            return title
        return _DataframeFamiliarityTitle.add_label_to_title(title, self.label)


def make_duplicates_title(response: str) -> str:
    response = remove_special_characters(response)
    return f"{_DataframeColumnPrefixes.DUPLICATES.value}_{response}"


def make_projection_titles(response: str) -> t.Tuple[str, str]:
    response = remove_special_characters(response)
    prefix = f"{_DataframeColumnPrefixes.PROJECTION.value}_{response}"
    return f"{prefix}_x", f"{prefix}_y"


def get_response_from_column_name(name: str) -> t.Optional[str]:
    if any((name.startswith(_FamiliarityPrefixes.SPLIT_MIDDLE_KEYWORD.value),
            name.startswith(_FamiliarityPrefixes.OVERALL.value))):
        return _DataframeFamiliarityTitle.get_response_name_from_title(title=name)
    if name.startswith(_DataframeColumnPrefixes.DUPLICATES.value):
        return name[len(_DataframeColumnPrefixes.DUPLICATES.value) + 1:]
    if name.startswith(_DataframeColumnPrefixes.PROJECTION.value):
        return name[len(_DataframeColumnPrefixes.PROJECTION.value) + 1:-2]
    return None


def _create_titles_from_label_mapping(label_mapping: t.Mapping[str, t.Sequence[str]]
                                      ) -> t.Sequence[_DataframeFamiliarityTitle]:
    """
    Create a ``_DataframeFamiliarityTitle`` object for each label dimension / label combination

    Args:
        label_mapping: mapping of unique label dimensions to a list of labels for that dimension

    Return:
        ordered tuple of ``prefixes``, ``suffixes`` and ``labels``
    """
    return [
        _DataframeFamiliarityTitle(
            type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
            label_dimension=label_type,
            label=label
        )
        for label_type, labels in label_mapping.items()
        for label in labels
    ]


@t.final
@dataclass(frozen=True)
class _DatasetReportMapping:
    """
    Map the ``DatasetReport`` columns back into response names, label dimensions, etc.
    This is used specifically by the ``_DatasetReportDNIViewer`` to interpret the columns.
    """

    id: str
    "Name of the ID column"

    response_names: t.Sequence[str]
    label_dimensions: t.Sequence[str]

    duplicates_columns: t.Mapping[str, str]
    "names of duplicate information columns indexed by response_name"

    projection_columns: t.Mapping[str, t.Tuple[str, str]]
    "names of projection (x, y) columns indexed by response_name"

    familiarity_columns: t.Mapping[str, str]
    "names of familiarity information columns indexed by response_name"

    familiarity_columns_by_label: t.Mapping[str, t.Mapping[str, str]]
    "names of duplicate information columns indexed by response_name and label dimension"

    def __init__(self, columns: t.Sequence[str]) -> None:
        object.__setattr__(self, "id", "id")

        # label dimensions are columns that are not id or having one of the
        # known column names
        label_dimensions = []
        for name in columns:
            if not any((name == "id",
                        name.startswith(_DataframeColumnPrefixes.PROJECTION.value),
                        name.startswith(_DataframeColumnPrefixes.DUPLICATES.value),
                        name.startswith(_FamiliarityPrefixes.OVERALL.value),
                        name.startswith(_FamiliarityPrefixes.SPLIT_FAMILIARITY.value))):
                label_dimensions.append(name)
        object.__setattr__(self, "label_dimensions", label_dimensions)

        # extract the response names
        response_names = set()
        for name in columns:
            response_name = get_response_from_column_name(name)
            if response_name is not None:
                response_names.add(response_name)

        object.__setattr__(self, "response_names", response_names)

        # build the different possible columns -- only include columns that actually exist
        duplicates_columns: t.MutableMapping[str, str] = {}
        projection_columns: t.MutableMapping[str, t.Tuple[str, str]] = {}
        familiarity_columns: t.MutableMapping[str, str] = {}
        familiarity_columns_by_label: t.MutableMapping[str, dict] = defaultdict(dict)

        for response_name in response_names:
            duplicate_column = make_duplicates_title(response_name)
            if duplicate_column in columns:
                duplicates_columns[response_name] = duplicate_column

            projection_x, projection_y = make_projection_titles(response_name)
            if projection_x in columns and projection_y in columns:
                projection_columns[response_name] = (projection_x, projection_y)

            title = _DataframeFamiliarityTitle.from_overall_familiarity()
            familiarity_column = title.make_title(response_name)
            if familiarity_column in columns:
                familiarity_columns[response_name] = familiarity_column

            for label in label_dimensions:
                title = _DataframeFamiliarityTitle.from_dimension_and_label(label)
                by_label_column = title.make_title(response_name)
                if by_label_column in columns:
                    familiarity_columns_by_label[response_name][label] = by_label_column

        object.__setattr__(self, "duplicates_columns", duplicates_columns)
        object.__setattr__(self, "projection_columns", projection_columns)
        object.__setattr__(self, "familiarity_columns", familiarity_columns)
        object.__setattr__(self, "familiarity_columns_by_label", familiarity_columns_by_label)

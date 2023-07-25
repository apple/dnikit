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

import pytest
from dataclasses import dataclass
import logging
import pathlib
import random
import string

import numpy as np

try:
    import pandas as pd
except ImportError:
    pytest.skip("Pandas not available", allow_module_level=True)

from dnikit.base import (
    Batch,
    Producer,
)
import dnikit.typing._types as t
from dnikit.samples import StubProducer
from dnikit.introspectors import (
    DimensionReduction,
    DatasetReport,
    ReportConfig
)
from dnikit.introspectors._report._report_builder_introspectors import (
    _SummaryBuilder,
    _DuplicatesBuilder,
    _FamiliarityBuilder,
    _ProjectionBuilder,
)
from dnikit.introspectors._report._dataframe_formatting import (
    _DataframeColumnPrefixes,
    _DataframeFamiliarityTitle
)
from dnikit.introspectors._report._familiarity_wrappers import (
    _SplitFamiliarity,
    _NamedFamiliarity
)
from dnikit._availability import _umap_available


_logger = logging.getLogger("dnikit.tests")


@dataclass(frozen=True)
class MyProducer(Producer):

    color_options = ['black', 'black', 'white', 'grey', '']
    dataset_options = ['train', 'test']

    _producer: Producer

    def __init__(self, dataset_size: int, include_labels: bool = True) -> None:
        seed = 27

        np.random.seed(seed)
        random.seed(seed)
        random_s = np.random.RandomState(seed=seed)

        data = {
            "layer1": np.concatenate((
                # first cluster
                random_s.normal(10, .01, (int(dataset_size / 10), 10)),  # 10% duplicates
                random_s.normal(100, 50, (dataset_size - int(dataset_size / 5), 10)),
                # second cluster
                random_s.normal(500, .01, (int(dataset_size / 10), 10)),  # 10% duplicates
            )),
        }

        identifiers = [
            ''.join(random.choices(string.ascii_letters, k=10))
            for _ in range(dataset_size)
        ]
        if include_labels:
            metadata = {
                Batch.StdKeys.LABELS: {
                    'color': random.choices(self.color_options, k=dataset_size),
                    'dataset': random.choices(self.dataset_options, k=dataset_size),
                    '': ['label'] * dataset_size
                },
                Batch.StdKeys.IDENTIFIER: identifiers
            }
        else:
            metadata = {Batch.StdKeys.IDENTIFIER: identifiers}

        object.__setattr__(self, "_producer", StubProducer(data, metadata))

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        return self._producer(batch_size)


def test_summary_builder() -> None:
    dataset_size = 20
    producer = MyProducer(dataset_size=dataset_size)
    summary = _SummaryBuilder.introspect(producer, batch_size=5)

    # There should be 4 columns for {dataset_size} data samples
    assert summary.data.shape == (dataset_size, 4)

    # Make sure the keys are what is expected (ID column title is from SummaryBuilder)
    id_title = _DataframeColumnPrefixes.ID.value
    assert set(summary.data.keys()) == {id_title, 'color', 'dataset', ''}

    # Check unique labels
    assert summary.unique_labels['color'].keys() == set(producer.color_options)
    assert summary.unique_labels['dataset'].keys() == set(producer.dataset_options)
    assert summary.unique_labels[''].keys() == {'label'}

    # Check first row of data is valid
    assert summary.data[id_title][0][0] in string.ascii_letters
    assert summary.data['color'][0] in producer.color_options
    assert summary.data['dataset'][0] in producer.dataset_options


def test_summary_builder_counts() -> None:
    data = {
        "field1": np.random.randn(200).reshape((10, 20))
    }
    metadata = {
        Batch.StdKeys.LABELS: {
            "class": [
                "cat", "dog", "dog", "dog", "",
                "cat", "dog", "cow", "dog", "cat",
            ],
            "": ["label"] * 10
        },
        Batch.StdKeys.IDENTIFIER: list(range(10)),
    }
    producer = StubProducer(data, metadata)
    summary = _SummaryBuilder.introspect(producer, batch_size=5)

    assert summary.unique_labels == {
        "class": {"cat": 3, "dog": 5, "cow": 1, "": 1},
        "": {"label": 10}
    }
    assert summary.filtered_labels(3) == {
        "class": ["cat", "dog"], "": ["label"],
    }


def test_nonstr_labels() -> None:
    # Test that non-string label values work when input to DatasetReport.
    float_labels: t.List[float] = list(np.random.random_sample((10,)))
    mixed_labels: t.List[t.Hashable] = [
        "cat", (1, 2, 3.5), 0.75, "cat", "",
        (1, 2, 3.5), 4, ("a", 1, "b", 2), False, 4,
    ]
    data = {
        "field1": np.random.randn(200).reshape((10, 20))
    }
    metadata = {
        Batch.StdKeys.LABELS: {
            "nums": list(range(10)),
            "floats": float_labels,
            "mix": mixed_labels
        },
        Batch.StdKeys.IDENTIFIER: list(range(10)),
    }
    producer = StubProducer(data, metadata)
    report = DatasetReport.introspect(producer, batch_size=5, config=ReportConfig())

    assert report.data.shape == (10, 8)
    assert tuple(report.data['nums']) == tuple([str(e) for e in range(10)])
    assert (tuple(report.data['floats']) ==
           tuple([str(e) for e in float_labels]))
    assert (tuple(report.data['mix']) ==
           tuple([str(e) for e in mixed_labels]))


def test_duplicates_builder() -> None:
    dataset_size = 20
    producer = MyProducer(dataset_size=dataset_size)
    duplicates = _DuplicatesBuilder.introspect(producer, batch_size=5)

    assert duplicates.data.shape == (20, 1)
    assert duplicates.data["duplicates_layer1"] is not None

    # there is a cluster at the start and one at the end
    match = np.full((20, ), -1)
    match[0] = 0
    match[1] = 0
    match[18] = 1
    match[19] = 1
    assert np.all(duplicates.data["duplicates_layer1"] == match)


@pytest.mark.skipif(not _umap_available(),
                    reason="UMAP not available so skipping projection builder test")
def test_projection_builder() -> None:
    dataset_size = 20
    producer = MyProducer(dataset_size=dataset_size)
    model = DimensionReduction.introspect(producer,
                                          strategies=DimensionReduction.Strategy.UMAP(2))
    projection = _ProjectionBuilder.introspect(producer, batch_size=5, projection_model=model)

    assert projection.data["projection_layer1_x"].shape == (20,)
    assert projection.data["projection_layer1_y"].shape == (20,)


def test_familiarity_builder_and_wrappers() -> None:
    dataset_size = 20
    producer = MyProducer(dataset_size=dataset_size)

    # Note: if this starts to fail, it's because there are likely less than
    #    5 data samples with label "color" == "black", due to ordering of random.sample (producer)
    named_split_familiarity: _NamedFamiliarity = _SplitFamiliarity.introspect(
        producer, label_type='color', label='black', batch_size=5)
    familiarity_split = _FamiliarityBuilder.introspect(
        producer, batch_size=5,
        named_fam=named_split_familiarity
    )
    assert familiarity_split.data["splitFamiliarity_layer1_byAttr_color_black"].shape == (20,)

    named_familiarity_overall = _NamedFamiliarity.from_overall_familiarity(producer, batch_size=5)
    familiarity_overall = _FamiliarityBuilder.introspect(
        producer, batch_size=5,
        named_fam=named_familiarity_overall
    )
    assert familiarity_overall.data["familiarity_layer1"].shape == (20,)


def test_condense_dataframes_by_label_dim() -> None:
    # set up data to test
    label_dimension = "color"
    labels = ["blue", "red", ""]
    responses = ["response_a", "response_b_"]

    label_0_title = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension=label_dimension, label=labels[0]
    )
    label_1_title = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension=label_dimension, label=labels[1]
    )
    label_2_title = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension=label_dimension, label=labels[2]
    )
    label_3_title = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension='', label='label'
    )

    frames = [
        # Dataframe 1 is overall familiarity for two responses (a and b_)
        pd.DataFrame.from_dict({
            _DataframeFamiliarityTitle.from_overall_familiarity().make_title(
                responses[0]): [1, 2, 3],
            _DataframeFamiliarityTitle.from_overall_familiarity().make_title(
                responses[1]): [4, 5, 6]
        }),
        # Dataframe 2 is label 0 familiarity for two responses (a and b_)
        pd.DataFrame.from_dict({
            label_0_title.make_title(responses[0]): [7, 8, 9],
            label_0_title.make_title(responses[1]): [10, 11, 12]
        }),
        # Dataframe 3 is label 1 familiarity for two responses (a and b_)
        pd.DataFrame.from_dict({
           label_1_title.make_title(responses[0]): [70, 80, 90],
           label_1_title.make_title(responses[1]): [100, 110, 120]
        }),
        # Dataframe 4 is label 2 familiarity for two responses (a and b_)
        pd.DataFrame.from_dict({
            label_2_title.make_title(responses[0]): [700, 800, 900],
            label_2_title.make_title(responses[1]): [1000, 1100, 1200]
        }),
        # Dataframe 5 is label 2 familiarity for two responses (a and b_)
        pd.DataFrame.from_dict({
            label_3_title.make_title(responses[0]): [9, 10, 11],
            label_3_title.make_title(responses[1]): [90, 100, 110]
        }),
    ]

    condensed = _FamiliarityBuilder.condense_dataframes_by_label_dim(
        frames=frames,
        label_dimension=label_dimension,
        labels=labels
    )
    assert len(condensed.keys()) == 2, "Should only have two split columns"

    resp_a = condensed['splitFamiliarity_response_a_byAttr_color']
    assert resp_a[0] == {'blue': 7, 'red': 70, '': 700}
    assert resp_a[1] == {'blue': 8, 'red': 80, '': 800}
    assert resp_a[2] == {'blue': 9, 'red': 90, '': 900}

    resp_b = condensed['splitFamiliarity_response_b__byAttr_color']
    assert resp_b[0] == {'blue': 10, 'red': 100, '': 1000}
    assert resp_b[1] == {'blue': 11, 'red': 110, '': 1100}
    assert resp_b[2] == {'blue': 12, 'red': 120, '': 1200}

    condensed = _FamiliarityBuilder.condense_dataframes_by_label_dim(
        frames=frames,
        label_dimension='',
        labels=['label']
    )
    assert len(condensed.keys()) == 2, "Should only have two split columns"

    resp_a_2 = condensed['splitFamiliarity_response_a_byAttr_']
    assert resp_a_2[0] == {'label': 9}
    assert resp_a_2[1] == {'label': 10}
    assert resp_a_2[2] == {'label': 11}

    resp_b_2 = condensed['splitFamiliarity_response_b__byAttr_']
    assert resp_b_2[0] == {'label': 90}
    assert resp_b_2[1] == {'label': 100}
    assert resp_b_2[2] == {'label': 110}


@pytest.mark.parametrize(
    'configuration, columns, needs_umap',
    [
        # Only summary
        (
                {"familiarity": None, "duplicates": None, "projection": None},
                ['color', 'dataset', ''],
                False
        ),
        # Summary + duplicates
        (
                {"familiarity": None, "projection": None},
                ['color', 'dataset', '',
                 'duplicates_layer1'],
                False
        ),
        # Summary + familiarity
        (
                {"duplicates": None, "projection": None, "split_familiarity_min": 5},
                ['color', 'dataset', '',
                 'splitFamiliarity_layer1_byAttr_color', 'splitFamiliarity_layer1_byAttr_dataset',
                 'splitFamiliarity_layer1_byAttr_',
                 'familiarity_layer1'],
                False
        ),
        # Summary + no split familiarity
        (
                {"duplicates": None, "projection": None},
                ['color', 'dataset', '',
                 'familiarity_layer1'],
                False
        ),
        # Summary + projection
        (
                {"duplicates": None, "familiarity": None},
                ['color', 'dataset', '',
                 'projection_layer1_x', 'projection_layer1_y'],
                True
        ),
        # Summary + duplicates + projection
        (
                {"familiarity": None},
                ['color', 'dataset', '',
                 'duplicates_layer1', 'projection_layer1_x', 'projection_layer1_y'],
                True
        ),
        # Summary + duplicates + familiarity
        (
                {"projection": None, "split_familiarity_min": 5},
                ['color', 'dataset', '',
                 'duplicates_layer1',
                 'splitFamiliarity_layer1_byAttr_color', 'splitFamiliarity_layer1_byAttr_dataset',
                 'splitFamiliarity_layer1_byAttr_',
                 'familiarity_layer1'
                 ],
                False,
        ),
        # Summary + familiarity + projection
        (
                {"duplicates": None, "split_familiarity_min": 5},
                ['color', 'dataset', '',
                 'projection_layer1_x', 'projection_layer1_y',
                 'splitFamiliarity_layer1_byAttr_color', 'splitFamiliarity_layer1_byAttr_dataset',
                 'splitFamiliarity_layer1_byAttr_',
                 'familiarity_layer1'
                 ],
                True
        ),
        # Everything
        (
                {"split_familiarity_min": 5},  # Use default everything
                ['color', 'dataset', '',
                 'duplicates_layer1', 'projection_layer1_x', 'projection_layer1_y',
                 'splitFamiliarity_layer1_byAttr_color', 'splitFamiliarity_layer1_byAttr_dataset',
                 'splitFamiliarity_layer1_byAttr_',
                 'familiarity_layer1'],
                True
        ),
    ])
def test_dataset_report_combinations(configuration: t.Any,
                                     columns: t.Sequence[str],
                                     needs_umap: bool) -> None:
    if not needs_umap or _umap_available():
        dataset_size = 21
        producer = MyProducer(dataset_size=dataset_size)
        report_config = ReportConfig(
            dim_reduction=None,  # always use default dimension reduction strategy
            **configuration
        )
        report = DatasetReport.introspect(producer, config=report_config, batch_size=10)
        assert set(report.data.columns) == set([_DataframeColumnPrefixes.ID.value, *columns])
    else:
        logging.debug("Skipping projection tests in report as umap-learn is not installed.")


def test_no_labels_report() -> None:
    dataset_size = 20
    producer = MyProducer(dataset_size=dataset_size, include_labels=False)

    summary = _SummaryBuilder.introspect(producer, batch_size=5)
    assert summary.data.shape == (dataset_size, 1)
    assert summary.unique_labels == {}
    assert summary.filtered_labels(10) == {}

    split_fam = _SplitFamiliarity.get_label_introspectors(
        label_mapping={},
        batch_size=5
    )
    assert split_fam == []

    report = DatasetReport.introspect(producer)
    assert set(report.data.columns) == {
        "id", "duplicates_layer1", "projection_layer1_x",
        "projection_layer1_y", "familiarity_layer1"
    }


@pytest.fixture(scope='session')
def overall_title() -> _DataframeFamiliarityTitle:
    return _DataframeFamiliarityTitle(
        type=_DataframeColumnPrefixes.FAMILIARITY.value.OVERALL,
        label_dimension=None,
        label=None
    )


@pytest.fixture(scope='session')
def split_title() -> _DataframeFamiliarityTitle:
    return _DataframeFamiliarityTitle(
        type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
        label_dimension='color',
        label='blue'
    )


@pytest.fixture(scope='session')
def partial_title() -> _DataframeFamiliarityTitle:
    return _DataframeFamiliarityTitle(
        type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
        label_dimension='shape',
        label=None
    )


@pytest.fixture(scope='session')
def empty_string_title() -> _DataframeFamiliarityTitle:
    return _DataframeFamiliarityTitle(
        type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
        label_dimension='shape',
        label=''
    )


def test_title_instantiation(overall_title: _DataframeFamiliarityTitle,
                             split_title: _DataframeFamiliarityTitle,
                             partial_title: _DataframeFamiliarityTitle,
                             empty_string_title: _DataframeFamiliarityTitle) -> None:
    cls_overall_instantiation = _DataframeFamiliarityTitle.from_overall_familiarity()
    assert overall_title == cls_overall_instantiation

    from_dim_label_overall_instantiation = _DataframeFamiliarityTitle.from_dimension_and_label()
    assert overall_title == from_dim_label_overall_instantiation

    from_dim_label_split_instantiation = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension='color',
        label='blue'
    )
    assert split_title == from_dim_label_split_instantiation

    from_dim_label_partial_instantiation = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension='shape',
    )
    assert partial_title == from_dim_label_partial_instantiation

    from_dim_label_empty_string_instantiation = _DataframeFamiliarityTitle.from_dimension_and_label(
        label_dimension='shape',
        label=''
    )
    assert empty_string_title == from_dim_label_empty_string_instantiation


def test_title_utils(overall_title: _DataframeFamiliarityTitle,
                     split_title: _DataframeFamiliarityTitle,
                     partial_title: _DataframeFamiliarityTitle,
                     empty_string_title: _DataframeFamiliarityTitle) -> None:

    response = 'response_1'

    # check title format for all different kinds of instantiation
    overall_result = overall_title.make_title(response)
    assert overall_result == 'familiarity_response_1'

    split_result = split_title.make_title(response)
    assert split_result == 'splitFamiliarity_response_1_byAttr_color_blue'

    empty_string_result = empty_string_title.make_title(response)
    assert empty_string_result == 'splitFamiliarity_response_1_byAttr_shape_'

    partial_result = partial_title.make_title(response)
    assert partial_result == 'splitFamiliarity_response_1_byAttr_shape'

    # check adding label to existing string title
    partial_revised = _DataframeFamiliarityTitle.add_label_to_title(partial_result, 'circle')
    assert partial_revised == 'splitFamiliarity_response_1_byAttr_shape_circle'

    # check split familiarity suffix
    suffix = _DataframeFamiliarityTitle._format_split_suffix('shape')
    assert suffix == '_byAttr_shape'

    # test get response from label
    assert _DataframeFamiliarityTitle.get_response_name_from_title(
        title='splitFamiliarity_response_a_byAttr_color_blue') == 'response_a'
    assert _DataframeFamiliarityTitle.get_response_name_from_title(
        title='splitFamiliarity_response_b__byAttr_color_blue') == 'response_b_'


def test_bad_title() -> None:
    with pytest.raises(Exception):
        _ = _DataframeFamiliarityTitle(
            type=_DataframeColumnPrefixes.FAMILIARITY.value.SPLIT_FAMILIARITY,
            label_dimension=None,
            label='invalid-to-provide-label-here'
        )


def test_save_and_load_report(tmp_path: pathlib.Path) -> None:
    dataset_size = 20
    producer = MyProducer(dataset_size=dataset_size)
    report_config = ReportConfig(
        familiarity=None,
        duplicates=None,
        projection=None
    )
    report = DatasetReport.introspect(producer, config=report_config, batch_size=10)

    save_path = tmp_path/'report'
    report.to_disk(save_path)

    report2 = DatasetReport.from_disk(save_path)
    assert report.data.equals(report2.data)

    with pytest.raises(FileExistsError):
        report.to_disk(save_path)  # didn't set overwrite flag
    with pytest.raises(FileNotFoundError):
        _ = DatasetReport.from_disk('./bad_directory')

    # Ensure this doesn't throw an error
    report.to_disk(save_path, overwrite=True)

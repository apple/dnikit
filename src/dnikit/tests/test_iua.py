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

from copy import deepcopy

import pytest
import numpy as np

from dnikit.base import Producer
from dnikit.samples import StubProducer
from dnikit.introspectors import IUA
from dnikit._availability import (
    _pandas_available,
    _matplotlib_available,
)


RESPONSE_LENGTH = 84

STUB_DATA = {
    'layers_a': np.random.randn(RESPONSE_LENGTH, 3),
    'layers_b': np.random.randn(RESPONSE_LENGTH, 83),
    'layers_c': np.random.randn(RESPONSE_LENGTH, 11),
    'layers_d': np.random.randn(RESPONSE_LENGTH, 7)
}

ZEROS_INDXS = {
    'layers_a': ([0, 0, 10, 80, 4, 60, 78], [0, 1, 2, 1, 0, 2, 1]),
    'layers_b': ([0, 0, 10, 11, 80, 80, 80, 80, 4, 60, 78], [78, 4, 2, 45, 1, 2, 3, 4, 0, 2, 1])
}


@pytest.fixture
def producer() -> Producer:
    return StubProducer(STUB_DATA)


@pytest.fixture
def iua(producer: Producer) -> IUA:
    return IUA.introspect(producer)


@pytest.fixture
def zeros_response_producer() -> Producer:
    zeros_stub_data = deepcopy(STUB_DATA)

    # Set some of the stub units to zero
    zeros_stub_data['layers_a'][ZEROS_INDXS['layers_a']] = 0

    zeros_stub_data['layers_b'][ZEROS_INDXS['layers_b']] = 0

    return StubProducer(zeros_stub_data)


def test_iua_result_shapes(iua: IUA) -> None:
    iua_results = iua.results
    # Check that the correct type and number of values was tracked
    # per statistic
    for response_name in STUB_DATA.keys():
        response_result = iua_results[response_name]
        # The number of inactive unit counts per layer should match the number
        # of responses
        assert (len(response_result.inactive) == RESPONSE_LENGTH)
        # The peron unit level statistics should have the same shape as
        # the responses in stub_data
        assert (len(response_result.unit_inactive_count) == STUB_DATA[response_name].shape[1])
        assert (len(response_result.unit_inactive_proportion) == STUB_DATA[response_name].shape[1])


def test_iua_result_statistics(zeros_response_producer: Producer) -> None:
    # Ground truth results
    a_layer_counts = [0] * RESPONSE_LENGTH
    layer_a_rows, layer_a_cols = ZEROS_INDXS['layers_a'][0], ZEROS_INDXS['layers_a'][1]
    for response, unit in zip(layer_a_rows, layer_a_cols):
        a_layer_counts[response] += 1

    b_layer_counts = [0] * RESPONSE_LENGTH
    layer_b_rows, layer_b_cols = ZEROS_INDXS['layers_b'][0], ZEROS_INDXS['layers_b'][1]
    for response, unit in zip(layer_b_rows, layer_b_cols):
        b_layer_counts[response] += 1

    ground_truth_mean = {'layers_a': np.mean(a_layer_counts),
                         'layers_b': np.mean(b_layer_counts)}
    ground_truth_std = {'layers_a': np.std(a_layer_counts),
                        'layers_b': np.std(b_layer_counts)}

    ground_truth_layer_counts = {'layers_a': a_layer_counts,
                                 'layers_b': b_layer_counts}

    a_unit_counts = np.zeros_like(STUB_DATA['layers_a'][1])
    for response, unit in zip(layer_a_rows, layer_a_cols):
        a_unit_counts[unit] += 1
    b_unit_counts = np.zeros_like(STUB_DATA['layers_b'][1])
    for response, unit in zip(layer_b_rows, layer_b_cols):
        b_unit_counts[unit] += 1

    ground_truth_unit_counts = {'layers_a': a_unit_counts,
                                'layers_b': b_unit_counts}

    ground_truth_unit_proportions = {'layers_a': np.divide(a_unit_counts, RESPONSE_LENGTH),
                                     'layers_b': np.divide(b_unit_counts, RESPONSE_LENGTH)}

    iua_results = IUA.introspect(zeros_response_producer).results

    # Confirm IUA output matches ground truth
    for layer_name in ZEROS_INDXS.keys():
        layer_result = iua_results[layer_name]

        assert (layer_result.mean_inactive == ground_truth_mean[layer_name])

        assert (layer_result.std_inactive == ground_truth_std[layer_name])

        assert (layer_result.inactive == ground_truth_layer_counts[layer_name])

        assert np.array_equal(
            layer_result.unit_inactive_count,
            ground_truth_unit_counts[layer_name]
        )

        assert np.array_equal(
            layer_result.unit_inactive_proportion,
            ground_truth_unit_proportions[layer_name]
        )


@pytest.mark.skipif(not _pandas_available(), reason="Pandas not installed")
def test_iua_show_table(iua: IUA) -> None:
    # test all responses for table show
    results = IUA.show(iua)
    assert results is not None
    assert len(results.columns) == 3
    assert set(results.columns) == {'response', 'mean inactive', 'std inactive'}
    assert len(results) == 4

    # test single layer argument
    results_less = IUA.show(iua, response_names=['layers_a'])
    assert results_less is not None
    assert len(results_less.columns) == 3
    assert set(results_less.columns) == {'response', 'mean inactive', 'std inactive'}
    assert len(results_less) == 1
    assert results_less['response'][0] == 'layers_a'

    # test multi layer argument
    results_multiple = IUA.show(iua, response_names=['layers_a', 'layers_b'])
    assert results_multiple is not None
    assert len(results_multiple.columns) == 3
    assert set(results_multiple.columns) == {'response', 'mean inactive', 'std inactive'}
    assert len(results_multiple) == 2
    assert results_multiple['response'][0] == 'layers_a'
    assert results_multiple['response'][1] == 'layers_b'


@pytest.mark.skipif(not _matplotlib_available(), reason="Matplotlib not installed")
def test_iua_show_chart(iua: IUA) -> None:
    # Check base chart show method
    assert IUA.show(iua, vis_type=IUA.VisType.CHART) is not None

    # single response
    plots = IUA.show(iua, vis_type=IUA.VisType.CHART, response_names=['layers_a'])
    assert not isinstance(plots, np.ndarray)
    with pytest.raises(TypeError):
        # This is a strange way to test, but this will return a single AxisSubplot (not numpy array)
        assert len(plots) == 1

    # multiple responses
    plots = IUA.show(
        iua, vis_type=IUA.VisType.CHART, response_names=['layers_a', 'layers_b'])
    assert isinstance(plots, np.ndarray)
    assert len(plots) == 2

    # Bad response name input
    with pytest.raises(ValueError):
        IUA.show(
            iua, vis_type=IUA.VisType.CHART,
            response_names=['layers_a', 'bad_response']
        )

    # No responses input
    with pytest.raises(ValueError):
        IUA.show(iua, vis_type=IUA.VisType.CHART, response_names=[])


def test_iua_show_invalid_vis(iua: IUA) -> None:
    with pytest.raises(ValueError):
        IUA.show(iua, vis_type='invalid')

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
import pytest

# Some tests use TF and dnikit_tensorflow, only run those if TF and dnikit_tensorflow are installed
try:
    import tensorflow as tf
    from dnikit_tensorflow import load_tf_model_from_memory
    from dnikit_tensorflow.samples import get_simple_cnn_model
    from dnikit_tensorflow._tensorflow._tensorflow_protocols import running_tf_1
except ImportError:
    # This is so there is a stub for this function for mypy, but flake8 rationally recommends
    #   not defining functions using lambdas. This is never run, so it's left here.
    running_tf_1 = lambda: False  # noqa: E731

from dnikit.base import pipeline, ResponseInfo
from dnikit._availability import (
    _tensorflow_available,
    _pandas_available,
    _matplotlib_available,
)
from dnikit.introspectors import (
    PFA,
    PFAKLDiagnostics,
    PFAEnergyDiagnostics,
    PFARecipe
)
from dnikit.introspectors._pfa._pfa_units import _get_corr_and_inactive_units, _DirectionalStrategy
from dnikit.exceptions import DNIKitException
from dnikit.samples import (
    StubImageDataset,
    StubProducer
)
from dnikit.processors import (
    Pooler,
    FieldRenamer,
)

import dnikit.typing._types as t


def test_covariance_with_dead_units() -> None:
    # This test was introduced after issues #229 and #234

    # Generate random responses for 100 samples and 3 units
    acts = np.random.rand(100, 3)

    # Simulate having a dead unit that always produces 0 as response
    acts[:, 0] = 0.0

    # Run PFA simulation
    response_name = 'unit_with_dead_units'
    source = StubProducer({response_name: acts})
    pfa = PFA.introspect(source, batch_size=1)
    pfa_result = pfa._internal_result

    # Check that covariance for first unit is zero
    covariance_mat = pfa_result[response_name].covariances
    assert np.isclose(covariance_mat[0, 0], 0.0), 'Covariance was supposed to be zero but it is not'
    assert np.isclose(covariance_mat[0, 1], 0.0)
    assert np.isclose(covariance_mat[0, 2], 0.0)
    assert np.isclose(covariance_mat[1, 0], 0.0)
    assert np.isclose(covariance_mat[2, 0], 0.0)


def test_correlation_coefficients_with_nan() -> None:
    # This test was introduced after issues #229 and #234

    # Generate random responses for 100 samples and 3 units
    acts = np.random.rand(100, 3)

    # Simulate having a dead unit that always produces 0 as response
    acts[:, 0] = 0.0

    # Run PFA simulation
    response_name = 'unit_with_dead_units'
    source = StubProducer({response_name: acts})
    pfa = PFA.introspect(source, batch_size=1)
    pfa_result = pfa._internal_result

    # Given the preceding responses, the correlation coefficients will of the form
    # [[NaN, NaN, NaN],
    #  [NaN, 1, x],
    #  [NaN, x, 1]]
    # Except that in numpy np.cov returns exactly 0. This used to be handled differently
    # by forcing them to be 1. However this has changed and inactive units are
    # explicitly computed in PFA.introspect(). The
    # unit selection is done by always appending those units at the beginning of the list.
    # the specific value for the NaN is now specified based on the type of unit selection

    # correlation_coeff = pfa_result._get_non_diagonal_correlation_mat(response_name)
    correlation_coeff, inactive_units = _get_corr_and_inactive_units(pfa_result[response_name])

    assert inactive_units[0] == 0
    assert np.isnan(correlation_coeff[0, 0])
    assert np.isnan(correlation_coeff[0, 1])
    assert np.isnan(correlation_coeff[0, 2])
    assert np.isnan(correlation_coeff[1, 0])
    assert np.isnan(correlation_coeff[2, 0])


@pytest.mark.skipif(not _tensorflow_available(),
                    reason="TensorFlow and/or dnikit_tensorflow not installed.")
class TestPFAforCNN:

    def setup_method(self) -> None:
        """ set up any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        # Note, must clear session before setting random seed
        tf.keras.backend.clear_session()
        np.random.seed(1234)

        simple_cnn_config_override = {
            "num_classes": 10
        }

        if running_tf_1():
            sess = tf.Session()
            tf.random.set_random_seed(751994)
            _ = get_simple_cnn_model(simple_cnn_config_override)
            sess.run(tf.global_variables_initializer())

            self.model = load_tf_model_from_memory(session=sess)
        else:
            tf.random.set_seed(751994)
            keras_model = get_simple_cnn_model(simple_cnn_config_override)
            self.model = load_tf_model_from_memory(model=keras_model)

    def test_pfa_on_keras_model(self) -> None:
        dataset = StubImageDataset(
            dataset_size=192,
            image_width=32,
            image_height=32,
            channel_count=3
        )

        requested_responses = [
            info.name
            for info in self.model.response_infos.values()
            if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
        ]

        producer = pipeline(
            dataset,
            FieldRenamer({"images": "input_1:0"}),
            self.model(requested_responses),
            Pooler(dim=(1, 2), method=Pooler.Method.MAX)
        )

        pfa = PFA.introspect(batch_size=151, producer=producer)
        pfa_kl_recipe = pfa.get_recipe(strategy=PFA.Strategy.KL())

        assert (pfa_kl_recipe is not None)

        for energy_level in [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]:
            pfa_en_recipe = pfa.get_recipe(
                strategy=PFA.Strategy.Energy(energy_threshold=energy_level, min_kept_count=3)
            )
            assert (pfa_en_recipe is not None)

        for energy_level in [-1, 2]:
            with pytest.raises(ValueError):
                _ = pfa.get_recipe(
                    strategy=PFA.Strategy.Energy(energy_threshold=energy_level, min_kept_count=7)
                )

        with pytest.raises(DNIKitException):
            # This cannot be called on an instance of PFA, but rather should be called on
            #    the output of `pfa.get_recipe`. Mypy will complain because this is incorrect,
            #    but check for it anyway
            PFA.show(pfa)  # type: ignore


class TestUnitSelection:
    """Tests Different unit selection strategies"""

    def build_correlation(self) -> np.ndarray:
        cor_mat = np.array([[1.0, 0.7, 0.2, 0.3, 0.9],
                            [0.7, 1.0, 0.1, 0.2, 0.6],
                            [0.2, 0.1, 1.0, 0.8, 0.3],
                            [0.3, 0.2, 0.8, 1.0, 0.3],
                            [0.9, 0.6, 0.3, 0.3, 1.0]])
        np.fill_diagonal(cor_mat, np.nan)
        return cor_mat

    def build_correlation_with_inactive_units(self) -> np.ndarray:
        cor_mat = np.array([[1.0, 0.7, np.nan, 0.3, np.nan],
                            [0.7, 1.0, np.nan, 0.2, np.nan],
                            [np.nan, np.nan, 1.0, np.nan, np.nan],
                            [0.3, 0.2, np.nan, 1.0, np.nan],
                            [np.nan, np.nan, np.nan, np.nan, 1.0]])
        np.fill_diagonal(cor_mat, np.nan)
        return cor_mat

    def build_equal_correlation(self) -> np.ndarray:
        cor_mat = np.ones((4, 4))
        np.fill_diagonal(np.array(cor_mat), np.nan)
        return cor_mat

    def _helper_test_unit_selection_for_strategy(self, unit_strategy: _DirectionalStrategy
                                                 ) -> t.Tuple[t.List[int], t.List[int]]:
        cor_mat = self.build_correlation()

        correlated_indices_1 = unit_strategy._select_units_given_corr(
            cor_mat,
            found_indices=np.empty((0, 0), dtype=int),
            num_units_to_keep=1,
        )

        with pytest.raises(ValueError):
            _ = unit_strategy._select_units_given_corr(
                cor_mat,
                found_indices=np.empty((0, 0), dtype=int),
                num_units_to_keep=0,
            )

        with pytest.raises(DNIKitException):
            _ = unit_strategy._select_units_given_corr(
                cor_mat,
                found_indices=np.empty((0, 0), dtype=int),
                num_units_to_keep=cor_mat.shape[0] + 1,
            )

        cor_mat = self.build_equal_correlation()
        correlated_indices_2 = unit_strategy._select_units_given_corr(
            cor_mat,
            found_indices=np.empty((0, 0), dtype=int),
            num_units_to_keep=1,
        )

        return correlated_indices_1.tolist(), correlated_indices_2.tolist()

    def test_unit_selection_given_corr_max_l1(self) -> None:
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1

        # After marking unit 0 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6, 0.9
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4, 1.2
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6, 1.3
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1, 1.2

        # After marking unit 3 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6, 0.9, 0.7
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4, 1.2, 0.4
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6, 1.3
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1, 1.2, 0.9

        # After marking unit 4 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6, 0.9, 0.7, 0.1
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4, 1.2, 0.4, 0.1
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6, 1.3
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1, 1.2, 0.9

        l1_max_strategy = PFA.UnitSelectionStrategy.L1Max()
        correlated_indices_1, correlated_indices_2 = self._helper_test_unit_selection_for_strategy(
            l1_max_strategy
        )

        assert correlated_indices_1 == [0, 3, 4, 1], correlated_indices_1
        assert correlated_indices_2 == [0, 1, 2]

    def test_unit_selection_given_corr_min_l1(self) -> None:
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1

        # After marking unit 2 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1, 1.9
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6, 1.5
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4,
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6, 0.8
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1, 1.8

        # After marking unit 3 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1, 1.9, 1.6
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6, 1.5, 1.3
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6, 0.8
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1, 1.8, 1.5

        # After marking unit 1 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], # ==> sum 2.1, 1.9, 1.6, 0.9
        #            [0.7, 1.0, 0.1, 0.2, 0.6], # ==> sum 1.6, 1.5, 1.3
        #            [0.2, 0.1, 1.0, 0.8, 0.3], # ==> sum 1.4
        #            [0.3, 0.2, 0.8, 1.0, 0.3], # ==> sum 1.6, 0.8
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] # ==> sum 2.1, 1.8, 1.5, 0.9

        l1_min_strategy = PFA.UnitSelectionStrategy.L1Min()
        correlated_indices_1, correlated_indices_2 = self._helper_test_unit_selection_for_strategy(
            l1_min_strategy
        )

        assert correlated_indices_1 == [2, 3, 1, 0], correlated_indices_1
        assert correlated_indices_2 == [0, 1, 2]

    def test_unit_selection_given_corr_abs_min(self) -> None:
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], 0.2
        #            [0.7, 1.0, 0.1, 0.2, 0.6], 0.1 -> 0.2 <== chosen
        #            [0.2, 0.1, 1.0, 0.8, 0.3], 0.1 -> 0.2
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.2
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] 0.3

        # After marking unit 1 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], 0.2 -> 0.3 <== chosen
        #            [0.7, 1.0, 0.1, 0.2, 0.6], <= correlated
        #            [0.2, 0.1, 1.0, 0.8, 0.3], 0.2 -> 0.3
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.3
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] 0.3
        #                   ^

        # After marking unit 0 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], <= correlated
        #            [0.7, 1.0, 0.1, 0.2, 0.6], <= correlated
        #            [0.2, 0.1, 1.0, 0.8, 0.3], 0.3 -> 0.8
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.3 -> 0.8
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] 0.3 -> 0.3 <== chosen
        #              ^    ^

        # After marking unit 4 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], <= correlated
        #            [0.7, 1.0, 0.1, 0.2, 0.6], <= correlated
        #            [0.2, 0.1, 1.0, 0.8, 0.3], 0.8 <== chosen
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.8
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] <== correlated
        #              ^    ^              ^

        abs_min_strategy = PFA.UnitSelectionStrategy.AbsMin()
        correlated_indices_1, correlated_indices_2 = self._helper_test_unit_selection_for_strategy(
            abs_min_strategy
        )

        assert correlated_indices_1 == [1, 0, 4, 2], correlated_indices_1
        assert correlated_indices_2 == [0, 1, 2]

    def test_unit_selection_given_corr_abs_max(self) -> None:
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], 0.9 -> 0.7 <== chosen
        #            [0.7, 1.0, 0.1, 0.2, 0.6], 0.7
        #            [0.2, 0.1, 1.0, 0.8, 0.3], 0.8
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.8
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] 0.9 -> 0.6

        # After marking unit 0 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], <= correlated
        #            [0.7, 1.0, 0.1, 0.2, 0.6], 0.6
        #            [0.2, 0.1, 1.0, 0.8, 0.3], 0.8 -> 0.3 <= chosen
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.8 -> 0.3
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] 0.6
        #              ^

        # After marking unit 2 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], <= correlated
        #            [0.7, 1.0, 0.1, 0.2, 0.6], 0.6 -> 0.2
        #            [0.2, 0.1, 1.0, 0.8, 0.3], <= correlated
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.3
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] 0.6 -> 0.3 <= chosen
        #              ^         ^

        # After marking unit 4 as highly correlated
        # cor_mat = [[1.0, 0.7, 0.2, 0.3, 0.9], <= correlated
        #            [0.7, 1.0, 0.1, 0.2, 0.6], 0.2 <= chosen
        #            [0.2, 0.1, 1.0, 0.8, 0.3], <= correlated
        #            [0.3, 0.2, 0.8, 1.0, 0.3], 0.2
        #            [0.9, 0.6, 0.3, 0.3, 1.0]] <= correlated
        #              ^         ^         ^

        abs_max_strategy = PFA.UnitSelectionStrategy.AbsMax()
        correlated_indices_1, correlated_indices_2 = self._helper_test_unit_selection_for_strategy(
            abs_max_strategy
        )

        assert correlated_indices_1 == [0, 2, 4, 1], correlated_indices_1
        assert correlated_indices_2 == [0, 1, 2]

    def _unit_selection_for_inactive_units_given_strategy(self, unit_strategy: _DirectionalStrategy
                                                          ) -> t.List[int]:
        cor_mat = self.build_correlation_with_inactive_units()

        correlated_indices_1 = unit_strategy._select_units_given_corr(
            cor_mat,
            found_indices=np.array([2, 4]),
            num_units_to_keep=1,
        )

        correlated_indices_2 = unit_strategy._select_units_given_corr(
            cor_mat,
            found_indices=np.array([2, 4]),
            num_units_to_keep=3,
        )
        assert correlated_indices_2.tolist() == [2, 4]

        with pytest.raises(DNIKitException):
            _ = unit_strategy._select_units_given_corr(
                cor_mat,
                found_indices=np.array([2, 4]),
                num_units_to_keep=4,
            )

        return correlated_indices_1.tolist()

    def test_unit_selection_given_corr_and_inactive_units_l1_max(self) -> None:
        # cor_mat = np.array([[np.nan, 0.7,    np.nan, 0.3,    np.nan], 1.0 <= chosen
        #                     [0.7,    np.nan, np.nan, 0.2,    np.nan], 0.9
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [0.3,    0.2,    np.nan, np.nan, np.nan], 0.5
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # after marking unit 0 as highly correlated
        # cor_mat = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], <= correlated
        #                     [np.nan, np.nan, np.nan, 0.2,    np.nan], 0.2 <= chosen
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [np.nan, 0.2,    np.nan, np.nan, np.nan], 0.2
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # L1 Max should mark [0, 1] as correlated
        l1_max_strategy = PFA.UnitSelectionStrategy.L1Max()
        correlated_indices = self._unit_selection_for_inactive_units_given_strategy(l1_max_strategy)
        assert correlated_indices == [2, 4, 0, 1], correlated_indices

    def test_unit_selection_given_corr_and_inactive_units_l1_min(self) -> None:
        # cor_mat = np.array([[np.nan, 0.7,    np.nan, 0.3,    np.nan], 0.3
        #                     [0.7,    np.nan, np.nan, 0.2,    np.nan], 0.2 -> 07
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [0.3,    0.2,    np.nan, np.nan, np.nan], 0.2 -> 0.3 <= chosen
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # after marking unit 3 as highly correlated
        # cor_mat = np.array([[np.nan, 0.7,    np.nan, np.nan  np.nan], 0.7 <= chosen
        #                     [0.7,    np.nan, np.nan, np.nan, np.nan], 0.7
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan], <= correlated
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # L1 Min should mark [3, 0] as correlated
        l1_min_strategy = PFA.UnitSelectionStrategy.L1Min()
        correlated_indices = self._unit_selection_for_inactive_units_given_strategy(l1_min_strategy)
        assert correlated_indices == [2, 4, 3, 0], correlated_indices

    def test_unit_selection_given_corr_and_inactive_units_abs_max(self) -> None:
        # cor_mat = np.array([[np.nan, 0.7,    np.nan, 0.3,    np.nan], 0.7 -> 0.3 <= chosen
        #                     [0.7,    np.nan, np.nan, 0.2,    np.nan], 0.7 -> 0.2
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [0.3,    0.2,    np.nan, np.nan, np.nan], 0.3
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # after marking unit 0 as highly correlated
        # cor_mat = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], <= correlated
        #                     [np.nan, np.nan, np.nan, 0.2,    np.nan], 0.2 <= chosen
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [np.nan, 0.2,    np.nan, np.nan, np.nan], 0.2
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # ABS Max should mark [0, 1] as correlated
        abs_max_strategy = PFA.UnitSelectionStrategy.AbsMax()
        correlated_indices = self._unit_selection_for_inactive_units_given_strategy(
            abs_max_strategy
        )
        assert correlated_indices == [2, 4, 0, 1], correlated_indices

    def test_unit_selection_given_corr_and_inactive_units_abs_min(self) -> None:
        # cor_mat = np.array([[np.nan, 0.7,    np.nan, 0.3,    np.nan], 0.3
        #                     [0.7,    np.nan, np.nan, 0.2,    np.nan], 0.2 -> 0.7
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [0.3,    0.2,    np.nan, np.nan, np.nan], 0.2 -> 0.3 <= chosen
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # after marking unit 3 as highly correlated
        # cor_mat = np.array([[np.nan, 0.7,    np.nan, np.nan  np.nan], 0.7 <= chosen
        #                     [0.7,    np.nan, np.nan, np.nan, np.nan], 0.7
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan],
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan], <= correlated
        #                     [np.nan, np.nan, np.nan, np.nan, np.nan]])

        # ABS Min should mark [3, 0] as correlated
        abs_min_strategy = PFA.UnitSelectionStrategy.AbsMin()
        correlated_indices = self._unit_selection_for_inactive_units_given_strategy(
            abs_min_strategy
        )
        assert correlated_indices == [2, 4, 3, 0], correlated_indices

    def make_stub_correlated_and_inactive_responses(self, n_images: int
                                                    ) -> t.Mapping[str, np.ndarray]:
        stub_data: t.Dict[str, np.ndarray] = {}

        for ii in range(n_images):
            random_state = np.random.RandomState(seed=ii)

            activations = {
                'layer_0': random_state.uniform(-1.0, 1.0, (1, 20, 20, 128)),
                'layer_1': random_state.uniform(-1.0, 1.0, (1, 10, 10, 64)),
                'layer_2': random_state.uniform(-1.0, 1.0, (1, 5, 5, 32))
            }

            def correlate(layer: str, source: int, target: int) -> None:
                """ Copy activations and add some noise to not fully correlate """
                amp = .01
                noise = random_state.normal(
                    0.0, amp, activations[layer][..., source].shape)
                activations[layer][...,
                                   target] = activations[layer][..., source] + noise

            # Correlate some units
            for trgt in range(1, 15):
                random_state.seed(ii)
                correlate('layer_0', 0, trgt)
                correlate('layer_1', 0, trgt)
                correlate('layer_2', 0, trgt)

            def deactivate(layer: str, target: int, constant_value: float) -> None:
                """ Make some units constant, i.e., inactive """
                activations[layer][..., target] = (
                    np.ones(activations[layer][..., target].shape) * constant_value
                )

            # Make some units inactive
            for trgt in range(15, 25):
                random_state.seed(ii)
                deactivate('layer_0', trgt, 0.0)
                deactivate('layer_1', trgt, 1.0)
                deactivate('layer_2', trgt, 5.0)

            for layer_name, resp in activations.items():
                if layer_name in stub_data:
                    stub_data[layer_name] = np.concatenate(
                        (stub_data[layer_name], activations[layer_name]),
                        axis=0
                    )
                else:
                    stub_data[layer_name] = activations[layer_name]
        return stub_data

    def test_unit_selection_with_inactive_units(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_stub_correlated_and_inactive_responses(200)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=100)

        fs_criterions = PFA.UnitSelectionStrategy.get_algos()
        for crit in fs_criterions:
            recs = pfa.get_recipe(strategy=PFA.Strategy.Energy(energy_threshold=0.98),
                                  unit_strategy=crit)

            # validate that the first 10 units are the inactive ones
            for layer in response_names:
                assert recs[layer].number_inactive_units == 10
                assert recs[layer].maximally_correlated_units[0:10] == list(range(15, 25))


@pytest.mark.regression
class TestPFARegression:
    """Tests PFA against legacy implementation"""

    def make_legacy_stub_responses(self, n_images: int) -> t.Mapping[str, np.ndarray]:
        stub_data: t.Dict[str, np.ndarray] = {}

        for i in range(n_images):
            random_state = np.random.RandomState(seed=i)

            activations = {
                'layer_0': random_state.uniform(-1.0, 1.0, (1, 20, 20, 128)),
                'layer_1': random_state.uniform(-1.0, 1.0, (1, 10, 10, 64)),
                'layer_2': random_state.uniform(-1.0, 1.0, (1, 5, 5, 32))
            }

            def correlate(layer: str, source: int, target: int) -> None:
                """ Copy activations and add some noise to not fully correlate """
                amp = .01
                noise = random_state.normal(
                    0.0, amp, activations[layer][..., source].shape)
                activations[layer][...,
                                   target] = activations[layer][..., source] + noise

            # Correlate some units
            for trgt in range(1, 15):
                random_state.seed(i)
                correlate('layer_0', 0, trgt)
                correlate('layer_1', 0, trgt)
                correlate('layer_2', 0, trgt)

            for layer_name in ['layer_0', 'layer_1', 'layer_2']:
                if layer_name in stub_data:
                    stub_data[layer_name] = np.concatenate(
                        (stub_data[layer_name], activations[layer_name]),
                        axis=0
                    )
                else:
                    stub_data[layer_name] = activations[layer_name]
        return stub_data

    def test_pfa_kl_linear_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(128)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=128)
        recommendations_by_layer = pfa.get_recipe(strategy=PFA.Strategy.KL())

        assert set(recommendations_by_layer.keys()) == set(response_names), \
            "The recipe should have recommendations for all responses named {}, " \
            "and those only".format(response_names)

        # validate recipe for layer_0
        assert recommendations_by_layer['layer_0'].original_output_count == 128
        assert recommendations_by_layer['layer_0'].recommended_output_count == 99
        assert isinstance(recommendations_by_layer['layer_0'].diagnostics, PFAKLDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_0'].diagnostics.kl_divergence, 1.129493867639542
        )
        assert np.isclose(
            recommendations_by_layer['layer_0'].diagnostics.units_ratio, 0.7672121140631339
        )

        # validate recipe for layer_1
        assert recommendations_by_layer['layer_1'].original_output_count == 64
        assert recommendations_by_layer['layer_1'].recommended_output_count == 52
        assert isinstance(recommendations_by_layer['layer_1'].diagnostics, PFAKLDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_1'].diagnostics.kl_divergence, 0.8441312319086356
        )
        assert np.isclose(
            recommendations_by_layer['layer_1'].diagnostics.units_ratio, 0.7970293429776533
        )

        # validate recipe for layer_2
        assert recommendations_by_layer['layer_2'].original_output_count == 32
        assert recommendations_by_layer['layer_2'].recommended_output_count == 20
        assert isinstance(recommendations_by_layer['layer_2'].diagnostics, PFAKLDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_2'].diagnostics.kl_divergence, 1.3327294476569362
        )
        assert np.isclose(
            recommendations_by_layer['layer_2'].diagnostics.units_ratio, 0.6154555670037302
        )

    def test_pfa_en_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(128)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=128)
        recommendations_by_layer = pfa.get_recipe(
            strategy=PFA.Strategy.Energy(energy_threshold=0.98))

        assert set(recommendations_by_layer.keys()) == set(response_names), \
            "The recipe should have recommendations for all responses named {}, " \
            "and those only".format(response_names)

        # validate recipe for layer_0
        assert recommendations_by_layer['layer_0'].original_output_count == 128
        assert recommendations_by_layer['layer_0'].recommended_output_count == 82
        assert isinstance(recommendations_by_layer['layer_0'].diagnostics, PFAEnergyDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_0'].diagnostics.total_kept_energy, 0.0037030691584144576
        )

        # validate recipe for layer_1
        assert recommendations_by_layer['layer_1'].original_output_count == 64
        assert recommendations_by_layer['layer_1'].recommended_output_count == 45
        assert isinstance(recommendations_by_layer['layer_1'].diagnostics, PFAEnergyDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_1'].diagnostics.total_kept_energy, 0.026573058327728254
        )

        # validate recipe for layer_2
        assert recommendations_by_layer['layer_2'].original_output_count == 32
        assert recommendations_by_layer['layer_2'].recommended_output_count == 17
        assert isinstance(recommendations_by_layer['layer_2'].diagnostics, PFAEnergyDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_2'].diagnostics.total_kept_energy, 0.1743278851257953
        )

    def test_pfa_en_violation_warning(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(128)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        min_count = stub_data['layer_2'].shape[-1] - 1

        pfa = PFA.introspect(producer, batch_size=128)
        recipes = pfa.get_recipe(
            strategy=PFA.Strategy.Energy(energy_threshold=0.01, min_kept_count=min_count))

        for recipe in recipes.values():
            assert recipe.recommended_output_count == min_count

    def test_pfa_not_enough_samples_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']

        # these responses had enough data and are produced
        expected_response_names = ['layer_1', 'layer_2']

        # these failed -- not enough data
        failed_responses = ['layer_0']

        stub_data = self.make_legacy_stub_responses(100)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        with pytest.warns(UserWarning):
            pfa = PFA.introspect(producer, batch_size=151)

        recommendations_by_layer = pfa.get_recipe(strategy=PFA.Strategy.KL())

        assert set(recommendations_by_layer.keys()) == set(expected_response_names), \
            "The recipe should have recommendations for all responses named {}, " \
            "and those only".format(response_names)

        # expect that there is a response that failed to process
        assert set(pfa.failed_responses) == set(failed_responses)

        # validate recipe for layer_1
        assert recommendations_by_layer['layer_1'].original_output_count == 64
        assert recommendations_by_layer['layer_1'].recommended_output_count == 50
        assert isinstance(recommendations_by_layer['layer_1'].diagnostics, PFAKLDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_1'].diagnostics.kl_divergence, 0.9482624890154597
        )
        assert np.isclose(
            recommendations_by_layer['layer_1'].diagnostics.units_ratio, 0.7719910682727285
        )

        # validate recipe for layer_2
        assert recommendations_by_layer['layer_2'].original_output_count == 32
        assert recommendations_by_layer['layer_2'].recommended_output_count == 20
        assert isinstance(recommendations_by_layer['layer_2'].diagnostics, PFAKLDiagnostics)
        assert np.isclose(
            recommendations_by_layer['layer_2'].diagnostics.kl_divergence, 1.375076807139795
        )
        assert np.isclose(
            recommendations_by_layer['layer_2'].diagnostics.units_ratio, 0.6032367018995977
        )

    def test_pfa_unit_selection_l1_max_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(200)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=100)
        recs = pfa.get_recipe(strategy=PFA.Strategy.Energy(energy_threshold=0.98),
                              unit_strategy=PFA.UnitSelectionStrategy.L1Max())

        # validate unit selection for layer_0
        assert recs['layer_0'].original_output_count == 128
        assert recs['layer_0'].maximally_correlated_units is not None
        assert recs['layer_0'].recommended_output_count == \
               128 - len(recs['layer_0'].maximally_correlated_units)
        assert sorted(recs['layer_0'].maximally_correlated_units[0:13]) == \
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]

        # validate unit selection for layer_1
        assert recs['layer_1'].original_output_count == 64
        assert recs['layer_1'].recommended_output_count == \
               64 - len(recs['layer_1'].maximally_correlated_units)
        assert sorted(list(recs['layer_1'].maximally_correlated_units)) == [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 30, 50
        ]

        # validate unit selection for layer_2
        assert recs['layer_2'].original_output_count == 32
        assert recs['layer_2'].recommended_output_count == \
               32 - len(recs['layer_2'].maximally_correlated_units)
        assert sorted(list(recs['layer_2'].maximally_correlated_units)) == [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26
        ]

    def test_pfa_unit_selection_l1_min_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(200)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=100)
        recs = pfa.get_recipe(strategy=PFA.Strategy.Energy(energy_threshold=0.98),
                              unit_strategy=PFA.UnitSelectionStrategy.L1Min())

        # validate unit selection for layer_0
        assert recs['layer_0'].original_output_count == 128
        assert recs['layer_0'].maximally_correlated_units is not None
        assert recs['layer_0'].recommended_output_count == 128 - len(
            recs['layer_0'].maximally_correlated_units)
        assert set(recs['layer_0'].maximally_correlated_units).issubset(set(range(15, 128)))

        # validate unit selection for layer_1
        assert recs['layer_1'].original_output_count == 64
        assert recs['layer_1'].recommended_output_count == 64 - len(
            recs['layer_1'].maximally_correlated_units)
        assert sorted(list(recs['layer_1'].maximally_correlated_units)) == [
            18, 20, 23, 25, 28, 32, 37, 42, 44, 46, 51, 52, 58, 59, 60, 61, 63
        ]

        # validate unit selection for layer_2
        assert recs['layer_2'].original_output_count == 32
        assert recs['layer_2'].recommended_output_count == 32 - len(
            recs['layer_2'].maximally_correlated_units)
        assert sorted(list(recs['layer_2'].maximally_correlated_units)) == [
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
        ]

    def test_pfa_unit_selection_abs_max_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(200)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=100)
        unit_strategy = PFA.UnitSelectionStrategy.AbsMax()
        recs = pfa.get_recipe(strategy=PFA.Strategy.Energy(energy_threshold=0.98),
                              unit_strategy=unit_strategy)

        # validate unit selection for layer_0
        assert recs['layer_0'].original_output_count == 128
        assert recs['layer_0'].maximally_correlated_units is not None
        assert recs['layer_0'].recommended_output_count == 128 - len(
            recs['layer_0'].maximally_correlated_units)
        assert sorted(list(recs['layer_0'].maximally_correlated_units)) == [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            19, 23, 24, 30, 31, 35, 38, 48, 55, 60, 65, 70,
            73, 75, 77, 80, 82, 85, 113
        ]

        # validate unit selection for layer_1
        assert recs['layer_1'].original_output_count == 64
        assert recs['layer_1'].recommended_output_count == 64 - len(
            recs['layer_1'].maximally_correlated_units)
        assert sorted(list(recs['layer_1'].maximally_correlated_units)) == [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 41, 46
        ]

        # validate unit selection for layer_2
        assert recs['layer_2'].original_output_count == 32
        assert recs['layer_2'].recommended_output_count == 32 - len(
            recs['layer_2'].maximally_correlated_units)
        assert sorted(list(recs['layer_2'].maximally_correlated_units)) == [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 27
        ]

        # test error in this implementation
        test_corr = np.array([[1.00, 0.90, 0.80, 0.99],
                              [0.90, 1.00, 0.75, 0.65],
                              [0.80, 0.75, 1.00, 0.99],
                              [0.98, 0.65, 0.99, 1.00]])

        assert sorted(unit_strategy._select_units_given_corr(
            test_corr,
            found_indices=np.empty((0, 0)),
            num_units_to_keep=2)
        ) == [0, 2]

    def test_pfa_unit_selection_abs_min_vs_legacy(self) -> None:
        response_names = ['layer_0', 'layer_1', 'layer_2']
        stub_data = self.make_legacy_stub_responses(200)
        producer = pipeline(StubProducer(stub_data),
                            Pooler(
                                dim=(1, 2),
                                method=Pooler.Method.MAX,
                                fields=response_names
                            ))

        pfa = PFA.introspect(producer, batch_size=100)
        recs = pfa.get_recipe(strategy=PFA.Strategy.Energy(energy_threshold=0.98),
                              unit_strategy=PFA.UnitSelectionStrategy.AbsMin())

        # validate unit selection for layer_0
        assert recs['layer_0'].original_output_count == 128
        assert recs['layer_0'].maximally_correlated_units is not None
        assert recs['layer_0'].recommended_output_count == 128 - len(
            recs['layer_0'].maximally_correlated_units)
        assert sorted(list(recs['layer_0'].maximally_correlated_units)) == [
            18, 24, 25, 28, 34, 35, 36, 39,
            53, 54, 57, 58, 61,
            62, 66, 67, 70, 78, 80, 83, 84,
            87, 89, 91, 95, 98,
            99, 113, 115, 116, 118, 120, 122
        ]

        # validate unit selection for layer_1
        assert recs['layer_1'].original_output_count == 64
        assert recs['layer_1'].recommended_output_count == 64 - len(
            recs['layer_1'].maximally_correlated_units)
        assert sorted(list(recs['layer_1'].maximally_correlated_units)) == [
            0, 20, 26, 28, 33, 37, 39, 41, 47, 49, 52, 53, 54, 59, 60, 61, 63
        ]

        # validate unit selection for layer_2
        assert recs['layer_2'].original_output_count == 32
        assert recs['layer_2'].recommended_output_count == 32 - len(
            recs['layer_2'].maximally_correlated_units)
        assert sorted(list(recs['layer_2'].maximally_correlated_units)) == [
            15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31
        ]


def test_pfa_kl_linear_interpolation_function() -> None:
    kl_divergences = np.abs(np.random.randn(31))
    max_kl_divergence = max(kl_divergences)

    for kl_divergence in kl_divergences:
        units_ratio = PFA.Strategy.KL.LinearInterpolation()(
            kl_divergence, max_kl_divergence)

        assert np.isclose(
            (1.0 - units_ratio) * max_kl_divergence,
            kl_divergence
        )


def recipe_results(kl: bool = True) -> t.Mapping[str, PFARecipe]:
    def get_diagnostic(kl: bool = True) -> t.Union[PFAKLDiagnostics, PFAEnergyDiagnostics, None]:
        if kl:
            return PFAKLDiagnostics(
                kl_divergence=1.2,
                units_ratio=.6
            )
        else:
            return PFAEnergyDiagnostics(
                total_kept_energy=.8
            )

    return {
        'layer_0': PFARecipe(
            original_output_count=10,
            recommended_output_count=6,
            maximally_correlated_units=[1, 2, 3, 4],
            number_inactive_units=4,
            diagnostics=get_diagnostic(kl)
        ),
        'layer_1': PFARecipe(
            original_output_count=20,
            recommended_output_count=10,
            maximally_correlated_units=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            number_inactive_units=10,
            diagnostics=get_diagnostic(kl)
        )
    }


@pytest.mark.skipif(not _pandas_available(), reason="Pandas not installed")
@pytest.mark.parametrize(
    'kl, include_columns, exclude_columns, resulting_columns',
    [
        # Default columns, Energy
        (
            False,
            None,
            None,
            ['layer name', 'original count', 'recommended count', "units to keep"],
        ),
        # Default columns, KL
        (
            True,
            None,
            None,
            ['layer name', 'original count', 'recommended count', "units to keep"],
        ),
        # All columns, Energy
        (
            False,
            [],
            None,
            ['layer name', 'original count', 'recommended count', 'units to keep', 'KL divergence',
             'PFA strategy', 'units ratio', 'kept energy'],
        ),
        # All columns, KL
        (
            True,
            [],
            None,
            ['layer name', 'original count', 'recommended count', 'units to keep', 'KL divergence',
             'PFA strategy', 'units ratio', 'kept energy'],
        ),
        # Specify include columns, KL
        (
            True,
            ['PFA strategy', 'units ratio', 'kept energy'],
            None,
            ['PFA strategy', 'units ratio', 'kept energy'],
        ),
        # Specify exclude columns, KL
        (
            True,
            [],
            ['layer name', 'original count', 'recommended count'],
            ['units to keep', 'KL divergence', 'PFA strategy', 'units ratio', 'kept energy'],
        ),
        # Specify bad include columns, KL
        (
            True,
            ['layer name', 'original count', 'recommended count', 'dummy_column'],
            None,
            ['layer name', 'original count', 'recommended count'],
        ),
        # Specify bad exclude columns, KL
        (
            True,
            None,
            ['layer name', 'original count', 'recommended count', 'dummy_column'],
            ['units to keep'],
        )
    ]
)
def test_pfa_show_table(kl: bool,
                        include_columns: t.Sequence[str],
                        exclude_columns: t.Sequence[str],
                        resulting_columns: t.Sequence[str]) -> None:
    results = PFA.show(
        recipe_result=recipe_results(kl),
        include_columns=include_columns,
        exclude_columns=exclude_columns,
    )

    assert len(results.columns) == len(resulting_columns)
    assert set(results.columns) == set(resulting_columns)

    if kl:
        if 'kind' in resulting_columns:
            assert results['kind'][0] == 'PFA KL'
        if 'kept energy' in resulting_columns:
            assert results['kept energy'][0] == 'N/A'
    else:
        if 'kind' in resulting_columns:
            assert results['kind'][0] == 'PFA Energy'
        if 'units ratio' in resulting_columns:
            assert results['units ratio'][0] == 'N/A'
        if 'KL divergence' in resulting_columns:
            assert results['KL divergence'][0] == 'N/A'


@pytest.mark.skipif(not _pandas_available(), reason="Pandas not installed")
def test_pfa_show_table_multi_result() -> None:
    results = PFA.show(
        recipe_result=[
            recipe_results(kl=True),
            recipe_results(kl=False)
        ]
    )

    assert len(results) == 4


@pytest.mark.skipif(not _pandas_available(), reason="Pandas not installed")
def test_pfa_show_table_exceptions() -> None:
    recipe = recipe_results()

    # completely cancel out things
    with pytest.raises(DNIKitException):
        PFA.show(
            recipe,
            include_columns=['kind'],
            exclude_columns=['kind']
        )
    with pytest.raises(DNIKitException):
        PFA.show(
            recipe,
            exclude_columns=[
                'layer name',
                'original count',
                'recommended count',
                "units to keep"
            ]
        )

    # emtpy arguments
    with pytest.raises(ValueError):
        PFA.show({})
    with pytest.raises(ValueError):
        PFA.show([])

    # Invalid vis type
    with pytest.raises(ValueError):
        PFA.show(recipe, vis_type='invalid')


@pytest.mark.skipif(not _matplotlib_available(), reason="Matplotlib not installed")
def test_pfa_show_chart() -> None:
    recipes = recipe_results()

    # Check base chart show method
    assert PFA.show(recipes, vis_type=PFA.VisType.CHART) is not None

    # Can only pass one recipe to PFA show for show chart
    with pytest.raises(DNIKitException):
        PFA.show((recipes, recipes), vis_type=PFA.VisType.CHART)

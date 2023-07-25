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

import dataclasses
import re
import pathlib

import pytest
import numpy as np
import tensorflow as tf

import dnikit.typing._types as t
from dnikit.samples import StubGatedAdditionDataset
from dnikit.base import pipeline, ResponseInfo, Model, Batch, PipelineStage
from dnikit.processors import (
    FieldRenamer,
    FieldRemover,
    SnapshotSaver,
)
from dnikit_tensorflow import load_tf_model_from_path
from dnikit_tensorflow.samples import get_simple_tf1_lstm_model
from dnikit_tensorflow._tensorflow._tensorflow_protocols import running_tf_1


# The only TF 2 test
@pytest.mark.skipif(running_tf_1(), reason="Skipping the TF2-only test.")
def test_lstm_model_with_tf2() -> None:
    simple_lstm_config_override = {
        'maximum_sequence_length': 50,
        'num_lstm_nodes_fw': 100,
        'num_lstm_nodes_bw': 100,
        'num_layers': 1,
    }
    with pytest.raises(EnvironmentError):
        _ = get_simple_tf1_lstm_model(simple_lstm_config_override)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    base_path = tmp_path_factory.mktemp('basic_lstm')

    simple_lstm_config_override = {
        'maximum_sequence_length': 50,
        'num_lstm_nodes_fw': 100,
        'num_lstm_nodes_bw': 100,
        'num_layers': 1,
    }

    with tf.Session() as sess:
        _ = get_simple_tf1_lstm_model(simple_lstm_config_override)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        _ = saver.save(sess, str(base_path))

    return base_path


def set_seed(seed: int = 1234) -> None:
    tf.random.set_random_seed(seed)
    np.random.seed(seed)


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 test.")
def test_instantiation(model_path: pathlib.Path) -> None:
    model = load_tf_model_from_path(path=model_path)
    assert (model is not None)


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 test.")
def test_response_meta(model_path: pathlib.Path) -> None:
    model = load_tf_model_from_path(path=model_path)
    set_seed(1234)

    # get memory and sequential type operations of forward LSTM cell
    forward_memory_c_seq_h_operations = [
        info
        for info in model.response_infos.values() if
        re.match(r'(.*)my_(seq_output|cell_outputs)_forward:0', info.name)
    ]

    # get memory and sequential type operations of backward LSTM cell
    backward_memory_c_seq_h_operations = [
        info
        for info in model.response_infos.values() if
        re.match(r'(.*)my_(seq_output|cell_outputs)_backward:0', info.name)
    ]

    # get all gate operations
    gate_operations = [
        info
        for info in model.response_infos.values() if
        re.match(r'(.*)my_gates_outputs_(forward|backward):0', info.name)
    ]

    # # there should be at least one forward cell operation
    assert (len(forward_memory_c_seq_h_operations) > 0)

    # # there should be at least one backward cell operation
    assert (len(backward_memory_c_seq_h_operations) > 0)

    # # there should be at least one gate operation
    assert (len(gate_operations) > 0)

    for operation in forward_memory_c_seq_h_operations:
        # all forward operations should have a name like '<scope>/my_(outputs|cell_state)'
        m = re.match(r'(?P<scope>.*)\/(outputs|cell_state)*', operation.name)
        assert (m is not None)

        name_scope = m.group('scope')
        assert (len(name_scope) > 0)

        # within the <scope> scope, there should be at least 1 response
        scope_response_infos = [
            info
            for info in model.response_infos.values()
            if name_scope in info.name
        ]
        assert (len(scope_response_infos) >= 1)

        # Of those responses, there should be at most two with name
        # '<scope>/my_(outputs|cell_state)_ta_intermediate_forward:0'
        sequential_responses_in_scope = [
            info
            for info in scope_response_infos
            if info.name.startswith(f'{name_scope}/my_output_ta_intermediate_forward:0')
            or info.name.startswith(f'{name_scope}/my_cell_state_ta_intermediate_forward:0')
        ]
        assert (len(sequential_responses_in_scope) == 1 or len(sequential_responses_in_scope) == 2)

        # And the names of those response should be '<scope>/my_outputs_ta_intermediate_forward:0'
        # or '<scope>/my_cell_state_ta_intermediate_forward:0'
        assert sequential_responses_in_scope[0].name in (
            f'{name_scope}/my_outputs_ta_intermediate_forward:0'
            f'{name_scope}/my_cell_state_ta_intermediate_forward:0'
        )


@t.final
@dataclasses.dataclass(frozen=True)
class Stacker(PipelineStage):
    """
    If the Batch responses have multiple samples for a given time step,
    this will convert them into batches -- basically unfolding the first dimension into the batch
    dimension. This is common in sequence models.  Specifically this will turn an BxTxC batch into a
    BTxC batch.

    It's possible to use a ``sequence_lengths`` callable that can access Batch metadata to provide
    per-values lengths.  For example the the batch was 3x5x40, the callable would produce 3 values
    between 0 and 5.  If it returned ``[1, 2, 5]`` the resulting batch would be 3*(1 + 2 + 5) x 40
    in length.

    Warnings:
        All responses must be the same batch length -- this must be applied to all
        layers in the batch.  All of the ``T`` dimensions must be the same.

        This PipelineStage changes the batch size and thus is forced to discard
        any attached ``metadata`` and ``snapshots`` from the Batch.
    """

    sequence_lengths: t.Optional[t.Callable[[Batch], t.Sequence[int]]] = None

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        # This is a PipelineStage rather than a Processor.  It needs access to the
        # batch metadata and needs to do a set of work (preparing the lengths) once per batch.

        def batch_processor(batch: Batch) -> Batch:
            # note: neither metadata nor snapshots can be copied into the new batch
            # because the batch size is changing
            builder = Batch.Builder()

            # get one of the responses to get comparison data for the dimensions
            any_response = next(iter(batch.fields.values()))

            assert len(any_response.shape) >= 3, "batches are required to be BxTxC (3+ dimensions)"
            batch_length = any_response.shape[0]
            value_length = any_response.shape[1]

            # now verify each of the responses
            for name, response in batch.fields.items():
                assert len(response.shape) >= 3, (
                    f"batches are required to be BxTxC (3+ dimensions): {name}"
                )
                assert response.shape[1] == value_length, (
                    f"batches must have the same T dimension: "
                    f"{value_length} vs {response.shape[1]} in {name}"
                )

            # produce the length array
            if self.sequence_lengths is None:
                lengths: t.Sequence[int] = [value_length] * batch_length
            else:
                lengths = self.sequence_lengths(batch)
                assert len(lengths) == batch_length, (
                    f"sequence_lengths returned {len(lengths)} values "
                    f"but {batch_length} are required"
                )
                test_lengths = np.array(lengths)
                assert np.alltrue(test_lengths <= value_length), (
                    f"all values from sequence_lengths "
                    f"must be <= {value_length}: {lengths}"
                )
                assert np.alltrue(test_lengths >= 0), (
                    f"all values from sequence_lengths "
                    f"must be >= 0: {lengths}"
                )

            # for each response in the incoming batch, remove the T dimension respecting the
            # requested lengths of each T.  Here, a list of np.ndarrays is concatenated.
            # e.g. looking at the shapes if there are lengths [1, 2, 3] and an input of
            # (3, 3, 2) -> concatenate((1, 1, 2), (1, 2, 2), (1, 3, 2)).
            for field, value in batch.fields.items():
                builder.fields[field] = np.concatenate([
                        value[batch_index, :length, ...]
                        for batch_index, length in enumerate(lengths)
                    ], axis=0)

            return builder.make_batch()

        return batch_processor


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 test.")
def test_response_generation(model_path: pathlib.Path) -> None:
    model = load_tf_model_from_path(path=model_path)
    set_seed(1234)

    dataset_size = 500
    dataset = StubGatedAdditionDataset(
        dataset_size=dataset_size, minimum_sequence_length=45, maximum_sequence_length=50)

    (
        forward_memory_c_seq_h_operations,
        backward_memory_c_seq_h_operations,
        final_ops,
        init_zero_cell_state_ops,
        data_to_input_fields_map
    ) = _get_operations(model, scope='')

    requested_responses = (
        forward_memory_c_seq_h_operations + backward_memory_c_seq_h_operations + final_ops
    )
    sequence_responses = forward_memory_c_seq_h_operations + backward_memory_c_seq_h_operations

    def sequence_lens_accessor(r_batch: Batch) -> t.Sequence[int]:
        return [int(data) for data in r_batch.snapshots["snapshot"].fields['my_seq_len:0']]

    producer = pipeline(
        dataset,
        FieldRenamer(data_to_input_fields_map),
        SnapshotSaver(),
        model(requested_responses),
        # trim it down to just the sequence responses -- it's necessary that the batch lengths
        # are consistent and these are the fields that will remain consistent.
        FieldRemover(fields=sequence_responses, keep=True),
        Stacker(sequence_lengths=sequence_lens_accessor),
    )

    total_response_data_points = 0

    for response_batch in producer(100):
        total_response_data_points += response_batch.batch_size

    assert dataset_size * 45 < total_response_data_points <= dataset_size * 50


def _get_operations(model: Model, scope: str = '') -> t.Tuple[
        t.List[str], t.List[str], t.List[str], t.List[str], t.Mapping[str, str]]:
    #
    # Get memory and sequential type operations of forward LSTM cell
    #
    forward_memory_c_seq_h_operations = [
        info
        for info in model.response_infos if
        re.match(scope + r'(.*)my_(seq_output|cell_outputs)_forward:0', info)
    ]
    #
    # Get memory and sequential type operations of backward LSTM cell
    #
    backward_memory_c_seq_h_operations = [
        info
        for info in model.response_infos if
        re.match(scope + r'(.*)my_(seq_output|cell_outputs)_backward:0', info)
    ]

    #
    # Get final output operations of both backward and forward cell
    #
    final_seq_h_operations = [
        info
        for info in model.response_infos if
        re.match(scope + r'(.*)my_final_output_(forward|backward):0', info)
    ]

    #
    # Get zero state memory and sequential operations
    #
    init_zero_cell_state_operations = [
        info
        for info in model.response_infos if
        re.match(scope + r'(.*)LSTMCellZeroState/zeros:0', info)
    ]
    #
    # Idiom to search for placeholder input layers, sequence length layer and output layers
    # within a DNIKit Model
    #
    potential_input_response_infos = [
        info.name
        for info in model.response_infos.values()
        if 'my_lstm_input' in info.name
        and info.layer.kind is ResponseInfo.LayerKind.PLACEHOLDER
        and info.shape == (None, 50, 2)
    ]
    input_layer_name = scope + 'my_lstm_input:0'
    assert (input_layer_name in potential_input_response_infos)

    potential_sequence_response_infos = [
        info.name
        for info in model.response_infos.values()
        if 'my_seq_len' in info.name
        and info.layer.kind is ResponseInfo.LayerKind.PLACEHOLDER
    ]
    input_sequence_len_name = scope + 'my_seq_len:0'
    assert (input_sequence_len_name in potential_sequence_response_infos)

    potential_output_response_infos = [
        info.name
        for info in model.response_infos.values()
        if 'my_lstm_output' in info.name
        and info.layer.kind is ResponseInfo.LayerKind.PLACEHOLDER
    ]
    output_layer_name = scope + 'my_lstm_output:0'
    assert (output_layer_name in potential_output_response_infos)

    data_to_input_fields_map = {
        'x': input_layer_name,
        'target': output_layer_name,
        'sequence_length': input_sequence_len_name,
    }

    return (
        forward_memory_c_seq_h_operations,
        backward_memory_c_seq_h_operations,
        final_seq_h_operations,
        init_zero_cell_state_operations,
        data_to_input_fields_map
    )

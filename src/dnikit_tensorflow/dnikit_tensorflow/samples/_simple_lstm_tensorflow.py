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
import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time
from tensorflow.compat.v1.nn.rnn_cell import RNNCell
import typing as t


def get_default_config() -> t.Dict[str, t.Any]:
    return {
        'name': 'tensorflow_simplelstm',
        'maximum_sequence_length': 50,
        'num_lstm_nodes_fw': 100,
        'num_lstm_nodes_bw': 100,
        'num_layers': 1,
        'gradient_clip': 10
    }


def get_model(config_override: t.Mapping[str, t.Any] = {}) -> "Model":
    """
    Create SimpleLSTM model with the default configuration overridden with keys specified in
    `config`
    """
    if tf.__version__[0] != '1':
        raise EnvironmentError(
            f"The simple lstm model sample can only be used with TF1, not TF {tf.__version__}.")

    config = get_default_config()
    config.update(config_override)
    maximum_sequence_length = config['maximum_sequence_length']
    num_lstm_nodes_fw = config['num_lstm_nodes_fw']
    num_lstm_nodes_bw = config['num_lstm_nodes_bw']
    num_layers = config['num_layers']
    gradient_clip = config['gradient_clip']

    return Model.build_model(
        maximum_sequence_length, num_lstm_nodes_fw, num_lstm_nodes_bw, num_layers, gradient_clip
    )


class MyLSTMCell(RNNCell):
    def __init__(self, num_units, initializer=None, forget_bias=1.0, activation=None, reuse=None):
        super(MyLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or tf.nn.tanh
        self._lstm_matrix_weights = self._lstm_matrix_biases = None

        self._state_size = tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def step(self, c_prev, m_prev, inputs, lstm_matrix_weights, lstm_matrix_biases):
        x = tf.concat([inputs, m_prev], 1)
        lstm_matrix = tf.matmul(x, lstm_matrix_weights)
        lstm_matrix = tf.nn.bias_add(lstm_matrix, lstm_matrix_biases)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        input_gate, new_input, forget_gate, output_gate = tf.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        input_gate_activation = tf.nn.sigmoid(input_gate)
        forget_gate_activation = tf.nn.sigmoid(forget_gate + self._forget_bias)
        output_gate_activation = tf.nn.sigmoid(output_gate)

        c = (forget_gate_activation * c_prev + input_gate_activation * self._activation(new_input))

        m = output_gate_activation * self._activation(c)

        new_state = tf.contrib.rnn.LSTMStateTuple(c, m)

        return m, (input_gate_activation, forget_gate_activation, output_gate_activation), new_state

    def call(self, inputs, state, scope=''):
        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        dtype = inputs.dtype
        if self._lstm_matrix_weights is None or self._lstm_matrix_biases is None:
            # lazy initialization, to enable use of e.g. dtype
            with tf.variable_scope(tf.get_variable_scope(), initializer=self._initializer):
                lstm_matrix_input_size = input_size + self._num_units
                lstm_matrix_output_size = 4 * self._num_units
                self._lstm_matrix_weights = tf.get_variable(
                    "kernel", [lstm_matrix_input_size, lstm_matrix_output_size],
                    dtype=dtype, initializer=self._initializer)
                self._lstm_matrix_biases = tf.get_variable(
                    "bias", [lstm_matrix_output_size],
                    dtype=dtype, initializer=tf.constant_initializer(0.0, dtype=dtype))

        return self.step(
            c_prev, m_prev, inputs, self._lstm_matrix_weights, self._lstm_matrix_biases)


class LSTMCell(MyLSTMCell):
    # just rename __class__ for TF's convenience
    pass


class BasicLSTMCell(MyLSTMCell):
    # just rename __class__ for TF's convenience
    pass


class Model:

    def __init__(self,
                 x_placeholder,
                 y_placeholder,
                 sequence_lengths_placeholder,
                 learning_rate_placeholder,
                 average_error,
                 loss,
                 minimizer,
                 cell_fw,
                 cell_bw,
                 sequence_outputs_fw,
                 sequence_outputs_bw,
                 final_outputs_fw,
                 final_outputs_bw,
                 memory_cells_fw,
                 memory_cells_bw,
                 bidirectional_gates,
                 final_fc):
        self.x_placeholder = x_placeholder
        self.y_placeholder = y_placeholder
        self.sequence_lengths_placeholder = sequence_lengths_placeholder
        self.learning_rate_placeholder = learning_rate_placeholder
        self.average_error = average_error
        self.loss = loss
        self.minimizer = minimizer
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        self.sequence_outputs_fw = sequence_outputs_fw
        self.sequence_outputs_bw = sequence_outputs_bw
        self.final_outputs_fw = final_outputs_fw
        self.final_outputs_bw = final_outputs_bw
        self.memory_cells_fw = memory_cells_fw
        self.memory_cells_bw = memory_cells_bw
        self.bidirectional_gates = bidirectional_gates
        self.final_fc = final_fc

    @staticmethod
    def _dynamic_rnn(batch_size,
                     lstm_size,
                     inputs_ta,
                     max_seq_len,
                     cur_batch_max_seq_len,
                     sequence_length_placeholder,
                     reverse):
        cell = LSTMCell(num_units=lstm_size,
                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        zero_state = cell.zero_state(batch_size, tf.float32)

        def forward_cond_fn(time, *_):
            return time < cur_batch_max_seq_len

        def backward_cond_fn(time, *_):
            return time > 0

        def forward_loop_fn(time,
                            cell_state,
                            outputs_ta,
                            final_outputs,
                            cell_state_ta,
                            final_memory_cell_states,
                            gates_ta):
            (cell_output, gates, cell_state) = cell(inputs_ta.read(time), cell_state)

            elements_finished = (time >= sequence_length_placeholder)
            cell_output = tf.where(elements_finished, zero_state.h, cell_output)
            outputs_ta = outputs_ta.write(time, cell_output)

            cell_state = tf.contrib.rnn.LSTMStateTuple(
                tf.where(elements_finished, zero_state.c, cell_state.c),
                tf.where(elements_finished, zero_state.h, cell_state.h))

            cell_state_ta = cell_state_ta.write(time, cell_state.c)

            final_outputs = tf.where(
                elements_finished, final_outputs, cell_state.h)
            final_memory_cell_states = tf.where(
                elements_finished, final_memory_cell_states, cell_state.c)

            stacked_gates = tf.stack(gates, axis=1)
            gates_ta = gates_ta.write(
                time,
                tf.where(elements_finished, tf.zeros_like(stacked_gates), stacked_gates)
            )
            return (
                time+1,
                cell_state,
                outputs_ta,
                final_outputs,
                cell_state_ta,
                final_memory_cell_states,
                gates_ta
            )

        def backward_loop_fn(time,
                             cell_state,
                             outputs_ta,
                             final_outputs,
                             cell_state_ta,
                             final_memory_cell_states,
                             gates_ta):
            (cell_output, gates, cell_state) = cell(inputs_ta.read(time-1), cell_state)

            elements_reached = (time <= sequence_length_placeholder)
            cell_state = tf.contrib.rnn.LSTMStateTuple(
                tf.where(elements_reached, cell_state.c, zero_state.c),
                tf.where(elements_reached, cell_state.h, zero_state.h))
            outputs_ta = outputs_ta.write(time-1, cell_state.h)
            cell_state_ta = cell_state_ta.write(time - 1, cell_state.c)

            final_outputs = tf.where(elements_reached, cell_state.h, final_outputs)
            final_memory_cell_states = tf.where(
                elements_reached, cell_state.c, final_memory_cell_states)

            stacked_gates = tf.stack(gates, axis=1)
            gates_ta = gates_ta.write(
                time - 1,
                tf.where(elements_reached, stacked_gates, tf.zeros_like(stacked_gates))
            )

            return (
                time-1,
                cell_state,
                outputs_ta,
                final_outputs,
                cell_state_ta,
                final_memory_cell_states,
                gates_ta
            )

        if not reverse:
            time = tf.constant(0, dtype=tf.int32, name='my-time')
        else:
            time = cur_batch_max_seq_len
        final_outputs = zero_state.h
        final_memory_cell_states = zero_state.c

        if reverse == 1:
            direction = 'backward'
        else:
            direction = 'forward'

        outputs_ta = tf.TensorArray(
            tf.float32, cur_batch_max_seq_len, name=f'my_outputs_ta_{direction}')
        cell_state_ta = tf.TensorArray(
            tf.float32, cur_batch_max_seq_len, name=f'my_cell_state_ta_{direction}')
        gates_ta = tf.TensorArray(
            tf.float32, cur_batch_max_seq_len, name=f'my_gates_ta_{direction}')
        (
            _, _,
            outputs_ta,
            final_outputs,
            cell_state_ta,
            final_memory_cell_states,
            gates_ta
        ) = tf.while_loop(
            forward_cond_fn if not reverse else backward_cond_fn,
            forward_loop_fn if not reverse else backward_loop_fn,
            loop_vars=[
                time,
                zero_state,
                outputs_ta,
                final_outputs,
                cell_state_ta,
                final_memory_cell_states,
                gates_ta
            ],
            # 32 is in line with what dynamic_rnn does internally
            parallel_iterations=32, name=f'my_while_loop_{direction}'
        )

        final_outputs = tf.identity(
            final_outputs, name=f'my_final_output_{direction}')
        final_memory_cell_states = tf.identity(
            final_memory_cell_states, name=f'my_final_cell_output_{direction}')

        sequence_outputs = outputs_ta.stack()
        sequence_outputs = _transpose_batch_time(sequence_outputs)

        sequence_outputs = tf.identity(
            sequence_outputs, name=f'my_outputs_ta_intermediate_{direction}')
        zero_padding = tf.zeros(
            [batch_size, max_seq_len-cur_batch_max_seq_len, lstm_size], tf.float32, name='my_zeros')
        sequence_outputs = tf.concat(
            [sequence_outputs, zero_padding], axis=1, name='my_seq_output_' + direction)
        sequence_outputs.set_shape([None, max_seq_len, lstm_size])
        outputs_ta.close().mark_used()

        cell_state_outputs = cell_state_ta.stack()
        cell_state_outputs = _transpose_batch_time(cell_state_outputs)
        cell_state_outputs = tf.identity(
            cell_state_outputs, name=f'my_cell_state_ta_intermediate_{direction}')
        cell_state_outputs = tf.concat(
            [cell_state_outputs, zero_padding], axis=1, name=f'my_cell_outputs_{direction}')
        cell_state_outputs.set_shape([None, max_seq_len, lstm_size])
        cell_state_ta.close().mark_used()

        gates_outputs = gates_ta.stack()
        gates_outputs = _transpose_batch_time(gates_outputs)
        gates_outputs = tf.identity(
            gates_outputs, name=f'my_gates_outputs_ta_intermediate_{direction}')
        gates_zero_padding = tf.zeros(
            [batch_size, max_seq_len-cur_batch_max_seq_len, 3, lstm_size], tf.float32)
        gates_outputs = tf.concat(
            [gates_outputs, gates_zero_padding], axis=1, name=f'my_gates_outputs_{direction}')
        gates_outputs.set_shape([None, max_seq_len, 3, lstm_size])
        gates_ta.close().mark_used()

        return (
            cell,
            sequence_outputs,
            final_outputs,
            cell_state_outputs,
            final_memory_cell_states,
            gates_outputs
        )

    @staticmethod
    def _bidirectional_dynamic_rnn(num_units_fw,
                                   num_units_bw,
                                   x,
                                   max_seq_len,
                                   sequence_length_placeholder):
        batch_size = tf.shape(x)[0]
        current_batch_max_seq_len = tf.reduce_max(sequence_length_placeholder)
        inputs = _transpose_batch_time(x)
        inputs_ta = tf.TensorArray(
            dtype=tf.float32, size=max_seq_len, clear_after_read=False, name='my-inputs-ta')
        inputs_ta = inputs_ta.unstack(inputs)

        with tf.variable_scope('bidirectional_lstm'):
            with tf.variable_scope('fw'):
                (
                    fw_cell,
                    fw_sequence_outputs,
                    fw_final_outputs,
                    fw_memory_cells,
                    fw_final_memory_status,
                    fw_gates
                ) = Model._dynamic_rnn(
                    batch_size,
                    num_units_fw,
                    inputs_ta,
                    max_seq_len,
                    current_batch_max_seq_len,
                    sequence_length_placeholder,
                    reverse=False
                )
            with tf.variable_scope('bw'):
                (
                    bw_cell,
                    bw_sequence_outputs,
                    bw_final_outputs,
                    bw_memory_cells,
                    bw_final_memory_status,
                    bw_gates
                ) = Model._dynamic_rnn(
                    batch_size,
                    num_units_bw,
                    inputs_ta,
                    max_seq_len,
                    current_batch_max_seq_len,
                    sequence_length_placeholder,
                    reverse=True
                )

        inputs_ta.close().mark_used()
        final_memory_status = tf.concat(
            (fw_final_memory_status, bw_final_memory_status), 1, name='my-final-memory-status'
        )
        bidirectional_gates = tf.concat((fw_gates, bw_gates), 3, name='my-bidirectional-gates')
        return (
            fw_cell,
            bw_cell,
            fw_sequence_outputs,
            bw_sequence_outputs,
            fw_final_outputs,
            bw_final_outputs,
            fw_memory_cells,
            bw_memory_cells,
            final_memory_status,
            bidirectional_gates
        )

    @staticmethod
    def build_model(max_sequence_length, num_units_fw, num_units_bw, num_layers,
                    gradient_clip=10, lstm_projections=None) -> "Model":
        # Placeholders
        x = tf.compat.v1.placeholder(
            tf.float32, name='my_lstm_input', shape=(None, max_sequence_length, 2))
        y = tf.compat.v1.placeholder(tf.float32, name='my_lstm_output', shape=None)
        sequence_lengths = tf.compat.v1.placeholder(tf.int32, name='my_seq_len',  shape=None)
        learning_rate_placeholder = tf.compat.v1.placeholder_with_default(0.0, shape=[])

        assert num_layers == 1  # Only single layer supported atm

        (
            cell_fw,
            cell_bw,
            sequence_outputs_fw,
            sequence_outputs_bw,
            final_outputs_fw,
            final_outputs_bw,
            memory_cells_fw,
            memory_cells_bw,
            final_memory_status,
            bidirectional_gates
        ) = Model._bidirectional_dynamic_rnn(
            num_units_fw,
            num_units_bw,
            x,
            max_sequence_length,
            sequence_lengths
        )

        if lstm_projections is not None:
            def _build_reconstruction_mat_and_bias(reconstructions, num_nodes_in):
                num_nodes_out = len(reconstructions.keys())

                weights = np.zeros([num_nodes_in, num_nodes_out])
                bias = np.zeros(num_nodes_out)

                for i in range(num_nodes_out):
                    weights[:, i] = reconstructions[i].coefficients
                    bias[i] = reconstructions[i].intercept
                return (
                    tf.cast(tf.constant(weights), tf.float32),
                    tf.cast(tf.constant(bias), tf.float32)
                )

            projections_fw, projections_bw = lstm_projections
            fw_projection_weights, fw_projection_bias = _build_reconstruction_mat_and_bias(
                projections_fw, num_units_fw
            )
            bw_projection_weights, bw_projection_bias = _build_reconstruction_mat_and_bias(
                projections_bw, num_units_bw
            )

            final_outputs_fw = tf.add(
                tf.matmul(final_outputs_fw, fw_projection_weights),
                fw_projection_bias
            )
            final_outputs_bw = tf.add(
                tf.matmul(final_outputs_bw, bw_projection_weights),
                bw_projection_bias
            )

        final_outputs = tf.concat((final_outputs_fw, final_outputs_bw), 1)
        final_fc = tf.layers.dense(final_outputs, units=1, activation=None, use_bias=True)
        final_output = tf.squeeze(final_fc)

        average_error = tf.reduce_mean(tf.abs(final_output - y))

        loss = tf.losses.mean_squared_error(y, final_output)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_placeholder)

        gradients_and_variables = optimizer.compute_gradients(loss)
        min_grad, max_grad = -gradient_clip, gradient_clip
        capped_gvs = [
            (tf.clip_by_value(grad, min_grad, max_grad) if grad is not None else None, var)
            for grad, var in gradients_and_variables
        ]
        minimizer = optimizer.apply_gradients(capped_gvs)

        return Model(
            x, y,
            sequence_lengths,
            learning_rate_placeholder,
            average_error,
            loss,
            minimizer,
            cell_fw,
            cell_bw,
            sequence_outputs_fw,
            sequence_outputs_bw,
            final_outputs_fw,
            final_outputs_bw,
            memory_cells_fw,
            memory_cells_bw,
            bidirectional_gates,
            final_fc
        )

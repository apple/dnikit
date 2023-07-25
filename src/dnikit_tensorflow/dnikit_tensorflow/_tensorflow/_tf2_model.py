#
# Copyright 2022 Apple Inc.
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
import logging

import tensorflow as tf
import numpy as np

import dnikit.typing._types as t
from dnikit.base import ResponseInfo
from dnikit.base._model import _ModelDetails


_logger = logging.getLogger("dnikit_tensorflow.TF2")


_KNOWN_OPS: t.Final[t.Mapping[str, ResponseInfo.LayerKind]] = {
    "Placeholder": ResponseInfo.LayerKind.PLACEHOLDER,
    "Softmax": ResponseInfo.LayerKind.SOFTMAX,
    "Relu": ResponseInfo.LayerKind.RELU,
    "Relu6": ResponseInfo.LayerKind.RELU6,
    "Conv2D": ResponseInfo.LayerKind.CONV_2D,
    "FusedBatchNormV3": ResponseInfo.LayerKind.BATCH_NORM
}

_LAYER_PREFIXES: t.Final[t.Mapping[str, ResponseInfo.LayerKind]] = {
    'conv': ResponseInfo.LayerKind.CONV_2D,
    'input': ResponseInfo.LayerKind.PLACEHOLDER,
    'dropout': ResponseInfo.LayerKind.DROPOUT,
    'global_average_pooling': ResponseInfo.LayerKind.AVERAGE_POOLING_2D
}


def _convert_tf_shape(shape: tf.TensorShape) -> t.Tuple[t.Optional[int], ...]:
    if shape.dims is None:
        return tuple()
    return tuple(dim for dim in shape.dims)


def _convert_tf_dtype(dtype: tf.dtypes.DType) -> np.dtype:
    return dtype.as_numpy_dtype if dtype.is_numpy_compatible else np.dtype(object)


def _remove_op_number(name: str) -> str:
    if ':' in name:
        name = name.split(':')[0]
    return name


def _extract_kind(full_name: str) -> str:
    return _remove_op_number(full_name.split('/')[-1])


def _extract_layer_prefix(full_name: str) -> str:
    return _remove_op_number(full_name.split('/')[0])


def _convert_tf_operation(layer_name: str) -> ResponseInfo.LayerKind:
    # First check known operations
    operation = _extract_kind(layer_name)
    if operation in _KNOWN_OPS:
        return _KNOWN_OPS[operation]

    # next, check layer name prefixes
    for layer_prefix, layer_kind in _LAYER_PREFIXES.items():
        if layer_name.startswith(layer_prefix):
            return layer_kind

    # Otherwise, layer is unknown
    return ResponseInfo.LayerKind.UNKNOWN


@t.final
@dataclass
class _Tensorflow2ModelDetails(_ModelDetails):
    """Class wrapping a Tensorflow 2 model so that it can be seamlessly used in DNIKit."""

    model: tf.keras.models.Model

    def __post_init__(self) -> None:
        _logger.info("Instantiating TF2 Model")
        _logger.info(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    def get_response_infos(self) -> t.Iterable[ResponseInfo]:
        # now go through all layers (including input) and return output
        for layer in self.model.layers:
            yield ResponseInfo(
                name=layer.name,
                dtype=_convert_tf_dtype(layer.output.type_spec.dtype),
                shape=_convert_tf_shape(layer.output.type_spec.shape),
                layer=ResponseInfo.Layer(
                    name=_remove_op_number(layer.output.name),
                    kind=_convert_tf_operation(layer.output.name),
                    typename=_extract_kind(layer.output.name)
                )
            )

    def get_input_layer_responses(self) -> t.Sequence[ResponseInfo]:
        """
        Get the names of possible input layers to the TF 2 :class:`Model`.
        """
        return [
            info
            for info in self.get_response_infos()
            if info.layer.kind is ResponseInfo.LayerKind.PLACEHOLDER
            and 'input' in info.name
        ]

    def run_inference(self,
                      inputs: t.Mapping[str, np.ndarray],
                      outputs: t.AbstractSet[str]) -> t.Mapping[str, np.ndarray]:
        self.model.trainable = False

        # Create a model object that takes in the same input as the original model, but
        #    reads the output of specific layers only (not just model output)
        inference_model = tf.keras.Model(
            inputs=[self.model.input],
            outputs=[
                self.model.get_layer(layer_name).output
                for layer_name in outputs
            ]
        )
        possible_inputs = [input_layer.name for input_layer in self.get_input_layer_responses()]
        for input_name in inputs.keys():
            if input_name not in possible_inputs:
                raise TypeError(
                    f'Invalid input "{input_name}". Valid inputs are {possible_inputs}.')

        results = inference_model(list(inputs.values()))

        # Inference on a single data sample collapses the batch dimension in the result, but
        #    the batch dimension needs to be in place for other parts of dnikit!
        if inputs[possible_inputs[0]].shape[0] == 1:
            r_val = {
                response_name: np.expand_dims(tensor.numpy(), axis=0)
                for response_name, tensor in zip(outputs, results)
            }
            return r_val

        # Otherwise, if only one response is needed, then there is only one result
        if len(outputs) == 1:
            return {list(outputs)[0]: results.numpy()}

        # Otherwise, there are multiple data samples, so just return tensors as-is (has batch dim)
        return {
            response_name: tensor.numpy()
            for response_name, tensor in zip(outputs, results)
        }

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

from dataclasses import dataclass
import logging

import tensorflow as tf
import numpy as np

import dnikit.typing._types as t
from dnikit.base import ResponseInfo
from dnikit.base._model import _ModelDetails


_logger = logging.getLogger("dnikit_tensorflow.TF1")


_KNOWN_OPS: t.Final[t.Mapping[str, ResponseInfo.LayerKind]] = {
    "Placeholder": ResponseInfo.LayerKind.PLACEHOLDER,
    "Softmax": ResponseInfo.LayerKind.SOFTMAX,
    "Relu": ResponseInfo.LayerKind.RELU,
    "Conv2D": ResponseInfo.LayerKind.CONV_2D
}


def _dim_to_int(dim: tf.compat.v1.Dimension) -> t.Optional[int]:
    return int(dim.value) if dim.value is not None else None


def _convert_tf_shape(shape: tf.TensorShape) -> t.Tuple[t.Optional[int], ...]:
    if shape.dims is None:
        return tuple()
    return tuple(_dim_to_int(dim) for dim in shape.dims)


def _convert_tf_dtype(dtype: tf.dtypes.DType) -> np.dtype:
    return dtype.as_numpy_dtype if dtype.is_numpy_compatible else np.dtype(object)


def _convert_tf_operation(operation: str) -> ResponseInfo.LayerKind:
    if operation in _KNOWN_OPS:
        return _KNOWN_OPS[operation]
    return ResponseInfo.LayerKind.UNKNOWN


@t.final
@dataclass(frozen=True)
class _Tensorflow1ModelDetails(_ModelDetails):
    """Class wrapping a Tensorflow 1 model so that it can be seamlessly used in DNIKit."""

    session: tf.compat.v1.Session

    def __post_init__(self) -> None:
        _logger.info("Instantiating TF1 Model")
        from tensorflow.python.client import device_lib
        devices = [x.name for x in device_lib.list_local_devices()]
        _logger.info("Devices available: ", devices)

    def get_response_infos(self) -> t.Iterable[ResponseInfo]:
        for op in self.session.graph.get_operations():
            for output_tensor in op.outputs:
                yield ResponseInfo(
                    name=output_tensor.name,
                    dtype=_convert_tf_dtype(output_tensor.dtype),
                    shape=_convert_tf_shape(output_tensor.shape),
                    layer=ResponseInfo.Layer(
                        name=op.name,
                        kind=_convert_tf_operation(op.type),
                        typename=op.type
                    )
                )

    def get_input_layer_responses(self) -> t.Sequence[ResponseInfo]:
        """
            Get the names of possible input layers to the current TF :class:`Session`.

            Note that this assumes only a single model is loaded into the session.
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
        # Run all the output operations in the graph and get the tensors back
        self.session.graph.add_to_collection(
            "IS_TRAINING", False)

        results = self.session.run(
            fetches=list(outputs),
            feed_dict=inputs
        )

        return {
            response_name: tensor
            for response_name, tensor in zip(outputs, results)
        }

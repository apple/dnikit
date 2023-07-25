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
import enum

import numpy as np

import dnikit.typing._types as t


@t.final
@dataclass(frozen=True)
class ResponseInfo:
    """
    Information about of a model's response.

    A response is the output tensor of a model operation (ie. layer or a module depending on the
    deep-learning framework used). Note that an operation may have more than one response.

    Args:
        name: see :attr:`name`
        dtype: see :attr:`dtype`
        shape: see :attr:`shape`
        layer: see :attr:`layer`
    """

    name: str
    """Name of the response."""

    dtype: np.dtype
    """Data type of the response."""

    shape: t.Tuple[t.Optional[int], ...]
    """Shape of the response. The first dimension will generally be `None`."""

    layer: "ResponseInfo.Layer"
    """Details about the layer that generates this response."""

    @t.final
    @dataclass(frozen=True)
    class Layer:
        """
        Class to hold information about the layer that generated the response.

        Args:
            name: see :attr:`name`
            kind: see :attr:`kind`
            typename: see :attr:`typename`
        """

        name: str
        """Name of the layer that generated the response."""

        kind: "ResponseInfo.LayerKind"
        """
        DNIKit-generic type of the layer that generated the response.

        Note:
            DNIKit only understands the most common type of layers. For other types of
            layers, it may be necessary to check the exact name provided by the framework used.

            See :attr:`typename` for more info.
        """

        typename: str
        """
        Framework-dependent name of the type of the layer that generated the response.

        Note:
            This property does not generalize across different deep learning frameworks (e.g.,
            tensorflow or pytorch). Use with care for code that needs to work across
            different frameworks.
        """

    class LayerKind(enum.Enum):
        """DNIKit abstraction to represent several types of common deep learning layers."""
        UNKNOWN = 0

        # Basic layers
        LINEAR = 1000
        DENSE = 1000
        PLACEHOLDER = 1001

        # Convolutional layers
        CONV_1D = 2000
        CONV_2D = 2001
        CONV_3D = 2002
        CONV_TRANSPOSE_1D = 2003
        CONV_TRANSPOSE_2D = 2004
        CONV_TRANSPOSE_3D = 2005

        # Pooling layers
        MAX_POOLING_1D = 3000
        MAX_POOLING_2D = 3001
        MAX_POOLING_3D = 3003
        AVERAGE_POOLING_1D = 3004
        AVERAGE_POOLING_2D = 3005
        AVERAGE_POOLING_3D = 3006

        # Recurrent layers
        RNN = 4000
        LSTM = 4001
        GRU = 4002

        # Normalization layers
        BATCH_NORM = 5000
        BATCH_NORM_2D = 5001
        BATCH_NORM_3D = 5002
        LAYER_NORM = 5003

        # Regularization layers
        DROPOUT = 6000
        DROPOUT_2D = 6001
        DROPOUT_3D = 6002

        # Activation non-linear layers
        SIGMOID = 8000
        TANH = 8001
        RELU = 8002
        LEAKY_RELU = 8003
        PRELU = 8004
        ELU = 8005
        SOFTMAX = 8006
        ATTENTION = 8007
        RELU6 = 8008

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

import tensorflow as tf

from dnikit.base import Model
import dnikit.typing as dt
from ._tf1_model import _Tensorflow1ModelDetails
from ._tf1_file_loaders import (
    _TF1SavedModelLoader,
    _TF1ProtobufLoader,
    _TF1CheckpointLoader,
    _TF1KerasArchAndWeightsLoader,
    _TF1KerasWholeModelLoader,
)
from ._tensorflow_protocols import LoadingChain
from ._tensorflow_file_loaders import _clear_keras_session

_TF1LoadingChain = LoadingChain(
    loading_chain=[
        _TF1SavedModelLoader,
        _TF1ProtobufLoader,
        _TF1CheckpointLoader,
        _TF1KerasArchAndWeightsLoader,
        _TF1KerasWholeModelLoader,
    ]
)


def load_tf_1_model_from_memory(session: tf.compat.v1.Session) -> Model:
    """
    Initialize a TensorFlow :class:`Model` from a model loaded in ``memory``

    Args:
        session: The TensorFlow session which contains the graph to execute.

    Returns:
        A TensorFlow :class:`Model`.
    """
    return Model(_Tensorflow1ModelDetails(session=session))


def load_tf_1_model_from_path(path: dt.PathOrStr) -> Model:
    """
    Initialize a TensorFlow :class:`Model` from a model serialized in ``path``

    Currently accepted serialized model formats:

    - TensorFlow SavedModel
    - TensorFlow checkpoint (pass checkpoint prefix as ``path`` param)
    - TensorFlow protobuf
    - Keras whole models
    - Keras models with separate architecture and weights files

    Args:
        path: Model path (for single model file) or directory that contains all the model files.

    Returns:
        A DNIKit TensorFlow :class:`Model`.
    """
    path = dt.resolve_path_or_str(path)

    # clear tf.keras backend session so a new TF session can be loaded
    _clear_keras_session()

    loader = _TF1LoadingChain.get_loader(path)
    return Model(loader.load(pathname=path))

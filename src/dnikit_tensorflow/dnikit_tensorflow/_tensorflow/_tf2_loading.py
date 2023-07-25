#
# Copyright 2023 Apple Inc.
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
from ._tf2_model import _Tensorflow2ModelDetails
from ._tf2_file_loaders import (
    _TF2SavedKerasModelLoader,
    _TF2KerasArchAndWeightsLoader,
    _TF2KerasWholeModelLoader,
)
from ._tensorflow_protocols import LoadingChain


TF2LoadingChain = LoadingChain(
    loading_chain=[
        _TF2SavedKerasModelLoader,
        _TF2KerasArchAndWeightsLoader,
        _TF2KerasWholeModelLoader,
    ]
)


def load_tf_2_model_from_memory(model: tf.keras.models.Model) -> Model:
    """
    Initialize a TensorFlow :class:`Model` from a model loaded in ``memory``

    Args:
        model: The TensorFlow Keras model

    Returns:
        A TensorFlow :class:`Model`.
    """
    return Model(_Tensorflow2ModelDetails(model=model))


def load_tf_2_model_from_path(path: dt.PathOrStr) -> Model:
    """
    Initialize a TensorFlow :class:`Model` from a model serialized in ``path``

    Currently accepted serialized model formats:

    - TensorFlow Keras SavedModel
    - Keras whole models (h5)
    - Keras models with separate architecture and weights files

    Args:
        path: Model path (for single model file) or directory that contains all the model files.

    Returns:
        A DNIKit TensorFlow :class:`Model`.
    """
    path = dt.resolve_path_or_str(path)

    # clear session before loading new model
    tf.keras.backend.clear_session()

    loader = TF2LoadingChain.get_loader(path)
    return Model(loader.load(pathname=path))

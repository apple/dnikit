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

import typing as t

import tensorflow as tf

from dnikit.base import Model
import dnikit.typing as dt

from ._tensorflow_protocols import running_tf_1
if running_tf_1():
    from ._tf1_loading import load_tf_1_model_from_memory as tf_memory_load
    from ._tf1_loading import load_tf_1_model_from_path as tf_path_load
else:
    # Using type ignores here because the signature of these functions changes (input param)
    from ._tf2_loading import load_tf_2_model_from_memory as tf_memory_load  # type: ignore
    from ._tf2_loading import load_tf_2_model_from_path as tf_path_load  # type: ignore


def load_tf_model_from_memory(*, session: t.Optional[tf.compat.v1.Session] = None,
                              model: t.Optional[tf.keras.models.Model] = None) -> Model:
    """
    Initialize a TensorFlow :class:`Model <dnikit.base.Model>` from a model loaded in ``memory``.
    This function is supported for both TF2 and TF1, but different parameters are required.
    For TF2, only pass parameter ``model``. For TF1, only pass parameter ``session``.

    Args:
        session: Pass only this parameter when running TensorFlow 1. This is the session
            that contains the graph to execute.
        model: Pass only this parameter when running TensorFlow 2. This is the TF Keras model.

    Returns:
        A TensorFlow :class:`Model <dnikit.base.Model>`.
    """
    if running_tf_1():
        error_message = 'For TF1 (currently installed), please pass param `session`'
    else:
        error_message = 'For TF2 (currently installed), please pass param `model`'

    # Raise errors for incorrect
    if session is None and model is None:
        raise ValueError(error_message)
    if session is not None and model is not None:
        raise ValueError(error_message + ' only.')
    if running_tf_1() and session is None:
        raise ValueError(error_message)
    if not running_tf_1() and model is None:
        raise ValueError(error_message)

    # Load TF1 with "session"
    if running_tf_1():
        assert session is not None
        return tf_memory_load(session)

    # else, load TF2 with "model"
    assert model is not None
    return tf_memory_load(model)


def load_tf_model_from_path(path: dt.PathOrStr) -> Model:
    """
    Initialize a TensorFlow :class:`Model <dnikit.base.Model>` from a model serialized in ``path``

    Currently accepted serialized model formats, depending on if TF 1 or TF 2 is running.

    TF2 Supported formats:
        - TensorFlow Keras SavedModel
        - Keras whole models (h5)
        - Keras models with separate architecture and weights files

    TF1 Supported formats:
        - TensorFlow SavedModel
        - TensorFlow checkpoint (pass checkpoint prefix as ``path`` param)
        - TensorFlow protobuf
        - Keras whole models
        - Keras models with separate architecture and weights files

    Note:
        The keras loaders are currently using ``tf.keras`` instead of ``keras`` natively, and so
        issues might appear when trying to load models saved with native ``keras`` (not tf.keras).
        In this case, load the model outside of DNIKit with ``keras`` and pass it to load with
        :func:`load_tf_model_from_memory <dnikit_tensorflow.load_tf_model_from_memory>`.

    Args:
        path: Model path (for single model file) or directory that contains all the model files.

    Returns:
        A DNIKit TensorFlow :class:`Model <dnikit.base.Model>`.
    """
    return tf_path_load(path)

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

import pathlib

import tensorflow as tf
import keras

import dnikit.typing._types as t
from ._tensorflow_protocols import _TFLoader


def _clear_keras_session() -> None:
    """
    Clear out the global tf.keras session (which causes issues if not cleared when trying run a
    second session)

    More information:
        tf.keras is used in the DNIKit loaders to load Keras models into a TensorFlow session and
        graph. Keras is not used directly.

    Warning:
        tf.keras is separate from keras in versions of - be sure to clear the appropriate session!
        Ex: pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/
        - Original keras was not subsumed into tensorflow to ensure compatibility and so that they
          could both organically develop.
        - Keras 2.3.0 is the first release of Keras that brings keras in sync with tf.keras
    """
    tf.keras.backend.clear_session()
    keras.backend.clear_session()


def resolve_directory(pathname: pathlib.Path) -> pathlib.Path:
    # If specified pathname is not a directory, replace with its parent directory
    if not pathname.is_dir():
        return pathname.parent
    return pathname


class _TFKerasArchAndWeightsLoader(_TFLoader, t.Protocol):
    @staticmethod
    def _get_arch_and_file_paths(pathname: pathlib.Path
                                 ) -> t.Tuple[t.Optional[pathlib.Path], t.Optional[pathlib.Path]]:
        keras_weight_file_extensions = ['.hdf', '.h5', '.hdf5', '.he5']
        keras_architecture_file_extensions = ['.json', '.yml', '.hdf5', '.he5']

        architecture_paths = [path for path in pathname.iterdir() if
                              path.suffix in keras_architecture_file_extensions]
        weights_paths = [path for path in pathname.iterdir() if
                         path.suffix in keras_weight_file_extensions]

        arch_path: t.Optional[pathlib.Path] = None
        if len(architecture_paths) == 1:
            arch_path = architecture_paths[0]

        weights_path: t.Optional[pathlib.Path] = None
        if len(weights_paths) == 1:
            weights_path = weights_paths[0]

        return arch_path, weights_path

    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        if not pathname.is_dir():
            return False
        arch_path, weights_path = _TFKerasArchAndWeightsLoader._get_arch_and_file_paths(pathname)

        return arch_path is not None and weights_path is not None


class _TFKerasWholeModelLoader(_TFLoader, t.Protocol):
    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        return pathname.suffix == '.h5'

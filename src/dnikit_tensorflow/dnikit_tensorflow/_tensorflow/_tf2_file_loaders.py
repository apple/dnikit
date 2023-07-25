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

import pathlib

import tensorflow as tf

import dnikit.typing._types as t
from ._tf2_model import _Tensorflow2ModelDetails
from ._tensorflow_file_loaders import (
    _TFKerasArchAndWeightsLoader,
    _TFKerasWholeModelLoader,
    resolve_directory,
)
from ._tensorflow_protocols import _TFLoader


@t.final
class _TF2SavedKerasModelLoader(_TFLoader):
    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        pathname = resolve_directory(pathname)
        return tf.saved_model.contains_saved_model(str(pathname))

    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow2ModelDetails:
        pathname = resolve_directory(pathname)
        model = tf.keras.models.load_model(pathname)

        return _Tensorflow2ModelDetails(
            model=model,
        )


@t.final
class _TF2KerasArchAndWeightsLoader(_TFKerasArchAndWeightsLoader):
    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow2ModelDetails:
        arch_path, weights_path = _TFKerasArchAndWeightsLoader._get_arch_and_file_paths(pathname)
        assert arch_path is not None and weights_path is not None

        model = tf.keras.models.model_from_json(arch_path.read_text())
        model.load_weights(str(weights_path))

        return _Tensorflow2ModelDetails(
            model=model
        )


@t.final
class _TF2KerasWholeModelLoader(_TFKerasWholeModelLoader):
    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow2ModelDetails:
        model = tf.keras.models.load_model(str(pathname), compile=False)

        return _Tensorflow2ModelDetails(model=model)

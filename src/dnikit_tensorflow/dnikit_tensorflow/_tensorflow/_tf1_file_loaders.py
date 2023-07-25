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

import dnikit.typing._types as t
from ._tf1_model import _Tensorflow1ModelDetails
from ._tensorflow_file_loaders import (
    _TFKerasArchAndWeightsLoader,
    _TFKerasWholeModelLoader,
    resolve_directory,
)
from ._tensorflow_protocols import _TFLoader


def _get_default_session_config() -> tf.compat.v1.ConfigProto:
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = False
    return session_config


@t.final
class _TF1SavedModelLoader(_TFLoader):
    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        pathname = resolve_directory(pathname)
        return tf.compat.v1.saved_model.loader.maybe_saved_model_directory(str(pathname))

    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow1ModelDetails:

        pathname = resolve_directory(pathname)

        tf_session = tf.compat.v1.Session(config=_get_default_session_config())
        tf.compat.v1.saved_model.loader.load(
            tf_session,
            [tf.compat.v1.saved_model.tag_constants.SERVING],
            str(pathname)
        )
        return _Tensorflow1ModelDetails(
            session=tf_session,
        )


@t.final
class _TF1ProtobufLoader(_TFLoader):
    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        return pathname.suffix in ['.pb', '.proto']

    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow1ModelDetails:

        tf_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(pathname), 'rb') as f:
            tf_graph_def.ParseFromString(f.read())

        tf.import_graph_def(tf_graph_def, name='')
        tf_session = tf.compat.v1.Session(config=_get_default_session_config())
        tf_session.run(tf.global_variables_initializer())
        return _Tensorflow1ModelDetails(
            session=tf_session,
        )


@t.final
class _TF1CheckpointLoader(_TFLoader):
    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        return pathname.with_suffix('.meta').is_file()

    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow1ModelDetails:

        saver = tf.train.import_meta_graph(str(pathname.with_suffix('.meta')))
        tf_session = tf.compat.v1.Session(config=_get_default_session_config())
        tf_session.run(tf.global_variables_initializer())
        saver.restore(tf_session, str(pathname))
        return _Tensorflow1ModelDetails(
            session=tf_session
        )


@t.final
class _TF1KerasArchAndWeightsLoader(_TFKerasArchAndWeightsLoader):
    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow1ModelDetails:

        arch_path, weights_path = _TFKerasArchAndWeightsLoader._get_arch_and_file_paths(pathname)
        assert arch_path is not None and weights_path is not None

        # Note: The keras loaders are currently using tf.keras instead of Keras
        #    Issues might appear when trying to load models saved with native Keras
        tf_session = tf.compat.v1.Session(config=_get_default_session_config())
        tf.keras.backend.set_session(tf_session)
        tf.reset_default_graph()

        # Load the network in the current session
        _k_network = tf.keras.models.model_from_json(arch_path.read_text())
        _k_network.load_weights(str(weights_path))
        tf.keras.backend.set_learning_phase(0)  # set learning phase to test

        return _Tensorflow1ModelDetails(
            session=tf.keras.backend.get_session()
        )


@t.final
class _TF1KerasWholeModelLoader(_TFKerasWholeModelLoader):
    @staticmethod
    def load(pathname: pathlib.Path) -> _Tensorflow1ModelDetails:
        # Note: The keras loaders are currently using tf.keras instead of keras
        #    Issues might arise when trying to load models saved with native keras
        tf_session = tf.compat.v1.Session(config=_get_default_session_config())
        tf.keras.backend.set_session(tf_session)
        tf.reset_default_graph()
        tf.keras.models.load_model(str(pathname), compile=False)
        tf.keras.backend.set_learning_phase(0)  # set learning phase to test

        return _Tensorflow1ModelDetails(
            session=tf.keras.backend.get_session()
        )

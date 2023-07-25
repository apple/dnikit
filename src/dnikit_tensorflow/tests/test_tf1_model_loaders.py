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

import pytest
import numpy as np
import keras
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import dnikit.typing._types as t
from dnikit.base import Model, pipeline, ResponseInfo
from dnikit.samples import StubImageDataset
from dnikit.processors import FieldRenamer
from dnikit_tensorflow import load_tf_model_from_memory
from dnikit_tensorflow.samples import get_simple_cnn_model
from dnikit_tensorflow._tensorflow._tf1_model import _Tensorflow1ModelDetails
from dnikit_tensorflow._tensorflow._tf1_loading import (
    _TF1LoadingChain,
)

from dnikit_tensorflow._tensorflow._tf1_file_loaders import (
    _TF1SavedModelLoader,
    _TF1ProtobufLoader,
    _TF1CheckpointLoader,
    _TF1KerasArchAndWeightsLoader,
    _TF1KerasWholeModelLoader,
    _get_default_session_config,
)
from dnikit_tensorflow._tensorflow._tensorflow_file_loaders import _TFLoader, _clear_keras_session
from dnikit_tensorflow._tensorflow._tensorflow_protocols import running_tf_1


@pytest.fixture
def keras_model() -> keras.models.Model:
    # Note: This is run for every test. The scope of this can be changed in the future to reduce
    # redundant computation.
    _clear_keras_session()
    simple_cnn_config_override = {
        "num_classes": 10
    }
    return get_simple_cnn_model(simple_cnn_config_override)


def _test_model(model: Model) -> None:
    dataset_size = 4
    requested_response = next(
        info.name
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
    )
    # Doesn't really matter which layer to use, but this is what is expected
    assert requested_response == 'conv0_conv/Conv2D:0'

    input_layer_name = 'input_1:0'
    response_producer = pipeline(
        StubImageDataset(dataset_size=dataset_size, image_width=32, image_height=32),
        FieldRenamer({'images': input_layer_name}),
        model(requested_response)
    )
    # Trigger the pipeline to make sure that the model can run inference, not just load
    _ = list(response_producer(batch_size=1))


def _get_new_session() -> tf.compat.v1.Session:
    sess = tf.compat.v1.Session(config=_get_default_session_config())
    sess.run(tf.global_variables_initializer())
    return sess


def _test_loading(model_path: pathlib.Path) -> t.Type[_TFLoader]:
    loader = _TF1LoadingChain.get_loader(model_path)

    # load saved model using DNIKit TF saved model loader
    loaded_model = loader.load(model_path)

    # Check that what was loaded is correct
    assert loaded_model is not None
    assert isinstance(loaded_model, _Tensorflow1ModelDetails)
    _test_model(Model(loaded_model))

    return loader


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_load_tf_saved_model(tmp_path: pathlib.Path, keras_model: keras.models.Model) -> None:
    # Save model as "saved model" format
    model_path = tmp_path / 'saved_model/'
    sess = tf.keras.backend.get_session()
    sess.run(tf.global_variables_initializer())
    tf.saved_model.simple_save(
        sess,
        str(model_path),
        inputs={'input_image': keras_model.input},
        outputs={out.name: out for out in keras_model.outputs}
    )

    loader = _test_loading(model_path)
    assert loader == _TF1SavedModelLoader


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_load_protobuf_model(tmp_path: pathlib.Path, keras_model: keras.models.Model) -> None:
    # Save model in pb format
    sess = _get_new_session()
    const_graph = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph.as_graph_def(),
        output_node_names=['activation_8/Softmax']
    )
    graph_io.write_graph(
        graph_or_graph_def=const_graph,
        logdir=str(tmp_path),
        name='cifar.pb',
        as_text=False
    )

    model_path = tmp_path / 'cifar.pb'
    loader = _test_loading(model_path)
    assert loader == _TF1ProtobufLoader


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_load_checkpoint_model(tmp_path: pathlib.Path, keras_model: keras.models.Model) -> None:
    # Save model as a checkpoint
    saver = tf.train.Saver()
    sess = _get_new_session()
    saver.save(sess, str(tmp_path))

    loader = _test_loading(tmp_path)
    assert loader == _TF1CheckpointLoader


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_load_keras_arch_and_weights(tmp_path: pathlib.Path,
                                     keras_model: keras.models.Model) -> None:
    # Save model in separate architecture and weights files
    tmp_path.joinpath('model.json').write_text(keras_model.to_json())
    keras_model.save_weights(str(tmp_path/'model.h5'))

    loader = _test_loading(tmp_path)
    assert loader == _TF1KerasArchAndWeightsLoader


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_tf_load_keras_whole_model(tmp_path: pathlib.Path, keras_model: keras.models.Model) -> None:
    # Save model in whole model h5 format
    model_path = tmp_path / 'simple_cnn.h5'
    keras_model.save(str(model_path))

    loader = _test_loading(model_path)
    assert loader == _TF1KerasWholeModelLoader


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_tf_load_from_memory(keras_model: keras.models.Model) -> None:
    sess = _get_new_session()
    loaded_model = load_tf_model_from_memory(session=sess)
    assert loaded_model is not None
    assert isinstance(loaded_model._details, _Tensorflow1ModelDetails)
    _test_model(loaded_model)


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_tf_keras_whole_model_layers(keras_model: keras.models.Model) -> None:
    # Load sample model from memory
    sess = _get_new_session()
    model = load_tf_model_from_memory(session=sess)
    assert isinstance(model._details, _Tensorflow1ModelDetails)

    # Get input tensors for sample model
    input_tensor_names = [
        info.name
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.PLACEHOLDER
        and "input" in info.name
    ]
    assert input_tensor_names is not None

    # Get conv tensors for sample model
    conv_tensor_names = [
        info.name
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
    ]
    assert conv_tensor_names is not None

    for conv_tensor in conv_tensor_names:
        assert conv_tensor.endswith('Conv2D:0')
        assert conv_tensor.startswith('conv')


@pytest.mark.skipif(not running_tf_1(), reason="Skipping TF1 tests because TF2 is running.")
def test_tf_keras_whole_model_response_generation(keras_model: keras.models.Model) -> None:
    # Load sample model from memory
    sess = tf.keras.backend.get_session()
    model = load_tf_model_from_memory(session=sess)
    assert isinstance(model._details, _Tensorflow1ModelDetails)

    with model._details.session as keras_sess:
        input_tensors = []
        conv_tensor_names = []
        b_size = 100
        for op in model._details.session.graph.get_operations():
            for output_index in range(len(op.outputs)):
                output_tensor = op.outputs[output_index]
                if op.type == 'Placeholder' and "input" in output_tensor.name:
                    input_tensors.append(output_tensor)
                if op.type == 'Conv2D':
                    conv_tensor_names.append(output_tensor.name)

        feed_dict = {}
        for input_tensor in input_tensors:
            feed_dict[input_tensor.name] = np.random.randn(
                b_size,
                *[input_tensor.shape[idx].value for idx in
                    range(1, len(input_tensor.shape))]
            )

        keras_sess.run(tf.global_variables_initializer())
        results = keras_sess.run(fetches=conv_tensor_names, feed_dict=feed_dict)

        assert results is not None
        assert len(results) == len(conv_tensor_names)
        for result in results:
            assert result.shape[0] == b_size

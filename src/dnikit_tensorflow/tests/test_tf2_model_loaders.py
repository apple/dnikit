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

import pytest
import numpy as np
import tensorflow as tf

import dnikit.typing._types as t
from dnikit.base import Model, pipeline, ResponseInfo
from dnikit.samples import StubImageDataset
from dnikit.processors import FieldRenamer
from dnikit_tensorflow import load_tf_model_from_memory
from dnikit_tensorflow.samples import get_simple_cnn_model
from dnikit_tensorflow._tensorflow._tf2_model import _Tensorflow2ModelDetails
from dnikit_tensorflow._tensorflow._tf2_loading import (
    TF2LoadingChain
)

from dnikit_tensorflow._tensorflow._tf2_file_loaders import (
    _TF2SavedKerasModelLoader,
    _TF2KerasArchAndWeightsLoader,
    _TF2KerasWholeModelLoader,
)
from dnikit_tensorflow._tensorflow._tensorflow_protocols import _TFLoader, running_tf_1


@pytest.fixture
def keras_model() -> tf.keras.models.Model:
    # Note: This is run for every test. The scope of this can be changed in the future to reduce
    # redundant computation.
    tf.keras.backend.clear_session()
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
    assert requested_response == 'conv0_conv'

    response_producer = pipeline(
        StubImageDataset(dataset_size=dataset_size, image_width=32, image_height=32),
        FieldRenamer({'images': list(model.input_layers.keys())[0]}),
        model(requested_response)
    )
    # Trigger the pipeline to make sure that the model can run inference, not just load
    _ = list(response_producer(batch_size=1))


def _test_loading(model_path: pathlib.Path) -> t.Type[_TFLoader]:
    loader = TF2LoadingChain.get_loader(model_path)

    # load saved model using DNIKit TF saved model loader
    tf.keras.backend.clear_session()
    loaded_model = loader.load(model_path)

    # Check that what was loaded is correct
    assert loaded_model is not None
    assert isinstance(loaded_model, _Tensorflow2ModelDetails)
    _test_model(Model(loaded_model))

    return loader


@pytest.mark.skipif(running_tf_1(), reason="Skipping TF2 tests because TF1 is running.")
def test_load_tf_keras_saved_model(tmp_path: pathlib.Path,
                                   keras_model: tf.keras.models.Model) -> None:
    # Save model as "saved model" format
    model_path = tmp_path / 'saved_model'
    keras_model.save(str(model_path))

    loader = _test_loading(model_path)
    assert loader == _TF2SavedKerasModelLoader


@pytest.mark.skipif(running_tf_1(), reason="Skipping TF2 tests because TF1 is running.")
def test_tf_load_keras_whole_model(tmp_path: pathlib.Path,
                                   keras_model: tf.keras.models.Model) -> None:
    model_path = tmp_path / 'saved_model.h5'
    keras_model.save(str(model_path))

    loader = _test_loading(model_path)
    assert loader == _TF2KerasWholeModelLoader


@pytest.mark.skipif(running_tf_1(), reason="Skipping TF2 tests because TF1 is running.")
def test_load_keras_arch_and_weights(tmp_path: pathlib.Path,
                                     keras_model: tf.keras.models.Model) -> None:
    # Save model in separate architecture and weights files
    tmp_path.joinpath('model.json').write_text(keras_model.to_json())
    keras_model.save_weights(str(tmp_path/'model.h5'))

    loader = _test_loading(tmp_path)
    assert loader == _TF2KerasArchAndWeightsLoader


@pytest.mark.skipif(running_tf_1(), reason="Skipping TF2 tests because TF1 is running.")
def test_tf_load_from_memory(keras_model: tf.keras.models.Model) -> None:
    loaded_model = load_tf_model_from_memory(model=keras_model)
    assert loaded_model is not None
    assert isinstance(loaded_model._details, _Tensorflow2ModelDetails)
    _test_model(loaded_model)


@pytest.mark.skipif(running_tf_1(), reason="Skipping TF2 tests because TF1 is running.")
def test_tf_keras_whole_model_responses(keras_model: tf.keras.models.Model) -> None:
    # Load sample model from memory
    model = load_tf_model_from_memory(model=keras_model)
    assert isinstance(model._details, _Tensorflow2ModelDetails)

    # Get input tensors for sample model
    input_tensors = model.input_layers
    assert input_tensors is not None
    assert len(input_tensors) == 1
    input_name = list(input_tensors.keys())[0]

    # Get conv tensors for sample model
    conv_tensor_names = [
        info.name
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
    ]
    assert conv_tensor_names is not None

    for conv_tensor in conv_tensor_names:
        assert conv_tensor.startswith('conv')

    shape_dim = []
    for idx in range(1, len(input_tensors[input_name].shape)):
        dim = input_tensors[input_name].shape[idx]
        if dim is not None:
            shape_dim.append(int(dim))
        else:
            raise ValueError("cannot have none shape")

    b_size = 16
    inputs = {
        input_name: np.random.randn(
            b_size,
            *shape_dim
        )
    }
    results = model._details.run_inference(
        inputs=inputs,
        outputs=set(conv_tensor_names)
    )

    assert results is not None
    assert len(results) == len(conv_tensor_names)
    for result_name, result in results.items():
        assert result_name in conv_tensor_names
        assert result.shape[0] == b_size

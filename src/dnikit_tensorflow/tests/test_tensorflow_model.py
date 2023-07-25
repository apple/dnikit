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
import tensorflow as tf
import numpy as np

from dnikit.base import pipeline, ResponseInfo, Batch, PipelineStage, Model
from dnikit.base._model import _ModelPipelineStage, _ModelDetails
from dnikit.processors import FieldRenamer
from dnikit.samples import StubImageDataset
from dnikit_tensorflow import load_tf_model_from_path
from dnikit_tensorflow.samples import get_simple_cnn_model
import dnikit.typing as dt
import dnikit.typing._types as t
from dnikit.exceptions import DNIKitException
from dnikit_tensorflow._tensorflow._tensorflow_protocols import running_tf_1


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    simple_cnn_config_override = {
        "num_classes": 10
    }

    tf.keras.backend.clear_session()
    model = get_simple_cnn_model(simple_cnn_config_override)

    base_path = tmp_path_factory.mktemp('model')

    if running_tf_1():
        weights_path = base_path / 'simple_cnn.h5'
        model.save_weights(str(weights_path))

        arch_path = base_path / 'simple_cnn.json'
        arch_path.write_text(model.to_json())
        model_path = base_path
    else:
        model_path = base_path / 'model.h5'
        model.save(model_path)

    return model_path


def test_instantiation(model_path: pathlib.Path) -> None:
    model = load_tf_model_from_path(model_path)
    assert(model is not None)


def _check_tf1_names(conv2d_operations: t.Sequence[ResponseInfo], model: Model) -> None:
    for conv2d_operation in conv2d_operations:
        assert conv2d_operation.name.endswith('Conv2D:0')
        assert conv2d_operation.layer.name.endswith('Conv2D')

        assert conv2d_operation.name.startswith('conv')
        assert conv2d_operation.layer.name.startswith('conv')


def _check_tf2_names(conv2d_operations: t.Sequence[ResponseInfo], model: Model) -> None:
    for conv2d_operation in conv2d_operations:
        # all operations of type Conv2D should have a name like 'conv..../...'
        assert conv2d_operation.name.startswith('conv')


def test_response_meta(model_path: pathlib.Path) -> None:
    model = load_tf_model_from_path(model_path)
    # get all operations of type Conv2D
    conv2d_operations = [
        info
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
    ]

    # there should be at least one conv2d operation
    assert(len(conv2d_operations) > 0)

    if running_tf_1():
        _check_tf1_names(conv2d_operations, model)
    else:
        _check_tf2_names(conv2d_operations, model)


def test_response_generation(model_path: pathlib.Path) -> None:
    model = load_tf_model_from_path(model_path)
    dataset_size = 107
    dataset = StubImageDataset(
        dataset_size=dataset_size, image_width=32, image_height=32)

    requested_responses = [
        info.name
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
    ]

    # Search for input layers within a DNIKit Model
    input_layer_name = list(model.input_layers.keys())[0]

    # Prepare pipeline
    response_producer = pipeline(
        dataset,
        FieldRenamer({'images': input_layer_name}),
        model(requested_responses)
    )

    total_response_data_points = 0

    for response_batch in response_producer(batch_size=10):
        total_response_data_points += response_batch.batch_size

    assert(total_response_data_points == dataset_size)


def test_auto_field_renamer(model_path: pathlib.Path) -> None:

    class StubFieldsInjector(PipelineStage):
        # Adds fields to a batch with stub data

        def __init__(self, *, fields: dt.OneOrMany[str]):
            super().__init__()
            self._fields = dt.resolve_one_or_many(fields, str)

        def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
            def batch_processor(batch: Batch) -> Batch:
                builder = Batch.Builder(base=batch)
                for field_name in self._fields:
                    builder.fields[field_name] = np.random.randint(0, 3, (batch.batch_size,))
                return builder.make_batch()

            return batch_processor

    model = load_tf_model_from_path(model_path)
    dataset_size = 107
    dataset = StubImageDataset(dataset_size=dataset_size, image_width=32, image_height=32)

    requested_responses = [
        info.name
        for info in model.response_infos.values()
        if info.layer.kind is ResponseInfo.LayerKind.CONV_2D
    ]

    # Prepare pipeline without explicit FieldRenamer:
    response_producer = pipeline(
        dataset,  # produces batches with a single field, "images"
        model(requested_responses)  # the model expects an 'input_1:0' key in fields
    )

    # Generate responses
    total_response_data_points = 0
    for response_batch in response_producer(batch_size=10):
        total_response_data_points += response_batch.batch_size

        # Check fields of the response batch are all in requested_responses:
        assert all(key in requested_responses for key in response_batch.fields.keys())

    assert total_response_data_points == dataset_size

    # Try erroneous pipelines: adding an extra field should trigger TypeError
    with pytest.raises(TypeError) as excinfo:
        response_producer = pipeline(
            dataset,
            StubFieldsInjector(fields="extra_feature"),
            model(requested_responses)
        )

        for response_batch in response_producer(batch_size=10):
            break
    if running_tf_1():
        assert "feed_dict" in str(excinfo.value)
    else:
        assert "Invalid input" in str(excinfo.value)

    # Fake a model that expects "2 inputs" and try to trigger error with
    # misnamed batch fields:
    class _ModelDetailsStub(_ModelDetails):
        def get_input_layer_responses(self) -> t.Sequence[ResponseInfo]:
            return [
                ResponseInfo(name="hello", dtype=np.dtype(np.int16), shape=(None, 1),
                             layer=ResponseInfo.Layer(
                                    name="layer1",
                                    kind=ResponseInfo.LayerKind.PLACEHOLDER,
                                    typename="layer1"
                             )),
                ResponseInfo(name="world", dtype=np.dtype(np.int16), shape=(None, 1),
                             layer=ResponseInfo.Layer(
                                 name="layer2",
                                 kind=ResponseInfo.LayerKind.PLACEHOLDER,
                                 typename="layer2"
                             ))
            ]

        def get_response_infos(self) -> t.Iterable[ResponseInfo]:
            return []

        def run_inference(self,
                          inputs: t.Mapping[str, np.ndarray],
                          outputs: t.AbstractSet[str]) -> t.Mapping[str, np.ndarray]:
            return inputs

    model_pipeline_stub = _ModelPipelineStage(_ModelDetailsStub(), set())
    with pytest.raises(DNIKitException) as excinfo2:
        response_producer = pipeline(
            dataset,
            StubFieldsInjector(fields="world"),
            model_pipeline_stub
        )

        for response_batch in response_producer(batch_size=10):
            break
    assert "Field names must match expected input names to perform inference" in str(excinfo2.value)

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

import pytest
import numpy as np

from dnikit.base import Model, Batch, TrainTestSplitProducer
from dnikit.exceptions import DNIKitException
from dnikit_tensorflow import (
    TFModelExamples,
    TFModelWrapper,
    TFDatasetExamples
)
from dnikit_tensorflow._tensorflow._tensorflow_protocols import running_tf_1
from dnikit_tensorflow._tensorflow._tensorflow_file_loaders import _clear_keras_session


def test_load_example_model() -> None:

    if running_tf_1():
        layer_name = "conv_pw_13/Conv2D:0"
    else:
        layer_name = "conv_pw_13"

    _clear_keras_session()
    mobilenet = TFModelExamples.MobileNet()
    assert isinstance(mobilenet, TFModelWrapper)
    assert isinstance(mobilenet.model, Model)
    assert len(mobilenet.model.input_layers) == 1
    assert layer_name in mobilenet.model.response_infos
    assert mobilenet.preprocessing is not None
    assert mobilenet.postprocessing is None


def test_load_example_dataset() -> None:
    cifar10 = TFDatasetExamples.CIFAR10(attach_metadata=True)

    # Assert num of items in x_train, test etc is correct
    (x_train, y_train), (x_test, y_test) = cifar10.split_dataset
    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000, 1)
    assert cifar10._samples.shape == (60000, 32, 32, 3)
    assert cifar10._labels.shape == (60000,)  # np.squeeze should've been run
    assert cifar10._dataset_ids.shape == (60000,)

    # Produce a batch
    for batch in cifar10(batch_size=16):
        assert batch.batch_size == 16
        assert "samples" in batch.fields
        assert batch.fields["samples"].shape == (16, 32, 32, 3)
        assert Batch.StdKeys.IDENTIFIER in batch.metadata
        assert Batch.StdKeys.LABELS in batch.metadata
        assert "label" in batch.metadata[Batch.StdKeys.LABELS]
        assert "dataset" in batch.metadata[Batch.StdKeys.LABELS]
        break

    # Test subset creation
    subset = cifar10.subset(labels=["frog", "bird", "deer"], datasets=["test"], max_samples=300)

    # Assert num of items in x_train, test etc in subset is correct
    (x_train, y_train), (x_test, y_test) = subset.split_dataset
    assert x_train.shape == (0,)
    assert y_train.shape == (0,)
    assert x_test.shape == (3000, 32, 32, 3)
    assert y_test.shape == (3000, 1)
    assert subset._samples.shape == (3000, 32, 32, 3)
    assert subset._labels.shape == (3000,)
    assert subset._dataset_ids.shape == (3000,)
    assert subset.max_samples == 300

    # Produce batches and check that it terminates correctly
    num_runs = 0
    for batch in subset(batch_size=30):
        assert batch.batch_size == 30
        assert "samples" in batch.fields
        assert batch.fields["samples"].shape == (30, 32, 32, 3)
        assert Batch.StdKeys.IDENTIFIER in batch.metadata
        assert Batch.StdKeys.LABELS in batch.metadata
        assert "label" in batch.metadata[Batch.StdKeys.LABELS]
        assert "dataset" in batch.metadata[Batch.StdKeys.LABELS]

        # Check that all data has only label idxs matching frogs, birds and deer:
        assert all((label in (2, 4, 6)) for label in batch.metadata[Batch.StdKeys.LABELS]["label"])

        # Check that all data is from test set
        assert all(d == 1 for d in batch.metadata[Batch.StdKeys.LABELS]["dataset"])
        num_runs += 1

    # Assert that subset terminates w/ num of items equal to "max_samples" (set to 300 earlier)
    assert num_runs == 10  # batch_size of 30 * 10 runs = 300 samples

    # Test shuffle
    assert subset._permutation is None
    subset.shuffle()
    assert isinstance(subset._permutation, np.ndarray)


def test_load_custom_traintest_dataset() -> None:
    # Load custom x_train, y_train, etc split
    producer = TrainTestSplitProducer(
        split_dataset=((np.random.rand(100, 3, 32, 32), np.random.rand(100, 1)),
                       (np.random.rand(16, 3, 32, 32), np.random.rand(16, 1))),
        attach_metadata=True
    )

    # Assert num of items in x_train, test etc is correct
    (x_train, y_train), (x_test, y_test) = producer.split_dataset
    assert x_train.shape == (100, 3, 32, 32)
    assert y_train.shape == (100, 1)
    assert x_test.shape == (16, 3, 32, 32)
    assert y_test.shape == (16, 1)
    assert producer._samples.shape == (116, 3, 32, 32)
    assert producer._labels.shape == (116,)
    assert producer._dataset_ids.shape == (116,)

    # Produce a batch
    for batch in producer(batch_size=16):
        assert batch.batch_size == 16
        assert "samples" in batch.fields
        assert batch.fields["samples"].shape == (16, 3, 32, 32)
        assert Batch.StdKeys.IDENTIFIER in batch.metadata
        assert Batch.StdKeys.LABELS in batch.metadata
        assert "label" in batch.metadata[Batch.StdKeys.LABELS]
        assert "dataset" in batch.metadata[Batch.StdKeys.LABELS]
        break

    with pytest.raises(DNIKitException):
        # Mismatching train feature and label length
        TrainTestSplitProducer(
            split_dataset=((np.random.rand(100, 3), np.random.rand(32)),
                           (np.random.rand(10, 3), np.random.rand(26)))
        )

    with pytest.raises(DNIKitException):
        # Empty train and test sets
        TrainTestSplitProducer(
            split_dataset=((np.empty((0,)), np.empty((0,))),
                           (np.empty((0,)), np.empty((0,))))
        )

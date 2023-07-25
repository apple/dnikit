#
# Copyright 2021 Apple Inc.
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

import numpy as np
import pytest

import dnikit.typing._types as t
from dnikit.base import (
    Batch,
    ImageFormat,
    peek_first_batch,
    pipeline,
    Producer
)
from dnikit.processors import (
    ImageGammaContrastProcessor,
    ImageGaussianBlurProcessor,
    ImageResizer,
    ImageRotationProcessor,
    Transposer
)
from dnikit.samples import StubImageDataset

cv2 = pytest.importorskip("cv2")

_ARRAY_DIMS = (5, 224, 224, 3)


@pytest.mark.parametrize('size, pixel_format, expected_size',
                         [
                             # Images should get resized to the given size
                             # and be in their original format
                             ((224, 224), ImageFormat.CHW, (3, 224, 224)),
                             ((16, 16), ImageFormat.HWC, (16, 16, 3)),
                         ])
def test_image_resizer(size: t.Tuple[int, int],
                       pixel_format: ImageFormat, expected_size: t.Tuple[int, int, int]) -> None:
    # source data
    dataset = StubImageDataset(dataset_size=11, image_width=32, image_height=32)

    # transpose it into the expected format from HWC.  the pixel_format tells how to
    # transpose the fields in the image itself.  shift it by one for the batch dimension
    dims = list(map(lambda x: x + 1, list(pixel_format.value)))
    dims.insert(0, 0)
    transpose = Transposer(dim=dims)

    processor = ImageResizer(pixel_format=pixel_format, size=size)
    producer = pipeline(dataset, transpose, processor)
    for batch in producer(5):
        assert batch.fields['images'].shape[1:] == expected_size


def make_producer(array: np.ndarray) -> Producer:
    def producer(batch_size: int) -> t.Iterable[Batch]:
        yield Batch({"data": array})

    return producer


@pytest.fixture
def image_uint8_array() -> np.ndarray:
    return np.random.randint(0, 256, _ARRAY_DIMS, np.uint8)


@pytest.fixture
def pre_defined_uint8_array() -> np.ndarray:
    return np.array([[[0, 0], [100, 100], [0, 0]],
                     [[100, 100], [100, 100], [100, 100]],
                     [[0, 0], [100, 100], [0, 0]]], dtype=float).reshape((1, 3, 3, 2))


def test_gaussian_blur(image_uint8_array: np.ndarray, pre_defined_uint8_array: np.ndarray) -> None:
    # With 0 sigma, output data must be unperturbed
    producer = pipeline(make_producer(image_uint8_array), ImageGaussianBlurProcessor(sigma=0))
    result = peek_first_batch(producer).fields["data"]
    assert np.array_equal(image_uint8_array, result)

    # Blurring uniform array with random blur value must increase magnitude of input uniformly
    # Sigma limits (standard deviation of gaussian kernel): 0 -> no blur, 3 -> high blur
    input_array = np.ones(_ARRAY_DIMS) * np.random.rand()
    processor = ImageGaussianBlurProcessor(sigma=np.random.uniform(0, 3))
    producer = pipeline(make_producer(input_array), processor)
    result = peek_first_batch(producer).fields["data"]
    assert np.min(result) == np.max(result)

    # Test blurring on pre-defined array
    producer = pipeline(make_producer(pre_defined_uint8_array), ImageGaussianBlurProcessor(sigma=1))
    result = peek_first_batch(producer).fields["data"]
    assert np.allclose(result, np.array([[[[74.275, 74.275], [75.005, 75.005], [74.275, 74.275]],
                                          [[75.005, 75.005], [75.714, 75.714], [75.005, 75.005]],
                                          [[74.275, 74.275], [75.005, 75.005], [74.275, 74.275]]]]))


def test_gamma_contrast(image_uint8_array: np.ndarray, pre_defined_uint8_array: np.ndarray) -> None:
    # With 0 gamma, output data must be unperturbed
    producer = pipeline(make_producer(image_uint8_array), ImageGammaContrastProcessor(gamma=1))
    result = peek_first_batch(producer).fields["data"]
    assert np.array_equal(image_uint8_array, result)

    # Test contrasting on null array
    # Note: Pixels are scaled using formula: 255*((v/255)**gamma)
    # Gamma limits: 1 -> no contrast, 2 -> high contrast
    input_array = np.zeros(_ARRAY_DIMS).astype(np.uint8)
    producer = pipeline(
        make_producer(input_array),
        ImageGammaContrastProcessor(gamma=np.random.uniform(1, 2))
    )
    result = peek_first_batch(producer).fields["data"]
    assert np.array_equal(input_array, result)

    # Test contrasting on pre-defined array
    producer = pipeline(
        make_producer(pre_defined_uint8_array),
        ImageGammaContrastProcessor(gamma=0.5)
    )
    result = peek_first_batch(producer).fields["data"]
    assert np.array_equal(result, 1.59 * pre_defined_uint8_array)


def test_rotation(image_uint8_array: np.ndarray, pre_defined_uint8_array: np.ndarray) -> None:
    # Test rotation on pre-defined array
    data_producer = pipeline(
        make_producer(pre_defined_uint8_array),
        ImageRotationProcessor(angle=30, pixel_format=ImageFormat.HWC)
    )
    result = peek_first_batch(data_producer).fields["data"]
    expected = np.array([[
        [[41.015625, 41.015625], [17.578125, 17.578125], [13.671875, 13.671875]],
        [[61.328125, 61.328125], [94.140625, 94.140625], [55.859375, 55.859375]],
        [[76.171875, 76.171875], [87.109375, 87.109375], [25.390625, 25.390625]]
    ]])
    assert np.allclose(result, expected)

    # Test equivalent rotations in opposite directions
    angles = [45, 90, 180, 275, 360]
    data_producer = make_producer(image_uint8_array)

    for angle in angles:
        # Absolute angle
        producer_pos = pipeline(data_producer,
                                ImageRotationProcessor(angle=angle, pixel_format=ImageFormat.HWC))
        result_pos = peek_first_batch(producer_pos).fields["data"]

        # angle - 360
        producer_neg = pipeline(
            data_producer,
            ImageRotationProcessor(angle=angle - 360, pixel_format=ImageFormat.HWC)
        )
        result_neg = peek_first_batch(producer_neg).fields["data"]

        assert np.array_equal(result_pos, result_neg)

        # Rotation 0 and 360 must return unperturbed output
        if angle == 360:
            assert np.array_equal(image_uint8_array, result_pos)

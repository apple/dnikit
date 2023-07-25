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
import typing as t

import pytest
import numpy as np
try:
    import cv2
except ImportError:
    pass

from dnikit.exceptions import DNIKitException
from dnikit.base import ImageProducer, Batch
from dnikit._availability import _opencv_available

_IMAGE_RES = (120, 160)
_NUM_IMAGES = 8


def _write_image(path: pathlib.Path, image: np.ndarray, depth: str) -> None:
    if depth == "color":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif depth == "alpha":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), image)


def _create_image(depth: str, image_format: str) -> np.ndarray:
    shape: t.Sequence[int]
    if depth == "grayscale":
        shape = _IMAGE_RES
    elif depth == "color":
        shape = _IMAGE_RES + (3, )
    elif depth == "alpha":
        shape = _IMAGE_RES + (4, )
    else:
        assert False, "Unknown image depth"
    # Use a random image to create pngs
    if image_format == "png":
        return np.random.randint(0, 255, shape, dtype=np.uint8)
    # Compression artifacts make it difficult to test jpegs
    elif image_format == "jpeg":
        img = np.zeros(shape, dtype=np.uint8)
        half_height = _IMAGE_RES[0] // 2
        half_width = _IMAGE_RES[1] // 2
        img[:half_height, :half_width, ...] = 1
        img[half_height:, half_width:, ...] = 1
        return img
    else:
        assert False, "Unknown image format"


@pytest.fixture(params=["grayscale", "color", "alpha"])
def depth(request: t.Any) -> str:
    return request.param


@pytest.fixture(params=["jpeg", "png"])
def image_format(request: t.Any) -> str:
    return request.param


@pytest.fixture
def image(depth: str, image_format: str) -> np.ndarray:
    return _create_image(depth, image_format)


@pytest.fixture
def tmp_image_path(image: np.ndarray,
                   depth: str,
                   image_format: str,
                   tmp_path: pathlib.Path) -> t.Generator[pathlib.Path, None, None]:
    if depth == "alpha" and image_format == "jpeg":
        pytest.skip("JPEGs cannot encode image with alpha channels")

    # Populate temporary directory
    for i in range(_NUM_IMAGES):
        ext = image_format if i % 2 else image_format.upper()
        image_path = tmp_path / f"{i}.{ext}"
        _write_image(image_path, image, depth)
    # yield temporary dir
    yield tmp_path


@pytest.fixture(params=[1, 2, 5, 8, 10])
def batch_size(request: t.Any) -> int:
    return request.param


@pytest.mark.skipif(not _opencv_available(), reason="OpenCV is not installed.")
def test_non_existing_folder() -> None:
    invalid_path = pathlib.Path.cwd() / "oiwelfsdkajlasdjflkjlksdjflk"
    assert not invalid_path.exists()

    with pytest.raises(NotADirectoryError):
        ImageProducer(invalid_path)


@pytest.mark.skipif(not _opencv_available(), reason="OpenCV is not installed.")
def test_empty_folder(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DNIKitException):
        ImageProducer(tmp_path)


@pytest.mark.skipif(not _opencv_available(), reason="OpenCV is not installed.")
def test_image_loading(tmp_image_path: pathlib.Path,
                       image_format: str,
                       image: np.ndarray) -> None:
    image_producer = ImageProducer(tmp_image_path)
    # Check discovery found all images
    assert len(image_producer.image_paths) == _NUM_IMAGES

    # Produce batches
    batches = list(image_producer(_NUM_IMAGES))
    # Check a single batch was produced
    assert len(batches) == 1
    # Inspect batch
    batch = batches[0]

    image_paths = batch.metadata[Batch.StdKeys.PATH]

    assert len(batch.fields["images"]) == _NUM_IMAGES
    assert len(batch.fields) == 1
    # Check image loading worked
    for i in range(_NUM_IMAGES):
        new_image = batch.fields["images"][i]
        new_image = np.squeeze(new_image)
        path = image_paths[i]
        assert new_image.shape == image.shape
        assert np.all(new_image.squeeze() == image), "Image failed to load correctly"
        assert f"{i}.{image_format}" in str(path).lower()


@pytest.mark.skipif(not _opencv_available(), reason="OpenCV is not installed.")
def test_batch_generation(tmp_image_path: pathlib.Path, batch_size: int) -> None:
    image_producer = ImageProducer(tmp_image_path)
    # Generate batches
    batches = list(image_producer(batch_size))
    # Check batches look correct
    assert len(batches) == np.ceil(_NUM_IMAGES / batch_size)
    images_loaded = np.sum([batch.batch_size for batch in batches])
    assert images_loaded == _NUM_IMAGES


@pytest.mark.skipif(not _opencv_available(), reason="OpenCV is not installed.")
def test_invalid_batch_size(tmp_image_path: pathlib.Path) -> None:
    image_producer = ImageProducer(tmp_image_path)

    with pytest.raises(ValueError):
        list(image_producer(batch_size=0))

    with pytest.raises(ValueError):
        list(image_producer(batch_size=-1))


@pytest.mark.skipif(not _opencv_available(), reason="OpenCV is not installed.")
def test_mismatched_dimensions(tmp_path: pathlib.Path) -> None:
    # Write images with two different depths
    image_format = "png"
    for depth in ("color", "alpha"):
        image = _create_image(depth, image_format)
        image_path = tmp_path / f"{depth}.{image_format}"
        _write_image(image_path, image, image_format)
    # Create data producer
    image_producer = ImageProducer(tmp_path)
    # Check that it raises with mismatched dimensions
    with pytest.raises(DNIKitException):
        list(image_producer(1))

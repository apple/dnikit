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

import dataclasses
import enum
import pathlib

import numpy as np
try:
    import cv2  # This is an optional dnikit dependency
except ImportError:
    pass

from ._batch._batch import Batch
from ._producer import Producer
from dnikit._availability import _opencv_available
from dnikit._logging import _Logged
from dnikit.exceptions import DNIKitException
import dnikit.typing as dt
import dnikit.typing._types as t

_DEFAULT_EXTS: t.Final[t.AbstractSet[str]] = {"png", "jpeg", "jpg", "tiff", "bmp"}


def _gather_images(directory: pathlib.Path,
                   extensions: t.AbstractSet[str],
                   recursive: bool) -> t.Sequence[pathlib.Path]:
    # Remove any leading periods
    extensions = {ext.lstrip(".") for ext in extensions}
    glob_prefix = "**" if recursive else "*"  # ** -> recursive search

    result = []
    for ext in extensions:
        for ext_variation in (ext.lower(), ext.upper()):
            glob_pattern = f"{glob_prefix}/*.{ext_variation}"
            result += list(directory.glob(glob_pattern))

    return sorted(result)


def _load_image(image_path: pathlib.Path) -> np.ndarray:
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    # Get image properties
    grayscale = len(image.shape) == 2
    color = len(image.shape) == 3 and image.shape[-1] == 3
    color_alpha = len(image.shape) == 3 and image.shape[-1] == 4
    # Transform to follow more common expectations
    if grayscale:
        image = np.expand_dims(image, axis=-1)
    elif color:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    elif color_alpha:
        cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA, dst=image)
    else:
        assert False, f"Unknown image shape: {image.shape}, expected 2–4 channels"
    return image


@t.final
class ImageFormat(enum.Enum):
    """Layout of the pixel data. Value is argument to np.transpose to put it into ``HWC`` format."""
    HWC = (0, 1, 2)
    """Height x Width x Channels"""
    CHW = (2, 1, 0)
    """Channels x Height x Width"""


@t.final
class PixelFormat(enum.Enum):
    """
    An enum that describes the pixel format of the image data.
    """
    BGRA = enum.auto()
    BGR = enum.auto()
    RGBA = enum.auto()
    RGB = enum.auto()
    GRAY = enum.auto()

    @property
    def to_opencv(self) -> t.Optional[int]:
        """Return the `cv2.cvtColor` code to transform to the cv2 write pixel format (e.g. BGRA)."""
        if not _opencv_available():
            raise DNIKitException("OpenCV not available, was dnikit['image'] installed?")
        return {
            PixelFormat.BGRA: None,
            PixelFormat.BGR: None,
            PixelFormat.RGBA: cv2.COLOR_RGBA2BGRA,
            PixelFormat.RGB: cv2.COLOR_RGB2BGR,
            PixelFormat.GRAY: None,
        }[self]

    @property
    def alpha_channel(self) -> t.Optional[int]:
        """Return the channel that contains alpha information, if any."""
        return {
            PixelFormat.BGRA: 3,
            PixelFormat.BGR: None,
            PixelFormat.RGBA: 3,
            PixelFormat.RGB: None,
            PixelFormat.GRAY: None,
        }[self]


@t.final
@dataclasses.dataclass(frozen=True)
class ImageProducer(Producer, _Logged):
    """
    ``ImageProducer`` is a data :class:`Producer` that streams images loaded from the filesystem.

    The images are loaded in ``NHWC`` format with ``C=1`` for grayscale images,
    ``C=3`` (RGB) for color images and ``C=4`` (RGBA) with images with transparency.

    Note:
        OpenCV is used to load images and therefore this class supports every format that library
        supports. Check the `OpenCV docs <https://docs.opencv.org/>`_ for supported formats.

    Warning:
        All images must have the same height, width and number of channels (``HWC``). Otherwise
        batch creation in :func:`__call__` will fail.

    Args:
        directory: root directory where images are located. As with other :class:`Producer`
            images are only loaded to memory when batches are requested.
        extensions: **[keyword arg, optional]** one, many or none extensions to be used to discover
            images when traversing the filesystem. If no extensions are provided default list will
            be used (which includes jpeg, jpg, png, bmp, and tiff).
        recursive: **[keyword arg, optional]** if ``True``, all subdirectories of ``directory``
            will be traversed, otherwise only ``directory`` will be used to discover images
            (defaults to ``True``).
        field: **[keyword arg, optional]** the key under which the images will be stored in the
            resulting data :class:`Batch` (defaults to "images").

    Raises:
        NotADirectoryError: if the ``directory`` is not a directory.
        DNIKitException: if no images are found in the given ``directory``.
        DNIKitException: if OpenCV is not available.
    """

    image_paths: t.Sequence[pathlib.Path]
    """
    Return the sequence of image paths that were found upon initialization.

    These images will be used to create batches by :func:`__call__`.
    """

    field: str
    """
    Name of the batch field where the loaded images will be stored.

    The same name will also be used to store the image paths as metadata keyed
    with :attr:`Batch.StdKeys.PATH`.
    """

    def __init__(self,
                 directory: pathlib.Path, *,
                 extensions: dt.OneManyOrNone[str] = None,
                 recursive: bool = True,
                 field: str = "images"):
        super().__init__()

        if not _opencv_available():
            raise DNIKitException("OpenCV not available, was dnikit['image'] installed?")

        # First check directory does exist
        if not directory.is_dir():
            raise NotADirectoryError(f"Invalid directory: {directory}")

        exts = dt.resolve_one_many_or_none(extensions, str) or _DEFAULT_EXTS
        image_paths = _gather_images(directory.resolve(), exts, recursive)

        if not image_paths:
            raise DNIKitException(
                f"No images with extensions {', '.join(exts)}]"
                f"found in directory {directory}"
            )

        self.logger.debug(
            f"Found {len(image_paths)} images found in {directory}. "
            f"(searched extensions: {', '.join(exts)}"
        )

        # store instance properties
        object.__setattr__(self, "image_paths", image_paths)
        object.__setattr__(self, "field", field)

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        """
        Produce data :class:`Batch` of the images found of the the size requested.
        :attr:`Batch.StdKeys.PATH` and :attr:`Batch.StdKeys.IDENTIFIER` will both be set.

        Args:
            batch_size: size of the batch to be streamed

        Returns:
            a data :class:`Batch` encapsulating the data loaded from file.

        Raises:
            ValueError: if ``batch_size`` is a non-positive number
            DNIKitException: if images do not all have the same dimensions
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size has to be a greater than 0, got {batch_size}")

        num_images = len(self.image_paths)
        expected_shape = None
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            image_paths = list(self.image_paths[start:end])
            batch = None
            for i, image_path in enumerate(image_paths):
                image = _load_image(image_path)
                # Check shape of this image fits expectations
                if expected_shape is None:
                    expected_shape = image.shape
                elif image.shape != expected_shape:
                    raise DNIKitException(
                        f"Invalid shape for image in: {image_path}, got: {image.shape}, "
                        f"expected: {expected_shape}"
                    )

                # Initialize batch data if necessary
                if batch is None:
                    batch_shape = (end-start, ) + expected_shape
                    batch = np.empty(batch_shape, dtype=image.dtype)

                # Store image in batch
                batch[i, ...] = image

            assert batch is not None
            builder = Batch.Builder({self.field: batch})

            builder.metadata[Batch.StdKeys.IDENTIFIER] = image_paths
            builder.metadata[Batch.StdKeys.PATH] = image_paths

            yield builder.make_batch()

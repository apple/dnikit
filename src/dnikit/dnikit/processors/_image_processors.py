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

try:
    import cv2
except ImportError:
    pass

from ._base_processor import Processor
from dnikit._availability import _opencv_available
from dnikit.base import ImageFormat
from dnikit.exceptions import DNIKitException
import dnikit.typing as dt
import dnikit.typing._types as t


def _raise_if_opencv_not_available() -> None:
    if not _opencv_available():
        raise DNIKitException("OpenCV not available, was dnikit['image'] installed?")


@t.final
class ImageResizer(Processor):
    """
    Initialize an ``ImageResizer``.  This uses `OpenCV <https://docs.opencv.org>`_ to
    resize images.  This can convert  responses with the structure ``BxHxWxC``
    (see :class:`ImageFormat <dnikit.base.ImageFormat>` for alternatives) to a
    new ``HxW`` value.  This does not honor aspect ratio -- the new image will be exactly the
    size given. This uses the default `OpenCV <https://docs.opencv.org>`_ interpolation,
    ``INTER_LINEAR``.

    Args:
        pixel_format: **[keyword arg]** the layout of the pixel data, see
            :class:`ImageFormat <dnikit.base.ImageFormat>`
        size: **[keyword arg]** the size to scale to, ``(width, height)``
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>`
            name, or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be
            processed. If ``fields`` param is ``None``, then all
            :attr:`fields <dnikit.base.Batch.fields>` will be resized.

    Raises:
        DNIKitException: if `OpenCV <https://docs.opencv.org>`_ is not installed.
        ValueError: if ``size`` elements (``(width, height)``) are not positive
    """

    def __init__(self, *, pixel_format: ImageFormat, size: t.Tuple[int, int],
                 fields: dt.OneManyOrNone[str] = None) -> None:
        if size[0] <= 0:
            raise ValueError('Width (size[0]) must be positive.')
        if size[1] <= 0:
            raise ValueError('Height (size[1]) must be positive.')

        _raise_if_opencv_not_available()

        def func(data: np.ndarray) -> np.ndarray:
            assert len(data.shape) == 4, "Image data must be in BxHWC or BxCHW"

            resized_data = []

            # iterate the sub-images of the batch
            for i, original_image_data in enumerate(data):
                hwc_image_data = np.transpose(original_image_data, axes=pixel_format.value)
                resized_hwc_image_data = cv2.resize(hwc_image_data, dsize=size)
                resized_data.append(np.transpose(resized_hwc_image_data, axes=pixel_format.value))

            return np.array(resized_data)

        super().__init__(func, fields=fields)


@t.final
class ImageGaussianBlurProcessor(Processor):
    """
    :class:`Processor` that blurs images in a data field from a :class:`Batch <dnikit.base.Batch>`.
    ``BxCHW`` and ``BxHWC`` images accepted with non-normalized values (between 0 and 255).

    Args:
        sigma: **[optional]** blur filter size; recommended values between 0 and 3, but values
            beyond this range are acceptable.
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>` name,
            or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be processed. If
            ``fields`` param is ``None``, then all :attr:`fields <dnikit.base.Batch.fields>`
            will be processed.

    Raises:
        :class:`DNIKitException`: if `OpenCV <https://docs.opencv.org>`_ is not installed.
        ValueError: if ``sigma`` is not positive
    """

    def __init__(self, sigma: float = 0., *, fields: dt.OneManyOrNone[str] = None) -> None:
        _raise_if_opencv_not_available()
        if sigma < 0.:
            raise ValueError(f"Sigma must be a positive value, got {sigma}")

        # Compute gaussian kernel size automatically when sigma > 0.
        # When sigma = 0, minimum kernel size is (1,1).
        k_size = None if sigma else (1, 1)

        def func(data: np.ndarray) -> np.ndarray:
            assert len(data.shape) == 4, "Image data must be in BxHWC or BxCHW"
            assert np.amin(data) >= 0 and np.amax(data) <= 255, (
                "Pixel values must be within 0 and 255"
            )

            result = np.zeros_like(data)

            # Iterate through images in the batch
            for i, original_image_data in enumerate(data):
                # Blur image
                result[i, ...] = cv2.GaussianBlur(
                    original_image_data, sigmaX=sigma, sigmaY=sigma, ksize=k_size
                )

            return np.array(result)

        super().__init__(func, fields=fields)


@t.final
class ImageGammaContrastProcessor(Processor):
    """
    :class:`Processor` that gamma corrects images in a data field from a
    :class:`Batch <dnikit.base.Batch>`. ``BxCHW`` and ``BxHWC`` images accepted with
    non-normalized values (between 0 and 255). Image (I) is contrasting using formula
    ``(I/255)^gamma*255``.

    Args:
        gamma: **[optional]** contrast filter
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>` name,
            or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be processed. If
            ``fields`` param is ``None``, then all :attr:`fields <dnikit.base.Batch.fields>` will be
            processed.

    Raises:
        :class:`DNIKitException`: if `OpenCV <https://docs.opencv.org>`_ is not installed.
    """

    def __init__(self, gamma: float = 1., *, fields: dt.OneManyOrNone[str] = None) -> None:
        _raise_if_opencv_not_available()

        # Create look up table mapping pixel values [0, 255] to corresponding gamma adjusted values
        lut = np.array([
            ((idx / 255.0) ** gamma) * 255
            for idx in np.arange(0, 256)
        ]).astype(np.uint8)

        def func(data: np.ndarray) -> np.ndarray:
            assert len(data.shape) == 4, "Image data must be in BxHWC or BxCHW"
            assert np.amin(data) >= 0 and np.amax(data) <= 255, (
                "Pixel values must be within 0 and 255"
            )

            result = np.zeros_like(data)

            # cv2.LUT accepts only uint8 data
            data = data.astype(np.uint8)

            # Iterate through images in the batch
            for i, original_image_data in enumerate(data):
                # Replace original image values between 0 and 255 with gamma adjusted values in LUT
                result[i, ...] = cv2.LUT(original_image_data, lut)

            return result

        super().__init__(func, fields=fields)


@t.final
class ImageRotationProcessor(Processor):
    """
    :class:`Processor` that performs image rotation along y-axis on data in a data field from a
    :class:`Batch <dnikit.base.Batch>`. ``BxCHW`` and ``BxHWC`` images accepted with non-normalized
    values (between 0 and 255).

    Args:
        angle: **[optional]** angle (in degrees) of image rotation; positive values mean
            counter-clockwise rotation
        pixel_format: **[optional]** the layout of the pixel data, see
            :class:`ImageFormat <dnikit.base.ImageFormat>`
        cval: **[keyword arg, optional]** RGB color value to fill areas outside image; defaults
            to ``(0, 0, 0)`` (black)
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>` name,
            or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be processed.
            If ``fields`` param is ``None``, then all :attr:`fields <dnikit.base.Batch.fields>`
            will be processed.

    Raises:
        DNIKitException: if `OpenCV <https://docs.opencv.org>`_ is not installed.
    """

    def __init__(self,
                 angle: float = 0.,
                 pixel_format: ImageFormat = ImageFormat.HWC, *,
                 cval: t.Tuple[int, int, int] = (0, 0, 0),
                 fields: dt.OneManyOrNone[str] = None) -> None:
        _raise_if_opencv_not_available()

        def func(data: np.ndarray) -> np.ndarray:
            assert len(data.shape) == 4, "Image data must be in BxHWC or BxCHW"
            assert np.amin(data) >= 0 and np.amax(data) <= 255, (
                "Pixel values must be within 0 and 255"
            )

            result = np.zeros_like(data)

            # Iterate through images in the batch
            for i, original_image_data in enumerate(data):
                # Compute center of image
                hwc_image_data = np.transpose(original_image_data, axes=pixel_format.value)
                (img_w, img_h) = tuple(hwc_image_data.shape[1::-1])
                center = (img_w / 2, img_h / 2)

                # Get 2D rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
                rotated_image = cv2.warpAffine(hwc_image_data, rotation_matrix, (img_w, img_h),
                                               flags=cv2.INTER_LINEAR, borderValue=cval)
                result[i, ...] = np.transpose(rotated_image, axes=pixel_format.value)

            return result

        super().__init__(func, fields=fields)

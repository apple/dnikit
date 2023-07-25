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

import numpy as np
import math

from dnikit.base import Batch, Producer
import dnikit.typing._types as t


@t.final
class StubImageDataset(Producer):
    """
    This class generates random images of the requested size and yields DNIBatches according to
    the requested batch size.

    Data returned in the format N, H, W, C.
    """

    def __init__(self, dataset_size: int, image_width: int = 640, image_height: int = 480,
                 channel_count: int = 3) -> None:
        """
        Args:
            dataset_size: Requested number of images in the stub dataset
            image_width: Width of the images
            image_height: Height of the images
            channel_count: Number of channels of the images (typically 1 or 3, but not restrictive)
        """
        assert image_width > 0, "The image width be greater than 0."
        assert image_height > 0, "The image height must be greater than 0."
        assert channel_count > 0, "The number of channels must be greater than 0."
        self.dataset_size = dataset_size
        self.image_width = image_width
        self.image_height = image_height
        self.channel_count = channel_count

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        """
            Generates data batches according to the requested size.

            Args
                batch_size: dimension of the batch.

            Returns
                a :class:Iterable[DataBatch] that contains a DataBatch.
        """

        if batch_size <= 0:
            raise ValueError('Batch size has to be a positive number: {}'.format(batch_size))

        batch_count = math.ceil(self.dataset_size / batch_size)
        last_batch_size = self.dataset_size % batch_size

        for batch_index in range(batch_count):
            if batch_index < batch_count - 1 or last_batch_size == 0:
                effective_batch_size = batch_size
            else:
                effective_batch_size = last_batch_size

            if effective_batch_size > 0:
                batch_map = {
                    "images": np.random.randn(
                        effective_batch_size,
                        self.image_height,
                        self.image_width,
                        self.channel_count
                    )
                }
                yield Batch(batch_map)


@t.final
class StubGatedAdditionDataset(Producer):
    """
    This class creates a stub dataset for the Gated Addition task, which consists of the following
    fields:

    * `x` = An array of length T, where T is a random number between `minimum_sequence_length`
    and `maximum_sequence_length`. The first row contains random numbers between 0 and 1. The second
    is an array of indicators of length T, with all indicators of value 0 but two of them of
    value 1.
    * `target`: The target value, sum(x[indicators==1])
    * `sequence_length`: The sequence length T of each sequence.

    """
    def __init__(self, dataset_size: int, minimum_sequence_length: int = 100,
                 maximum_sequence_length: int = 100) -> None:
        """
        Initializer of the gated addition stub dataset.

        Args:
            dataset_size: The dataset size, ie. number of sequences generated.
            minimum_sequence_length: The minimum sequence length considered.
            maximum_sequence_length: The maximum sequence length considered.
        """
        assert minimum_sequence_length <= maximum_sequence_length
        self.dataset_size = dataset_size
        self.minimum_sequence_length = minimum_sequence_length
        self.maximum_sequence_length = maximum_sequence_length

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        """
        A generator of data batches containing sequences and labels.

        Args:
            batch_size: The requested batch size.

        Yields:
            :class:`Batch`

        """
        batch_count = math.ceil(self.dataset_size / batch_size)
        random_state = np.random.RandomState(751994)

        for batch_index in range(batch_count):
            effective_batch_size = np.minimum(
                batch_size, self.dataset_size - batch_index * batch_size
            )

            if effective_batch_size > 0:
                assert self.minimum_sequence_length >= 2, "The sequence length must be at least two"
                sequence_lengths = random_state.randint(
                    self.minimum_sequence_length,
                    self.maximum_sequence_length + 1,
                    effective_batch_size
                )

                # These are over generated beyond sequence boundaries -
                # but they are expected to get masked anyway
                x1 = random_state.uniform(
                    -1.0, 1.0, (effective_batch_size, self.maximum_sequence_length)
                )

                x2 = np.zeros((effective_batch_size, self.maximum_sequence_length))

                for i in range(effective_batch_size):
                    # Fast path for sequence length equal to minimum
                    # (for the common case that max == min)
                    if sequence_lengths[i] == self.minimum_sequence_length:
                        this_marker_choice = self.minimum_sequence_length
                    else:
                        this_marker_choice = sequence_lengths[i]
                    marker_indices = random_state.choice(
                        this_marker_choice,
                        min(2, self.minimum_sequence_length),
                        replace=False
                    )
                    x2[i, marker_indices] = 1

                # Note: Hochreiter et al.'s variant of the problem introduce some additional rules
                # on marker indices (Not included).
                x = np.stack((x1, x2), axis=2)

                target_shape = (effective_batch_size, 2 if self.minimum_sequence_length >= 2 else 1)
                target_sums = np.sum(
                    x1[x2 == 1].reshape(target_shape),
                    axis=1
                )
                # Scale to [0,1]
                y = 0.5 + target_sums / 4.0

                builder = Batch.Builder({
                    "x": x.astype(np.float32),
                    "target": y.astype(np.float32),
                    "sequence_length": sequence_lengths,
                })

                yield builder.make_batch()

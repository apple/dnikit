#
# Copyright 2022 Apple Inc.
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

import numpy as np

from ._batch._batch import Batch
from ._producer import Producer
from dnikit.exceptions import DNIKitException
import dnikit.typing as dt
import dnikit.typing._types as t


@dataclasses.dataclass
class TrainTestSplitProducer(Producer):
    """
    Produce :class:`Batches <Batch>` from a train/test split of the form:

    ``(x_train, y_train), (x_test, y_test)``

    where variables are numpy arrays, the `x` arrays represent features,
    and the `y` arrays represent labels.

    For instance, for MNIST, features array ``x_train`` might be of shape
    (60000, 28, 28, 1) with corresponding labels array ``y_train`` might be of shape (60000, 1).

    Only one of ``x_train``, ``x_test`` can be empty (size 0 NumPy array).

    .. note::
        This format is the direct output of calling
        :func:`load_data` on a `tf.keras.dataset <https://keras.io/api/datasets/>`_.
        One can initialize a dataset from :mod:`tf.keras.datasets` simply by writing:

        ``TrainTestSplitProducer(tf.keras.datasets.cifar10.load_data())``

    Args:
        split_dataset: see :attr:`split_dataset`
        attach_metadata: **[optional]** see :attr:`attach_metadata`
        max_samples: **[optional]** see :attr:`max_samples`
    """

    split_dataset: dt.TrainTestSplitType
    """
    The underlying dataset, stored in NumPy arrays of the following tuple format:

    ``(x_train, x_test), (y_train, y_test)``

    Note that the left side has to represent the "train" set, and the right has to be "test".
    """

    attach_metadata: bool = True
    """Whether to attach metadata to batches (e.g., labels) or not."""

    max_samples: int = -1
    """Max data samples to pull from. Set to -1 to pull all samples."""

    _samples: np.ndarray = dataclasses.field(init=False)
    _labels: np.ndarray = dataclasses.field(init=False)
    _dataset_ids: np.ndarray = dataclasses.field(init=False)
    _permutation: t.Optional[np.ndarray] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # Verify type of data matches expectation
        if not (isinstance(self.split_dataset, tuple) and
                len(self.split_dataset) == 2 and
                all(isinstance(tup, tuple) and len(tup) == 2 for tup in self.split_dataset) and
                all(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                    for (x, y) in self.split_dataset)):
            raise TypeError(f"Expected tuple of type ((np.ndarray, np.ndarray), "
                            f"(np.ndarray, np.ndarray)) for split_dataset; "
                            f"received {str(self.split_dataset)}.")

        (x_train, y_train), (x_test, y_test) = self.split_dataset

        # Checks for empty data and initializes appropriately
        # This is necessary because np.concatenate complains when given an empty ndarray.
        # NOTE: Because of the error checking in the .subset method,
        #       only one of x_train or x_test can be empty.
        if x_test.size == 0 and x_train.size == 0:
            raise DNIKitException("Only one of x_train or x_test can be empty.")
        elif (x_test.size > 0 and x_train.size > 0 and
              x_train.shape[1:] != x_test.shape[1:]):
            raise DNIKitException(
                "Individual items for x_train and x_test must be of the same shape.")
        elif x_test.shape[0] != y_test.shape[0]:
            raise DNIKitException("x_test and y_test must be of the same length.")
        elif x_train.shape[0] != y_train.shape[0]:
            raise DNIKitException("x_train and y_train must be of the same length.")
        elif x_test.size == 0:
            self._samples = np.squeeze(x_train)
            self._labels = np.squeeze(y_train)
            self._dataset_ids = np.full(len(x_train), 0)
        elif x_train.size == 0:
            self._samples = np.squeeze(x_test)
            self._labels = np.squeeze(y_test)
            self._dataset_ids = np.full(len(x_test), 1)
        else:
            self._samples = np.squeeze(np.concatenate((x_train, x_test)))
            self._labels = np.squeeze(np.concatenate((y_train, y_test)))
            self._dataset_ids = np.concatenate((np.full(len(x_train), 0),
                                                np.full(len(x_test), 1)))
        if self.max_samples < 0 or self.max_samples > len(self._samples):
            # If max_samples is less than 0 or greater than the dataset, sample the whole dataset
            self.max_samples = len(self._samples)

        self._permutation = None

    def shuffle(self) -> None:
        """
        Shuffle the dataset, to produce randomized samples in batches.
        Note: this shuffling will not transfer to subsets.
        """
        self._permutation = np.random.permutation(len(self._samples))

    def subset(self, labels: dt.OneManyOrNone[t.Hashable] = None,
               datasets: dt.OneManyOrNone[str] = None,
               max_samples: t.Optional[int] = None) -> 'TrainTestSplitProducer':
        """
        Filter the data samples by specific labels or datasets, returning
        a new :class:`TrainTestSplitProducer` that only produces the filtered data.

        Args:
            labels: **[optional]** a single label or list of labels in ``y_train, y_test`` to
                include. If ``None``, includes all.
            datasets: **[optional]** a single str or list of dataset names to include
                      (only ``"train"`` and ``"test"`` are acceptable).
                      If None, includes both train and test sets.
            max_samples: **[optional]** how many data samples to include (-1 for all).
                         If not set, will use the existing instance's ``max_samples``.

        Returns:
            a new :class:`TrainTestSplitProducer` of the same class that
            produces only the filtered data
        """

        labels = dt.resolve_one_many_or_none(labels, t.Hashable)  # type: ignore
        datasets = dt.resolve_one_many_or_none(datasets, str)

        # Error catching for empty lists and tuples (which would return an empty Producer)
        if datasets is not None and len(datasets) == 0:
            raise ValueError("'datasets' field is of length 0. Maybe it should be None?")
        if labels is not None and len(labels) == 0:
            raise ValueError("'labels' field is of length 0. Maybe it should be None?")

        (x_train, y_train), (x_test, y_test) = self.split_dataset

        if datasets is not None:
            if "train" not in datasets:
                x_train = np.empty((0,))
                y_train = np.empty((0,))
            if "test" not in datasets:
                x_test = np.empty((0,))
                y_test = np.empty((0,))

        if labels is not None:
            train_filter = [y in labels for y in np.squeeze(y_train)]
            test_filter = [y in labels for y in np.squeeze(y_test)]
            x_train = x_train[train_filter]
            y_train = y_train[train_filter]
            x_test = x_test[test_filter]
            y_test = y_test[test_filter]

        # Return a copy of the same class type as self, with the filtered dataset fields
        return dataclasses.replace(
            self,
            split_dataset=((x_train, y_train), (x_test, y_test)),
            max_samples=(self.max_samples if max_samples is None else max_samples)
        )

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        """
        Produce generic :class:`Batch` es from the loaded data,
        running through training and test sets.

        Args:
            batch_size: the length of batches to produce

        Return:
            yields :class:`Batches <dnikit.base.Batch>` of the split_dataset of size ``batch_size``.
            If ``self.attach_metadata`` is True, attaches metadata in format:

            - :class:`Batch.StdKeys.IDENTIFIER`: A NumPy array of ints representing unique indices
            - :class:`Batch.StdKeys.LABELS`: A dict with:
                - "label": a NumPy array of label features (format specific to each dataset)
                - "dataset": a NumPy array of ints either 0 (for "train") or 1 (for "test")

        """
        for ii in range(0, self.max_samples, batch_size):
            jj = min(ii + batch_size, self.max_samples)

            if self._permutation is None:
                indices = list(range(ii, jj))
            else:
                indices = self._permutation[ii:jj].tolist()

            # Create batch from data already in memory
            builder = Batch.Builder(
                fields={"samples": self._samples[indices, ...]}
            )

            if self.attach_metadata:
                # Use pathname as the identifier for each data sample, excluding base data directory
                builder.metadata[Batch.StdKeys.IDENTIFIER] = indices
                # Add class and dataset labels
                builder.metadata[Batch.StdKeys.LABELS] = {
                    "label": np.take(self._labels, indices),
                    "dataset": np.take(self._dataset_ids, indices)
                }

            yield builder.make_batch()

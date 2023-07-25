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

import tensorflow as tf

from dnikit.base import TrainTestSplitProducer
import dnikit.typing._types as t
import dnikit.typing._dnikit_types as dt


class _KerasDatasetLoader(TrainTestSplitProducer):
    """
    A wrapper for loading a :mod:`tf.keras.dataset` from the return
    value of :func:`load_data`
    """

    def __init__(self,
                 split_dataset: t.Optional[dt.TrainTestSplitType] = None,
                 attach_metadata: bool = True,
                 max_samples: int = -1) -> None:
        # Load the dataset:
        if split_dataset is None:
            split_dataset = self.load_dataset()
        super().__init__(
            split_dataset=split_dataset,
            attach_metadata=attach_metadata,
            max_samples=max_samples
        )

    @staticmethod
    def load_dataset() -> dt.TrainTestSplitType:
        """Overload this with a custom data loader call."""
        raise NotImplementedError


class _KerasDatasetWithStrLabels(_KerasDatasetLoader):
    """
    An abstract class for supporting filtering by string labels.

    The common format for labels are integer values, however in code
    it's convenient to refer to labels with strings e.g. "cow", "dog", etc.
    This class enables users to do that without having to convert the underlying
    label data from integers to strings.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._name_to_label_idx = self.str_to_label_idx()

    def str_to_label_idx(self) -> t.List[str]:
        """Overload this to load a custom map from str labels to int indices."""
        raise NotImplementedError

    def _label_idx_for_name(self, name: str) -> int:
        idx = self._name_to_label_idx.index(name)
        if idx > -1:
            return idx
        else:
            raise LookupError(f"The label '{name}' is not in the dataset.")

    def subset(self, labels: dt.OneManyOrNone[t.Hashable] = None,
               datasets: dt.OneManyOrNone[str] = None,
               max_samples: t.Optional[int] = None) -> 'TrainTestSplitProducer':
        # Overrides subset to allow labels to be a list of class names "automobile", "truck," etc

        def convert_label(lbl: t.Hashable) -> t.Hashable:
            # Cast string labels into ints representing the label idx (leave others untouched)
            if type(lbl) == str:
                return self._label_idx_for_name(lbl)  # type: ignore
            return lbl

        # Cast str labels into label indices
        if isinstance(labels, t.Iterable):
            labels = [convert_label(lbl) for lbl in labels]

        return super().subset(labels, datasets, max_samples)


@t.final
class CIFAR10(_KerasDatasetWithStrLabels):
    """
    The CIFAR10 dataset, loaded from `tf.keras.dataset <https://keras.io/api/datasets/>`_,
    that produces :class:`Batches <dnikit.base.Batch>`.

    Metadata labels for this dataset:
        ``['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']``

    Args:
        split_dataset: **[optional]** It is unlikely this parameter will be overridden. This is the
            dataset as defined by ``(x_train, y_train), (x_test, y_test)``, but by default,
            is set up to load the CIFAR10 dataset from
            `tf.keras.dataset <https://keras.io/api/datasets/>`_
        attach_metadata: **[optional]** attach :attr:`metadata <dnikit.base.Batch.metadata`
            to :class:`Batch <dnikit.base.Batch>` produced by this
            :class:`Producer <dnikit.base.Producer>`, under metadata key
            :class:`Batch.StdKeys.LABELS <dnikit.base.Batch.StdKeys.LABELS>`
        max_samples: **[optional]** number of samples this :class:`Producer <dnikit.base.Producer>`
            should yield (helpful for testing pipelines with a small number of data samples)
    """
    @staticmethod
    def load_dataset() -> dt.TrainTestSplitType:
        return tf.keras.datasets.cifar10.load_data()

    def str_to_label_idx(self) -> t.List[str]:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


@t.final
class CIFAR100(_KerasDatasetWithStrLabels):
    """
    Load CIFAR100 dataset, loaded from `tf.keras.dataset <https://keras.io/api/datasets/>`_,
    that produces :class:`Batches <dnikit.base.Batch>`.

    Metadata labels for this dataset (``fine``):
        ``['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
        'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
        'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
        'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']``

    Metadata labels for this dataset (``coarse``):
        ``['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects',
        'large_carnivores', 'large_man-made_outdoor_things',
        'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']``

    Args:
        split_dataset: **[optional]** It is unlikely this parameter will be overridden. This is the
            dataset as defined by ``(x_train, y_train), (x_test, y_test)``, but by default,
            is set up to load the CIFAR100 dataset from
            `tf.keras.dataset <https://keras.io/api/datasets/>`_
        attach_metadata: **[optional]** attach :attr:`metadata <dnikit.base.Batch.metadata>`
            to :class:`Batch <dnikit.base.Batch>` produced by this
            :class:`Producer <dnikit.base.Producer>`, under metadata key
            :class:`Batch.StdKeys.LABELS <dnikit.base.Batch.StdKeys.LABELS>`
        max_samples: **[optional]** number of samples this :class:`Producer <dnikit.base.Producer>`
            should yield (helpful for testing pipelines with a small number of data samples)
        label_mode: **[optional]** either ``fine`` or ``coarse`` to determine granularity of
            :attr:`metadata <dnikit.base.Batch.metadata>` labels
            (see `tf.keras.dataset documentation <https://keras.io/api/datasets/>`_)
    """

    def __init__(self,
                 split_dataset: t.Optional[dt.TrainTestSplitType] = None,
                 attach_metadata: bool = True,
                 max_samples: int = -1,
                 label_mode: str = "fine"):
        # Load the dataset:
        self.label_mode = label_mode
        if split_dataset is None:
            split_dataset = self.load_dataset(label_mode=label_mode)
        super().__init__(split_dataset=split_dataset,
                         attach_metadata=attach_metadata,
                         max_samples=max_samples)

    @staticmethod
    def load_dataset(label_mode: str = "fine") -> dt.TrainTestSplitType:
        return tf.keras.datasets.cifar100.load_data(label_mode=label_mode)

    def str_to_label_idx(self) -> t.List[str]:
        if self.label_mode == "fine":
            return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
                    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
                    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        elif self.label_mode == "coarse":
            return ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
                    'household_electrical_devices', 'household_furniture', 'insects',
                    'large_carnivores', 'large_man-made_outdoor_things',
                    'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
                    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
                    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']
        raise ValueError("label_mode must be either 'fine' or 'coarse'")


@t.final
class MNIST(_KerasDatasetLoader):
    """
    Load MNIST dataset, loaded from `tf.keras.dataset <https://keras.io/api/datasets/>`_,
    that produces :class:`Batches <dnikit.base.Batch>`.

    Metadata labels for this dataset:
        ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

    Args:
        split_dataset: **[optional]** It is unlikely this parameter will be overridden. This is the
            dataset as defined by ``(x_train, y_train), (x_test, y_test)``, but by default,
            is set up to load the MNIST dataset from
            `tf.keras.dataset <https://keras.io/api/datasets/>`_
        attach_metadata: **[optional]** attach :attr:`metadata <dnikit.base.Batch.metadata>`
            to :class:`Batch <dnikit.base.Batch>` produced by this
            :class:`Producer <dnikit.base.Producer>`, under metadata key
            :class:`Batch.StdKeys.LABELS <dnikit.base.Batch.StdKeys.LABELS>`
        max_samples: **[optional]** number of samples this :class:`Producer <dnikit.base.Producer>`
            should yield (helpful for testing pipelines with a small number of data samples)
    """

    @staticmethod
    def load_dataset() -> dt.TrainTestSplitType:
        return tf.keras.datasets.mnist.load_data()


@t.final
class FashionMNIST(_KerasDatasetWithStrLabels):
    """
    Load FashionMNIST dataset, loaded from `tf.keras.dataset <https://keras.io/api/datasets/>`_,
    that produces :class:`Batches <dnikit.base.Batch>`.

    Metadata labels for this dataset:
        ``['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
        'Sneaker', 'Bag', 'Ankle boot']``

    Args:
        split_dataset: **[optional]** It is unlikely this parameter will be overridden. This is the
            dataset as defined by ``(x_train, y_train), (x_test, y_test)``, but by default,
            is set up to load the FashionMNIST dataset from
            `tf.keras.dataset <https://keras.io/api/datasets/>`_
        attach_metadata: **[optional]** attach :attr:`metadata <dnikit.base.Batch.metadata>`
            to :class:`Batch <dnikit.base.Batch>` produced by this
            :class:`Producer <dnikit.base.Producer>`, under metadata key
            :class:`Batch.StdKeys.LABELS <dnikit.base.Batch.StdKeys.LABELS>`
        max_samples: **[optional]** number of samples this :class:`Producer <dnikit.base.Producer>`
            should yield (helpful for testing pipelines with a small number of data samples)
    """
    @staticmethod
    def load_dataset() -> dt.TrainTestSplitType:
        return tf.keras.datasets.fashion_mnist.load_data()

    def str_to_label_idx(self) -> t.List[str]:
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@t.final
class TFDatasetExamples:
    """
    Example TF Datasets, each bundled as a DNIKit :class:`Producer <dnikit.base.Producer>`.
    Loaded from `tf.keras.dataset <https://keras.io/api/datasets/>`_.
    """

    CIFAR10: t.Final = CIFAR10
    CIFAR100: t.Final = CIFAR100
    MNIST: t.Final = MNIST
    FashionMNIST: t.Final = FashionMNIST

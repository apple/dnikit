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

import dataclasses
import dnikit.typing._types as t

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from dnikit.base import Producer, Batch

# see ProducerTorchDataset class doc and mapping below
PRODUCER_TORCH_MAPPING = t.Union[str, Batch.DictMetaKey, Batch.MetaKey,
                                 t.Callable[[Batch.ElementType], t.Any]]

# see TorchProducer class doc and mapping below
TORCH_PRODUCER_MAPPING = t.Union[str, Batch.DictMetaKey, Batch.MetaKey,
                                 t.Callable[[t.Any, Batch.Builder], None]]


@dataclasses.dataclass(frozen=True)
class ProducerTorchDataset(IterableDataset):
    """
    Adaptor that transforms any :class:`Producer <dnikit.base.Batch>` into a
    :class:`PyTorch IterableDataset <torch.utils.data.IterableDataset>`.
    The Producer can be something simple like a :class:`ImageProducer <dnikit.base.ImageProducer>`
    or a more complex :func:`pipeline <dnikit.base.pipeline>` of stages.

    Instances are given a :attr:`mapping` that describes how to transform the structured
    data in a :class:`Batch.ElementType <dnikit.base.Batch.ElementType>`  (type of single
    :class:`batch.elements <dnikit.base.Batch.elements>`) into an unstructured tuple
    that PyTorch expects from a Dataset. This same mapping can be used to map the positional values
    from PyTorch back into a dnikit Producer via :class:`TorchProducer`.

    This class also supports an optional :attr:`transforms` that works similar
    to the ``transforms`` attr on
    `PyTorch image datasets <https://pytorch.org/vision/stable/datasets.html>`_.

    See Also
        - :class:`TorchProducer` -- converts a PyTorch Dataset/DataLoader into a
          :class:`Producer <dnikit.base.Producer>`

    Args:
        producer: see :attr:`producer`
        mapping: see :attr:`mapping`
        batch_size: **[optional]** see :attr:`batch_size`
        transforms: **[optional]** see :attr:`transforms`
    """

    producer: Producer
    """
    The Producer to represent as a PyTorch Dataset.
    """

    mapping: t.Sequence[PRODUCER_TORCH_MAPPING]
    """
    Describes how to map a :class:`Batch.ElementType <dnikit.base.Batch.ElementType>` to the Dataset
    result. Typically the first value returned from a Dataset is an array-like piece of data, e.g.
    an ``image`` :attr:`field <dnikit.base.Batch.fields>` in a typical
    :class:`Batch <dnikit.base.Batch>`.

    The mapping supports several different types of values:

    - string -- names a `batch.fields` to copy into the output
    - :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>` /
      :class:`MetaKey <dnikit.base.Batch.MetaKey>` -- names a
      :attr:`batch.metadata <dnikit.base.Batch.metadata>` to copy into the output
    - callable -- custom code to produce a custom result

    For example:

    .. code-block:: python

        # consider a Batch.ElementType with data like this:
        im = np.random.randint(255, size=(64, 64), dtype=np.uint8)
        fields = {
            "image": im,
            "image2": im,
        }
        key1 = Batch.DictMetaKey[dict]("KEY1")
        metadata = {
            key1: {"k1": "v1", "k2": "v2"}
        }

        # it's possible to define the mapping like this:

        def transform(element: Batch.ElementType) -> np.ndarray:
            # note: pycharm requires a writable copy of the ndarray
            return element.fields["image"].reshape((128, 32)).copy()

        ds = ProducerTorchDataset(producer, ["image", "image2", key1, transform])

    In this example the Dataset will produce two ndarrays, a dictionary and a reshaped ndarray.
    """

    batch_size: int = 100
    """
    The size of batch to read from the producer.  This is independent of the downstream
    batch size in PyTorch.
    """

    transforms: t.Optional[t.Mapping[str, t.Callable[[torch.Tensor], torch.Tensor]]] = None
    """
    Optional transforms (https://pytorch.org/vision/stable/transforms.html).  This
    is a mapping from field name to a Tensor transform, e.g. image and audio transforms.

    Typical PyTorch Datasets provide a ``transform`` and ``target_transform`` to transform
    the first and second values.  This class requires passing in specific field names for the
    transforms to apply to.

    For example:

    .. code-block:: python

        dataset = ProducerTorchDataset(
            producer, ["image", "mask", "heights"],
            transforms={
                "image": transforms.RandomCrop(32, 32),
                "mask": transforms.Compose([
                     transforms.CenterCrop(10),
                     transforms.ColorJitter(),
                ]),
            })
    """

    def __post_init__(self) -> None:
        # since str is-a Sequence, double check to make sure the caller
        # didn't pass in a bare str by accident
        assert not isinstance(self.mapping, str), (
            "mapping should be a list of strings, callables and DictMetaKeys, not a single str.")

    def __iter__(self) -> t.Iterator:
        transforms = self.transforms or {}

        for batch in self.producer(self.batch_size):
            for element in batch.elements:
                result: t.List[t.Any] = []  # this is a t.Any to allow for custom mappings

                for i, mapping in enumerate(self.mapping):
                    if isinstance(mapping, str):
                        # take a copy of the field data -- torch.Tensor requires that the
                        # data be writable.
                        data = element.fields[mapping].copy()

                        if mapping in transforms:
                            tensor = data if isinstance(data, torch.Tensor) else torch.Tensor(data)
                            transformed = transforms[mapping](tensor)
                            data = transformed.detach().cpu().numpy()

                        result.append(data)

                    elif isinstance(mapping, Batch.MetaKey):
                        # simple sequence
                        data = element.metadata[mapping]
                        result.append(data)

                    elif isinstance(mapping, Batch.DictMetaKey):
                        # metadata is a dictionary of arrays
                        meta_data = element.metadata[mapping]

                        # dictionary with single field -> list
                        if len(meta_data) == 1:
                            single_field_data = meta_data[next(iter(meta_data))]
                            result.append(single_field_data)
                        else:
                            result.append(meta_data)

                    elif callable(mapping):
                        # custom mapping
                        result.append(mapping(element))

                    else:
                        raise ValueError(f'mapping "{mapping}" is an '
                                         f'unhandled type: {type(mapping)}')

                yield tuple(result)


@dataclasses.dataclass(frozen=True)
class TorchProducer(Producer):
    """
    Adaptor that transforms a PyTorch
    `DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ into a DNIKit
    :class:`Producer <dnikit.base.Producer>`. This enables reuse of PyTorch Datasets with
    DNIKit :func:`pipelines <dnikit.base.pipeline>`.

    Instances are given a :attr:`mapping` that describes how to transform the unstructured
    tuple that a PyTorch Dataset produces into a structured DNIKit
    :class:`Batch <dnikit.base.Batch>`.
    This same mapping can be used to match a :class:`Batch <dnikit.base.Batch>` into a
    Dataset in :class:`ProducerTorchDataset`.

    See Also
        - :class:`ProducerTorchDataset` -- :class:`Producer <dnikit.base.Producer>` into a
          PyTorch Dataset

    Args:
        data_loader: see :attr:`data_loader`
        mapping: see :attr:`mapping`
        anonymous_field_name: see :attr:`anonymous_field_name`
    """

    data_loader: DataLoader
    """
    The PyTorch DataLoader to adapt to a :class:`Producer <dnikit.base.Producer>`.
    """

    mapping: t.Sequence[TORCH_PRODUCER_MAPPING]
    """
    This mapping defines how the positional values in a PyTorch Dataset map
    back to a structured :class:`Batch <dnikit.base.Batch>`.  This is essentially the same mapping
    used in :class:`ProducerTorchDataset` -- the same mapping could be used
    to round-trip the data between PyTorch and dnikit.

    The values in the mapping correspond to the positions in the Dataset result
    and convert values as follows:

    - string -- map a Tensor into a :attr:`batch.fields <dnikit.base.Batch.fields>`
      :class:`numpy.ndarray`
    - :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>` /
      :class:`MetaKey <dnikit.base.Batch.MetaKey>` -- map a value into
      :attr:`batch.metadata <dnikit.base.Batch.metadata>`
    - callable -- perform custom conversion and update the
      :class:`Batch.Builder <dnikit.base.Batch.Builder>`
    - None -- discard a value

    For example, given a Dataset that produced data like this:

    .. code-block:: python

        yield ndarray, ndarray, 50, {"k1": "v1", "k2": "v2"}

    it can be mapped into dnikit :attr:`metadata <dnikit.base.Batch.metadata>` like this:

    .. code-block:: python

        key1 = Batch.DictMetaKey[int]("KEY1")
        key2 = Batch.DictMetaKey[t.Mapping[str, str]]("KEY2")
        producer = TorchProducer(loader, ["image", None, key1, key2])

    That will map the first field into ``batch.fields["image"]`` as an
    :class:`numpy.ndarray`.  The second field will be discarded.  The third and fourth fields
    will come across as :attr:`metadata <dnikit.base.Batch.metadata>` like this:

    .. code-block:: python

        element.metadata[key1] == { "_": 50 }
        element.metadata[key2] == {"k1": "v1", "k2": "v2"}

    If the Dataset only produces image data, a single mapping will be sufficient: ``["image"]``

    """

    anonymous_field_name: str = "_"
    """
    The field name to use when mapping non-dictionary metadata to
    :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>`.
    For example, if a PyTorch Dataset produces:

    .. code-block:: python

        yield ndarray, [10, 20, 30]

    it can be mapped into a :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>` like this:

    .. code-block:: python

        key1 = Batch.DictMetaKey[t.List[int]]("KEY1")
        producer = TorchProducer(loader, ["image", key1])

        element = next(iter(producer(1))).elements[0]

        # this is how the metadata is surfaced
        element.metadata[key1] == { "_": [10, 20, 30] }

    Ideally a :class:`MetaKey <dnikit.base.Batch.MetaKey>` is used in these cases.
    """

    def __post_init__(self) -> None:
        # since str is-a Sequence, double check to make sure the caller
        # didn't pass in a bare str by accident
        assert not isinstance(self.mapping, str), ("mapping should be a list of strings, "
                                                   "callables and metadata keys, not a single str.")

    @property
    def batch_size(self) -> int:
        return self.data_loader.batch_size or 100

    def _transform(self, value: t.Any, mapping: TORCH_PRODUCER_MAPPING,
                   batch: Batch.Builder) -> None:
        if isinstance(mapping, str):
            # field data
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif isinstance(value, list):
                value = np.array(value)
            elif isinstance(value, np.ndarray):
                pass
            else:
                raise ValueError(f'field mapping "{mapping}" is an unhandled type: {type(value)}')

            batch.fields[mapping] = value

        elif isinstance(mapping, Batch.MetaKey):
            # simple metadata, use the value directly (unwrapping tensors as needed)

            if isinstance(value, torch.Tensor):
                batch.metadata[mapping] = value.tolist()
            elif isinstance(value, t.Sequence):
                batch.metadata[mapping] = value
            else:
                raise ValueError(f'metadata mapping "{mapping}" '
                                 f'is an unhandled type: {type(value)}')

        elif isinstance(mapping, Batch.DictMetaKey):
            # dictionary metadata
            #
            # if a Dataset returns:
            #
            # <number> -> Tensor([...])
            # ndarray -> Tensor([ndarray, ndarray, ...])
            #
            # <str> -> [...]
            # [a, b, ...] -> [ Tensor([a, ..]), Tensor([b, ...]), ...] or [["a", ...], ["b", ...]]
            # {k1: v1, k2: v2} -> {k1: Tensor[v1, ...], k2: Tensor[v2, ...]}

            if isinstance(value, torch.Tensor):
                # map to an array of values
                batch.metadata[mapping] = {self.anonymous_field_name: value.tolist()}
            elif isinstance(value, t.Sequence):
                value_is_list_tensor_or_tuple = (
                    isinstance(value[0], list) or
                    isinstance(value[0], torch.Tensor) or
                    isinstance(value[0], tuple)
                )
                if value_is_list_tensor_or_tuple:
                    # transpose the lists
                    #     -- lists of [a, b, c, ...] are expected, not [a, a, a, ...]
                    batch.metadata[mapping] = {
                        self.anonymous_field_name: list(map(list, zip(*value)))}
                else:
                    batch.metadata[mapping] = {self.anonymous_field_name: list(value)}

            elif isinstance(value, dict):
                # turn the keys into fields in the metadata -- the dicts at the element
                # level will match what the Dataset returned
                batch.metadata[mapping] = {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                raise ValueError(f'metadata mapping "{mapping}" '
                                 f'is an unhandled type: {type(value)}')

        elif mapping is None:
            # skip the field in the tuple
            pass

        elif callable(mapping):
            mapping(value, batch)

        else:
            raise ValueError(f'cannot handle mapping of type: {type(mapping)}')

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        if batch_size != self.batch_size:
            raise ValueError('The Torch DataLoader used in this instance produces batches '
                             f'of size {self.batch_size}, '
                             f'requested batch size: {batch_size}')

        for data in self.data_loader:
            batch = Batch.Builder()

            if isinstance(data, list):
                # if the Dataset returns multiple values, data will be a list
                for value, mapping in zip(data, self.mapping):
                    self._transform(value, mapping, batch)

            else:
                # a single value -- this is the result of a Dataset that returns a single value
                self._transform(data, self.mapping[0], batch)

            yield batch.make_batch()

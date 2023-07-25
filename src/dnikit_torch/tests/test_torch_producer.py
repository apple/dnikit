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

import numpy as np
import pytest
import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms

import dnikit.typing._types as t
import dnikit_torch
from dnikit.base import Batch, Producer


@dataclasses.dataclass(frozen=True)
class MetaDataset(torch_data.IterableDataset):
    metadata: t.Sequence[t.Any]

    def __iter__(self) -> t.Iterator:
        for _ in range(10):
            yield tuple([np.arange(20).reshape((4, 5)), *self.metadata])


def test_print() -> None:
    # useful to examine how the DataLoader transforms the metadata
    ds = MetaDataset([7])
    loader = torch_data.DataLoader(ds, batch_size=2, shuffle=False)

    print(next(iter(loader)))


def _run_producer(metadata: t.Sequence[t.Any], mapping: t.Sequence[t.Any]) -> Batch:
    ds = MetaDataset(metadata)
    loader = torch_data.DataLoader(ds, batch_size=2, shuffle=False)

    producer = dnikit_torch.TorchProducer(loader, mapping)
    batch = next(iter(producer(2)))
    return batch


def test_producer_field() -> None:
    # convert field data
    batch = _run_producer([], ["image"])
    assert len(batch.fields["image"]) == 2
    assert np.all(batch.fields["image"][0] == np.arange(20).reshape((4, 5)))


def test_producer_single() -> None:
    # single integer metadata
    key1 = Batch.DictMetaKey[int]("KEY1")
    batch = _run_producer([7], ["image", key1])
    assert batch.metadata[key1] == {"_": [7, 7]}


def test_producer_two_metadata() -> None:
    # two integer metadata
    key1 = Batch.DictMetaKey[int]("KEY1")
    key2 = Batch.DictMetaKey[int]("KEY2")
    batch = _run_producer([7, 8], ["image", key1, key2])
    assert batch.metadata[key1] == {"_": [7, 7]}
    assert batch.metadata[key2] == {"_": [8, 8]}


def test_producer_skip_metadata() -> None:
    # single integer metadata
    key2 = Batch.DictMetaKey[int]("KEY2")
    batch = _run_producer([7, 8], ["image", None, key2])
    assert batch.metadata[key2] == {"_": [8, 8]}


def test_producer_string() -> None:
    # single string metadata
    key1 = Batch.DictMetaKey[int]("KEY1")
    batch = _run_producer(["a"], ["image", key1])
    assert batch.metadata[key1] == {"_": ["a", "a"]}


def test_producer_array_int() -> None:
    # array of ints
    key1 = Batch.DictMetaKey[int]("KEY1")
    batch = _run_producer([[7, 8]], ["image", key1])
    assert batch.metadata[key1] == {"_": [[7, 8], [7, 8]]}


def test_producer_array_str() -> None:
    # array of str
    key1 = Batch.DictMetaKey[int]("KEY1")
    batch = _run_producer([["cat", "dog"]], ["image", key1])
    assert batch.metadata[key1] == {"_": [["cat", "dog"], ["cat", "dog"]]}


def test_producer_dict() -> None:
    # dictionary
    key1 = Batch.DictMetaKey[dict]("KEY1")
    d = {"k1": "v1", "k2": "v2"}
    batch = _run_producer([d], ["image", key1])
    assert batch.metadata[key1] == {"k1": ["v1", "v1"], "k2": ["v2", "v2"]}
    assert batch.elements[0].metadata[key1] == d


@dataclasses.dataclass(frozen=True)
class RepeatedProducer(Producer):

    fields: t.Mapping[str, np.ndarray]
    metadata: t.Mapping[Batch.DictMetaKey, t.Mapping[str, t.Any]]
    simple_metadata: t.Optional[t.Mapping[Batch.MetaKey, t.Any]] = None

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        batch = Batch.Builder()

        batch.fields = {
            k: np.array(list(v) * batch_size).reshape(batch_size, *v.shape)
            for k, v in self.fields.items()
        }

        if len(self.metadata) > 0:
            for mdk, mdv in self.metadata.items():
                batch.metadata[mdk] = {
                    k: [v] * batch_size
                    for k, v in mdv.items()
                }
        if self.simple_metadata is not None and len(self.simple_metadata) > 0:
            for k, v in self.simple_metadata.items():
                batch.metadata[k] = [v] * batch_size

        yield batch.make_batch()


def _run_dataset(fields: t.Optional[t.Mapping[str, np.ndarray]],
                 metadata: t.Mapping[Batch.DictMetaKey, t.Mapping[str, t.Any]],
                 mapping: t.Sequence[t.Any]) -> t.Sequence[t.Any]:
    if fields is None:
        fields = {"image": np.arange(10).reshape((5, 2))}
    producer = RepeatedProducer(fields, metadata)
    ds = dnikit_torch.ProducerTorchDataset(producer, mapping)
    loader = torch_data.DataLoader(ds, batch_size=2, shuffle=False)

    v = next(iter(loader))
    return v


def test_dataset_field() -> None:
    im = np.arange(10).reshape((5, 2))
    fields = {"image": im}
    v = _run_dataset(fields, {}, ["image"])

    # v is [Tensor(im, im)]
    assert len(v) == 1
    assert np.all(v[0][0].numpy() == im)


def test_dataset_metadata_int() -> None:
    key1 = Batch.DictMetaKey[int]("KEY1")
    metadata = {
        key1: {"_": 7}
    }

    v = _run_dataset(None, metadata, ["image", key1])
    # v is [Tensor Tensor(7, 7)]
    assert len(v) == 2
    assert v[1][0] == 7


def test_dataset_metadata_str() -> None:
    key1 = Batch.DictMetaKey[str]("KEY1")
    metadata = {
        key1: {"_": "foo"}
    }

    v = _run_dataset(None, metadata, ["image", key1])
    # v is [Tensor ["foo", ...]]
    assert len(v) == 2
    assert v[1][0] == "foo"


def test_dataset_metadata_int_array() -> None:
    key1 = Batch.DictMetaKey[list]("KEY1")
    metadata = {
        key1: {"_": [1, 2, 3]}
    }

    v = _run_dataset(None, metadata, ["image", key1])
    # v is [Tensor [Tensor[1, 1], ...]]
    assert len(v) == 2
    assert torch.all(v[1][0] == torch.Tensor([1, 1]))


def test_dataset_metadata_str_array() -> None:
    key1 = Batch.DictMetaKey[list]("KEY1")
    metadata = {
        key1: {"_": ["a", "b"]}
    }

    v = _run_dataset(None, metadata, ["image", key1])
    # v is [Tensor [(a, a), ...]]
    assert len(v) == 2
    assert v[1] == [("a", "a"), ("b", "b")]


def test_dataset_metadata_dict() -> None:
    key1 = Batch.DictMetaKey[dict]("KEY1")
    metadata = {
        key1: {"k1": "v1", "k2": "v2"}
    }

    v = _run_dataset(None, metadata, ["image", key1])
    # v is [Tensor {"k1": ["v1", "v1"}, ...}]
    assert len(v) == 2
    assert v[1] == {"k1": ["v1", "v1"], "k2": ["v2", "v2"]}


def test_instructions_list() -> None:
    im = np.arange(10).reshape((5, 2))
    fields = {"image": im}

    # this should assert because the mapping isn't a list of mappings (but it *is* a sequence)
    with pytest.raises(Exception):
        _run_dataset(fields, {}, "image")


def test_transforms() -> None:
    im = np.random.randint(255, size=(64, 64), dtype=np.uint8)
    fields = {
        "image": im,
        "image2": im,
    }

    producer = RepeatedProducer(fields, {})
    ds = dnikit_torch.ProducerTorchDataset(
        producer, ["image", "image2"],
        transforms={
            "image": transforms.RandomCrop(32, 32),
            "image2": transforms.RandomCrop(16, 16),
        })
    loader = torch_data.DataLoader(ds, batch_size=2, shuffle=False)
    v = next(iter(loader))

    assert len(v) == 2
    assert v[0].shape == (2, 32, 32)
    assert v[1].shape == (2, 16, 16)


def test_producer_torch_callable() -> None:
    """
    Example of using a callable to do custom transformations.
    """

    im = np.random.randint(255, size=(64, 64), dtype=np.uint8)
    fields = {"image": im}

    def transform(element: Batch.ElementType) -> np.ndarray:
        # note: pycharm requires a writable copy of the ndarray
        return element.fields["image"].reshape((128, 32)).copy()

    v = _run_dataset(fields, {}, ["image", transform])

    assert len(v) == 2

    # original data
    assert v[0].shape == (2, 64, 64)

    # custom transform
    assert v[1].shape == (2, 128, 32)


def test_round_trip_dnikit_pycharm_dnikit() -> None:
    """
    Example of turning a Producer -> Dataset -> Producer
    """
    im = np.random.randint(255, size=(64, 64), dtype=np.uint8)
    fields = {
        "image": im,
        "image2": im,
    }
    key1 = Batch.DictMetaKey[dict]("KEY1")
    key2 = Batch.MetaKey[int]("KEY2")
    metadata = {
        key1: {"k1": "v1", "k2": "v2"},
    }
    simple_metadata = {
        key2: 7,
    }

    # the dnikit producer
    producer = RepeatedProducer(fields, metadata, simple_metadata)

    # convert to a Dataset
    ds = dnikit_torch.ProducerTorchDataset(
        producer, ["image", "image2", key1, key2],
        transforms={
            "image": transforms.RandomCrop(32, 32),
            "image2": transforms.RandomCrop(16, 16),
        })
    loader = torch_data.DataLoader(ds, batch_size=3, shuffle=False)

    # convert back into a producer -- use the same mapping as shown previously
    producer2 = dnikit_torch.TorchProducer(loader, ["image", "image2", key1, key2])

    # read out the first batch
    batch = next(iter(producer2(3)))

    assert len(batch.fields) == 2

    # these were transformed earlier in the ProducerTorchDataset
    assert batch.fields["image"].shape == (3, 32, 32)
    assert batch.fields["image2"].shape == (3, 16, 16)

    assert batch.metadata[key1] == {
        "k1": ["v1"] * 3,
        "k2": ["v2"] * 3,
    }
    assert batch.metadata[key2] == [7, 7, 7]


def test_example() -> None:
    # example use of the API

    # simple dataset where the innards are visible
    class ExampleDataset(torch_data.IterableDataset):  # type: ignore

        def __iter__(self) -> t.Iterator:
            for i in range(10):
                yield (
                    np.random.random((10, 10)),
                    i * i,
                    np.arange(20),
                    "test"
                )

    # create the dataset and loader
    ds = ExampleDataset()
    loader = torch_data.DataLoader(ds, batch_size=5, shuffle=False)

    # metadata keys for DNIKit
    squares = Batch.DictMetaKey[int]("SQUARES")
    names = Batch.MetaKey[str]("NAMES")

    # producer with mapping
    producer = dnikit_torch.TorchProducer(loader, [
        "image",
        squares,
        "vector",
        names,
    ])

    # producer makes batches with:
    # fields = {
    #   image: ndarray
    #   vector: ndarray
    # }
    # metadata = {
    #   squares: t.Mapping[str, t.List[int]]
    #   names: t.List[str]
    # }

    # it's possible to go the other way too
    _ = dnikit_torch.ProducerTorchDataset(
        producer, [
            "image",
            squares,
            "vector",
            names,
        ],
        transforms={
            "image": transforms.RandomCrop(5, 5),
        })

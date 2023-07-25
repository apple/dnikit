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

import typing as t
from dataclasses import dataclass
from functools import partial
import random
import string

import numpy as np
import pytest

from dnikit.base import Batch, Producer, pipeline, peek_first_batch
from dnikit.processors import (
    Processor,
    MeanStdNormalizer,
    Transposer,
    FieldRemover,
    FieldRenamer,
    Flattener,
    SnapshotSaver,
    SnapshotRemover,
    Pooler,
    Concatenator,
    PipelineDebugger,
    Composer,
)
from dnikit.base._producer import _accumulate_batches

RESPONSE_NAMES = ['resp_a', 'resp_b']
NUM_BATCHES = 3
BATCH_SIZE = 5
STR_META_KEY = Batch.DictMetaKey[str]('STR_META_KEY')


def stub_dataset(batch_size: int) -> t.Iterable[Batch]:
    rs = np.random.RandomState(1234)
    batch_shape = (batch_size, 20, 20, 6)
    for _ in range(NUM_BATCHES):
        batch = {r: rs.normal(0, 1, batch_shape) for r in RESPONSE_NAMES}
        yield Batch(batch)


@pytest.mark.parametrize('method, dim, expected_shape, raises',
                         [
                             (Pooler.Method.MAX, (1, 2), (BATCH_SIZE, 6), False),
                             (Pooler.Method.SUM, (1, 2), (BATCH_SIZE, 6), False),
                             (Pooler.Method.AVERAGE, (1, 2), (BATCH_SIZE, 6), False),
                             (Pooler.Method.MAX, 3, (BATCH_SIZE, 20, 20), False),
                             (Pooler.Method.SUM, 3, (BATCH_SIZE, 20, 20), False),
                             (Pooler.Method.AVERAGE, 3, (BATCH_SIZE, 20, 20), False),
                             (Pooler.Method.MAX, (3, 4), (BATCH_SIZE, 20, 20), True),
                             (Pooler.Method.MAX, 4, (BATCH_SIZE, 20, 20), True),
                         ])
def test_pooling_args(method: Pooler.Method,
                      dim: t.Tuple[int, ...],
                      expected_shape: t.Tuple[int, ...],
                      raises: bool) -> None:
    pooler = Pooler(dim=dim, fields=RESPONSE_NAMES, method=method)
    resp_producer = pipeline(stub_dataset, pooler)

    def iterate() -> None:
        num_batches_yielded = 0
        for batch in resp_producer(BATCH_SIZE):
            num_batches_yielded += 1
            for r in RESPONSE_NAMES:
                assert batch.fields[r].shape == expected_shape
        assert num_batches_yielded == NUM_BATCHES

    if raises:
        with pytest.raises(AssertionError):
            iterate()
    else:
        iterate()


@pytest.mark.parametrize('method, dim, pool_fn',
                         [
                             (Pooler.Method.MAX, (1, 2), partial(np.max, axis=(1, 2))),
                             (Pooler.Method.SUM, (1, 2), partial(np.sum, axis=(1, 2))),
                             (Pooler.Method.AVERAGE, (1, 2), partial(np.mean, axis=(1, 2))),
                             (Pooler.Method.MAX, 3, partial(np.max, axis=3)),
                             (Pooler.Method.SUM, 3, partial(np.sum, axis=3)),
                             (Pooler.Method.AVERAGE, 3, partial(np.mean, axis=3)),
                         ])
def test_pooling_values(method: Pooler.Method,
                        dim: t.Tuple[int, ...],
                        pool_fn: t.Callable[[np.ndarray], np.ndarray]) -> None:
    pooler = Pooler(dim=dim, fields=RESPONSE_NAMES, method=method)
    resp_producer = pipeline(stub_dataset, pooler)
    for original_batch, processed_batch in zip(stub_dataset(BATCH_SIZE), resp_producer(BATCH_SIZE)):
        for r in RESPONSE_NAMES:
            np.testing.assert_almost_equal(
                pool_fn(original_batch.fields[r]),
                processed_batch.fields[r],
                decimal=6
            )


def test_partial_fields_pooling() -> None:
    # specify a field and do only that one -- pass the others through
    pooler = Pooler(dim=(1, 2), fields=RESPONSE_NAMES[0], method=Pooler.Method.MAX)
    resp_producer = pipeline(stub_dataset, pooler)
    for batch in resp_producer(BATCH_SIZE):
        assert batch.fields[RESPONSE_NAMES[0]].shape == (BATCH_SIZE, 6)
        assert batch.fields[RESPONSE_NAMES[1]].shape == (BATCH_SIZE, 20, 20, 6)


def test_none_fields_pooling() -> None:
    # no fields specified means do all of them
    pooler = Pooler(dim=(1, 2), method=Pooler.Method.MAX)
    resp_producer = pipeline(stub_dataset, pooler)
    for batch in resp_producer(BATCH_SIZE):
        assert batch.fields[RESPONSE_NAMES[0]].shape == (BATCH_SIZE, 6)
        assert batch.fields[RESPONSE_NAMES[1]].shape == (BATCH_SIZE, 6)


@pytest.mark.parametrize('processor, data, expected',
                         [
                             (MeanStdNormalizer(mean=6, std=2), np.arange(12).reshape(2, 3, 2),
                              # data is [0 - 11], expected is np.divide(data - 6, 2)
                              [[[-3, -2.5], [-2, -1.5], [-1, -0.5]],
                               [[0, 0.5], [1, 1.5], [2, 2.5]]]),

                             (Transposer(dim=[0, 2, 1]), np.arange(12).reshape(2, 3, 2),
                              # reorder the 1 and 2 dimensions -- now it is shape (2, 2, 3)
                              [[[0, 2, 4], [1, 3, 5]],
                               [[6, 8, 10], [7, 9, 11]]]),

                             (Processor(lambda x: x + 2), np.arange(12).reshape(2, 3, 2),
                              # data is [0 - 11], expected is +2 to each element
                              [[[2, 3], [4, 5], [6, 7]],
                               [[8, 9], [10, 11], [12, 13]]]),

                             (Flattener('C'), np.arange(12).reshape(2, 3, 2),
                              # data is [0 - 11], expected output shape is (2, 6)
                              [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),

                             (Flattener('F'), np.arange(12).reshape(2, 3, 2),
                              # data is [0 - 11], expected output shape is (2, 6)
                              [[0, 2, 4, 1, 3, 5], [6, 8, 10, 7, 9, 11]])
                         ])
def test_processors_single_batch(processor: Processor,
                                 data: np.ndarray,
                                 expected: np.ndarray) -> None:
    # test processor with a single batch, verify against expected data

    def source(array: np.ndarray) -> Producer:
        def producer(batch_size: int) -> t.Iterable[Batch]:
            yield Batch({"data": array})
        return producer

    p = pipeline(source(data), processor)
    result = peek_first_batch(p, 1000).fields["data"]
    assert np.allclose(result, expected)


def test_field_remover() -> None:
    processor = FieldRemover(fields=RESPONSE_NAMES[0])
    resp_producer = pipeline(stub_dataset, processor)
    for batch in resp_producer(BATCH_SIZE):
        assert set(batch.fields.keys()) == set([RESPONSE_NAMES[1]])


def test_field_renamer() -> None:
    processor = FieldRenamer(mapping={RESPONSE_NAMES[0]: "cow"})
    resp_producer = pipeline(stub_dataset, processor)
    for batch in resp_producer(BATCH_SIZE):
        assert set(batch.fields.keys()) == set([RESPONSE_NAMES[1], "cow"])


def test_concatenator() -> None:
    def data(batch_size: int) -> t.Iterable[Batch]:
        batch = {
            "a": np.arange(18).reshape(3, 3, 2),
            "b": np.arange(100, 118).reshape(3, 3, 2),
            "c": np.arange(200, 218).reshape(3, 3, 2),
        }
        yield Batch(batch)

    producer = pipeline(data, Concatenator(dim=1, output_field="new", fields=["c", "a"]))

    # get the first batch
    batch = peek_first_batch(producer, 1000)

    # 4 fields in the response
    assert set(batch.fields.keys()) == {"a", "b", "c", "new"}

    # this is what concatenating c and a looks like
    expected = [[[200, 201],
                 [202, 203],
                 [204, 205],
                 [0, 1],
                 [2, 3],
                 [4, 5]],

                [[206, 207],
                 [208, 209],
                 [210, 211],
                 [6, 7],
                 [8, 9],
                 [10, 11]],

                [[212, 213],
                 [214, 215],
                 [216, 217],
                 [12, 13],
                 [14, 15],
                 [16, 17]]]

    assert np.allclose(batch.fields["new"], expected)
    assert batch.fields["new"].shape == (3, 6, 2)


@dataclass(frozen=True)
class _TestMetadata:
    name: str


META_RESPONSE_NAMES = ['resp_a', 'resp_b', 'resp_c']
META_KEY = Batch.DictMetaKey[_TestMetadata]("META_KEY")


def stub_dataset_metadata(batch_size: int) -> t.Iterable[Batch]:
    rs = np.random.RandomState(1234)
    batch_shape = (batch_size, 20, 20, 6)
    for _ in range(1):
        data = {r: rs.normal(0, 1, batch_shape) for r in META_RESPONSE_NAMES}
        builder = Batch.Builder(data)
        builder.metadata[META_KEY] = {
            r: [_TestMetadata(r) for _ in range(batch_size)]
            for r in META_RESPONSE_NAMES
        }

        yield builder.make_batch()


def test_snapshot() -> None:
    # simple snapshot
    producer = pipeline(stub_dataset_metadata, SnapshotSaver())
    batch = peek_first_batch(producer, 5)

    expected_fields = set(META_RESPONSE_NAMES)
    assert set(batch.fields.keys()) == expected_fields

    assert len(batch.snapshots) == 1
    assert batch.snapshots["snapshot"] is not None
    assert set(batch.snapshots["snapshot"].fields.keys()) == expected_fields

    # the metadata holds a string that is the original name of the field
    for i in range(3):
        response_name = META_RESPONSE_NAMES[i]
        assert batch.metadata[META_KEY][response_name][0].name == response_name
        assert (
            batch.snapshots["snapshot"].metadata[META_KEY][response_name][0].name == response_name
        )


def test_snapshot_fields() -> None:
    # snapshot with limited fields
    producer = pipeline(
        stub_dataset_metadata,
        SnapshotSaver(save="snap", fields=[META_RESPONSE_NAMES[1]])
    )
    batch = peek_first_batch(producer, 5)

    expected_fields = set(META_RESPONSE_NAMES)
    expected_fields_snapshot = {META_RESPONSE_NAMES[1]}
    assert set(batch.fields.keys()) == expected_fields

    assert len(batch.snapshots) == 1
    assert batch.snapshots["snap"] is not None
    assert set(batch.snapshots["snap"].fields.keys()) == expected_fields_snapshot

    # the metadata holds a string that is the original name of the field
    for i in range(3):
        response_name = META_RESPONSE_NAMES[i]
        assert batch.metadata[META_KEY][response_name][0].name == response_name

    response_name = META_RESPONSE_NAMES[1]
    assert batch.snapshots["snap"].metadata[META_KEY][response_name][0].name == response_name


def test_snapshot_remove() -> None:
    # snapshot with limited fields
    producer = pipeline(
        stub_dataset_metadata,
        SnapshotSaver(save="snap"),
        SnapshotRemover(snapshots=["snap"])
    )
    batch = peek_first_batch(producer, 5)

    expected_fields = set(META_RESPONSE_NAMES)
    assert set(batch.fields.keys()) == expected_fields

    assert len(batch.snapshots) == 0


def test_rename_fields() -> None:
    # simple rename
    producer = pipeline(
        stub_dataset_metadata,
        FieldRenamer(mapping={META_RESPONSE_NAMES[0]: "new"})
    )
    batch = peek_first_batch(producer)

    expected_fields = {META_RESPONSE_NAMES[1], META_RESPONSE_NAMES[2], "new"}

    # the field got renamed
    assert set(batch.fields.keys()) == expected_fields

    # so did the metadata associated with this field
    assert set(batch.metadata[META_KEY].keys()) == set(META_RESPONSE_NAMES)


def test_remove_field() -> None:
    # remove a single field
    producer = pipeline(stub_dataset_metadata, FieldRemover(fields=META_RESPONSE_NAMES[1]))
    batch = peek_first_batch(producer)

    # only two fields left
    expected_fields = {META_RESPONSE_NAMES[0], META_RESPONSE_NAMES[2]}

    assert set(batch.fields.keys()) == expected_fields
    assert set(batch.metadata[META_KEY].keys()) == set(META_RESPONSE_NAMES)


def test_pipeline_debugger() -> None:
    # Test of PipelineDebugger
    #
    # Normally this would be used inside a pipeline like this:
    producer = pipeline(stub_dataset_metadata, SnapshotSaver(save="snap"), PipelineDebugger())
    batch = peek_first_batch(producer, 5)

    output = PipelineDebugger.dump(batch)
    for i in range(3):
        assert META_RESPONSE_NAMES[i] in output
    assert "Metadata" in output
    assert "Snapshots" in output


def test_composer_none() -> None:
    def f(b: Batch) -> t.Optional[Batch]:
        return None

    producer = pipeline(stub_dataset_metadata, Composer(f))
    batch = peek_first_batch(producer, 5)
    assert batch.batch_size == 0


def test_composer_subset() -> None:
    def f(b: Batch) -> t.Optional[Batch]:
        return b.elements[[0]]

    producer = pipeline(stub_dataset_metadata, Composer(f))
    batch = peek_first_batch(producer, 5)
    assert batch.batch_size == 1


@dataclass
class CategoryProducer(Producer):
    dataset_size: int = 20
    seed: int = 64

    color_options = ['black', 'white', "grey", "lilac"]
    dataset_options = ['train', 'test']

    def __post_init__(self) -> None:
        # To make things easier, assume everything is / 4
        assert self.dataset_size % 4 == 0, "Dataset size must be / 4"
        assert len(self.color_options) % 2 == 0, "color_options size must be even"
        assert len(self.color_options) <= 4, "color_options size must be <= 4"
        assert len(self.dataset_options) % 2 == 0, "dataset_options size must be even"

        np.random.seed(self.seed)
        random.seed(self.seed)

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        assert batch_size % 2 == 0, "batch size must be / 4"

        for start in range(0, self.dataset_size, batch_size):
            n_samples = min(batch_size, self.dataset_size - start)
            assert n_samples % 2 == 0, "n_samples must be / 4"

            builder = Batch.Builder({
                "layer1": np.random.randn(n_samples, 3, 3)
            })
            builder.metadata[STR_META_KEY] = {
                'color': self.color_options * (n_samples//4),
                'dataset': self.dataset_options * (n_samples//2),
                'identifier': [
                    ''.join(random.choices(string.ascii_letters, k=10))
                    for _ in range(n_samples)
                ]
            }
            yield builder.make_batch()


def test_composer_from_element_filter() -> None:
    dataset_size = 20
    producer = CategoryProducer(dataset_size=dataset_size)

    # Element-wise filter for testing (should filter out 50% of data)
    def element_filter(batch_element: Batch.ElementType) -> bool:
        return batch_element.metadata[STR_META_KEY]['dataset'] == 'train'
    composer = Composer.from_element_filter(element_filter)

    accumulated_batch = _accumulate_batches(pipeline(producer, composer))
    assert accumulated_batch.batch_size == dataset_size / 2


def test_composer_from_dict_metadata() -> None:
    dataset_size = 20
    producer = CategoryProducer(dataset_size=dataset_size)

    # Should filter out half the data
    composer = Composer.from_dict_metadata(
        metadata_key=STR_META_KEY,
        label_dimension='dataset',
        label='test'
    )
    accumulated_batch = _accumulate_batches(pipeline(producer, composer))
    assert accumulated_batch.batch_size == dataset_size / 2
    for element in accumulated_batch.elements:
        assert element.metadata[STR_META_KEY]['dataset'] == 'test'

    # Should filter out 3/4 of the data
    composer = Composer.from_dict_metadata(
        metadata_key=STR_META_KEY,
        label_dimension='color',
        label='lilac'
    )
    accumulated_batch = _accumulate_batches(pipeline(producer, composer))
    assert accumulated_batch.batch_size == dataset_size // 4
    for element in accumulated_batch.elements:
        assert element.metadata[STR_META_KEY]['color'] == 'lilac'

    # Does not exist label
    composer = Composer.from_dict_metadata(
        metadata_key=STR_META_KEY,
        label_dimension='color',
        label='square'
    )
    accumulated_batch = _accumulate_batches(pipeline(producer, composer))
    assert accumulated_batch.batch_size == 0

    # Does not exist label dimension
    composer = Composer.from_dict_metadata(
        metadata_key=STR_META_KEY,
        label_dimension='shape',
        label='lilac'
    )
    accumulated_batch = _accumulate_batches(pipeline(producer, composer))
    assert accumulated_batch.batch_size == 0

    # Does not exist metadata key
    composer = Composer.from_dict_metadata(
        metadata_key=Batch.DictMetaKey[str]('Test'),  # does not exist
        label_dimension='color',
        label='lilac'
    )
    accumulated_batch = _accumulate_batches(pipeline(producer, composer))
    assert accumulated_batch.batch_size == 0

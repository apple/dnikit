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

import pytest
import numpy as np

from dnikit.introspectors import DimensionReduction
from dnikit.samples import StubProducer
from dnikit.base import pipeline, Producer


@pytest.fixture
def dataset_size() -> int:
    return 50


@pytest.fixture
def source(dataset_size: int) -> Producer:
    random_s = np.random.RandomState(seed=42)

    response_stub_data = {
        "a": random_s.randn(dataset_size * 5, 20),
        "b": random_s.randn(dataset_size * 5, 40),
        "c": random_s.randn(dataset_size * 5, 60),
    }

    return StubProducer(response_stub_data)


def test_pca(dataset_size: int, source: Producer) -> None:
    pca_components_by_response = {"a": 10, "b": 15, "c": 20}
    pca = DimensionReduction.introspect(
        source,
        strategies={
            name: DimensionReduction.Strategy.PCA(dimensions)
            for name, dimensions in pca_components_by_response.items()
        }
    )

    producer = pipeline(source, pca)

    for batch in producer(batch_size=dataset_size):
        actual_batch_size = batch.batch_size
        assert set(batch.fields.keys()) == {"a", "b", "c"}

        for response_name, reduced_dim in pca_components_by_response.items():
            assert batch.fields[response_name].shape == (actual_batch_size, reduced_dim)


def test_pca_single(dataset_size: int, source: Producer) -> None:
    # specify a single strategy that applies to all the layers
    dim = 10
    pca = DimensionReduction.introspect(source,
                                        strategies=DimensionReduction.Strategy.PCA(dim))
    producer = pipeline(source, pca)

    for batch in producer(batch_size=dataset_size):
        actual_batch_size = batch.batch_size
        assert set(batch.fields.keys()) == {"a", "b", "c"}

        for data in batch.fields.values():
            assert data.shape == (actual_batch_size, dim)


def test_pca_partial(dataset_size: int, source: Producer) -> None:
    # specify some fields
    pca_components_by_response = {"a": 10, "b": 20}
    pca = DimensionReduction.introspect(
        source,
        strategies={
            name: DimensionReduction.Strategy.PCA(dimensions)
            for name, dimensions in pca_components_by_response.items()
        }
    )
    producer = pipeline(source, pca)

    for batch in producer(batch_size=dataset_size):
        actual_batch_size = batch.batch_size

        # still has all the keys
        assert set(batch.fields.keys()) == {"a", "b", "c"}

        # this should be untouched
        assert batch.fields["c"].shape == (actual_batch_size, 60)

        for name, expected in pca_components_by_response.items():
            assert batch.fields[name].shape == (actual_batch_size, expected)


def test_pca_standard(dataset_size: int, source: Producer) -> None:
    # use StandardPCA
    dim = 10
    pca = DimensionReduction.introspect(source,
                                        strategies=DimensionReduction.Strategy.StandardPCA(dim))
    producer = pipeline(source, pca)

    for batch in producer(batch_size=dataset_size):
        actual_batch_size = batch.batch_size
        assert set(batch.fields.keys()) == {"a", "b", "c"}

        for data in batch.fields.values():
            assert data.shape == (actual_batch_size, dim)


def test_tsne(dataset_size: int, source: Producer) -> None:
    # use tSNE (a one-shot reducer)
    dim = 2
    tsne = DimensionReduction.introspect(source,
                                         strategies=DimensionReduction.Strategy.TSNE(dim))
    producer = pipeline(source, tsne)

    for batch in producer(batch_size=dataset_size):
        actual_batch_size = batch.batch_size
        assert set(batch.fields.keys()) == {"a", "b", "c"}

        for data in batch.fields.values():
            assert data.shape == (actual_batch_size, dim)


# umap is slow to load and is not normally installed unless required
@pytest.mark.slow
def test_umap(dataset_size: int, source: Producer) -> None:
    try:
        components_by_response = {"a": 10, "b": 2, "c": 5}
        reducer = DimensionReduction.introspect(
            source,
            strategies={
                name: DimensionReduction.Strategy.UMAP(dimensions)
                for name, dimensions in components_by_response.items()
            })

        producer = pipeline(source, reducer)

        for batch in producer(batch_size=dataset_size):
            actual_batch_size = batch.batch_size
            assert set(batch.fields.keys()) == {"a", "b", "c"}

            for response_name, reduced_dim in components_by_response.items():
                assert batch.fields[response_name].shape == (actual_batch_size, reduced_dim)

    except ImportError:
        pytest.skip("Did not run UMAP test, failure to import")


@pytest.mark.slow
def test_pacmap(dataset_size: int, source: Producer) -> None:
    try:
        dim = 10
        pacmap = DimensionReduction.introspect(
            source,
            strategies=DimensionReduction.Strategy.PaCMAP(dim)
        )
        producer = pipeline(source, pacmap)

        for batch in producer(batch_size=dataset_size):
            actual_batch_size = batch.batch_size
            assert set(batch.fields.keys()) == {"a", "b", "c"}

            for data in batch.fields.values():
                assert data.shape == (actual_batch_size, dim)
    except ImportError:
        pytest.skip("Did not run PacMap test, failure to import")

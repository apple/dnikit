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

import os
import contextlib

from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA as SKPCA
from sklearn.manifold import TSNE as SKTSNE

from ._protocols import DimensionReductionStrategyType
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t


@t.final
@dataclass(frozen=True)
class PCA(DimensionReductionStrategyType):
    """
    Principal Component Analysis based dimension reduction using
    :class:`SKLearn IncrementalPCA <sklearn.decomposition.IncrementalPCA>`.

    Note:
        This does not require reading all of the responses into memory to compute the
        model.  A larger batch size will improve the quality of the fit at the cost of additional
        memory.  The incremental approach produces an approximation of PCA, but is documented
        to be very close and testing backs this up.

    :class:`DimensionReduction.Strategy.StandardPCA
    <dnikit.introspectors.DimensionReduction.Strategy.StandardPCA>` can be used if exact
    computation of PCA is necessary.

    Args:
        target_dimensions: **[optional]** Target dimensionality of the data.
    """

    target_dimensions: int = 2
    """Target dimensionality of the data."""

    _pca: IncrementalPCA = field(init=False)

    def __post_init__(self) -> None:
        # This allows setting an attribute within a frozen dataclass
        object.__setattr__(self, '_pca', IncrementalPCA(n_components=self.target_dimensions))

    def default_batch_size(self) -> int:
        # a batch size of at least the target dimension is required but
        # the results get better if it is larger
        return max(self.target_dimensions * 5, 500)

    def check_batch_size(self, batch_size: int) -> None:
        if self.target_dimensions > batch_size:
            raise DNIKitException(
                f'DimensionReduction.Strategy.PCA (IncrementalPCA) requires that the'
                f'batch_size ({batch_size}) must be larger or equal to the '
                f'target_dimensions ({self.target_dimensions}).')

    def fit_incremental(self, data: np.ndarray) -> None:
        # only fit if the data is >= the target dimensions, else there isn't
        # enough data to fit (this might happen in the last batch of the producer)
        if len(data) >= self.target_dimensions:
            self._pca.partial_fit(data)

    def fit_complete(self) -> None:
        pass

    @property
    def is_one_shot(self) -> bool:
        return False

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self._pca.transform(data)

    def transform_one_shot(self) -> np.ndarray:
        raise DNIKitException("transform_one_shot() not implemented, call transform()")

    def _clone(self) -> DimensionReductionStrategyType:
        return replace(self)


@dataclass(frozen=True)
class _Accumulator:
    """
    Abstract class to provide accumulation of data for DimensionReductionStrategyType.
    """
    _accumulate: t.List[np.ndarray] = field(default_factory=list, init=False)

    def default_batch_size(self) -> int:
        # the batch size doesn't matter -- use the default from accumulate_batches()
        return 1024

    def check_batch_size(self, batch_size: int) -> None:
        # no requirements
        pass

    def fit_incremental(self, data: np.ndarray) -> None:
        self._accumulate.append(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        raise DNIKitException("transform() not implemented, call transform_one_shot()")

    def transform_one_shot(self) -> np.ndarray:
        raise DNIKitException("transform_one_shot() not implemented, call transform()")


@t.final
@dataclass(frozen=True)
class StandardPCA(_Accumulator, DimensionReductionStrategyType):
    """
    Principal Component Analysis based dimension reduction using
    :class:`SKLearn PCA <sklearn.decomposition.PCA>`.

    This dimension reduction strategy requires reading all of the data into
    memory before producing the projection.

    :class:`DimensionReduction.Strategy.PCA
    <dnikit.introspectors.DimensionReduction.Strategy.PCA>` is preferred for its lower memory use.

    Args:
        target_dimensions: **[optional]** Target dimensionality of the data.
    """

    target_dimensions: int = 2
    """Target dimensionality of the data."""

    _pca: SKPCA = field(init=False)

    def __post_init__(self) -> None:
        # This allows setting an attribute within a frozen dataclass
        object.__setattr__(self, '_pca', SKPCA(n_components=self.target_dimensions))

    def fit_complete(self) -> None:
        all_data = np.concatenate(self._accumulate)
        self._accumulate.clear()
        self._pca.fit(all_data)

    @property
    def is_one_shot(self) -> bool:
        return False

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self._pca.transform(data)

    def _clone(self) -> DimensionReductionStrategyType:
        return replace(self)


@t.final
@dataclass(frozen=True)
class TSNE(_Accumulator, DimensionReductionStrategyType):
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE) using
    :class:`SKLearn t-SNE <sklearn.manifold.TSNE>`.

    This dimension reduction strategy requires reading all of the data into
    memory before producing the projection.  Typically the input data
    should be reduced from high dimension to low, e.g. 1024 -> 40, before
    applying t-SNE.

    Args:
        target_dimensions: **[optional]** Target dimensionality of the data.
        kwargs: **[optional]** Any additional :class:`SKLearn t-SNE <sklearn.manifold.TSNE>` args.
    """

    target_dimensions: int = 2
    """Target dimensionality of the data."""

    _parameters: t.Optional[t.Mapping[str, t.Any]] = None
    """Optional parameters for the TSNE algorithm -- pass as kwargs"""

    _tsne: SKTSNE = field(init=False)

    def __init__(self, target_dimensions: int = 2, *,
                 _parameters: t.Optional[t.Mapping[str, t.Any]] = None, **kwargs: t.Any) -> None:
        super().__init__()
        object.__setattr__(self, 'target_dimensions', target_dimensions)
        object.__setattr__(self, '_parameters', _parameters or kwargs)
        object.__setattr__(self, '_tsne',
                           SKTSNE(n_components=self.target_dimensions, **self._parameters))

    def fit_complete(self) -> None:
        all_data = np.concatenate(self._accumulate)
        self._accumulate.clear()
        self._tsne.fit(all_data)

    @property
    def is_one_shot(self) -> bool:
        return True

    def transform_one_shot(self) -> np.ndarray:
        return self._tsne.embedding_

    def _clone(self) -> DimensionReductionStrategyType:
        return replace(self)


@t.final
@dataclass(frozen=True)
class PaCMAP(_Accumulator, DimensionReductionStrategyType):
    """
    PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction
    method built with `PaCMAP <https://github.com/YingfanWang/PaCMAP>`_.
    PaCMAP can be used for visualization, preserving both local and global
    structure of the data in original space.

    This dimension reduction strategy requires reading all of the data into
    memory before producing the projection.  Typically the input data
    should be reduced from high dimension to low, e.g. 1024 -> 40, before
    applying PaCMAP.

    Args:
        target_dimensions: **[optional]** Target dimensionality of the data.
        kwargs: **[optional]** Any additional `PaCMAP <https://github.com/YingfanWang/PaCMAP>`_
            keyword args
    """

    target_dimensions: int = 2
    """Target dimensionality of the data."""

    _parameters: t.Optional[t.Mapping[str, t.Any]] = None
    """Optional parameters for the PaCMAP algorithm -- pass as kwargs"""

    _pacmap: t.Any = field(init=False)

    def __init__(self, target_dimensions: int = 2, *,
                 _parameters: t.Optional[t.Mapping[str, t.Any]] = None, **kwargs: t.Any) -> None:
        # This was moved here so that it is not imported when dnikit is imported, but rather
        #    only if/when PaCMAP is actually used. Caution due to numba SIGSEGV on task cleanup.
        try:
            from pacmap import PaCMAP as PaCMAPAlg
        except ImportError:
            raise ImportError(
                "pacmap not available, was dnikit['dimreduction'] or pacmap installed?")

        super().__init__()
        object.__setattr__(self, 'target_dimensions', target_dimensions)
        object.__setattr__(self, '_parameters', _parameters or kwargs)
        object.__setattr__(self, '_pacmap',
                           # Note: first argument is named `n_dims` for pacmap <= 1.6.0
                           #   and `n_components` for pacmap >= 1.6.1, so leave it un-named here
                           PaCMAPAlg(self.target_dimensions, **self._parameters))

    def fit_complete(self) -> None:
        all_data = np.concatenate(self._accumulate)
        self._accumulate.clear()
        self._pacmap.fit(all_data)

    @property
    def is_one_shot(self) -> bool:
        return True

    def transform_one_shot(self) -> np.ndarray:
        return self._pacmap.embedding_

    def _clone(self) -> DimensionReductionStrategyType:
        return replace(self)


@t.final
@dataclass(frozen=True)
class UMAP(_Accumulator, DimensionReductionStrategyType):
    """
    UMAP based dimension reduction using umap-learn (https://umap-learn.readthedocs.io).

    This dimension reduction strategy requires reading all of the data into
    memory before producing the projection.  Typically the input data
    should be reduced from high dimension to low, e.g. 1024 -> 40, before
    applying UMAP.

    Args:
        target_dimensions: **[optional]** Target dimensionality of the data.
        kwargs: **[optional]** Any additional `umap-learn <https://umap-learn.readthedocs.io>`_
            args.

    Raises:
        DNIKitException: if a layer's response shape does not have exactly 2 dimensions.
    """
    target_dimensions: int = 2
    """
    The dimension of the space to embed into. This defaults to 2 to provide straightforward
    visualization, but can reasonably be set to any integer value in the range 2 to 100.
    (from https://umap-learn.readthedocs.io)
    """

    _parameters: t.Optional[t.Mapping[str, t.Any]] = None
    """Optional parameters for the UMAP algorithm -- pass as kwargs"""

    _umap: t.Any = field(init=False)

    def __init__(self, target_dimensions: int = 2, *,
                 _parameters: t.Optional[t.Mapping[str, t.Any]] = None, **kwargs: t.Any) -> None:
        # This was moved here so that it is not imported when dnikit is imported, but rather
        #    only if/when UMAP is actually used. Caution due to numba SIGSEGV on task cleanup.
        try:
            from umap import UMAP as ULearnUMAP
        except ImportError:
            raise ImportError(
                "UMAP not available, was dnikit['dimreduction'] or umap-learn installed?")

        super().__init__()
        object.__setattr__(self, 'target_dimensions', target_dimensions)
        object.__setattr__(self, '_parameters', _parameters or kwargs)
        object.__setattr__(self, '_umap',
                           ULearnUMAP(
                               n_components=self.target_dimensions,
                               verbose=False, **self._parameters))

    def fit_complete(self) -> None:
        all_data = np.concatenate(self._accumulate)
        self._accumulate.clear()
        self._umap.fit(all_data)

    @property
    def is_one_shot(self) -> bool:
        return False

    def transform(self, data: np.ndarray) -> np.ndarray:
        # File umap.umap_.py has a print statement `print("inside function\n", graph)`
        # that clutters DNIKit stdout. Hiding it.
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return self._umap.transform(data)

    def _clone(self) -> DimensionReductionStrategyType:
        return replace(self)

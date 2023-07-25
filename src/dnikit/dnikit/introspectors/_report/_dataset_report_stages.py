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

from dataclasses import dataclass, field
from functools import partial
import logging

try:
    import pandas as pd
except ImportError:
    class PandasStub:
        DataFrame = None
    pd = PandasStub()

from dnikit.base import (
    multi_introspect,
    Producer,
)
from dnikit.introspectors._dim_reduction._dimension_reduction import (
    DimensionReduction,
    OneOrManyDimStrategies
)
from dnikit.introspectors._duplicates import (
    Duplicates,
    DuplicatesThresholdStrategyType,
)
from dnikit.introspectors._familiarity._familiarity import Familiarity
from dnikit.introspectors._familiarity._protocols import FamiliarityStrategyType
import dnikit.typing._types as t
from dnikit._availability import _umap_available

from ._report_builder_introspectors import (
    _SummaryBuilder,
    _DuplicatesBuilder,
    _ProjectionBuilder,
    _FamiliarityBuilder,
)
from ._familiarity_wrappers import _SplitFamiliarity, _NamedFamiliarity


_logger = logging.getLogger("dnikit.introspectors.DatasetReport")


def _projection_default() -> t.Optional[OneOrManyDimStrategies]:
    if not _umap_available():
        _logger.warning("UMAP not available, not running projection in report."
                        "To fix, install dnikit['dimreduction'].")
        return None
    return DimensionReduction.Strategy.UMAP(2)


@t.final
@dataclass(frozen=True)
class ReportConfig:
    """
    Configuration for which components to build into the :class:`DatasetReport`, and what strategies
    to use to build those components. Default config corresponds to running all components with
    default strategies (:attr:`projection`, :attr:`duplicates`, and :attr:`familiarity`).

    When running familiarity, "split" familiarity is also run, which means that a
    familiarity model is built for each label, for each label category, and then that subgroup of
    data is evaluated according to the model.

    Args:
        projection: **[optional]** see :attr:`projection`
        duplicates: **[optional]** see :attr:`duplicates`
        familiarity: **[optional]** see :attr:`familiarity`
        dim_reduction: **[optional]** see :attr:`dim_reduction`
        split_familiarity_min: **[optional]** see :attr:`split_familiarity_min`
    """

    projection: t.Optional[OneOrManyDimStrategies] = field(default_factory=_projection_default)
    """
    Skip :class:`projection <DimensionReduction>` if None, else provide a
    :class:`DimensionReduction.Strategy` that projects down to 2 dimensions, for visualization
    (default is :class:`DimensionReduction.Strategy.UMAP`).
    """

    duplicates: t.Optional[DuplicatesThresholdStrategyType] = field(
        default_factory=lambda: Duplicates.ThresholdStrategy.Slope())
    """
    Skip :class:`Duplicates` if None, else :class:`Duplicates.ThresholdStrategy` (default is
    :class:`Slope <Duplicates.ThresholdStrategy.Slope>`).
    """

    familiarity: t.Optional[FamiliarityStrategyType] = field(
        default_factory=lambda: Familiarity.Strategy.GMM())
    """
    Skip :class:`Familiarity` if None, else provide :class:`Familiarity.Strategy` to apply to
    overall and split familiarity."""

    dim_reduction: t.Optional[OneOrManyDimStrategies] = None
    """
    If None, default to :class:`DimensionReduction.Strategy.PCA` before running :attr:`familiarity`,
    :attr:`duplicates`, and/or :attr:`projection``. Else provide
    :class:`DimensionReduction.Strategy`.
    """

    split_familiarity_min: int = 50
    """
    If running :class:`Familiarity`, min data that must exist per-label for fitting individual
    models to subgroups of data determined by label ("split" familiarity).
    """

    @property
    def n_stages(self) -> int:
        """
        How many stages of :func:`multi introspect <dnikit.base.multi_introspect>` need to be
        run (not counting stub intropectors)
        """
        if self.familiarity or self.projection:
            return 3
        elif self.duplicates:
            return 2
        else:
            return 1

    @property
    def use_dim_reduction(self) -> bool:
        """True if overall dimension reduction needs to be run"""
        return True if (self.familiarity or self.duplicates or self.projection) else False


@t.final
@dataclass(frozen=True)
class _Stage1Result:
    """
    Result from stage one of computation: summary and dimensionality reduction
    """

    summary_result: _SummaryBuilder
    """Holds pandas dataframe of identifiers and metadata labels data"""

    overall_dim_reduction: t.Optional[DimensionReduction]
    """Dimensionality reduction model fit as a precursor to duplicates and familiarity"""


@t.final
@dataclass(frozen=True)
class _Stage2Result:
    """
    Result from stage one of computation: duplicates, projection, and familiarity fitting
    """

    duplicates_result: t.Optional[_DuplicatesBuilder]
    """Pandas dataframe to categorize samples that are near duplicates of each other"""

    overall_familiarity: t.Optional[_NamedFamiliarity]
    """Familiarity model fit on overall dataset per response"""

    split_familiarity: t.Optional[t.Sequence[_NamedFamiliarity]]
    """Familiarity models fit per each label dimension and label and response"""

    projection_reduction: t.Optional[DimensionReduction]
    """Dimensionality reduction model fit to project responses into 2 coordinates"""

    filtered_labels: t.Optional[t.Mapping[str, t.Sequence[str]]]
    """Hold filtered models used to fit the familiarity models"""


@t.final
@dataclass(frozen=True)
class _Stage3Result:
    """
    Result from stage one of computation: just familiarity scoring
    """

    overall_familiarity_result: t.Optional[_FamiliarityBuilder]
    """Scores extracted from overall dataset's familiarity model, per model response"""

    split_familiarity_frames: t.Optional[pd.DataFrame]
    """Scores for each label-wise familiarity model, per model response"""

    projection_result: t.Optional[_ProjectionBuilder]
    """2 coordinates for each data sample after projecting from higher dimension"""


def _introspector_stub(producer: Producer, *, batch_size: int) -> None:
    """Stub introspector to use as a placeholder during the various multi introspect stages"""
    for _ in producer(batch_size):
        pass
    return None


def _stage_1_runner(producer: Producer, batch_size: int, config: ReportConfig) -> _Stage1Result:
    """
    Run Stage 1 of introspectors in the dataset report

    Args:
        producer: producer of batches
        batch_size: batch size to use
        config: ReportConfig to show which components to run

    Return:
        :class:`_Stage1Result` object
    """
    # Always run summary
    summary_intr = partial(_SummaryBuilder.introspect, batch_size=batch_size)

    # Run dimension reduction if running fam, duplicates, or projection
    run_overall_dim_reduction = config.familiarity or config.duplicates or config.projection
    if run_overall_dim_reduction:
        overall_dim_reduction_intr: t.Callable[[Producer], t.Any] = partial(
            DimensionReduction.introspect,
            strategies=config.dim_reduction,
            batch_size=batch_size,
        )
    else:
        overall_dim_reduction_intr = partial(
            _introspector_stub, batch_size=batch_size)

    # Run multi introspect
    results = multi_introspect(
        summary_intr,
        overall_dim_reduction_intr,
        producer=producer
    )

    # Break up results and return
    return _Stage1Result(
        summary_result=results[0],
        overall_dim_reduction=results[1] if run_overall_dim_reduction else None,
    )


def _stage_2_runner(producer: Producer, batch_size: int, config: ReportConfig,
                    stage1: _Stage1Result) -> _Stage2Result:
    """
    Run Stage 2 of introspectors in the dataset report

    Args:
        producer: producer of batches
        batch_size: batch size to use
        config: ReportConfig to show which components to run
        stage1: results from Stage1

    Return:
        :class:`_Stage2Result` object
    """

    if config.familiarity:
        filtered_labels = stage1.summary_result.filtered_labels(config.split_familiarity_min)
    else:
        filtered_labels = None

    # Define introspectors based on config
    if config.duplicates:
        duplicates_intr: t.Callable[[Producer], t.Any] = partial(
            _DuplicatesBuilder.introspect, batch_size=batch_size, threshold=config.duplicates)
    else:
        duplicates_intr = partial(_introspector_stub, batch_size=batch_size)

    # Fit projection reduction
    if config.projection:
        projection_intr: t.Callable[[Producer], t.Any] = partial(
            DimensionReduction.introspect, strategies=config.projection, batch_size=batch_size)
    else:
        projection_intr = partial(_introspector_stub, batch_size=batch_size)

    if config.familiarity:
        overall_intr: t.Callable[[Producer], t.Any] = partial(
            _NamedFamiliarity.from_overall_familiarity,
            strategy=config.familiarity,
            batch_size=batch_size,
        )

        assert filtered_labels is not None

        split_intrs: t.Sequence[t.Callable[[Producer], t.Any]] = (
            _SplitFamiliarity.get_label_introspectors(
                label_mapping=filtered_labels,
                strategy=config.familiarity,
                batch_size=batch_size
            ))
    else:
        overall_intr = partial(_introspector_stub, batch_size=batch_size)
        split_intrs = tuple([partial(_introspector_stub, batch_size=batch_size)])

    # Run multi introspect
    results = multi_introspect(
        duplicates_intr,
        projection_intr,
        overall_intr,
        *split_intrs,
        producer=producer
    )

    # Break up results and return
    return _Stage2Result(
        duplicates_result=results[0] if config.duplicates else None,
        projection_reduction=results[1] if config.projection else None,
        overall_familiarity=results[2] if config.familiarity else None,
        split_familiarity=results[3:] if (config.familiarity and filtered_labels) else None,
        filtered_labels=filtered_labels
    )


def _stage_3_runner(producer: Producer, batch_size: int, config: ReportConfig,
                    stage2: _Stage2Result) -> _Stage3Result:
    """
    Run Stage 3 of introspectors in the dataset report

    Args:
        producer: producer of batches
        batch_size: batch size to use
        config: ReportConfig to show which components to run
        stage2: results from Stage2

    Return:
        :class:`_Stage3Result` object
    """

    if config.projection:
        assert stage2.projection_reduction is not None
        projection_intr: t.Callable[[Producer], t.Any] = partial(
            _ProjectionBuilder.introspect,
            batch_size=batch_size,
            projection_model=stage2.projection_reduction
        )
    else:
        projection_intr = partial(_introspector_stub, batch_size=batch_size)

    if config.familiarity:
        assert stage2.overall_familiarity
        overall_intr: t.Callable[[Producer], t.Any] = partial(
            _FamiliarityBuilder.introspect, batch_size=batch_size,
            named_fam=stage2.overall_familiarity
        )

        # This could be an empty list if there are no labels provided or too few data per label
        if stage2.split_familiarity:
            split_intrs: t.Iterable[t.Callable[[Producer], t.Any]] = (
                partial(_FamiliarityBuilder.introspect, batch_size=batch_size, named_fam=fam)
                for fam in stage2.split_familiarity
            )
        else:
            split_intrs = tuple([partial(_introspector_stub, batch_size=batch_size)])
    else:
        overall_intr = partial(_introspector_stub, batch_size=batch_size)
        split_intrs = tuple([partial(_introspector_stub, batch_size=batch_size)])

    # Run multi introspect
    results = multi_introspect(
        projection_intr,
        overall_intr,
        *split_intrs,
        producer=producer
    )
    projection_result = results[0]
    overall_familiarity_result = results[1]
    split_results = results[2:]

    # Combine split-familiarity results into the format required by Symphony
    if stage2.split_familiarity:
        # Filtered labels should exist if split familiarity was run
        assert stage2.filtered_labels

        combined_frames = _FamiliarityBuilder.condense_dataframes_by_labels(
            results=split_results,
            label_mapping=stage2.filtered_labels
        )
    else:
        combined_frames = None

    # Break up results and return
    return _Stage3Result(
        projection_result=projection_result if config.projection else None,
        overall_familiarity_result=overall_familiarity_result if config.familiarity else None,
        split_familiarity_frames=combined_frames
    )

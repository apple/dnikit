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

from dataclasses import dataclass, replace
import pathlib

try:
    import pandas as pd
except ImportError:
    class PandasStub:
        DataFrame = None
    pd = PandasStub()

from dnikit._availability import _pandas_available
from dnikit.base import (
    Batch,
    Introspector,
    peek_first_batch,
    pipeline,
    Producer
)
from dnikit.exceptions import DNIKitException
from dnikit.introspectors._dim_reduction._dimension_reduction import DimensionReduction
from dnikit.introspectors._dim_reduction._protocols import DimensionReductionStrategyType
from dnikit.processors import (
    Cacher,
    Composer
)
import dnikit.typing as dt
import dnikit.typing._types as t
from ._dataset_report_stages import (
    ReportConfig,
    _stage_1_runner,
    _stage_2_runner,
    _stage_3_runner
)


_DEFAULT_DIMENSIONS = 40


def _check_pandas_is_installed() -> None:
    if not _pandas_available():
        raise DNIKitException("pandas not available, was dnikit['dataset_report'] "
                              "or dnikit['dataset_report_base'] installed?")


@t.final
@dataclass(frozen=True)
class DatasetReport(Introspector):
    """
    A report built to inspect a dataset for a given model from the perspective of fairness.

    Like other :class:`introspectors <dnikit.base.Introspector>`, use
    :func:`DatasetReport.introspect <introspect>` to instantiate,
    or load a saved report using :func:`DatasetReport.from_disk <from_disk>`.

    This report is particularly useful for introspecting datasets that have various class
    labels attached. See overall `DatasetReport` page in docs to learn more.

    The following components can be run (default to all), configured using a :class:`ReportConfig`.
    - Summarize overall dataset, including by metadata labels, if they exist
    - Find near duplicate data samples, see :class:`Duplicates`
    - Find most / least representative data overall and per metadata label, see :class:`Familiarity`
    - Project the data down to visualize overall in a 2D scatterplot

    The input :class:`Producer <dnikit.base.Producer>` to this class's instantiation is expected to
    have :attr:`fields <dnikit.base.Batch.fields>` of model responses (likely a layer towards the
    end of the model but not the last response). These responses can come either from loading data
    and running it through a DNIKit :class:`Model <dnikit.base.Model>`, or by loading the responses
    directly from file into a :class:`Producer <dnikit.base.Producer>`.
    In each :attr:`Batch's metadata <dnikit.base.Batch.metadata>`, this report looks for
    identifiers and optional labels attached as metadata using
    :class:`Batch.StdKeys.IDENTIFIER <dnikit.base.Batch.StdKeys.IDENTIFIER>` and
    :class:`Batch.StdKeys.LABELS <dnikit.base.Batch.StdKeys.LABELS>` metadata keys.

    Note:
        For the moment, the :class:`Batch.StdKeys.IDENTIFIER <dnikit.base.Batch.StdKeys.IDENTIFIER>`
        should be a path to the image data.

    This class creates a :class:`pandas.DataFrame` full of the data needed to build the
    `Symphony <https://github.com/apple/ml-symphony>`_
    UI for the ``DatasetReport``, which can then be exported into a standalone static site
    to explore. The different components built in the UI interact with each other. See
    `Symphony's documentation <https://apple.github.io/ml-symphony/>`_ to learn more
    about the UI and how to use it with the output of the ``DatasetReport``:


    .. highlight:: python
    .. code-block:: python

        # Build all components of the dataset report using default configuration.
        #    This output can then be used to visualize the results with Symphony:
        #        (1) as a standalone web dashboard to explore interactively
        #        (2) inline in a Jupyter notebook to explore interactively
        #    Please see the Symphony documentation for an example:
        #    https://apple.github.io/ml-symphony/
        report = DatasetReport.introspect(producer)

    Args:
        data: do not instantiate ``DatasetReport`` directly, use
            :func:`DatasetReport.introspect <introspect>`
    """

    data: pd.DataFrame
    """:class:`pandas.DataFrame` of introspection results for responses and report components"""

    _report_save_data_path: pathlib.Path = pathlib.Path('report_save_data.pkl')
    """Filename of :attr:`report.data <data>` save path."""

    def __post_init__(self) -> None:
        _check_pandas_is_installed()

    @staticmethod
    def _get_response_names_from_producer(producer: Producer) -> t.Sequence[str]:
        batch = peek_first_batch(producer, batch_size=1)
        return list(batch.fields.keys())

    @staticmethod
    def _guess_dimension_strategies(producer: Producer
                                    ) -> t.Mapping[str, DimensionReductionStrategyType]:
        batch = peek_first_batch(producer, batch_size=1)
        data: t.Dict[str, DimensionReductionStrategyType] = {}
        for response_name, response_value in batch.fields.items():
            if response_value.shape[1] > _DEFAULT_DIMENSIONS:
                data[response_name] = DimensionReduction.Strategy.PCA(_DEFAULT_DIMENSIONS)
        return data

    @staticmethod
    def introspect(producer: Producer, *,
                   config: t.Optional[ReportConfig] = None,
                   batch_size: int = 1024) -> 'DatasetReport':
        """
        Build relevant ``DatasetReport`` components from input
        :class:`Producer <dnikit.base.Producer>`.

        Args:
            producer: response producer (separate caching not needed as responses are cached
                in this function)
            config: **[keyword arg, optional]** :class:`ReportConfig`. Set components to ``None``
                to omit them from report.
            batch_size: **[keyword arg, optional]** number of samples to batch at once

        Returns:
            a :class:`DatasetReport` whose results can be exported into different formats
        """
        _check_pandas_is_installed()

        if config is None:
            config = ReportConfig(
                dim_reduction=DatasetReport._guess_dimension_strategies(producer))
        elif config.use_dim_reduction and not config.dim_reduction:
            # Recreate same config as input, but replace the dim_reduction strategy with default
            config = replace(
                config,
                # Set up dim reduction strategy guesses based on producer
                dim_reduction=DatasetReport._guess_dimension_strategies(producer)
            )

        data: pd.DataFrame
        cacher: Cacher = Cacher()

        # It's necessary here to convert LABELS metadata for each batch to str values,
        # since both the _SummaryBuilder and _SplitFamiliarity filter assume a str type.
        def convert_labels_metadata_to_str(b: Batch) -> t.Optional[Batch]:
            if Batch.StdKeys.LABELS not in b.metadata:
                return b

            builder = Batch.Builder(base=b)
            builder.metadata[Batch.StdKeys.LABELS] = {
                key: [str(v) for v in values]
                for key, values in b.metadata[Batch.StdKeys.LABELS].items()
            }
            return builder.make_batch()
        producer = pipeline(producer, Composer(convert_labels_metadata_to_str))

        # Cache responses immediately when running exactly 2 stages (duplicates)
        if config.n_stages == 2:
            assert not cacher.cached
            producer = pipeline(producer, cacher)

        # Compute the dataset report in three calls to multi-introspect. Call those now.
        # Stage 1 is summary & both dimension reduction
        stage_1_results = _stage_1_runner(producer, batch_size, config)

        # For familiarity or projection, 3 full stages, so cache and use dim reduction
        if config.n_stages == 3:
            assert stage_1_results.overall_dim_reduction
            assert not cacher.cached
            # attach dim reduction here so the reduced producer can be used everywhere onwards
            producer = pipeline(producer, stage_1_results.overall_dim_reduction, cacher)

        # Stage 2 is projection pt1, duplicates, and familiarity
        stage_2_results = _stage_2_runner(producer, batch_size, config, stage_1_results)

        # Stage 3 is projection pt2 and familiarity
        stage_3_results = _stage_3_runner(producer, batch_size, config, stage_2_results)

        # concatenate result dataframes from all components together into one dataframe
        #    first with split-familiarity frame, overall fam frame, and summary frame
        all_frames = [
            result.data for result in [
                stage_1_results.summary_result,
                stage_2_results.duplicates_result,
                stage_3_results.projection_result,
                stage_3_results.overall_familiarity_result,
            ]
            if result is not None
        ]
        # handle case of data frame from partial familiarity
        if stage_3_results.split_familiarity_frames is not None:
            all_frames.append(stage_3_results.split_familiarity_frames)

        # join everything together into one dataframe
        data = pd.concat(all_frames, axis=1)

        return DatasetReport(
            data=data
        )

    @staticmethod
    def from_disk(directory: dt.PathOrStr) -> 'DatasetReport':
        """
        Create ``DatasetReport`` object from a report save directory

        Args:
            directory: path of directory where report file has been saved

        Return:
            Instance of ``DatasetReport``
        """
        _check_pandas_is_installed()

        directory = dt.resolve_path_or_str(directory)

        data_path = directory / DatasetReport._report_save_data_path
        if not data_path.exists():
            raise FileNotFoundError(
                f"{directory} missing necessary file: {DatasetReport._report_save_data_path}. "
                f"Try saving report using report.to_disk() method."
            )

        return DatasetReport(
            data=pd.read_pickle(directory / DatasetReport._report_save_data_path)
        )

    def to_disk(self, directory: dt.PathOrStr = './report_save', *,
                overwrite: bool = False) -> None:
        """
        Save Dataset Report's :attr:`data` to a directory, to avoid running introspect again when
        visualizing or sharing results.

        Args:
            directory: **[optional]** directory to save data within
            overwrite: **[keyword arg, optional]** True to overwrite any existing report save files
                in this directory
        """
        directory = dt.resolve_path_or_str(directory)
        directory.mkdir(parents=True, exist_ok=True)

        data_path = directory / self._report_save_data_path
        if not overwrite and data_path.exists():
            raise FileExistsError(
                'Report file already exists at this path.'
                'Set "overwrite=True" to overwrite the report saved here.'
            )

        self.data.to_pickle(data_path)

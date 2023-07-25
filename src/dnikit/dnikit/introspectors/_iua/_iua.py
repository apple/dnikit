#
# Copyright 2019 Apple Inc.
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

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from dnikit.base import Introspector, Producer
from dnikit.exceptions import DNIKitException
from dnikit._availability import _pandas_available, _matplotlib_available
import dnikit.typing._types as t

try:
    from pandas import DataFrame as pdDataFrame
except ImportError:
    pdDataFrame = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes as mplAxes
except ImportError:
    plt = None
    mplAxes = None


@t.final
@dataclass(frozen=True)
class IUA(Introspector):
    """
    An introspector that evaluates responses to compute for inactive unit statistics.

    Like other :class:`introspectors <dnikit.base.Introspector>`, use
    :func:`IUA.introspect <introspect>` to instantiate.
    """

    _layer_counts: t.Mapping[str, t.List[float]]
    _unit_counts: t.Mapping[str, np.ndarray]
    _total_probe_counts: int

    @t.final
    @dataclass(frozen=True)
    class Result:
        """
        Per-response ``IUA`` Result

        Args:
            mean_inactive: see :attr:`mean_inactive`
            std_inactive: see :attr:`std_inactive`
            inactive: see :attr:`inactive`
            unit_inactive_count: see :attr:`unit_inactive_count`
            unit_inactive_proportion: see :attr:`unit_inactive_proportion`
        """

        mean_inactive: float
        """mean inactive units in the batch"""

        std_inactive: float
        """standard deviation in number of inactive units across batch inputs"""

        inactive: t.Sequence[float]
        """
        sequence tracking the number of inactive units in the layer per batch input,
        used to compute :attr:`mean_inactive` and :attr:`std_inactive`
        """

        unit_inactive_count: t.Sequence[float]
        """sequence tracking the number of times each unit was inactive across batch inputs"""

        unit_inactive_proportion: t.Sequence[float]
        """sequence tracking the proportion of times each unit was inactive across batch inputs"""

    @t.final
    class VisType:
        """Type of visualization modality for IUA, available to visualize via :func:`IUA.show()`"""

        CHART: t.Final = 'chart'
        """Charts showing inactive units per layer"""

        TABLE: t.Final = 'table'
        """Table of all IUA result data"""

    @staticmethod
    def introspect(producer: Producer, *,
                   batch_size: int = 32, rtol: float = 1e-05, atol: float = 1e-08) -> "IUA":
        """
        Compute inactive unit statistics (mean, standard deviation, counts, and unit
        frequency) for each layer (:attr:`field <dnikit.base.Batch.fields>`) in the input
        ``producer`` of model responses.

        Args:
            producer: The producer of the model responses to be introspected
            batch_size: **[keyword arg, optional]** number of inputs to pull from ``producer``
                at a time
            rtol : **[keyword arg, optional]** float relative tolerance parameter
                (see doc for :func:`numpy.isclose`).
            atol : **[keyword arg, optional]** float absolute tolerance parameter
                (see doc for :func:`numpy.isclose`).

        Returns:
            an ``IUA`` instance that can provide information about inactive units in the model
        """

        # Dictionaries tracking the inactive unit counts used to compute the
        # IUA statistics
        layer_counts: t.MutableMapping[str, list] = defaultdict(list)
        unit_counts = dict()

        # Counts number of input probes
        total_probe_counts = 0
        batch_probe_counts = 0
        batch_count = 0

        # Get batch_size many responses to evaluate per iteration
        for resp_batch in producer(batch_size):
            # Track the number of batches for counting the total number of responses
            batch_count += 1
            # Evaluate unit inactivity per layer
            for layer_name, responses in resp_batch.fields.items():
                # Find the inactive units (response of 0.) in the layer
                inactive_units = np.isclose(
                    responses, np.zeros_like(responses), rtol=rtol, atol=atol
                )

                # Count of inactive units per batch item
                # Retain the batch size dimension (first dimension in inactive_units)
                # and flatten out the rest
                collapsed_dims = (inactive_units.shape[0], -1)
                inactive_counts = np.sum(inactive_units.reshape(collapsed_dims), axis=1)

                batch_probe_counts = len(inactive_units)

                # Track count of inactive units for this layer per
                # probe input, e.g. item in batch
                layer_counts[layer_name] += list(inactive_counts)

                # Number of times each unit was inactive in this batch
                unit_inactive_counts = np.sum(inactive_units, axis=0)

                # Check if the layer has been added to the unit counts dictionary
                if layer_name not in unit_counts:
                    unit_counts[layer_name] = np.zeros_like(unit_inactive_counts, dtype=np.intc)

                # Update the count tracking the number of times each unit is inactive
                unit_counts[layer_name] += unit_inactive_counts

            # Update the total number of response probes with those from the
            # the current batch
            total_probe_counts += batch_probe_counts

        return IUA(layer_counts, unit_counts, total_probe_counts)

    @property
    def results(self) -> t.Mapping[str, "IUA.Result"]:
        """A per-layer :class:`IUA.Result` encapsulating Inactive Unit Analysis results."""
        return {
            n: IUA.Result(
                # return type of numpy mean and std is np floating[Any], cast to float
                mean_inactive=float(np.mean(counts)),
                std_inactive=float(np.std(counts)),
                inactive=counts,
                unit_inactive_count=self._unit_counts[n].tolist(),

                # Total number of probe inputs for which a unit was inactive to get the proportion
                # of time spent inactive
                unit_inactive_proportion=np.divide(
                    self._unit_counts[n],
                    self._total_probe_counts
                ).tolist()
            )
            for n, counts in self._layer_counts.items()
        }

    @staticmethod
    def _validate_response_input(iua_results: t.Mapping[str, Result],
                                 response_names: t.Optional[t.Sequence[str]] = None
                                 ) -> t.Sequence[str]:
        # If there are no layers specified, plot all layers in results
        result_keys = iua_results.keys()
        if response_names is None:
            list_of_responses = list(result_keys)
        else:
            # Validate input response names (need to have corresponding results for them)
            for response in response_names:
                if response not in result_keys:
                    raise ValueError(
                        f'Invalid response passed: {response}. Try one of: {result_keys}'
                    )
            list_of_responses = list(response_names)
        list_of_responses.sort()

        if len(list_of_responses) == 0:
            raise ValueError(
                "Empty list of layers specified. Pass no `layer names` param to default"
                " to plotting all layers."
            )
        return list_of_responses

    @staticmethod
    def _show_dataframe(iua_results: t.Mapping[str, Result],
                        responses: t.Sequence[str]) -> pdDataFrame:
        if not _pandas_available():
            raise DNIKitException("PIL not available, was 'dnikit[notebook]' installed?")

        # Pull IUA results into a pandas dataframe
        all_results = pdDataFrame({
             'response': response,
             'mean inactive': iua_results[response].mean_inactive,
             'std inactive': iua_results[response].std_inactive
         } for response in responses)

        # Sort all values by 'response' column
        all_results = all_results.sort_values('response')

        # reset the index of rows after sorting
        return all_results.reset_index(drop=True)

    @staticmethod
    def _show_chart(iua_results: t.Mapping[str, Result],
                    responses: t.Sequence[str]) -> mplAxes:

        if not _matplotlib_available():
            raise DNIKitException("matplotlib not available, was 'dnikit[notebook]' installed?")

        fig, axs = plt.subplots(
            len(responses), figsize=(7, 70)
        )

        for axis_id, response_name in enumerate(responses):
            if len(responses) == 1:
                single_axis = axs
            else:
                single_axis = axs[axis_id]

            # Double check that the data is valid
            if np.sum(iua_results[response_name].unit_inactive_count) == 0:
                # If there's no data, create a black image (no activations)
                shape = (
                    len(iua_results[response_name].unit_inactive_count),
                    len(iua_results[response_name].unit_inactive_count)
                )
                single_axis.imshow(np.zeros(shape))
                added_text = "- no inactive units"
            else:
                # Else create heatmap
                single_axis.imshow(
                    np.sum(iua_results[response_name].unit_inactive_count, axis=0),
                    cmap='hot',
                    interpolation='nearest'
                )
                added_text = ""
            single_axis.set_title(f"{response_name} Inactive Unit Proportions {added_text}")

        return axs

    @staticmethod
    def show(iua: "IUA", *, vis_type: str = VisType.TABLE,
             response_names: t.Optional[t.Sequence[str]] = None) -> t.Union[mplAxes, pdDataFrame]:
        """
        Create table or chart to visualize IUA results in iPython / Jupyter notebook.

        Note: Requires `pandas <https://pandas.pydata.org/docs/>`_
            (``vis_type`` is :class:`IUA.VisType.TABLE`) or
            `matplotlib <https://matplotlib.org/stable/>`_
            (``vis_type`` is :class:`IUA.VisType.CHART`), which can be installed with
            ``pip install "dnikit[notebook]"``

        Args:
            iua: result of :func:`IUA.introspect`, instance of :class:`IUA`
            vis_type: **[keyword arg, optional]** determines visualization type.
                `IUA.VisType.TABLE` for pandas dataframe result or
                `IUA.VisType.CHART` for matplotlib pyplot of inactive units
            response_names: **[keyword arg, optional]** For `IUA.VisType.CHART` vis. Sequence of
                responses (:attr:`field <dnikit.base.Batch.fields>` names) to
                visualize (defaults to None for showing all responses)

        Return:
            :class:`pandas.DataFrame` or :class:`matplotlib.axes.Axes` of ``IUA`` results
        """
        iua_results = iua.results
        responses = IUA._validate_response_input(iua_results, response_names)

        if vis_type == IUA.VisType.TABLE:
            # Concatenate results for multiple recipe results
            return IUA._show_dataframe(iua_results, responses)
        elif vis_type == IUA.VisType.CHART:
            return IUA._show_chart(iua_results, responses)
        else:
            raise ValueError(
                'Unexpected input for parameter `vis_type`. Expected `IUA.VisType.CHART`'
                ' or `IUA.VisType.TABLE`'
            )

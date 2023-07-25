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

import dataclasses

import numpy as np

from ._batch._batch import Batch
from ._response_info import ResponseInfo
from ._pipeline import PipelineStage
import dnikit.typing as dt
import dnikit.typing._types as t
from dnikit.exceptions import DNIKitException


class _ModelDetails(t.Protocol):
    """
    Protocol to wrap models from other deep learning frameworks into DNIKit.

    Derived classes must implement :func:`run_inference` and :func:`get_response_infos`.

    Warning:
        As a private class, the API may be changed in a future release.

        To wrap a deep learning framework that DNIKit does not currently support,
        it's recommended to create a custom :class:`Producer <dnikit.base.Producer>` that yields the
        model responses, rather than creating a custom ``_ModelDetails``.

        This class is intended for code that will eventually be integrated into DNIKit.
    """
    def run_inference(self,
                      inputs: t.Mapping[str, np.ndarray],
                      outputs: t.AbstractSet[str]) -> t.Mapping[str, np.ndarray]:
        """
        Run inference on a single batch of input data and return its corresponding response.

        Note:
            This function runs inference upon being called.

        Args:
            inputs: A map of layer name to ``np.ndarray``.
            outputs: Selects responses to be collected from the model after feeding the
                input batch through.

        Returns:
            A map of response names to ``np.ndarray`` containing the requested model responses
            to the input data.
        """
        ...

    def get_response_infos(self) -> t.Iterable[ResponseInfo]:
        """
        Obtain :class:`ResponseInfo <dnikit.base.ResponseInfo>` for all layers in the model.
        """
        ...

    def get_input_layer_responses(self) -> t.Sequence[ResponseInfo]:
        """
        Obtain all input layers to the model as :class:`ResponseInfo <dnikit.base.ResponseInfo>`.
        """
        ...


@t.final
@dataclasses.dataclass(frozen=True)
class _ModelPipelineStage(PipelineStage):
    _details: _ModelDetails
    _requested_responses: t.AbstractSet[str]

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:

        potential_inputs = self._details.get_input_layer_responses()
        potential_input_names = set(ri.name for ri in potential_inputs)

        def process_batch(batch: Batch) -> Batch:
            # The batch fields to feed into model inference
            infer_fields = batch.fields

            # Auto-rename fields, if batch has a single field and model takes a single field:
            if len(potential_inputs) == 1 and len(batch.fields) == 1:
                batch_field = next(iter(batch.fields))
                input_response = next(ri for ri in potential_inputs)

                # Check that name of fields differ, but type and shape of fields match:
                if (batch_field != input_response.name and
                   batch.fields[batch_field].shape[1:] == input_response.shape[1:]):

                    # Rename field before passing dict to _ModelDetails.run_inference
                    infer_fields = {input_response.name: batch.fields[batch_field]}

            elif (len(potential_inputs) == len(batch.fields) and
                  potential_input_names != set(iter(batch.fields))):
                raise DNIKitException(
                    f"Model expects inputs named {', '.join([ri.name for ri in potential_inputs])} "
                    f"but batch contains fields named {', '.join([f for f in batch.fields])}. "
                    f"Field names must match expected input names to perform inference. "
                    f"(To change field names in a batch, try using a "
                    f"FieldRenamer in the pipeline. To import the FieldRenamer class, do "
                    f"'from dnikit.processors import FieldRenamer')")

            inference_result = self._details.run_inference(infer_fields, self._requested_responses)
            # Prepare output
            builder = Batch.Builder(base=batch)
            # set the output data
            builder.fields = dict(inference_result)
            return builder.make_batch()

        return process_batch


@t.final
@dataclasses.dataclass(frozen=True)
class Model:
    """
    Abstraction for deep learning models.

    A dnikit ``Model`` is in charge of generating a :class:`Batch` of responses when fed
    an input :class:`Batch`.

    In order to perform inference with an instance of ``Model``, one must call it in a
    :func:`pipeline()`, feeding it batches from a data :class:`Producer`.
    Note that dnikit avoids processing any data until an introspection algorithm needs it.
    See the example below for more information.

    More specifically, calling a ``Model`` instance
    produces a :class:`PipelineStage` for use in a :func:`pipeline()`.
    It's common to pass the model one or more requested responses (i.e., layers) to
    obtain when performing inference. See :func:`__call__()` for more information.

    Note that for a ``Model`` to perform inference on a :class:`Batch`, it must be able to link up
    the :class:`Batch` :attr:`field(s) <dnikit.base.Batch.fields>` to the expected input field(s)
    of the model. If a :class:`Batch`
    and a ``Model`` have only a single input field, this linking is attempted automatically.
    However, if the ``Model`` has two or more inputs, the number and name of the input
    :class:`Batch` fields **must match** the expected input field names for the model exactly.
    For instance, a TensorFlow model might expect two fields called ``input_1`` and ``input_2``.
    The :class:`Batch` that is fed into the model must have exactly two
    :attr:`fields <dnikit.base.Batch.fields>` with the
    same names and expected NumPy shapes. DNIKit has two helper methods to assist with modifying
    a :class:`Batch's <Batch>` :attr:`fields <dnikit.base.Batch.fields>`:
    :class:`FieldRenamer <dnikit.processors.FieldRenamer>` and
    :class:`FieldRemover <dnikit.processors.FieldRemover>`.

    Warning:
        Do not instantiate this class directly, either use one of the methods provided by
        dnikit_tensorflow, or, to implement a custom ``Model``,
        it's recommended to instead create a custom a :class:`Producer` that generates model
        responses,
        or follow the :class:`PipelineStage` protocol. If it's necessary to create a new
        ``Model`` for DNIKit, a custom
        :class:`_ModelDetails <dnikit.base._model._ModelDetails>` must be implemented.

    Example:
        .. code-block:: python

            data_producer = ... # Instantiate a valid Producer with input data
            model = ... # Instantiate a model with dnikit_tensorflow, etc...
            requested_responses = ["conv2d/0", "conv2d/1"]

            # Pipeline model with producer
            response_producer = pipeline(
                data_producer,
                model(requested_responses)
            )

            # Until now, no computation has occurred.
            # The following call will create a an input batch with 128 elements (using
            # data_producer) feed it as input to the model, run inference and collect the "conv2d/0"
            # and "conv2d/1" responses. This process is repeated for every iteration in the loop.
            batch = peek_first_batch(response_producer, batch_size=128)
            # Do processing with batch
            # frozenset(batch.keys()) == {"conv2d/0", "conv2d/1"}
            # batch.batch_size = 128
    """

    # Model fields
    _details: _ModelDetails
    _response_infos: t.Mapping[str, ResponseInfo] = dataclasses.field(init=False)
    _input_layers: t.Mapping[str, ResponseInfo] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize common elements for all dnikit ``Model``.
        """
        super().__init__()
        response_infos = {r.name: r for r in self._details.get_response_infos()}
        input_layers = {r.name: r for r in self._details.get_input_layer_responses()}
        object.__setattr__(self, "_response_infos", response_infos)
        object.__setattr__(self, "_input_layers", input_layers)

    @property
    def response_infos(self) -> t.Mapping[str, ResponseInfo]:
        """
        Get all possible responses in a model. Result is returned as a mapping between response
        names and the corresponding :class:`ResponseInfo`.
        """
        return self._response_infos

    @property
    def input_layers(self) -> t.Mapping[str, ResponseInfo]:
        """
        Get all potential input layers of the model. Result is returned as a mapping between
        input layer names and the corresponding :class:`ResponseInfo`.
        """
        return self._input_layers

    def __call__(self, requested_responses: dt.OneManyOrNone[str] = None) -> PipelineStage:
        """
        Used to obtain a :class:`PipelineStage`, which is necessary to run inference
        on input :class:`Batch`.

        Args:
            requested_responses: Determines which outputs from this model will be present in the
                :class:`Batch` output by the resulting :class:`PipelineStage`.
                The argument represents the name(s) of a single response or a collection of
                responses or ``None`` (the default).
                If ``None`` is used, all possible responses in the model will be selected (which may
                be expensive to compute!).

        Returns:
            a :class:`PipelineStage` that can be used with :func:`pipeline()` to run inference with
            the loaded ``Model``.
        """
        requested_responses = (
            dt.resolve_one_many_or_none(requested_responses, str)
            or frozenset(self._response_infos.keys())
        )
        return _ModelPipelineStage(self._details, requested_responses)

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

import abc

from ._batch._batch import Batch
from ._producer import Producer
from dnikit._logging import _Logged
import dnikit.typing._types as t
from dnikit.typing._dnikit_types import OneOrMany, resolve_one_or_many_to_list


class PipelineStage(_Logged, abc.ABC):
    """
    Protocol used to :func:`pipeline` operations (such as model inference, pre-processing and
    post-processing).

    :class:`PipelineStage` instances allow for declare how to transform a :class:`Batch` while
    delaying the actual computation until an algorithm's introspection function is called.

    Notable subclasses include :class:`dnikit.base.Model` and :class:`dnikit.processors.Processor`.

    Note:
        In order to implement a custom ``PipelineStage``,
        inherit from this class and implement :func:`_get_batch_processor()` and,
        in rare circumstances, :func:`_pipeline()`.
        Refer to the documentation of those methods for more information.
    """

    def _pipeline(self, producer: Producer) -> Producer:
        """
        Create a new :class:`Producer` which will yield instances :class:`Batch` with the
        appropriate processing/transformations applied.

        By default, the :class:`Producer` returned by this function will obtain a
        batch processor using :func:`_get_batch_processor()` and apply it to every batch
        produced by the input :class:`Producer`.
        """
        batch_processor = self._get_batch_processor()

        def new_producer(batch_size: int) -> t.Iterable[Batch]:
            for batch in producer(batch_size):
                yield batch_processor(batch)
        return new_producer

    @abc.abstractmethod
    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        """
        Get a function that will process/transform each of the batches produced by a
        :class:`Producer`.

        The batch processor must not modify its input ``batch``.

        Warning:
            The batch processor MUST be stateless. That is, its outputs must only depend on the
            input `batch`.
            If the ``PipelineStage`` has some state, the best way to ensure the batch processor
            is stateless is to make a local copy of all mutable variables.
            See examples below for details on how to do this.

        Note:
            If :func:`_pipeline()` is overridden, an empty implementation (``pass``) may be enough
            for :func:`_get_batch_processor()` as that is the only caller for this function.

        Example:
            .. code-block:: python

                # Implement a stateless PipelineStage
                class Simple(PipelineStage):
                    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
                        # Define the function that will process the batches
                        def simple_batch_operation(batch: Batch) -> Batch:
                            data = {"result" : batch["input"] * 2.0}
                            return Batch(data)
                        # Return batch operation
                        return simple_batch_operation

        Example:
            .. code-block:: python

                # Implement a stateful PipelineStage
                class Stateful(PipelineStage):
                    def __init__(self, factor: float):
                        # Note that users may change the value of this variable *after* pipelining
                        self.factor = factor

                    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
                        # First, get a local copy of factor.
                        factor = self.factor
                        # Then, take advantage of Python closures to define the batch operation.
                        def batch_operation(batch: Batch) -> Batch:
                            # Notice even if users change self.factor the result of this function
                            # depends only on its input
                            data = {"result" : batch["input"] * factor}
                            return Batch(data)
                        # Return batch operation
                        return batch_operation
        """
        raise NotImplementedError()


def pipeline(producer: Producer,
             *stages: OneOrMany[PipelineStage]) -> Producer:
    """
    Combine a :class:`Producer` with one or more :class:`PipelineStage` (normally :class:`Model` or
    :class:`Processor <dnikit.processors.Processor>`) to obtain a new :class:`Producer` which
    will yield :class:`Batch` from the input ``producer`` transformed the specified ``stages``.

    Args:
        producer: input :class:`Producer` which will be used to create initial batches.
        stages: one or more :class:`PipelineStage` that determine how to transform the batches
            produced by the input :class:`Producer`. Elements of `stages` may also be
            tuples or lists of :class:`PipelineStages <PipelineStage>` and will be
            flattened automatically.

    Example:
        .. code-block:: python

            in_producer = ... # Instantiate a valid input Producer
            model = ... # Instantiate a model
            postprocessor = Pooler(dim=-1, method=Pooler.Method.MAX) # create a max-pooler

            # Combine in_producer, model and postprocessor
            out_producer = pipeline(in_producer, model(), postprocessor)

            # Get a batch with 32 elements from the pipelined producer
            batch = peek_first_batch(out_producer, batch_size=32)
            # Batch now contains the max-pooled responses resulting from running inference on model
            # with the batches produced by in_producer.
    """
    for stage in stages:
        stage_as_list = resolve_one_or_many_to_list(stage, PipelineStage)  # type: ignore
        for s in stage_as_list:
            if not isinstance(s, PipelineStage):
                raise TypeError(f"Stage is of unsupported type: {type(stage)}")
            producer = s._pipeline(producer)

    return producer

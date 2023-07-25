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

from ._batch._batch import Batch
from ._batch._storage import _concatenate_batches
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t


class Producer(t.Protocol):
    """
    ``Producer`` is a protocol for a function that produces instances of :class:`Batch`. It is used
    by dnikit to delay loading batches into memory (and processing them!) until needed by an
    :class:`Introspector`.

    There are two main ways to implement a ``Producer`` as shown in this example:

    Example:
        .. code-block:: python

            # Implement a Producer as a free function
            def simple_producer(batch_size: int) -> t.Iterable[Batch]:
                for i in range(100):
                    # Load data (in this case random data is generated)
                    # Make sure the first dimension is of batch_size
                    data = numpy.random.randn(batch_size, 10)
                    # yield result
                    yield Batch({"input": data})

            # Implement a Producer as a class
            # Recommended (but not necessary): inherit from Producer.
            class ClassProducer(Producer):
                def __init__(self):
                    # Store stateful variables here
                    self._dims = (10, 1)

                def __call__(self, batch_size: int) -> t.Iterable[Batch]:
                    for i in range(100):
                        # Load data (in this case generate random data)
                        # Make sure the first dimension is of batch_size
                        data = numpy.random.randn(batch_size, *self._dims)
                        # yield result
                        yield Batch({"input": data})

    Warning:
        Make sure to have a finite number of batches the ``Producer`` will generate, as some
        :class:`Introspector` instances will try to consume all the batches of the producer and
        the program will stop responding indefinitely if there are infinite batches.
    """

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        """
        Signature for all producers.

        All ``Producers`` should yield at least one :class:`Batch` of size ``batch_size``. The last
        of the batches is allowed to have a size smaller than ``batch_size``.
        """
        ...


def _accumulate_batches(producer: Producer, *, batch_size: int = 1024) -> Batch:
    """
    Accumulate all batches produced by a :class:`Producer` into a single batch.

    Args:
        producer: used to obtain batches will be accumulated into a single batch.
        batch_size: the size of the batch to pull while iterating through the producer.
    Raises:
        DNIKitException: if producer did not produce any batch.
    Warning:
        This method may exhaust a system's memory since it will load **all** the batches of
        the producer into memory. Use with caution!
    """
    try:
        storage = _concatenate_batches([batch._storage for batch in producer(batch_size)])
        return Batch(_storage=storage)
    except ValueError as e:
        raise DNIKitException("Producer did not produce any batches") from e


def _resize_batches(batches: t.Iterable[Batch]) -> Producer:
    # iterator will keep track of the current batch that is being consumed
    iterator = iter(batches)

    # get_next_batch will yield one batch at a time and return None when done
    def get_next_batch() -> t.Optional[Batch]:
        return next(iterator, None)

    # producer will produce the resized batches
    def producer(batch_size: int) -> t.Iterable[Batch]:
        batch = get_next_batch()
        while batch is not None:
            # The batch size may be smaller/equal/greater than the size requested

            # If current batch is smaller than size requested
            if batch.batch_size < batch_size:
                # Accumulate enough batches to get to required size
                next_size = batch.batch_size
                to_accumulate = [batch._storage]
                done = False
                while next_size < batch_size and not done:
                    next_batch_to_accumulate = get_next_batch()
                    if next_batch_to_accumulate is not None:
                        to_accumulate.append(next_batch_to_accumulate._storage)
                        next_size += next_batch_to_accumulate.batch_size
                    else:
                        done = True
                # Combine accumulated batches
                storage = _concatenate_batches(to_accumulate)

                # If producer is done, return what is present and exit loop
                if done:
                    assert storage.batch_size <= batch_size
                    yield Batch(_storage=storage)
                    next_batch = None  # Signal outer loop to exit
                else:
                    next_batch = Batch(_storage=storage)

            # If current batch is of requested size
            elif batch.batch_size == batch_size:
                yield batch
                next_batch = get_next_batch()

            # If current batch is greater than requested size
            else:
                batch, next_batch = batch.elements[:batch_size], batch.elements[batch_size:]
                yield batch

            # Prepare for next iteration (will stop if next_batch is None)
            batch = next_batch
    return producer


def peek_first_batch(producer: Producer, batch_size: int = 1) -> Batch:
    """
    Helper function to examine the first :class:`Batch` (optionally giving a batch_size)
    of a :class:`Producer`. This is useful in debugging/inspecting the output of a
    :class:`pipeline` or data source.

    Args:
        producer: the :class:`Producer` to pull data from
        batch_size: **[optional]** the size of :class:`Batch` to pull

    Returns:
        The first :class:`Batch` from the :class:`Producer`
    """
    return next(iter(producer(batch_size)))


def _produce_elements(producer: Producer, batch_size: int = 32) -> t.Iterable[Batch.ElementType]:
    for batch in producer(batch_size):
        yield from batch.elements

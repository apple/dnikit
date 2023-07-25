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
import concurrent.futures
import threading
from typing import overload  # need this import as flake8 doesn't recognise DNIKit's t.overload

from ._batch._batch import Batch
from ._producer import Producer
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t


# Typing variables and helpers
_X = t.TypeVar("_X")
_Introspector = t.Callable[[Producer], _X]

_T1 = t.TypeVar("_T1")
_T2 = t.TypeVar("_T2")
_T3 = t.TypeVar("_T3")
_T4 = t.TypeVar("_T4")
_T5 = t.TypeVar("_T5")
_T6 = t.TypeVar("_T6")
_T7 = t.TypeVar("_T7")


# Helper class
@dataclasses.dataclass
class _ProducerSplitter:
    _producer: Producer
    _events: t.List[threading.Event] = dataclasses.field(default_factory=list)
    _first: bool = True
    _done: bool = False
    _failure: bool = False
    _batch: t.Optional[Batch] = None
    _batch_size: t.Optional[int] = None

    def make_producer(self) -> Producer:
        assert self._batch_size is None, (
            "Cannot create new producers after any Producer has been called"
        )
        if self._first:
            return self._make_first_producer()
        else:
            return self._make_subsequent_producer()

    def signal_failure(self) -> None:
        # Set failure
        self._failure = True
        for event in self._events:
            event.set()

    def _make_first_producer(self) -> Producer:
        """
        The first producer created by _ProducerSplitter is different from the others since
        it will actually trigger the underlying producer (with the appropriate batch_size!) and
        store the resulting batch in the instance.

        Another difference is that the first producer does not need to wait for its
        turn upon invocation (since it needs to retrieve the first batch).

        Finally, the first producer is in charge of indicating to other producers that
        there are no more batches to iterate over (which is why self._done is only set here).
        """
        # Logic checks (to avoid weird multithreading errors)
        assert self._first
        assert not self._done
        assert not self._failure
        assert self._batch is None
        assert self._batch_size is None

        # Make sure only one of these is created
        self._first = False
        # Instantiate a new event for this producer
        self._events.append(threading.Event())

        def _first_producer(batch_size: int) -> t.Iterable[Batch]:
            self._batch_size = batch_size
            for batch in self._producer(batch_size):
                # yield the batch to the introspector
                yield batch
                # save batch for next producer
                self._batch = batch
                # signal next producer and wait for turn
                self._signal_next_producer(index=0)
                self._wait_for_turn(index=0)

            self._done = True
            self._signal_next_producer(index=0)
        return _first_producer

    def _make_subsequent_producer(self) -> Producer:
        """
        These producers simply wait for their turn, yield the batch stored in the instance, signal
        the next producer and wait until the done flag is done.

        The batches are retrieved by the first producer. Similarly, it's the first producer
        responsibility to set the _done flag.

        Upon invocation these producers also need to wait until the first producer has stored
        the initial batch.
        """
        # Logic checks
        assert not self._first
        assert not self._done
        assert not self._failure
        assert self._batch is None
        assert self._batch_size is None

        # Add a new event for this producer
        self._events.append(threading.Event())
        index: t.Final = len(self._events) - 1

        def _subsequent_producer(batch_size: int) -> t.Iterable[Batch]:
            self._wait_for_turn(index)
            while not self._done:
                # Logic checks
                assert self._batch is not None
                assert self._batch_size is not None
                if batch_size != self._batch_size:
                    raise ValueError(
                        f"Mismatched batch_size, got {batch_size}, expected {self._batch_size}"
                    )
                # yield the batch to the introspector
                yield self._batch
                # Notify next producer, it's their turn (and wait for ours)
                self._signal_next_producer(index)
                self._wait_for_turn(index)
            # If this statement was reached, the producer is done. Signal next producer.
            self._signal_next_producer(index)
        return _subsequent_producer

    def _wait_for_turn(self, index: int) -> None:
        # Select appropriate event (previous to index)
        event = self._events[index - 1]
        # Wait for turn
        event.wait()
        # Check for failures after waking up
        assert not self._failure, "Early stopping due to exception in another introspector"
        # Prepare for next iteration
        event.clear()

    def _signal_next_producer(self, index: int) -> None:
        self._events[index].set()


# Overloads
@overload
def multi_introspect(in1: _Introspector[_T1],
                     *, producer: Producer) -> t.Tuple[_T1]: ...


@overload
def multi_introspect(in1: _Introspector[_T1],
                     in2: _Introspector[_T2],
                     *, producer: Producer) -> t.Tuple[_T1, _T2]: ...


@overload
def multi_introspect(in1: _Introspector[_T1],
                     in2: _Introspector[_T2],
                     in3: _Introspector[_T3],
                     *, producer: Producer) -> t.Tuple[_T1, _T2, _T3]: ...


@overload
def multi_introspect(in1: _Introspector[_T1],
                     in2: _Introspector[_T2],
                     in3: _Introspector[_T3],
                     in4: _Introspector[_T4],
                     *, producer: Producer) -> t.Tuple[_T1, _T2, _T3, _T4]: ...


@overload
def multi_introspect(in1: _Introspector[_T1],
                     in2: _Introspector[_T2],
                     in3: _Introspector[_T3],
                     in4: _Introspector[_T4],
                     in5: _Introspector[_T5],
                     *, producer: Producer) -> t.Tuple[_T1, _T2, _T3, _T4, _T5]: ...


@overload
def multi_introspect(in1: _Introspector[_T1],
                     in2: _Introspector[_T2],
                     in3: _Introspector[_T3],
                     in4: _Introspector[_T4],
                     in5: _Introspector[_T5],
                     in6: _Introspector[_T6],
                     *, producer: Producer) -> t.Tuple[_T1, _T2, _T3, _T4, _T5, _T6]: ...


@overload
def multi_introspect(in1: _Introspector[_T1],
                     in2: _Introspector[_T2],
                     in3: _Introspector[_T3],
                     in4: _Introspector[_T4],
                     in5: _Introspector[_T5],
                     in6: _Introspector[_T6],
                     in7: _Introspector[_T7],
                     *, producer: Producer) -> t.Tuple[_T1, _T2, _T3, _T4, _T5, _T6, _T7]: ...


# Generic overload for more than 7 arguments
@overload
def multi_introspect(*introspectors: _Introspector[t.Any],
                     producer: Producer) -> t.Tuple[t.Any, ...]: ...


# Actual implementation
def multi_introspect(in1: t.Optional[_Introspector[t.Any]] = None,
                     in2: t.Optional[_Introspector[t.Any]] = None,
                     in3: t.Optional[_Introspector[t.Any]] = None,
                     in4: t.Optional[_Introspector[t.Any]] = None,
                     in5: t.Optional[_Introspector[t.Any]] = None,
                     in6: t.Optional[_Introspector[t.Any]] = None,
                     in7: t.Optional[_Introspector[t.Any]] = None,
                     *introspectors: _Introspector[t.Any],
                     producer: Producer) -> t.Tuple[t.Any, ...]:
    """
    Execute one or more :class:`introspectors <Introspector>` concurrently reusing
    :class:`Batch` from a :class:`Producer`.

    This can be more memory efficient than running introspectors sequentially (as the
    :class:`Batches <Batch>` are produced just once).

    Args:
        introspectors: one or more introspectors (ie functions that take a :class:`Producer`,
            introspect the batches and return a result). The input arguments can *only* take
            a single :class:`Producer`, it may be necessary to use a lambda function, a closure,
            or :func:`functools.partial()` to pass other arguments.

        producer: **[keyword arg]** the :class:`Producer` whose :class:`Batch` will be reused
            for each of the ``introspectors``.

    Returns:
        A tuple with the result of each input ``introspector`` in the order they were passed.

    Raises:
        DNIKitException: if ``introspectors`` request batches of different sizes.
        DNIKitException: if either the ``producer`` or any of the ``introspectors`` raises. If any
            exception is raised, all ``introspectors`` will be stopped.

    Example:
        .. code:: python

            producer = ... # Instantiate a valid Producer

            pfa, familiarity = multi_introspect(
                PFA.introspect,
                Familiarity.introspect,
                producer=producer
            )
            # pfa and familiarity now contain the results of PFA and Familiarity

    Sometimes it's necessary to pass arguments to the introspectors, in which case it's
    recommended to use a lambda function:

    .. code:: python

        results = multi_introspect(
            lambda prod: Familiarity.introspect(prod, strategy=Familiarity.Strategy.GMM()),
            ...
            producer=producer
        )

    Alternatively, it's possible to use :func:`functools.partial()`:

    .. code:: python

        results = multi_introspect(
            functools.partial(Familiarity.introspect, strategy=Familiarity.Strategy.GMM())
            ...
            producer=producer
        )

    See also:
        :class:`dnikit.processors.Cacher` which will store :class:`Batches <Batch>` in the
        filesystem enabling use of the same :class:`Batches <Batch>` with two different
        :class:`introspectors <Introspector>`. If the dataset fits on the
        hard-drive, :class:`Cacher <dnikit.processors.Cacher>` may be a better option
        (since it will require less RAM).

    Note:
        **Implementation detail**: currently ``multi_introspect`` is implemented using Python
        threads to be able to preempt :class:`introspectors <Introspector>`.
        In fact, this function will start
        *one thread per introspector instance*. However, only one thread will be active at a
        time so, in all likelihood, it will not be necessary to make code thread-safe.

    Warning:
        Do not attempt to catch the :class:`AssertionError` in any of the input
        :class:`introspectors <Introspector>`, doing so may cause deadlock!
    """
    introspectors = tuple(
        x
        for x in (in1, in2, in3, in4, in5, in6, in7, *introspectors)
        if x is not None
    )
    num_introspectors = len(introspectors)
    # Instantiate a ProducerSplitter, and create a new producer for each introspector
    splitter = _ProducerSplitter(producer)
    producers = [splitter.make_producer() for _ in range(num_introspectors)]

    # Start thread pool (one thread per introspector, using threads to
    # interrupt execution NOT for parallelism)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_introspectors) as executor:
        # Submit all the introspectors (and keep track of the order!)
        futures = {
            executor.submit(introspector, producer): i
            for i, (introspector, producer) in enumerate(zip(introspectors, producers))
        }

        # Wait for results
        results = {}
        for future in concurrent.futures.as_completed(futures.keys()):
            try:
                i = futures[future]
                results[i] = future.result()
            except Exception as e:
                splitter.signal_failure()
                raise DNIKitException(
                    "Encountered exception when processing multiple introspectors") from e
        # Check that there are results for every introspector
        for i in range(num_introspectors):
            assert i in results, f"{i}th introspector did not produce results"

    # Re-assemble results as a tuple
    return tuple(results[i] for i in range(num_introspectors))

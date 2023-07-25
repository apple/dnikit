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

import itertools
import logging
import pathlib
import pickle
import shutil
import tempfile

from ._batch._batch import Batch
from ._pipeline import PipelineStage
from ._producer import Producer, _resize_batches
from dnikit._logging import _Logged
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t


def _get_pickled_files(storage_path: pathlib.Path) -> t.Iterator[pathlib.Path]:
    return storage_path.glob("*.pkl")


def _get_cache_dir_marker(storage_path: pathlib.Path) -> pathlib.Path:
    return storage_path / ".dni_cache_dir"


def _get_caching_done_marker(storage_path: pathlib.Path) -> pathlib.Path:
    return storage_path / ".cache.done"


def _has_cached_files(storage_path: pathlib.Path) -> bool:
    return (
        _get_cache_dir_marker(storage_path).exists()
        or _get_caching_done_marker(storage_path).exists()
        or bool(list(_get_pickled_files(storage_path)))
    )


def _create_cache_dir(storage_path: pathlib.Path) -> None:
    if not storage_path.is_dir():
        storage_path.mkdir(parents=True, exist_ok=True)
    _get_cache_dir_marker(storage_path).touch(exist_ok=False)


def _save_batch(storage_path: pathlib.Path,
                logger: logging.Logger,
                batch: Batch,
                index: int) -> None:
    # Saves a Batch to disk as a pickle file (eg: '127.pkl' if index is 127)
    filename = storage_path / f"{index}.pkl"
    filename.write_bytes(pickle.dumps(batch))
    logger.debug(f"Pickled batch with index: {index}")


def _get_batch_loader(storage_path: pathlib.Path,
                      logger: logging.Logger) -> Producer:
    # Collect all pickled files
    pickled_paths = list(_get_pickled_files(storage_path))
    # Sort them by index (which happens to be the stem of the file)
    pickled_paths = sorted(pickled_paths, key=lambda x: int(x.stem))

    def file_batch_loader(paths: t.Sequence[pathlib.Path]) -> t.Iterable[Batch]:
        for path in paths:
            logger.debug(f"Loading batch from: {path}")
            yield pickle.loads(path.read_bytes())

    return _resize_batches(file_batch_loader(pickled_paths))


def _mark_caching_done(storage_path: pathlib.Path) -> None:
    _get_caching_done_marker(storage_path).touch()


def _done_caching(storage_path: pathlib.Path) -> bool:
    return _get_caching_done_marker(storage_path).exists()


@t.final
class Cacher(PipelineStage):
    """
    ``Cacher`` is a :class:`PipelineStage <dnikit.base.PipelineStage>` that will cache to disk
    the batches produced by the previous :class:`Producer <dnikit.base.Producer>` in a pipeline
    created with :func:`pipeline() <dnikit.base.pipeline>`.

    The first time a pipeline with a ``Cacher`` is executed, ``Cacher`` store the batches to disk.
    Every time the pipeline is called after that, batches will be read directly from disk, without
    doing any computation for previous stages.

    Note that batches may be quite large and this may require a large portion of available
    disk space. Be mindful when using ``Cacher``.

    If the data from the ``producer`` does not have
    :attr:`Batch.StdKeys.IDENTIFIER <dnikit.base.Batch.StdKeys.IDENTIFIER>`, this class
    will assign a numeric identifier.  This cannot be used across calls to ``Cacher``
    but will be consistent for all uses of the ``pipelined_producer``.

    Example:
        .. code-block:: python

            producer = ... # create a valid dnikit Producer
            processor = ... # create a valid dnikit Processor
            cacher = Cacher()

            # Pipeline everything
            pipelined_producer = pipeline(producer, processor, cacher)

            # No results have been cached
            cacher.cached  # returns False

            # Trigger pipeline
            batches = list(pipelined_producer(batch_size=32)) # producer and processor are invoked.

            # Results have been cached
            cacher.cached  # returns True

            # Trigger pipeline again (fast, because batch_size has the same value as before)
            list(pipelined_producer(batch_size=32))  # producer and processor are NOT invoked

            # Trigger pipeline once more (slower, because batch_size is different from first time)
            list(pipelined_producer(batch_size=48))  # producer and processor are NOT invoked


    The typical use-case for this class is to cache the results of expensive computation (such as
    inference and post-processing) to avoid re-doing said computation more than once.

    Note:
        Just as with :class:`Model <dnikit.base.Model>`, and :class:`Processor` no computation
        (or in this case, caching) will be executed until the pipeline is triggered.

    See also:
        :func:`dnikit.base.multi_introspect()` which allows several introspectors to
        use the same batches without storing them in the file-system.
        :func:`multi_introspect() <dnikit.base.multi_introspect()>` may
        be a better option for very large datasets.

    Warning:
        ``Cacher`` has the ability to resize batches if batches of different sizes are requested
        (see example). However, doing so is relatively computationally expensive since it involves
        concatenating and splitting batches. Therefore it's recommended to use this feature
        sparingly.

    Warning:
        Unlike other :class:`PipelineStage <dnikit.base.PipelineStage>`, ``Cacher`` will raise
        a :class:`DNIKitException <dnikit.exceptions.DNIKitException>` if it is used
        with more than one pipeline. This is to avoid
        reading batches generated from another pipeline with different characteristics.

    Args:
        storage_path: **[optional ]** If set, ``Cacher`` will store batches in `storage_path`,
            otherwise it will create a random temporary directory.
    """

    @staticmethod
    def clear(storage_path: t.Optional[pathlib.Path] = None) -> None:
        """
        Clears files produced by :class:`Cacher` and
        :class:`CachedProducer <dnikit.base.CachedProducer>`.

        Args:
            storage_path: if ``None`` (default case), function will clear **all** dnikit caches
                under a system's temporary directory. Otherwise it will clear **all** dnikit
                caches under the specified directory.

        Raises:
            NotADirectoryError: if ``storage_path`` is not a valid directory.

        Warning:
             Make sure to only call this function once pipelines are no longer needed (or before
             pipelines are used at all). Otherwise, a cache that is already in
             use may be destroyed!
        """
        storage_path = pathlib.Path(tempfile.gettempdir()) if storage_path is None else storage_path
        if not storage_path.is_dir():
            raise NotADirectoryError(f"{storage_path} is not a valid directory")

        # Gather paths for deletion
        to_delete = [
            path
            # Consider storage_path and its children as candidates for deletion
            for path in itertools.chain([storage_path], storage_path.iterdir())
            # Will delete directories that have a cache marker
            if path.is_dir() and _get_cache_dir_marker(path).exists()
        ]
        for path in to_delete:
            shutil.rmtree(path)

    def __init__(self, storage_path: t.Optional[pathlib.Path] = None):
        """
        Initialize a ``Cacher``.

        Args:
            storage_path: If set, ``Cacher`` will store batches in `storage_path`, otherwise it will
                create a random temporary directory.
        """
        if storage_path is None:
            self._storage_path = pathlib.Path(tempfile.mkdtemp(prefix="dnikit-cacher-")).resolve()
        else:
            self._storage_path = storage_path.resolve()

        self._already_pipelined = False
        self._current_identifier = 0

        if _has_cached_files(self._storage_path):
            raise DNIKitException(
                f"Path {self._storage_path} already contains caching files."
            )
        _create_cache_dir(self._storage_path)

    @property
    def storage_path(self) -> pathlib.Path:
        """The (absolute) path where the batches are being cached."""
        return self._storage_path

    @property
    def cached(self) -> bool:
        """True if all the batches have already been cached."""
        return _done_caching(self._storage_path)

    def _add_identifier(self, batch: Batch) -> Batch:
        if Batch.StdKeys.IDENTIFIER in batch.metadata:
            return batch

        # add a numeric Batch.StdKeys.IDENTIFIER
        start = self._current_identifier
        end = start + batch.batch_size
        self._current_identifier = end

        builder = Batch.Builder(base=batch)
        builder.metadata[Batch.StdKeys.IDENTIFIER] = list(range(start, end))

        return builder.make_batch()

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        # No need to implement this one since this is overriding pipeline
        raise DNIKitException('Should never call this function in CachedProducer')

    def _pipeline(self, producer: Producer) -> Producer:
        # Do NOT use the same cacher in more than one pipeline
        if self._already_pipelined:
            raise DNIKitException(
                "Cacher already used in a pipeline. Either create a new Cacher to cache results "
                "from another pipeline or call as_producer() to create a CachedProducer which "
                "to reuse the results from this pipeline."
            )
        self._already_pipelined = True

        # NB new_producer is technically stateful, but:
        # * _storage_path is final and cannot be modified after being set
        # * _logger doesn't have any state
        # * _already_pipelined won't change anymore
        def new_producer(batch_size: int) -> t.Iterable[Batch]:
            if self.cached:  # If results are cached load from disk...
                self.logger.info('Using cached batches. Attempting to retrieve values..')
                batch_loader = _get_batch_loader(self._storage_path, self.logger)
                yield from batch_loader(batch_size)
            else:  # Otherwise load from producer and save batches to disk
                for index, batch in enumerate(producer(batch_size)):
                    # attach Batch.StdKeys.IDENTIFIER if not present
                    batch = self._add_identifier(batch)
                    _save_batch(self._storage_path, self.logger, batch, index)
                    yield batch
                _mark_caching_done(self._storage_path)
        return new_producer

    def as_producer(self) -> "CachedProducer":
        """
        Get a :class:`CachedProducer <dnikit.base.CachedProducer>` which loads the batches
        stored by this ``Cacher``.

        Raises:
            DNIKitException: if called before caching has been completed.
        """
        if not self.cached:
            raise DNIKitException("Caching must be complete before converting to a CachedProducer.")
        return CachedProducer(storage_path=self.storage_path)


@t.final
class CachedProducer(Producer, _Logged):
    """
    ``CachedProducer`` is a :class:`Producer` that reads batches already cached on disk.

    Refer to :class:`Cacher <dnikit.processors.Cacher>` to cache the
    :class:`batches <Batch>` produced by other :class:`producers <Producer>` and
    :class:`pipelines <pipeline>`.

    Note:
        Just as with :class:`ImageProducer` (and, in general, any :class:`Producer`) no computation
        (or in this loading of cached batches) will be executed until the :func:`__call__` method
        (or an associated pipeline) is invoked.

    Warning:
        ``CachedProducer`` has the ability to resize batches if batches of
        different sizes are requested in comparison to the ones stored on disk.
        However, doing so is relatively computationally
        expensive since it involves concatenating and splitting batches. Therefore it's recommended
        to use this feature sparingly.

    Args:
        storage_path: path in disk where the cached batches are stored.

    Raises:
        DNIKitException: if ``storage_path`` does not contain cached batches.
    """

    def __init__(self, storage_path: pathlib.Path):
        self._storage_path = storage_path.resolve()
        if not _get_caching_done_marker(storage_path).exists():
            raise DNIKitException(
                f"{storage_path} does not contain cached batches. Cannot create CachedProducer."
            )

    @property
    def storage_path(self) -> pathlib.Path:
        """The path from where this ``CachedProducer`` will read batches."""
        return self._storage_path

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        """
        Produce :class:`Batch` with ``batch_size`` elements by loading previously cached batches
        from disk.

        Args:
            batch_size: number of elements in every batch to be streamed.

        Raises:
            ValueError: if ``batch_size`` is a non-positive number
            DNIKitException: if cached files have been erased from disk since ``CachedProducer``
                             initialization.
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size has to be a greater than 0, got {batch_size}")
        if not _done_caching(self._storage_path):
            raise DNIKitException("Batch data cleared since CachedProducer was initialised")

        self.logger.info('Using cached batches. Attempting to retrieve values..')
        batch_loader = _get_batch_loader(self._storage_path, self.logger)
        yield from batch_loader(batch_size)

    def copy_to(self, new_path: pathlib.Path, *, overwrite: bool = False) -> "CachedProducer":
        """
        Copy the cached files used by this ``CachedProducer`` to another local path.

        Args:
            new_path: Destination path
            overwrite: **[keyword arg, optional]** overwrite cache files in ``new_path``
                (if they exist) [default=False]

        Returns:
            a new ``CachedProducer`` which will read elements from ``new_path``.

        Raises:
            DNIKitException: if batch data has been cleared from file after initialization or
                if ``new_path`` contains cache files and ``overwrite`` is False.
        """
        if not _done_caching(self._storage_path):
            raise DNIKitException("Batch data cleared since CachedProducer was initialised")

        if new_path.exists():
            if _has_cached_files(new_path) and not overwrite:
                raise DNIKitException(
                    f"Path {new_path} already contains caching files. "
                    f"Either call with overwrite=True or choose another destination directory."
                )
        else:
            new_path.mkdir(parents=True)

        # Deal with cache file markers
        _get_cache_dir_marker(new_path).touch()
        _get_caching_done_marker(new_path).touch()

        # Copy all pickle files
        for filename in _get_pickled_files(self._storage_path):
            new_filename = new_path / filename.name
            new_filename.write_bytes(filename.read_bytes())

        return CachedProducer(new_path)

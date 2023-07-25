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
from typing import overload

import numpy as np

from ._fields import _Fields
from ._storage import _BatchStorage, _subset_batch
from . import _metadata_storage as _meta
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t
import dnikit.typing._dnikit_types as dt

# Used only for metadata typing
_T = t.TypeVar("_T")


def _clean_snapshots(snapshots: t.Mapping[str, "Batch"]) -> t.Mapping[str, "_BatchStorage"]:
    # Creates a shallow copy of snapshots. Used when going from Builder to Batch,
    # otherwise it'd be possible to break the invariance constraint on Batch (since Builder
    # would hold on to a mutable reference of the snapshots in Batch)
    # Additionally, it avoids recursive snapshots (a snapshot is not allowed to have snapshots).
    return {
        name: _BatchStorage(
            fields=_Fields(dict(snap.fields)),
            metadata=snap.metadata._storage
        )
        for name, snap in snapshots.items()
    }


def _get_meta_keys_view(storage: _meta._AnyMetadataStorage
                        ) -> t.KeysView[t.Union["Batch.MetaKey", "Batch.DictMetaKey"]]:
    # Cast result, assume there's a 1:1 correspondence between
    # _MetaKeyProtocol <-> Batch.MetaKey & _DictMetaKeyProtocol <-> Batch.DictMetaKey
    return t.cast(
        t.KeysView[t.Union[Batch.MetaKey, Batch.DictMetaKey]],
        storage.keys()
    )


@t.final
@dataclasses.dataclass(frozen=True)
class Batch:
    """
    ``Batch`` is DNIKit's class to store input and output data.

    At its most basic, a ``Batch`` contains as an immutable dictionary between a label and a
    :class:`numpy.ndarray`. This dictionary is stored in the ``fields`` attribute of ``Batch``.
    Each field in ``fields`` is then composed of a label and a numpy array, with the label
    describing the data contained in the :class:`numpy.ndarray` (the label may be named after the
    layer that produced the response or for the input data name). The numpy array contains the data
    with its zero-th dimension representing the number  of elements (ie dimension 0 is **always**
    the batch dimension).

    :attr:`Batch.fields` can be used just like a ``Mapping[str, np.ndarray]``:

    .. code-block:: python

        # Print fields present in Batch
        print(list(batch.fields.keys()))

        # Retrieve batch data
        images = batch.fields["input_images"] # type is numpy.ndarray

        # Loop through elements in batch
        for k, v in batch.fields.items():
            print(f"field {k} has shape {v.shape}")

    It's possible to create a new ``Batch`` by simply passing a dictionary (or any other mapping)
    from ``str`` to ``numpy.ndarray``. For more advanced options (like ``snapshots`` or
    ``metadata``), see below use on how to use :class:`Batch.Builder` instead.

    .. code-block:: python

        new_batch = Batch({"images" : numpy.zeros((32, 3, 64, 64))})

    Warning:
        The number of elements (ie. the batch size) of all :class:`numpy.ndarray` in a ``Batch``
        **must** be the same for all arrays.

    Lookup the number of elements in a ``Batch`` by calling :attr:`Batch.batch_size`.

    Additionally, ``Batch`` has two extra attributes for advanced users: :attr:`Batch.metadata` and
    :attr:`Batch.snapshots`.

    :attr:`Batch.snapshots` behaves like an immutable mapping from a label to another ``Batch`` and
    is often used in conjunction with :func:`pipeline()` and
    :class:`SnapshotSaver <dnikit.processors.SnapshotSaver>` to store the
    state of a ``Batch`` before transforming it.

    .. code-block:: python

        # Check if a snapshot is available in batch
        assert "origin" in batch.snapshots

        # Retrieve the snapshot
        origin_batch = batch.snapshots["origin"]  # type is Batch

        # origin_batch is a regular Batch and may contain different fields
        original_data = origin_batch.fields["data"]  # type is np.ndarray

        # The batch_size of a snapshot always matches the current batch's batch_size.
        assert batch.batch_size == origin_batch.batch_size

    :attr:`Batch.metadata` is used to store metadata about certain aspects of the batch. There are
    many types of metadata and three built-in metadata keys (see :class:`Batch.StdKeys`) that can
    be used to attach and use metadata. Custom metadata keys can also be defined
    (see :class:`Batch.MetaKey` and :class:`Batch.DictMetaKey`). To use metadata, it's
    necessary to have both a meta key, and –if using a :class:`Batch.DictMetaKey`– a
    string identifier.

    .. code-block:: python

        # Flat metadata
        # ---------------
        # Instantiate a Batch.MetaKey (key to store and retrieve flat metadata in a batch)
        META_KEY = Batch.MetaKey[int]("META_KEY")

        # Retrieve metadata
        flat_metadata = batch.metadata[META_KEY]  # type is Sequence[int]

        # The number of elements in metadata is always the number of elements in a batch.
        assert len(flat_metadata) == batch.batch_size

        # Dict metadata
        # ---------------
        # Instantiate a Batch.DictMetaKey (key to store and retrieve dict-metadata in a batch)
        DICT_META_KEY = Batch.DictMetaKey[float]("DICT_META_KEY")

        # Retrieve metadata
        dict_metadata = batch.metadata[DICT_META_KEY]["key"]  # type is Sequence[float]

        # The number of elements in metadata is always the number of elements in a batch.
        assert len(dict_metadata) == batch.batch_size

    Use :class:`Batch.Builder` to make *new* instances of ``Batch`` with ``metadata`` or
    ``snapshots``. Note that DNIKit ships with many processors to modify metadata and snapshots
    in existing Batches, such as :class:`SnapshotSaver <dnikit.processors.SnapshotSaver>` and
    :class:`SnapshotRemover <dnikit.processors.SnapshotRemover>`.

    Warning:
        The number of elements of every metadata and every snapshot **must** always be the same
        as the number of elements in **every field**.

    Note:
        To add ``snapshots`` or ``metadata`` use :class:`Batch.Builder`.

    Args:
        fields: initial values for ``Batch`` in the form of a ``Mapping`` (dictionary) from
            ``str`` to ``numpy.ndarray``.
        _storage: internal use only, do not use.

    Raises:
        ValueError: If ``Batch`` is initialized without any data.
        DNIKitException: If the zero-th dimension (ie batch size) of any field, any metadata or
            any snapshot do not agree.
    """

    # Data members of Batch
    _storage: _BatchStorage

    @overload
    def __init__(self, fields: t.Mapping[str, np.ndarray]): ...
    @overload
    def __init__(self, *, _storage: _BatchStorage): ...

    def __init__(self,
                 fields: t.Optional[t.Mapping[str, np.ndarray]] = None, *,
                 _storage: t.Optional[_BatchStorage] = None):
        if _storage is None and not fields:
            raise ValueError("Cannot initialize Batch without any fields")
        elif _storage is not None and fields is not None:
            raise ValueError("Cannot provide both `fields` and `storage` arguments")
        elif fields is not None:
            storage = _BatchStorage(fields=_Fields(dict(fields)))
        elif _storage is not None:
            storage = _storage
        else:
            assert False, "Unreachable code"

        # Set frozen instance properties
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "_storage", storage)
        # Check invariants & freeze numpy arrays
        self._storage.check_invariants()
        self._storage.freeze_arrays()

    @property
    def fields(self) -> t.Mapping[str, np.ndarray]:
        """
        Retrieve ``fields`` contained in this batch.

        A ``field`` is just a :class:`mapping <collections.abc.Mapping>` between a
        label (:class:`str`) and a value (:class:`numpy.ndarray`). Fields can images, audio samples,
        temporal sequences, model responses, etc..
        """
        return self._storage.fields

    @property
    def snapshots(self) -> t.Mapping[str, "Batch"]:
        """
        Retrieve ``snapshots`` associated with this batch.

        Snapshots are a mapping from a label to a batch (``Mapping[str, Batch]``).
        Note that snapshots are read-only. To add snapshots to a :func:`pipeline`
        refer to :class:`SnapshotSaver <dnikit.processors.SnapshotSaver>`.
        """
        return {
            k: Batch(_storage=v)
            for k, v in self._storage.snapshots.items()
        }

    @property
    def metadata(self) -> "Batch.MetadataType":
        """
        :class:`Metadata <Batch.Metadata>` associated with this batch.

        See description of :class:`Batch.MetadataType` to check which operations are supported.
        """
        return Batch.MetadataType(self._storage.metadata)

    @property
    def batch_size(self) -> int:
        """
        Current batch size (i.e., number of samples in batch).

        Note:
            Batch size will be the batch size for all fields in the batch, for all fields in all
            types of metadata and for all fields in all snapshots.
        """
        return self._storage.batch_size

    @property
    def elements(self) -> "Batch.ElementsView":
        """
        Attribute that enables traversal of this ``Batch`` by element index.

        See description of :class:`Batch.ElementsView` to check what operations are supported.
        """
        return Batch.ElementsView(self._storage)

    # Batch.ElementsView
    # ----------------------------------------------------------------------------------------------
    @t.final
    @dataclasses.dataclass(frozen=True)
    class ElementsView:
        """
        Class to use :class:`Batch <dnikit.base.Batch>` elements by index.

        Do not instantiate this class directly, instead use
        :attr:`Batch.elements <dnikit.base.Batch.elements>`.

        ``Batch.ElementsView`` allows for:
        :func:`traversal <__iter__>`, as well as
        :func:`indexing and slicing <__getitem__>`.
        """
        _storage: _BatchStorage

        @overload
        def __getitem__(self, index: int) -> "Batch.ElementType": ...
        @overload
        def __getitem__(self, indices: t.Sequence[int]) -> "Batch": ...
        @overload
        def __getitem__(self, indices: slice) -> "Batch": ...

        def __getitem__(self, selector: t.Union[int, slice, t.Sequence[int]]
                        ) -> t.Union["Batch", "Batch.ElementType"]:
            """
            Indexing and slicing operator for ``Batch.ElementsView``.

            :attr:`Batch.elements <dnikit.base.Batch.elements>` can also be used to directly index
            one or more elements of the batch.

            Indexing with a single :class:`int` returns a
            :class:`Batch.ElementType <dnikit.base.Batch.ElementType>` corresponding to that
            element. Otherwise, indexing with a ``Sequence[int]`` of indices returns
            a :class:`Batch <dnikit.base.Batch>` with the selected elements.

            Just like with :class:`lists <list>`, negative items are read as starting from the end
            of the elements.

            .. code-block:: python

                element = batch.elements[42]  # type is Batch.ElementType
                subset = batch.elements[-1, 1, 2, 3, 5, 8]  # type is Batch

            Finally, :attr:`Batch.elements <dnikit.base.Batch.elements>` enables slicing the
            batch as well. Slicing operations always return a :class:`Batch <dnikit.base.Batch>`.

            .. code-block:: python

                # Get elements 10, 12, 14...30 of the Batch
                subset = batch.elements[10:30:2]  # type is Batch

            Args:
                selector: specifies how to select subsamples from Batch. See prior example.

            Returns:
                :class:`Batch <dnikit.base.Batch>` with specified subset, or single
                :class:`Batch element <dnikit.base.Batch.ElementType>`.
            """
            if isinstance(selector, int):
                return Batch.ElementType(self._storage, selector)
            # For advanced selector
            new_storage = _subset_batch(self._storage, selector)
            return Batch(_storage=new_storage)

        def __iter__(self) -> t.Iterator["Batch.ElementType"]:
            """
            Iteration operator for :class:`Batch.ElementsView <dnikit.base.Batch.ElementsView>`.

            :attr:`Batch.elements <dnikit.base.Batch.elements>` can be used in a for-loop or to
            get an iterator. Each element of the iteration will be of type
            :class:`Batch.ElementType <dnikit.base.Batch.ElementType>`.

            .. code-block:: python

                for element in batch.elements:
                    assert isinstance(element, Batch.ElementType)
                    # Can retrieve fields, snapshots & metadata for a single element
                    element.fields["data"]
            """
            return _BatchIterator(self._storage)

        def __len__(self) -> int:
            return self._storage.batch_size

    # Batch.Element
    # ----------------------------------------------------------------------------------------------
    @t.final
    @dataclasses.dataclass(frozen=True)
    class ElementType:
        """
        Class to hold a single element from a :class:`Batch <dnikit.base.Batch>`.

        This class should not be instantiated directly. Instead, receive
        a ``Batch.ElementType`` by iterating or indexing
        :attr:`Batch.elements <dnikit.base.Batch.elements>`.

        Just like :class:`Batch <dnikit.base.Batch>`, ``Batch.ElementType`` has three attributes:
        :attr:`fields <dnikit.base.Batch.ElementType.fields>`,
        :attr:`snapshots <dnikit.base.Batch.ElementType.snapshots>`
        and :attr:`metadata <dnikit.base.Batch.ElementType.metadata>`.

        The main difference is that these properties will return data about a single element in the
        :class:`Batch <dnikit.base.Batch>`.
        """
        _storage: _BatchStorage
        _index: int

        def __post_init__(self) -> None:
            assert self._index < self._storage.batch_size

        @property
        def fields(self) -> t.Mapping[str, t.Union[np.ndarray, np.number]]:
            """
            Retrieve ``fields`` contained in this batch element.

            See :attr:`Batch.fields <dnikit.base.Batch.fields>` for more information.

            Note:
                The return type of this operation depends whether the underlying field
                has more than one dimension. If the field, has dimensions ``B x D1 x .. x Dn``
                the return type will be a :class:`numpy.ndarray` with shape ``D1 x .. x Dn``,
                otherwise if the field has dimensions ``B`` then this property will return
                a :class:`numpy.number`.
            """
            return _Fields({
                field: value[self._index]
                for field, value in self._storage.fields.items()
            })

        @property
        def snapshots(self) -> t.Mapping[str, "Batch.ElementType"]:
            """
            Retrieve ``snapshots`` contained in this batch element.

            See :attr:`Batch.snapshots <dnikit.base.Batch.snapshots>` for more information.
            """
            return {
                name: Batch.ElementType(snapshot, self._index)
                for name, snapshot in self._storage.snapshots.items()
            }

        @property
        def metadata(self) -> "Batch.MetadataType.ElementType":
            """
            Retrieve ``metadata`` contained in this batch element.

            See :attr:`Batch.metadata <dnikit.base.Batch.metadata>` for more information.
            Similarly check
            :attr:`Batch.MetadataType.ElementType <dnikit.base.Batch.MetadataType.ElementType>`
            to see which operations are supported.
            """
            return Batch.MetadataType.ElementType(self._storage.metadata, self._index)

        def __str__(self) -> str:
            print_string = {field: value[self._index]
                            for field, value in self._storage.fields.items()}
            return f"Batch.ElementType(_storage={print_string}, _index={self._index})"

    # Batch.MetaKey
    # ----------------------------------------------------------------------------------------------
    @t.final
    @dataclasses.dataclass(frozen=True)
    class MetaKey(t.Generic[_T], _meta._MetaKeyTrait):
        """
        Class to represent different types of simple metadata in :class:`Batch <dnikit.base.Batch>`,
        where simple means a single metadata value per data sample for this metadata key.

        Most of the time, ``Batch.MetaKey`` will only be used to store and retrieve
        metadata from a :class:`Batch <dnikit.base.Batch>`.
        See :attr:`Batch.metadata <dnikit.base.Batch.metadata>` and
        :class:`Batch.MetadataType <dnikit.base.Batch.MetadataType>` to
        see how to do so.

        Metadata keyed by ``Batch.MetaKey`` will be stored as a
        :class:`Sequence <collections.abc.Sequence>` of payloads of the same length as the
        :attr:`batch (batch_size) <dnikit.base.Batch.batch_size>`.

        ``Batch.MetaKey`` can be instantiated just like a regular Python class:

        .. code-block:: python

            key_untyped = Batch.MetaKey("example")

        It's also possible to provide type annotations to inform a type checker of the payload type:

        .. code-block:: python

            key_typed = Batch.MetaKey[int]("example2")

        Note:
            ``Batch.MetaKey`` is a generic type that allows static-type analyzers to keep track
            the metadata payload type. For instance:

            - ``Batch.MetaKey[bool]``: signifies that the metadata will be stored as a
              :class:`sequences<collections.abc.Sequence>` of :class:`bool` instances.

            - ``Batch.MetaKey[Hashable]``: signifies that the metadata will be stored as
              :class:`sequences<collections.abc.Sequence>` of
              :class:`Hashable<collections.abc.Hashable>` instances.

            This payload type information is only used by static-type checkers and will not
            be verified at runtime.

        Warning:
            Make sure the name of the ``Batch.MetaKey`` is unique.

        Args:
            name: Unique name for this ``Batch.MetaKey``
        """

        name: str
        """Unique name for this ``Batch.MetaKey``"""

    # Batch.DictMetaKey
    # ----------------------------------------------------------------------------------------------
    @t.final
    @dataclasses.dataclass(frozen=True)
    class DictMetaKey(t.Generic[_T], _meta._DictMetaKeyTrait):
        """
        Class to represent different types of dictionary metadata in
        :class:`Batch <dnikit.base.Batch>`.

        Most of the time, ``Batch.DictMetaKey`` will only be used to store and retrieve
        metadata from a :class:`Batch`. See :attr:`Batch.metadata <dnikit.base.Batch.metadata>`
        and :class:`Batch.MetadataType <dnikit.base.Batch.MetadataType>` to see how to do so.

        Metadata keyed by ``Batch.DictMetaKey`` will be stored as a
        :class:`Mapping<collections.abc.Mapping>` from :class:`strings<str>` to
        :class:`Sequence <collections.abc.Sequence>` of payloads of the same length as the
        :attr:`batch (batch_size) <dnikit.base.Batch.batch_size>`.

        ``Batch.DictMetaKey`` can be instantiated just like a regular Python class:

        .. code-block:: python

            key_untyped = Batch.DictMetaKey("example")

        It's also possible to provide type annotations to inform a type checker of the payload type:

        .. code-block:: python

            key_typed = Batch.DictMetaKey[int]("example2")

        Note:
            ``DictMetaKey`` is a generic type that allows static-type analyzers to keep track
            the metadata payload type. For instance:

            - ``Batch.DictMetaKey[int]``: signifies that the metadata will be stored as a
              ``Mapping[str, Sequence[int]]``.

            - ``Batch.DictMetaKey[t.Union[str, float]]``: signifies that the metadata will be stored
              as ``Mapping[str, Sequence[t.Union[str, float]]``.

            This payload type information is only used by static-type checkers and will not
            be verified at runtime.

        Warning:
            Make sure the name of the ``Batch.DictMetaKey`` is unique.

        Args:
            name: Unique name for this ``Batch.DictMetaKey``
        """

        name: str
        """Unique name for this ``Batch.DictMetaKey``."""

    # Batch.Metadata
    # ----------------------------------------------------------------------------------------------
    @t.final
    @dataclasses.dataclass(frozen=True)
    class MetadataType:
        """
        Class to store metadata associated with a :class:`Batch <dnikit.base.Batch>`.

        Most likely, ``Batch.MetadataType`` will only need to be used in conjunction with the
        :attr:`metadata attribute <dnikit.base.Batch.metadata>`. from
        :class:`Batch <dnikit.base.Batch>`.

        ``Batch.Metadata`` supports :func:`indexing <dnikit.base.Batch.MetadataType.__getitem__>`,
        :func:`membership tests <dnikit.base.Batch.MetadataType.__contains__>`, and
        :func:`truth value tests <dnikit.base.Batch.MetadataType.__bool__>`.
        """
        _storage: _meta._MetadataStorage = dataclasses.field(default_factory=dict)

        @overload
        def __getitem__(self, key: "Batch.MetaKey[_T]") -> t.Sequence[_T]: ...

        @overload
        def __getitem__(self, key: "Batch.DictMetaKey[_T]") -> t.Mapping[str, t.Sequence[_T]]: ...

        def __getitem__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> t.Any:
            """
            Indexing operator for ``Batch.MetadataType``.

            Indexing behaves differently depending on whether a
            :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` is used.

            If a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` is used, the metadata will be
            stored as a :class:`Sequence <collections.abc.Sequence>` of the payloads type
            indicated in the :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>`:

            .. code-block:: python

                # Obtain a key for MetaKey, with payload int
                FLAT_META_KEY = Batch.MetaKey[int]("FLAT_META_KEY")

                # Retrieve metadata associated with FLAT_META_KEY
                my_metadata = batch.metadata[FLAT_META_KEY]  # type is Sequence[int]

            On the other hand, if a :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`
            is used, the metadata will be stored
            as a mapping from ``str`` to a sequence of payloads (as declared in the
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`):

            .. code-block:: python

                # Obtain a key for DictMetaKey, with payload float
                DICT_META_KEY = Batch.DictMetaKey[float]("DICT_META_KEY")

                # Retrieve metadata associated with DICT_META_KEY
                my_metadata = batch.metadata[DICT_META_KEY]  # type is Mapping[str, Sequence[float]]

                # Retrieve a specific entry in metadata
                field_metadata = my_metadata["key"]  # type is Sequence[float]

            Args:
                key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                    :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to use
                    metadata in the batch
            """
            return _meta._get_metadata_item(self._storage, key)

        def __contains__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> bool:
            """
            Membership operator for ``Batch.MetadataType``.

            It's possible to check whether a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` are
            present in :attr:`Batch.metadata <dnikit.base.Batch.metadata>` just as expected
            in Python:

            .. code-block:: python

                FLAT_META_KEY = Batch.MetaKey[int]("FLAT_META_KEY")
                DICT_META_KEY = Batch.DictMetaKey[float]("DICT_META_KEY")

                flat_metadata_present = FLAT_META_KEY in batch.metadata
                dict_metadata_present = DICT_META_KEY in batch.metadata

            Args:
                key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                    :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to see if
                    it exists in the batch metadata
            """
            return key in self._storage

        def __bool__(self) -> bool:
            """
            Truth value operator for ``Batch.MetadataType``

            It's possible to check whether
            :attr:`Batch.metadata <dnikit.base.Batch.metadata>` is empty
            by requesting its truth value (will return ``False`` if empty, ``True`` otherwise):

            .. code-block:: python

                bool(batch.metadata)  # False if empty, True otherwise
            """
            return bool(self._storage)

        def keys(self) -> t.KeysView[t.Union["Batch.MetaKey", "Batch.DictMetaKey"]]:
            """
            Return all instances of :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` and
            :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` contained
            within this instance.

            Warning:
                The :class:`MetaKeys <dnikit.base.Batch.MetaKey>` and
                :class:`DictMetaKeys <dnikit.base.Batch.DictMetaKey>`
                returned will be **type-erased** which means a type checker will not understand
                the type of the payload. It's recommended to use this method for debugging purposes
                or adding the type information back with :func:`typing.cast`.

            Returns:
                a view of each :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` and
                :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` contained in
                the metadata.
            """
            return _get_meta_keys_view(self._storage)

        # Batch.MetadataType.ElementType
        # ------------------------------------------------------------------------------------------
        @t.final
        @dataclasses.dataclass(frozen=True)
        class ElementType:
            """
            Class to store metadata associated with a single
            :class:`Batch <dnikit.base.Batch>` element.

            As a user, ``Batch.MetadataType.ElementType`` will really only be encountered through
            the :attr:`metadata attribute <dnikit.base.Batch.ElementType.metadata>` of
            :class:`Batch.ElementType <dnikit.base.Batch.ElementType>` which itself will
            likely only be used through :attr:`Batch.elements <dnikit.base.Batch.elements>`.

            ``Batch.MetadataType.ElementType`` supports the same
            :func:`indexing <dnikit.base.Batch.MetadataType.ElementType.__getitem__>`,
            :func:`membership tests <dnikit.base.Batch.MetadataType.ElementType.__contains__>` and
            :func:`truth value tests <dnikit.base.Batch.MetadataType.ElementType.__bool__>` as
            :class:`Batch.MetadataType <dnikit.base.Batch.MetadataType>`.

            The main difference is that where
            :class:`Batch.MetadataType <dnikit.base.Batch.MetadataType>` returns a sequence of
            elements, ``Batch.MetadataType.ElementType`` will return a single item.
            """

            _storage: _meta._MetadataStorage
            _index: int

            @overload
            def __getitem__(self, meta_key: "Batch.MetaKey[_T]") -> _T:
                ...

            @overload
            def __getitem__(self, meta_key: "Batch.DictMetaKey[_T]") -> t.Mapping[str, _T]:
                ...

            def __getitem__(self, meta_key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> t.Any:
                """
                Indexing operator for ``Batch.MetadataType.ElementType``.

                Behaves the same as
                :func:`Batch.MetadataType's indexing <dnikit.base.Batch.MetadataType.__getitem__>`
                operator, but returns a single instance of the payload rather than a sequence.

                .. code-block:: python

                    # Instantiate a MetaKey and  DictMetaKey
                    FLAT_META_KEY = Batch.MetaKey[int]("FLAT_META_KEY")

                    DICT_META_KEY = Batch.DictMetaKey[float]("DICT_META_KEY")

                    # Get a single batch_element
                    batch_element = batch.elements[0]

                    # Retrieve metadata
                    simple_metadata = batch_element.metadata[SIMPLE_META_KEY]
                    # type is int
                    dict_metadata = batch_element.metadata[DICT_META_KEY]
                    # type is Mapping[str, float]

                Args:
                    key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                        :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to use
                        metadata in the batch element
                """
                return _meta._get_metadata_element_item(self._storage, meta_key, self._index)

            def __contains__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> bool:
                """
                Membership operator for ``Batch.MetadataType.ElementType``.

                Behaves the same as the
                :func:`Batch.MetadataType membership <dnikit.base.Batch.MetadataType.__contains__>`
                operator.

                .. code-block:: python

                    flat_meta_key_present = FLAT_META_KEY in batch_element.metadata
                    dict_meta_key_present = DICT_META_KEY in batch_element.metadata

                Args:
                    key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                        :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to see if
                        it exists in the batch element's metadata
                """
                return key in self._storage

            def __bool__(self) -> bool:
                """
                Truth value operator for ``Batch.MetadataType.ElementType``

                It's possible to check whether an instance of
                ``Batch.MetadataType.ElementType`` is empty by
                requesting its truth value (will return ``False`` if empty, ``True`` otherwise):

                .. code-block:: python

                    bool(batch_element.metadata)  # False if empty, True otherwise
                """
                return bool(self._storage)

            def keys(self) -> t.KeysView[t.Union["Batch.MetaKey", "Batch.DictMetaKey"]]:
                """
                Return all instances of :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>`
                and :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`
                contained within this instance.

                Warning:
                    The :class:`MetaKeys <dnikit.base.Batch.MetaKey>` and
                    :class:`DictMetaKeys <dnikit.base.Batch.DictMetaKey>` returned will be
                    **type-erased** which means a type checker will not understand the type
                    of the payload. It's recommended to use this method for debugging purposes or
                    adding the type information back with :func:`typing.cast`.

                Returns:
                    a view of each :class:`Batch.MetaKey` and :class:`Batch.DictMetaKey` contained
                    in this ``Batch.MetadataType.ElementType``.
                """
                return _get_meta_keys_view(self._storage)

            def __str__(self) -> str:
                print_string = {field: value[self._index] for key in self._storage
                                for field, value in self._storage[key].items()}
                return (
                    f"Batch.MetadataType.ElementType(_storage={print_string}, "
                    f"_index={self._index})"
                )

    # Batch.Builder
    # ----------------------------------------------------------------------------------------------
    @t.final
    @dataclasses.dataclass
    class Builder:
        """
        ``Batch.Builder`` is a helper class to aid in the creation of new
        :class:`Batch <dnikit.base.Batch>` instances.

        From a high-level perspective, ``Batch.Builder`` has similar capabilities to
        :class:`Batch <dnikit.base.Batch>`, but all its attributes are **mutable**.

        That means that it's possible to directly add or modify ``fields`` in the builder:

        .. code-block:: python

            # Create a new builder
            builder = Batch.Builder()

            # Add a new field
            builder.fields["images"] = numpy.zeros((32, 64, 64, 3))

            # Modify existing field
            builder.fields["images"][:, :, :, 0] = 1.0

            # Remove field
            del builder.fields["images"]

        Similarly, it's possible to directly add or modify ``snapshots`` or ``metadata``:

        .. code-block:: python

            # Remove an existing snapshot
            del builder.snapshots["origin"]

            # Add a new snapshot
            builder.snapshots["snapshot"] = previous_batch

            # Add metadata
            FLAT_META_KEY = Batch.MetaKey[int]("FLAT_META_KEY")
            builder.metadata[FLAT_META_KEY] = list(range(32))

        After modifying ``Batch.Builder``, to obtain a fully-baked
        :class:`Batch <dnikit.base.Batch>`,
        call :func:`make_batch() <dnikit.base.Batch.Builder.make_batch>`.

        **Note:** the resulting ``Batch`` must have non-empty ``fields`` or ``metadata``.

        Arguments:
            fields: **[optional]** initial values for this ``Batch.Builder``.
            base: **[keyword arg, optional]** ``Batch`` instance whose fields, metadata
                and snapshots will be copied into this ``Batch.Builder``. This is useful to modify
                only a few aspects of a ``Batch`` but leave most of it intact. Using this
                argument alongside ``fields`` is not allowed.
        """
        fields: t.MutableMapping[str, np.ndarray]
        """
        Retrieve and set ``fields`` associated with this builder.

        This is a mutable version of :attr:`Batch.fields <dnikit.base.Batch.fields>`.
        """

        snapshots: t.MutableMapping[str, "Batch"]
        """
        Retrieve and set ``snapshots`` associated with this builder.

        This is a mutable version of :attr:`Batch.snapshots <dnikit.base.Batch.snapshots>`.
        """

        metadata: "Batch.Builder.MutableMetadataType"
        """
        :class:`metadata <dnikit.base.Batch.Builder.MutableMetadataType>`
        associated with this builder.

        This is a mutable version of :attr:`Batch.metadata <dnikit.base.Batch.metadata>`. See
        :class:`Batch.Builder.MutableMetadataType <dnikit.base.Batch.Builder.MutableMetadataType>`
        for a description of operation supported.
        """

        def __init__(self,
                     fields: t.Optional[t.MutableMapping[str, np.ndarray]] = None, *,
                     base: t.Optional["Batch"] = None) -> None:

            if fields is not None and base is not None:
                raise ValueError("Use either `fields` or `base` argument, not both")

            # catch Batch.Builder(batch) which looks correct but is using batch
            # as a generic dictionary (not permitted)
            if isinstance(fields, Batch):
                raise ValueError("Batch.Builder(batch) is not supported -- use "
                                 "Batch.Builder(base=batch) to create a builder from an "
                                 "existing batch.")

            self.fields = dict(fields) if fields is not None else {}
            self.snapshots = {}
            self.metadata = Batch.Builder.MutableMetadataType()
            if base is not None:
                self.fields.update(base.fields)
                self.snapshots.update(base.snapshots)
                _meta._update_metadata_storage(self.metadata._storage, base.metadata._storage)

        def make_batch(self) -> "Batch":
            """
            Return a fully-baked (and immutable) :class:`Batch <dnikit.base.Batch>` with the same
            data, metadata and snapshots as this instance of ``Batch.Builder``.
            """
            return Batch(_storage=_BatchStorage(
                fields=_Fields(dict(self.fields)),
                snapshots=_clean_snapshots(self.snapshots),
                metadata=_meta._copy_metadata_storage(self.metadata._storage)
            ))

        # Batch.Builder.Metadata
        # ------------------------------------------------------------------------------------------
        @t.final
        @dataclasses.dataclass
        class MutableMetadataType:
            """
            Class to store **mutable** metadata associated with a
            :class:`Batch.Builder <dnikit.base.Batch.Builder>`.

            Most likely ``Batch.Builder.MutableMetadataType`` will only have to be used in
            conjunction with the :attr:`metadata attribute <dnikit.base.Batch.Builder.metadata>`
            from :class:`Batch.Builder <dnikit.base.Batch.Builder>`.

            Just like :class:`Batch.MetadataType <dnikit.base.Batch.MetadataType>`,
            ``Batch.Builder.MutableMetadataType`` supports
            :func:`indexing <dnikit.base.Batch.Builder.MutableMetadataType.__getitem__>`,
            :func:`membership tests <dnikit.base.Batch.Builder.MutableMetadataType.__contains__>`,
            and :func:`truth value tests <dnikit.base.Batch.Builder.MutableMetadataType.__bool__>`.
            It also adds the ability to
            :func:`modify values <dnikit.base.Batch.Builder.MutableMetadataType.__setitem__>` and
            :func:`delete items <dnikit.base.Batch.Builder.MutableMetadataType.__delitem__>`.
            """
            _storage: _meta._MutableMetadataStorage = dataclasses.field(
                init=False,
                default_factory=_meta._new_mutable_metadata_storage
            )

            def _rename_fields(self,
                               mapping: t.Mapping[str, str],
                               meta_keys: t.AbstractSet["Batch.DictMetaKey"]) -> None:
                """
                Rename the keys associated with each
                :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>` to the mapping from old to
                new. If ``meta_keys`` is not ``None``, operation will only be applied to specified
                meta_keys.
                """
                _meta._rename_dict_metakey_fields(self._storage, mapping, meta_keys)

            def _remove_simple_meta_keys(self,
                                         meta_keys: t.AbstractSet["Batch.MetaKey"],
                                         keep: bool) -> None:
                """
                Delete the metadata associated with the given keys/field combination. If ``keep`` is
                ``True`` then only the mentioned keys/fields will be kept.
                """
                _meta._remove_meta_keys(self._storage, meta_keys, keep)

            def _remove_dict_meta_keys(self,
                                       meta_keys: t.AbstractSet["Batch.DictMetaKey"],
                                       keys: t.Optional[t.AbstractSet[str]],
                                       keep: bool) -> None:
                """
                Delete the metadata associated with the given keys/field combination. If ``keep`` is
                ``True`` then only the mentioned keys/fields will be kept.
                """
                _meta._remove_dict_meta_keys(self._storage, meta_keys, keys, keep)

            @overload
            def __getitem__(self, key: "Batch.MetaKey[_T]") -> t.MutableSequence[_T]:
                ...

            @overload
            def __getitem__(self, key: "Batch.DictMetaKey[_T]"
                            ) -> t.MutableMapping[str, t.MutableSequence[_T]]:
                ...

            def __getitem__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> t.Any:
                """
                Indexing operator for ``Batch.Builder.MutableMetadataType``.

                Behaves the same as the
                :func:`Batch.MetadataType indexing <dnikit.base.Batch.MetadataType.__getitem__>`
                operator (but returning mutable versions of the containers).

                .. code-block:: python

                    # Instantiate a MetaKey, ...
                    FLAT_META_KEY = Batch.MetaKey[int]("FLAT_META_KEY")

                    # ... and a DictMetaKey
                    DICT_META_KEY = Batch.DictMetaKey[float]("DICT_META_KEY")

                    # Retrieve metadata
                    flat_metadata = builder.metadata[FLAT_META_KEY]
                    # type is MutableSequence[int]
                    dict_metadata = builder.metadata[DICT_META_KEY]
                    # type is MutableMapping[str, MutableSequence[int]]

                Args:
                    key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                        :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to use
                        metadata in the batch
                """
                return _meta._get_metadata_item(self._storage, key)

            @overload
            def __setitem__(self, key: "Batch.MetaKey[_T]", value: t.Sequence[_T]) -> None:
                ...

            @overload
            def __setitem__(self, key: "Batch.DictMetaKey[_T]",
                            value: t.Mapping[str, t.Sequence[_T]]) -> None:
                ...

            def __setitem__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"],
                            value: t.Any) -> None:
                """
                Mutation operator for ``Batch.Builder.MutableMetadataType``.

                The behavior of metadata mutation varies depending whether a
                :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` is used.

                Metadata keyed by
                :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` can be set with:

                .. code-block:: python

                    # Instantiate a MetaKey
                    FLAT_META_KEY = Batch.MetaKey[int]("FLAT_META_KEY")

                    # Set metadata for FLAT_META_KEY
                    builder.metadata[FLAT_META_KEY] = [2, 7, 1, 8, ...]

                If using :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`,
                the metadata can be set like this:

                .. code-block:: python

                    # And a DictMetaKey
                    DICT_META_KEY = Batch.DictMetaKey[float]("DICT_META_KEY")

                    # Set all keys and values for a DICT_META_KEY
                    builder.metadata[DICT_META_KEY] = {
                        "digits": [1., 2., 3., 4., ...]
                        "tens": [10., 20., 30., 40., ...]
                    }
                    # Set a particular key for DICT_META_KEY
                    builder.metadata[DICT_META_KEY]["hundreds"] = [100., 200., 300., 400., ...]

                Args:
                    key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                        :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to indicate
                        what key to list the metadata under
                    value: metadata value to be used with ``key``
                """
                _meta._set_metadata_item(self._storage, key, value)

            def __delitem__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> None:
                """
                Deletion operator for ``Batch.Builder.MutableMetadataType``.

                The `del` keyword can be used in combination with normal indexing rules to delete
                any metadata elements:

                .. code-block:: python

                    # Delete FLAT_META_KEY
                    del builder.metadata[FLAT_META_KEY]
                    # Delete DICT_META_KEY
                    del builder.metadata[DICT_META_KEY]
                    # Delete a specific entry from a DICT_META_KEY
                    del builder.metadata[DICT_META_KEY]["key"]

                Args:
                    key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                        :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to indicate
                        which of the batch metadata to delete
                """
                del self._storage[key]

            def __contains__(self, key: t.Union["Batch.MetaKey", "Batch.DictMetaKey"]) -> bool:
                """
                Membership operator for ``Batch.Builder.MutableMetadataType``.

                Behaves the same as the
                :func:`Batch.MetadataType membership <dnikit.base.Batch.MetadataType.__contains__>`
                operator.

                .. code-block:: python

                    flat_meta_key_present = FLAT_META_KEY in builder.metadata
                    dict_meta_key_present = DICT_META_KEY in builder.metadata

                Args:
                    key: a :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` or a
                        :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>`, to see if
                        it exists in the batch metadata

                """
                return key in self._storage

            def __bool__(self) -> bool:
                """
                Truth value operator for ``Batch.Builder.MutableMetadataType``

                It's possible to whether an instance of
                ``Batch.Builder.MutableMetadataType`` is empty
                by requesting its truth value (will return ``False`` if empty, ``True`` otherwise):

                .. code-block:: python

                    bool(builder.metadata)  # False if empty, True otherwise
                """
                return bool(self._storage)

            def keys(self) -> t.KeysView[t.Union["Batch.MetaKey", "Batch.DictMetaKey"]]:
                """
                Return all instances of :class:`Batch.MetaKey` and :class:`Batch.DictMetaKey`
                contained within this instance.

                Warning:
                    The :class:`MetaKeys <dnikit.base.Batch.MetaKey>` and
                    :class:`DictMetaKeys <dnikit.base.Batch.DictMetaKey>` returned will be
                    **type-erased** which means a type checker will not understand the type of
                    the payload. It's recommended to use this method for debugging purposes or
                    to add the type information back with :func:`typing.cast`.

                Returns:
                    a view of each :class:`Batch.MetaKey <dnikit.base.Batch.MetaKey>` and
                    :class:`Batch.DictMetaKey <dnikit.base.Batch.DictMetaKey>` contained
                    in this ``Batch.Builder.MutableMetadataType``.
                """
                return _get_meta_keys_view(self._storage)

    # Batch.StdKeys
    # ----------------------------------------------------------------------------------------------
    @t.final
    class StdKeys:
        """
        Standard metadata keys that are useful across many domains.
        """

        # Note: see the end of the file for where these are initialized.
        # In the class scope it's not possible to refer to the outer type (Batch)
        # so these can't be initialized here.  A method scope _can_
        # refer to the outer type FWIW.

        IDENTIFIER: "Batch.MetaKey[t.Hashable]"
        """
        Metadata key that uniquely identifies a data element.  This should only be
        used to provide a unique identifier for each data element.  Code that
        needs to store per-element data can use this value as a key, for
        example in a dictionary.

        Use case 1:

        CIFAR image data is loaded from numpy arrays.  It would be reasonable
        to use the index into those arrays as the ``IDENTIFIER``.

        Use case 2:

        Images are loaded from files.  As long as files are not duplicated
        in the elements, the path to the file would make a good identifier.
        Note that it is better to use the path than the filename (the last
        part of the path) because that may not be unique enough.  Consider
        also storing the path in :attr:`PATH`.

        Use case 3:

        Face detection use cases might have a file and a crop rect.  The file
        may be used multiple times: once for each face.  In this case using
        a tuple to hold the path and crop rect will provide a unique identifier.
        Other metadata fields should be used if it's also necessary to use a
        crop rect or filename apart from the unique identifier (in particular
        do not try to decompose the ``IDENTIFIER``):

        .. code-block:: python

            builder.metadata[Batch.StdKeys.IDENTIFIER] = [
                (d.path, d.crop_x, d.crop_y, d.crop_w, d.crop_h)
                for d in dataset
            ]
            builder.metadata[Batch.StdKeys.PATH] = [
                d.path
                for d in dataset
            ]

            # probably a custom key for the crop rect as well
            CROP = Batch.MetaKey[t.Tuple[int, int, int, int]]("crop")
            builder.metadata[CROP] = [
                (d.crop_x, d.crop_y, d.crop_w, d.crop_h)
                for d in dataset
            ]

        Use case 4:

        If there isn't any suitable natural identifier, a UUID or sequence value (int)
        is a good way to go.

        The built in :class:`ImageProducer <dnikit.base.ImageProducer>` can provide both
        ``IDENTIFIER`` and :attr:`PATH`.  The built in :class:`Cacher <dnikit.processors.Cacher>`
        can provide a unique sequence ``IDENTIFIER`` for any producers.
        """

        PATH: "Batch.MetaKey[dt.PathOrStr]"
        """
        Metadata key that identifies the path (either a ``str`` or a :class:`pathlib.Path`)
        that the element represents.  This can be used by a
        :class:`PipelineStage <dnikit.base.PipelineStage>`
        to load data, or simply be used to associate results with an input file.

        The built in :class:`ImageProducer <dnikit.base.ImageProducer>` can provide both
        :attr:`IDENTIFIER` and ``PATH``.
        """

        LABELS: "Batch.DictMetaKey[t.Hashable]"
        """
        Metadata key to set of labels. Maps label_category_name (e.g. shape) to label (e.g. square).

        Example:
            For example, samples may be labeled across multiple dimensions:

            .. code-block:: python

                builder.metadata[Batch.StdKeys.LABELS] = {
                    "shape": [ "square", "square", "triangle", ... ],
                    "color": [ "blue", "red", "green", ... ],
                }

                labels_shapes = batch.metadata[Batch.StdKeys.LABELS]["shape"]
                labels_colors = batch.metadata[Batch.StdKeys.LABELS]["color"]
        """

        def __init__(self) -> None:
            raise DNIKitException("Do not instantiate Batch.StdKeys")


# _BatchIterator private helper
# --------------------------------------------------------------------------------------------------
@t.final
@dataclasses.dataclass
class _BatchIterator(t.Iterator[Batch.ElementType]):
    _storage: _BatchStorage
    _index: int = -1

    def __next__(self) -> Batch.ElementType:
        self._index += 1
        if self._index >= self._storage.batch_size:
            raise StopIteration()
        return Batch.ElementType(self._storage, self._index)


# initialize StdKeys -- unable to do this inline because of type visibility issues
Batch.StdKeys.IDENTIFIER = Batch.MetaKey[t.Hashable]('dnikit.base.Batch.StdKeys.identifier')
Batch.StdKeys.PATH = Batch.MetaKey[dt.PathOrStr]('dnikit.base.Batch.StdKeys.path')
Batch.StdKeys.LABELS = Batch.DictMetaKey[t.Hashable]('dnikit.base.Batch.StdKeys.labels')

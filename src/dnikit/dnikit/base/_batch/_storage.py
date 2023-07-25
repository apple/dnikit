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

import dataclasses
import itertools

import numpy as np

from ._fields import _Fields
from ._metadata_storage import _MetadataStorage
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t

_T = t.TypeVar("_T")
_Selector = t.Union[slice, t.Sequence[int]]


# BatchStorage definition
# ------------------------------------------------------------------------------
@t.final
@dataclasses.dataclass(frozen=True)
class _BatchStorage:
    fields: _Fields
    snapshots: t.Mapping[str, "_BatchStorage"] = dataclasses.field(default_factory=dict)
    metadata: _MetadataStorage = dataclasses.field(default_factory=dict)
    batch_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if len(self.fields):
            batch_size = len(next(iter(self.fields.values())))
        elif len(self.metadata):
            some_metadata = next(iter(self.metadata.values()))
            batch_size = len(next(iter(some_metadata.values())))
        else:
            raise DNIKitException("Must have non-empty fields or metadata.")

        object.__setattr__(self, "batch_size", batch_size)

    def freeze_arrays(self) -> None:
        for array in self.fields.values():
            array.flags.writeable = False

    def check_invariants(self) -> None:
        """
        Check that all fields in the Batch have same length
        """
        # Check invariants for batch._data
        for array in self.fields.values():
            if len(array) != self.batch_size:
                raise DNIKitException(
                    "This batch appears to have been corrupted."
                    "Its data fields do not all have the same length."
                )
        # Check invariants for batch.snapshots
        for snapshot_name, snapshot_batch in self.snapshots.items():
            if snapshot_batch.batch_size != self.batch_size:
                raise DNIKitException(
                    f"Batch for {snapshot_name} snapshot has invalid size. "
                    f"Got {snapshot_batch.batch_size}, expected {self.batch_size}"
                )
            if snapshot_batch.snapshots:
                raise DNIKitException(f"Snapshot {snapshot_name} contains snapshots itself.")

        # Check invariants for batch metadata
        for meta_key in self.metadata:
            for field, values in self.metadata[meta_key].items():
                if len(values) != self.batch_size:
                    raise DNIKitException(
                        f"Metadata field {field} of keyed by {meta_key} "
                        f"should have length {self.batch_size} instead of length {len(values)}"
                    )


# _concatenate_batch() definition and helpers
# ------------------------------------------------------------------------------
def _check_compatible(batches: t.Sequence[_BatchStorage]) -> None:
    # Alias to simplify code
    first = batches[0]
    # Ensure all fields are the same
    fields = frozenset(first.fields.keys())
    same_fields = all(
        fields == frozenset(b.fields.keys())
        for b in batches
    )
    if not same_fields:
        raise ValueError("Cannot concatenate batches with different fields")

    # Do the same for snapshots
    snapshots = frozenset(first.snapshots.keys())
    same_snapshots = all(
        snapshots == frozenset(b.snapshots.keys())
        for b in batches
    )
    if not same_snapshots:
        raise ValueError("Cannot concatenate batches with different snapshots")

    # Verify same metadata keys
    meta_keys = frozenset(first.metadata.keys())
    same_meta_keys = all(
        meta_keys == frozenset(b.metadata.keys())
        for b in batches
    )
    if not same_meta_keys:
        raise ValueError("Cannot concatenate batches with different metadata")

    # check all fields in metadata are the same
    for meta_key in meta_keys:
        meta_fields = frozenset(first.metadata[meta_key].keys())
        same_meta_fields = all(
            meta_fields == frozenset(b.metadata[meta_key].keys())
            for b in batches
        )
        if not same_meta_fields:
            raise ValueError("Cannot concatenate batches with different metadata fields.")


def _flatten_sequences(x: t.Iterable[t.Sequence[_T]]) -> t.Sequence[_T]:
    return list(itertools.chain.from_iterable(x))


def _concatenate_batches(batches: t.Sequence[_BatchStorage]) -> _BatchStorage:
    if not batches:
        raise ValueError("No batches passed to concatenate_batch")
    elif len(batches) == 1:
        return batches[0]

    _check_compatible(batches)

    # Alias to retrieve keys
    first = batches[0]

    # Concatenate fields
    fields = _Fields({
        f: np.concatenate([b.fields[f] for b in batches], axis=0)
        for f in first.fields.keys()
    })

    # Concatenate snapshots -- by recursively calling this function with every snapshot
    snapshots = {
        snapshot: _concatenate_batches([b.snapshots[snapshot] for b in batches])
        for snapshot in first.snapshots.keys()
    }

    # Concatenate metadata
    metadata = {
        meta_key: {
            key: _flatten_sequences(b.metadata[meta_key][key] for b in batches)
            for key in first.metadata[meta_key].keys()
        }
        for meta_key in first.metadata.keys()
    }

    return _BatchStorage(fields=fields, snapshots=snapshots, metadata=metadata)


# _subset_batch() definition and helpers
# ------------------------------------------------------------------------------
def _validate_selector(storage: _BatchStorage, selector: _Selector) -> None:
    if isinstance(selector, t.Sequence) and selector:
        if max(selector) >= storage.batch_size or min(selector) <= -storage.batch_size:
            raise IndexError(
                f"Selector {selector} out of range in batch with {storage.batch_size} elements"
            )


def _subset_sequence(seq: t.Sequence[_T], selector: _Selector) -> t.Sequence[_T]:
    if isinstance(selector, slice):
        return seq[selector]
    else:
        return [seq[i] for i in selector]


def _subset_batch(storage: _BatchStorage, selector: _Selector) -> _BatchStorage:
    # selector can be either an sequence of ints or a slice (a python builtin with
    # start, stop, step), either is used to create subset of the original storage.

    # Make sure selector is within range
    _validate_selector(storage, selector)

    # Fields from selected data samples
    fields = _Fields({
        field: value[selector, ...] for field, value in storage.fields.items()
    })

    # Add snapshots for select data samples -- call this function for every snapshot in batch
    snapshots = {
        name: _subset_batch(snap, selector)
        for name, snap in storage.snapshots.items()
    }

    # Add all metadata for specified data samples
    metadata = {
        meta_key: {
            key: _subset_sequence(value, selector)
            for key, value in meta_value.items()
        }
        for meta_key, meta_value in storage.metadata.items()
    }

    return _BatchStorage(fields=fields, snapshots=snapshots, metadata=metadata)

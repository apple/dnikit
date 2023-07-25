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
import enum
from functools import partial

import numpy as np

from ._base_processor import Processor
from dnikit.base import Batch, PipelineStage
from dnikit._dict_utils import rename_keys, delete_keys
import dnikit.typing as dt
import dnikit.typing._types as t


@t.final
class MeanStdNormalizer(Processor):
    """
    A :class:`Processor` that standardizes a :attr:`field <dnikit.base.Batch.fields>` of
    a :class:`Batch <dnikit.base.Batch>` by subtracting the mean and adjusting the standard
    deviation to 1.

    More precisely, if ``x`` is the data to be processed, the following processing
    is applied: ``(x - mean) / std``.

    Args:
        mean: **[keyword arg]** The mean to be applied
        std: **[keyword arg]** The standard deviation to be applied
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>` name,
            or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be processed.
            If ``fields`` param is ``None``, then all :attr:`fields <dnikit.base.Batch.fields>`
            will be processed.
    """

    def __init__(self, *, mean: float, std: float, fields: dt.OneManyOrNone[str] = None):
        def func(data: np.ndarray) -> np.ndarray:
            return np.divide(data - mean, std)

        super().__init__(func, fields=fields)


@t.final
class Transposer(Processor):
    """
    A :class:`Processor` that transposes dimensions in a data
    :attr:`field <dnikit.base.Batch.fields>` from a :class:`Batch <dnikit.base.Batch>`.
    This processor will reorder the dimensions of the data as specified in the :attr:`dim` param.

    Example:
        To reorder ``NCHW`` to ``NHWC`` (or vice versa), specify
        ``Transposer(dim=[0,3,1,2])``

    Args:
        dim: **[keyword arg]** the new order of the dimensions.  It is illegal to reorder the
            0th dimension.
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>` name,
            or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be transposed.
            If ``fields`` param is ``None``, then all :attr:`fields <dnikit.base.Batch.fields>`
            will be transposed.

    See also:
        :func:`numpy.transpose`

    Raises:
        ValueError: if input specifies reordering the 0th dimension
    """

    def __init__(self, *,
                 dim: t.Sequence[int],
                 fields: dt.OneManyOrNone[str] = None):

        if dim[0] != 0:
            raise ValueError("Unable to move the 0th (batch) dimension.")

        def func(data: np.ndarray) -> np.ndarray:
            return np.transpose(data, axes=dim)

        super().__init__(func, fields=fields)


@t.final
class FieldRemover(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that removes some
    :attr:`fields <dnikit.base.Batch.fields>` from a :class:`Batch <dnikit.base.Batch>`.

    Args:
        fields: **[keyword arg]** a single :attr:`field <dnikit.base.Batch.fields>` name, or an
            iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be removed.
        keep: **[keyword arg, optional]** if True, the ``fields`` input will be kept and all other
            will be removed
    """

    def __init__(self, *, fields: dt.OneOrMany[str], keep: bool = False):
        super().__init__()
        self._fields = dt.resolve_one_or_many(fields, str)
        self._keep = keep

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)
            delete_keys(builder.fields, keys=self._fields, keep=self._keep)

            self.logger.debug(
                f"Result of FieldRemover has fields {list(builder.fields.keys())}"
            )
            return builder.make_batch()
        return batch_processor


@t.final
class FieldRenamer(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that renames some
    :attr:`fields <dnikit.base.Batch.fields>` from a :class:`Batch <dnikit.base.Batch>`.

    Args:
        mapping: a dictionary (or similar) whose keys are the old
            :attr:`field <dnikit.base.Batch.fields>` names
            and values are the new :attr:`field <dnikit.base.Batch.fields>` names.
    """
    def __init__(self, mapping: t.Mapping[str, str]):
        self._mapping = mapping

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)
            builder.fields = rename_keys(builder.fields, self._mapping)
            self.logger.debug(
                f"Result of FieldRenamer has fields {frozenset(builder.fields.keys())}"
            )
            return builder.make_batch()
        return batch_processor


@t.final
class Flattener(Processor):
    """
    A :class:`Processor` that collapses array of shape ``BxN1xN2x..`` into ``BxN``

    Args:
        order: **[optional]** {``C``, ``F``, ``A``, ``K``}:

            ``C`` (default) means to flatten in row-major (C-style) order.

            ``F`` means to flatten in column-major (Fortran-style) order.

            ``A`` means to flatten in column-major order if it is Fortran contiguous in
            memory, row-major order otherwise.

            ``K`` means to flatten in the order the elements occur in memory.

        fields: **[optional]** a single :attr:`field <dnikit.base.Batch.fields>` name, or an
            iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be resized. If the
            ``fields`` param is ``None``, then all the :attr:`fields <dnikit.base.Batch.fields>`
            in the :class:`batch <dnikit.base.Batch>` will be resized.

    Raises:
        ValueError: if ``order`` param is not one of {``C``, ``F``, ``A``, ``K``}
    """

    def __init__(self, order: str = 'C', fields: dt.OneManyOrNone[str] = None) -> None:

        if order not in ['C', 'F', 'A', 'K']:
            raise ValueError(
                "``order`` param for ``Flattener`` must be `C`, `F`, `A`, or `K`}")

        def func(data: np.ndarray) -> np.ndarray:
            if order == 'C':
                return data.reshape((data.shape[0], -1))
            else:
                return np.array([d.flatten(order) for d in data])

        super().__init__(func, fields=fields)


@t.final
@dataclasses.dataclass(frozen=True)
class SnapshotSaver(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that attaches the current
    :class:`Batch <dnikit.base.Batch>` as the :attr:`snapshot <dnikit.base.Batch.snapshots>`.

    Args:
        save: **[optional]** see :attr:`save`
        fields: **[optional]** see :attr:`fields`
        keep: **[optional]** see :attr:`keep`
    """

    save: str = "snapshot"
    """save the current state of the :class:`batches <dnikit.base.Batch>` under the given key."""

    fields: dt.OneManyOrNone[str] = None
    """
    Optional list of :attr:`fields <dnikit.base.Batch.fields>` to include/remove in the
    saved :attr:`snapshot <dnikit.base.Batch.snapshots>` or ``None`` for all."""

    keep: bool = True
    """If ``True``, the ``fields`` list are the fields to keep, if ``False``, the ones to omit."""

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        fields = dt.resolve_one_many_or_none(self.fields, str)

        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)

            snapshot = Batch.Builder(base=batch)
            if fields:
                delete_keys(snapshot.fields, keys=fields, keep=self.keep)

            builder.snapshots[self.save] = snapshot.make_batch()

            return builder.make_batch()
        return batch_processor


@t.final
@dataclasses.dataclass(frozen=True)
class SnapshotRemover(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that removes snapshots from a
    :class:`Batch <dnikit.base.Batch>`. If used with no arguments, this
    will remove *all* :attr:`snapshots <dnikit.base.Batch.snapshots>`.

    Args:
        snapshots: **[optional]** see :attr:`snapshots` attribute
        keep: **[optional]** see :attr:`keep` attribute
    """

    snapshots: dt.OneManyOrNone[str] = None
    """List of :attr:`snapshots <dnikit.base.Batch.snapshots>` to keep/remove."""
    keep: bool = False
    """If ``True``, the listed ``snapshots`` are kept, else the ``snapshots`` will be removed."""

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        snapshots = dt.resolve_one_many_or_none(self.snapshots, str)

        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)
            if snapshots:
                delete_keys(builder.snapshots, keys=snapshots, keep=self.keep)
            else:
                builder.snapshots = {}

            return builder.make_batch()
        return batch_processor


@t.final
@dataclasses.dataclass(frozen=True)
class PipelineDebugger(PipelineStage):
    """
    A :class:`PipelineStage <dnikit.base.PipelineStage>` that can be used to inspect
    :class:`batches <dnikit.base.Batch>` in a :class:`pipeline <dnikit.base.pipeline>`.

    Args:
        label: **[optional]** see :attr:`label`
        first_only: **[optional]** see :attr:`first_only`
        dump_fields: **[optional]** see :attr:`dump_fields`
        fields: **[optional]** see :attr:`fields`
    """

    label: str = ""
    """Optional label to display."""
    first_only: bool = True
    """Show the first batch only."""
    dump_fields: bool = False
    """If ``True``, print the contents of the fields."""
    fields: dt.OneManyOrNone[str] = None
    """List of fields of interest.  Default is None which means all.  See ``dump_fields``"""

    @staticmethod
    def dump(batch: t.Union[Batch, Batch.Builder],
             label: str = "",
             dump_fields: bool = False,
             fields: dt.OneManyOrNone[str] = None) -> str:
        """
        Utility method to produce a dump of a :class:`Batch <dnikit.base.Batch>` or a
        :class:`Batch.Builder <dnikit.base.Batch.Builder>`.

        Args:
            batch: :class:`Batch <dnikit.base.Batch>` or
                :class:`Batch.Builder <dnikit.base.Batch.Builder>` to dump
            label: **[optional]** see :attr:`label`
            dump_fields: **[optional]** see :attr:`dump_fields`
            fields: **[optional]** see :attr:`fields`
        """
        prefix = f"{label} " if label else ""
        batch_size = f"batch_size={len(next(iter(batch.fields.values())))}" if batch.fields else ""
        result = f"{prefix}Batch({batch_size}) {{\n"

        fields = dt.resolve_one_many_or_none(fields, str)
        for name in sorted(batch.fields.keys()):
            data = batch.fields[name]
            if dump_fields and (fields is None or name in fields):
                result += f"{name}: {data.shape}\n{data}\n"
            else:
                result += f"{name}: {data.shape}\n"

        snapshots = batch.snapshots
        if snapshots:
            result += "\nSnapshots:\n"
            for key in sorted(snapshots.keys()):
                result += f"{key}: {sorted(snapshots[key].fields.keys())}\n"

        metadata = batch.metadata
        if metadata:
            result += "\nMetadata:\n"
            for meta_key in sorted(metadata.keys(), key=lambda x: x.name):
                result += f"{meta_key}"
                if isinstance(meta_key, Batch.DictMetaKey):
                    result += f"{meta_key}: {sorted(metadata[meta_key].keys())}"
                result += "\n"

        result += "}\n"
        return result

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        fields = dt.resolve_one_many_or_none(self.fields, str)

        # track whether the first value was processed.  note that it's necessary to capture
        # reference to something mutable -- an array in this case.
        first = [True]

        def batch_processor(batch: Batch) -> Batch:
            # show everything unless asked to show the first only
            show = not self.first_only or first[0]

            if show:
                print(
                    PipelineDebugger.dump(
                        batch, label=self.label, dump_fields=self.dump_fields, fields=fields
                    ),
                    end="\n\n"
                )

            # mark that the first item was handled
            first[0] = False

            return batch

        return batch_processor


@t.final
class Pooler(Processor):
    """
    A :class:`Processor` that pools the axes of a data field from a
    :class:`Batch <dnikit.base.Batch>` with a specific method.

    Args:
        dim: **[keyword arg]** The dimension (one or many) to be pooled.
            E.g., Spatial pooling is generally ``(1, 2)``.
        method: **[keyword arg]** Pooling method. See :class:`Pooler.Method` for full list
            of options.
        fields: **[keyword arg, optional]** a single :attr:`field <dnikit.base.Batch.fields>`
            name, or an iterable of :attr:`field <dnikit.base.Batch.fields>` names, to be pooled.
            If the ``fields`` param is ``None``, then all the
            :attr:`fields <dnikit.base.Batch.fields>` in the
            :class:`batch <dnikit.base.Batch>` will be pooled.
    """
    class Method(enum.Enum):
        MAX = enum.auto()
        SUM = enum.auto()
        AVERAGE = enum.auto()

    def __init__(self, *,
                 dim: dt.OneOrMany[int],
                 method: Method,
                 fields: dt.OneManyOrNone[str] = None):
        dims = tuple(dt.resolve_one_or_many(dim, int))
        assert 0 not in dims, "Unable to pool the 0th (batch) dimension."

        def func(data: np.ndarray) -> np.ndarray:
            assert data.shape, 'Data with no dimensions.'
            assert len(dims) <= len(data.shape), (
                f'data of dimension {data.shape}, too many dim {dims} selected.')
            assert max(dims) < len(data.shape), (
                f'dim {dims} out of data shape {data.shape}.')

            if method is self.Method.MAX:
                return np.max(data, axis=dims)
            elif method is self.Method.SUM:
                return np.sum(data, axis=dims)
            elif method is self.Method.AVERAGE:
                return np.mean(data, axis=dims)
            else:
                raise NotImplementedError(f'Pooling method {method.name} not implemented.')

        super().__init__(func, fields=fields)


@t.final
@dataclasses.dataclass(frozen=True)
class Concatenator(PipelineStage):
    """
    This :class:`PipelineStage <dnikit.base.PipelineStage>` will concatenate 2 or more
    :attr:`fields <dnikit.base.Batch.fields>` in the :attr:`Batch <dnikit.base.Batch>` and produce
    a new field with the given ``output_field``.

    Example:
        If there were fields ``M`` and ``N`` with dimensions ``BxM1xZ`` and ``BxN1xZ`` and
        they were concatenated along dimension 1,
        the result will have a new field of size ``Bx(M1+N1)xZ``.

    Args:
        dim: see :attr:`dim`
        output_field: see :attr:`output_field`
        fields: see :attr:`fields`
    """

    dim: int
    """the dimension to concatenate along"""
    output_field: str
    """name of the new :attr:`field <dnikit.base.Batch.fields>` (layer name) to hold the result"""
    fields: t.Sequence[str]
    """a sequence of :attr:`fields <dnikit.base.Batch.fields>` to concatenate, in order"""

    def __post_init__(self) -> None:
        assert self.dim != 0, "Unable to concatenate along dimension 0 (batch dimension)"
        assert len(self.fields) > 0, "Must specify fields to concatenate"

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        # note that this is a PipelineStage rather than a Processor -- it needs to read
        # multiple layers by name at once and add a new field

        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)

            # collect a list of the source fields in order and concatenate them
            builder.fields[self.output_field] = np.concatenate([
                batch.fields[field]
                for field in self.fields
            ], axis=self.dim)

            return builder.make_batch()

        return batch_processor


@t.final
class Composer(PipelineStage):
    """
    Apply a filter function to all :class:`batches <dnikit.base.Batch>`, e.g. composing filter(b).

    Args:
        filter: The filter function to apply to every :class:`batch <dnikit.base.Batch>` in the
            :func:`pipeline <dnikit.base.pipeline>`. The ``filter`` should take a single
            :class:`Batch <dnikit.base.Batch>` as input and return a transformed
            :class:`batch <dnikit.base.Batch>` (e.g. a subset) or ``None``
            (to produce an empty :class:`batch <dnikit.base.Batch>`).
        """

    def __init__(self, filter: t.Callable[[Batch], t.Optional[Batch]]) -> None:
        super().__init__()
        self._filter = filter

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            result = self._filter(batch)
            if result is None:
                result = batch.elements[[]]
            return result

        return batch_processor

    @classmethod
    def from_element_filter(cls, elem_filter: t.Callable[[Batch.ElementType], bool]) -> 'Composer':
        """
        Initialize a :class:`Composer` that filters batch data based on element-wise filter criteria

        Args:
            elem_filter: :attr:`Batch.element <dnikit.base.Batch.elements>`-wise validation fnc.
                Returns ``True`` if valid else ``False``

        Return:
            :class:`Composer` that filters batches to only elements that meet filter criteria
        """
        def batch_filter(batch: Batch,
                         element_filter: t.Callable[[Batch.ElementType], bool]
                         ) -> t.Optional[Batch]:
            return batch.elements[[
                i for i, element in enumerate(batch.elements)
                if element_filter(element)
            ]]

        return cls(filter=partial(batch_filter, element_filter=elem_filter))

    @classmethod
    def from_dict_metadata(cls, metadata_key: Batch.DictMetaKey[str],
                           label_dimension: str, label: str) -> 'Composer':
        """
        Initialize a :class:`Composer` to filter :class:`Batches <dnikit.base.Batch>` by
        restrictions on their :attr:`metadata <dnikit.base.Batch.metadata>`, as accessed
        with a :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>`, e.g.,
        :class:`Batch.StdKeys.LABELS <dnikit.base.Batch.StdKeys.LABELS>`.

        Args:
            metadata_key: :class:`DictMetaKey <dnikit.base.Batch.DictMetaKey>` to look for in
                batch's :attr:`elements <dnikit.base.Batch.elements>`
            label_dimension: label dimension in batch's ``metadata_key``
                :attr:`metadata <dnikit.base.Batch.metadata>` to filter by
            label: label value to filter by, for batch
                :attr:`metadata's <dnikit.base.Batch.metadata>`
                ``metadata_key`` and ``label_dimension``

        Return:
            :class:`Composer` that filters :class:`Batches <dnikit.base.Batch>` by
            dict :attr:`metadata <dnikit.base.Batch.metadata>` criteria
        """

        def elem_filter(elem: Batch.ElementType,
                        meta_key: Batch.DictMetaKey[str],
                        lbl_dim: str,
                        lbl: str) -> bool:
            """
            Returns True if element's metadata matches filter restrictions, False otherwise.
            """
            if meta_key not in elem.metadata:
                return False
            if lbl_dim not in elem.metadata[meta_key].keys():
                return False
            return elem.metadata[meta_key][lbl_dim] == lbl

        return cls.from_element_filter(elem_filter=partial(
            elem_filter,
            meta_key=metadata_key,
            lbl_dim=label_dimension,
            lbl=label
        ))

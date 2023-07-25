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

import numpy as np

from dnikit.base import Batch, PipelineStage
import dnikit.typing as dt
import dnikit.typing._types as t


class Processor(PipelineStage):
    """
    Class to apply transformations to the fields of :class:`Batch <dnikit.base.Batch>`.

    All other processors in DNIKit should inherit from this class. Note that this is *not* an
    abstract base class. A custom valid processor can be instantiated by simply passing a
    function, as shown in the next example:

    Example:
        .. code::

            def to_db_func(in: np.ndarray) -> np.ndarray:
                ref_value = 1e-5
                return 20 * np.log10(in/ref_value)

            processor = Processor(to_db_func)
            # processor can now be used with a pipeline.

    Args:
        func: transformation to be applied to selected fields.
        fields: **[keyword arg, optional]** a single field, or an iterable of fields, to be
            processed. If ``fields`` is ``None``, then all all
            :attr:`fields <dnikit.base.Batch.fields>` will be processed.
    """

    def __init__(self,
                 func: t.Callable[[np.ndarray], np.ndarray], *,
                 fields: dt.OneManyOrNone[str] = None):
        super().__init__()

        self._func = func
        self._fields = dt.resolve_one_many_or_none(fields, str)

    def _get_batch_processor(self) -> t.Callable[[Batch], Batch]:
        def batch_processor(batch: Batch) -> Batch:
            builder = Batch.Builder(base=batch)

            # Apply function for every selected field
            selected_fields = self._fields or batch.fields.keys()
            for f in selected_fields:
                builder.fields[f] = self._func(builder.fields[f])

            return builder.make_batch()

        return batch_processor

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

from dnikit.base import Producer, Batch
from dnikit.base._producer import _resize_batches
import dnikit.typing._types as t


@t.final
@dataclasses.dataclass(frozen=True)
class StubProducer(Producer):
    """Generates batches of responses from static data."""
    data: dataclasses.InitVar[t.Mapping[str, np.ndarray]]
    metadata: t.Optional[t.Mapping[t.Any, t.Any]] = None
    _large_batch: Batch = dataclasses.field(init=False)

    def __post_init__(self, data: t.Mapping[str, np.ndarray]) -> None:
        builder = Batch.Builder(fields={k: v for k, v in data.items()})
        if self.metadata is not None:
            for key, value in self.metadata.items():
                builder.metadata[key] = value

        object.__setattr__(self, "_large_batch", builder.make_batch())

    def __call__(self, batch_size: int) -> t.Iterable[Batch]:
        producer = _resize_batches((self._large_batch, ))
        yield from producer(batch_size)

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

from ._base_processor import Processor

from ._metadata_processors import (
    MetadataRemover,
    MetadataRenamer,
)

from ._processors import (
    MeanStdNormalizer,
    Transposer,
    FieldRemover,
    FieldRenamer,
    Flattener,
    SnapshotSaver,
    SnapshotRemover,
    PipelineDebugger,
    Pooler,
    Concatenator,
    Composer,
)
from ._image_processors import (
    ImageGammaContrastProcessor,
    ImageGaussianBlurProcessor,
    ImageResizer,
    ImageRotationProcessor
)

# Cacher is declared with CachedProducer since it shares much of the same functionality
# it's exposed via processors, since it behaves a lot more like a processor.
from dnikit.base._cached_producer import Cacher

__all__ = [
    "Processor",
    "MeanStdNormalizer",
    "Transposer",
    "FieldRemover",
    "FieldRenamer",
    "Flattener",
    "MetadataRemover",
    "MetadataRenamer",
    "SnapshotSaver",
    "SnapshotRemover",
    "PipelineDebugger",
    "Pooler",
    "Concatenator",
    "Cacher",
    "Composer",
    "ImageGammaContrastProcessor",
    "ImageGaussianBlurProcessor",
    "ImageResizer",
    "ImageRotationProcessor",
]

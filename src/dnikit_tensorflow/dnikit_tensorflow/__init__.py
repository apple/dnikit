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

"""TensorFlow extensions of DNIKit."""

__version__ = "2.0.0"

import dnikit
from ._tensorflow._tensorflow_loading import load_tf_model_from_path, load_tf_model_from_memory
from ._sample_models import TFModelExamples, TFModelWrapper
from ._sample_datasets import TFDatasetExamples

__all__ = [
    "load_tf_model_from_path",
    "load_tf_model_from_memory",
    "TFModelExamples",
    "TFModelWrapper",
    "TFDatasetExamples",
]

# Raise error if dnikit and dnikit_tensorflow versions are out of sync
assert __version__ == dnikit.__version__, (
    f'dnikit_tensorflow v{__version__} and '
    f'dnikit v{dnikit.__version__} should be the same versions.'
)

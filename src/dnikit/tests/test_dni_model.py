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

import pytest

from dnikit.base import Model


def test_abstract_class_instantiation() -> None:
    with pytest.raises(TypeError):
        # This line fails in two potential ways: 1) it is abstract (which is being caught here)
        # and 2) it is missing required arguments, which will be ignored here.
        Model()  # type: ignore

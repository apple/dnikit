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

import sys

# README: if using any of these functions, try to import
# the relevant package before using the functions. For instance
# try:
#     import cv2
# except ImportError:
#     pass
#
# _opencv_available() # Returns True if OpenCV is available, otherwise False


def _opencv_available() -> bool:
    return "cv2" in sys.modules


def _PIL_available() -> bool:
    return "PIL" in sys.modules


def _tensorflow_available() -> bool:
    return (
        "tensorflow" in sys.modules and
        "dnikit_tensorflow" in sys.modules
    )


def _matplotlib_available() -> bool:
    return "matplotlib" in sys.modules


def _umap_available() -> bool:
    # This is an inline import in the rest of the code,
    #  so in order to check if it's available, it's necessary to try to import first.
    #  There is a noqa because it's imported and not used in scope
    try:
        from umap import UMAP as ULearnUMAP  # noqa
    except ImportError:
        pass
    return "umap.umap_" in sys.modules


def _pandas_available() -> bool:
    return "pandas" in sys.modules

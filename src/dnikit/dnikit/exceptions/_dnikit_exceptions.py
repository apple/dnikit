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

import warnings


class DNIKitException(RuntimeError):
    """
    Raised when a DNIKit component fails

    Args:
        message: Error message to display
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:
        return f"{self.__class__}: {self.message}"


class DNIKitDeprecationWarning(DeprecationWarning):
    """
    Used to indicate a function or a class is deprecated and will be removed from DNIKit.

    By default, all deprecation warnings are disabled in Python. For developers
    using DNIKit, it's strongly recommended to call :func:`enable_deprecation_warnings()`
    at the beginning of any code.
    """


def enable_deprecation_warnings(*, error: bool = False) -> None:
    """
    DNIKit deprecation warnings will be shown in the console.

    Args:
        error: **[keyword arg, optional]** if ``True`` deprecation warnings will be treated as
            errors, that is they will be raised as an exception (default=False).
    """
    # Need to pass a literal as input to warnings.simplefilter: 'error', or 'default'
    if error:
        warnings.simplefilter('error', DNIKitDeprecationWarning)
    else:
        warnings.simplefilter('default', DNIKitDeprecationWarning)

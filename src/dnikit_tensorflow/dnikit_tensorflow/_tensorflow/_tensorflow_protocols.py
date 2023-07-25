#
# Copyright 2022 Apple Inc.
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
import pathlib

import tensorflow as tf

from dnikit.base._model import _ModelDetails
from dnikit.exceptions import DNIKitException
import dnikit.typing._types as t


def running_tf_1() -> bool:
    return tf.__version__[0] == '1'


class _TFLoader(t.Protocol):
    @staticmethod
    def can_load(pathname: pathlib.Path) -> bool:
        """
        Determine if the specified loader can load a model from the given ``pathname``.

        Args:
            pathname: file, if singular model file, or directory that contains multiple model files

        Returns:
            True, if loader can load model from given path, otherwise False
        """
        ...

    @staticmethod
    def load(pathname: pathlib.Path) -> _ModelDetails:
        """
        Load model from ``pathname`` into new TensorFlow graph and session with ``session_config``.

        Args:
            pathname: file or directory, expected to be same pathname passed to
                ``_TFLoader.can_load(pathname)``

        Returns:
            _Tensorflow1ModelDetails initialized with loaded TF graph and session
        """
        ...


@t.final
@dataclasses.dataclass
class LoadingChain:
    loading_chain: t.Sequence[t.Type[_TFLoader]]
    """Ordered sequence of loaders that follow :class:`_TFLoader` protocol"""

    def get_loader(self, pathname: pathlib.Path) -> t.Type[_TFLoader]:
        for loader in self.loading_chain:
            if loader.can_load(pathname):
                return loader

        raise DNIKitException(f'DNIKit unable to load TF model from path: {pathname}.')

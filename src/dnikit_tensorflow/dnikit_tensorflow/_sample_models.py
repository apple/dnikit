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
import os
import tempfile

import tensorflow as tf
import numpy as np

from dnikit.base import PipelineStage, Model, ResponseInfo
from dnikit.processors import Processor
import dnikit.typing._types as t
import dnikit.typing._dnikit_types as dt
from dnikit_tensorflow._tensorflow._tensorflow_loading import load_tf_model_from_path


@dataclasses.dataclass
class TFModelWrapper:
    """
    A wrapper for loading TensorFlow models into DNIKit :class:`Models <dnikit.base.Model>`
    and :class:`PipelineStages <dnikit.base.PipelineStage>`, with their pre- and post-processing
    functions built-in.

    Args:
        model: see :attr:`model`
        preprocessing: see :attr:`preprocessing`
        postprocessing: see :attr:`postprocessing`
    """

    model: Model
    """
    DNIKit :class:`Model <dnikit.base.Model>` to put into DNIKit
    :func:`pipeline <dnikit.base.pipeline>`
    """

    preprocessing: dt.OneManyOrNone[PipelineStage] = None
    """
    One or many DNIKit :class:`PipelineStages <dnikit.base.PipelineStage>` for pre-processing
    :class:`batches <dnikit.base.Batch>` for this model
    """

    postprocessing: dt.OneManyOrNone[PipelineStage] = None
    """
    One or many DNIKit :class:`PipelineStages <dnikit.base.PipelineStage>` for post-processing
    :class:`batches <dnikit.base.Batch>` after model output
    """

    @classmethod
    def from_keras(cls,
                   model: tf.keras.Model,
                   preprocessing: t.Callable[[np.ndarray], np.ndarray]) -> 'TFModelWrapper':
        """
        Convenience method for loading as :class:`TFModelWrapper`
        from Keras models and preprocessors.

        Note:
           When subclassing ``TFModelWrapper`` and there are additional pre-postprocessing steps
           to run outside of Keras's preprocessing, modify the respective attribute of the
           return object to add those steps as :class:`PipelineStages <dnikit.base.PipelineStage>`.

        Args:
            model: TensorFlow Keras model
            preprocessing: keras preprocessing function to transform data
        """
        return TFModelWrapper(
            model=cls.load_keras_model(model),
            preprocessing=Processor(preprocessing)
        )

    @staticmethod
    def load_keras_model(model: tf.keras.Model) -> Model:
        """
        Saves TF Keras model to disk and reloads it as a DNIKit :class:`Model <dnikit.base.Model>`.

        Args:
            model: TF Keras model
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.h5')
            model.save(model_path)
            dni_model = load_tf_model_from_path(model_path)
        return dni_model

    @property
    def response_infos(self) -> t.Mapping[str, ResponseInfo]:
        """
        Get all possible responses in a model. Result is returned as a mapping between response
        names and the corresponding :class:`ResponseInfo <dnikit.base.ResponseInfo>`.
        """
        return self.model._response_infos

    def __call__(self,
                 requested_responses: dt.OneManyOrNone[str] = None) -> (
            t.Union[dt.OneOrMany[PipelineStage], t.Sequence[dt.OneOrMany[PipelineStage]]]):
        """
        Generate a :class:`PipelineStage <dnikit.base.PipelineStage>` that preprocesses
        :class:`Batches <dnikit.base.Batch>` for the :class:`Model <dnikit.base.Model>`,
        runs the model with the requested responses, and postprocesses
        responses before returning them.

        Note:
            If the instance's ``postprocessing`` or ``preprocessing`` properties are None,
            it will ignore those steps`.

        Args:
            requested_responses: passed to the DNIKit :class:`Model <dnikit.base.Model>`.
                Determines which outputs from the model will be present in the
                :class:`Batch <dnikit.base.Batch>` output by the resulting
                :class:`PipelineStage <dnikit.base.PipelineStage>`.

        Returns:
            a single :class:`PipelineStage <dnikit.base.PipelineStage>` or list of
            :class:`PipelineStages <dnikit.base.PipelineStage>`
        """

        stages = [
            stage
            for stage in (self.preprocessing,
                          self.model(requested_responses=requested_responses),
                          self.postprocessing)
            if stage is not None
        ]

        if len(stages) == 1:
            return stages[0]
        return stages


@t.final
class TFModelExamples:
    """
    Out-of-the-box TF and Keras models with pre- and post-processing.
    """

    MobileNet: t.Callable[..., TFModelWrapper] = lambda: (
        TFModelWrapper.from_keras(tf.keras.applications.mobilenet.MobileNet(),
                                  tf.keras.applications.mobilenet.preprocess_input))
    """Load the MobileNet model and processing stages from Keras into DNIKit."""

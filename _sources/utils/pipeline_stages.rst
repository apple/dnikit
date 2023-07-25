.. _batch_processors:

================
Batch Processors
================

DNIKit provides a large number of various
:class:`Processors <dnikit.processors.Processor>` for batches. For instance,
a processor might resize the images in a batch, perform data augmentation,
remove batch fields, attach metadata, rename labels, etc. These processors
are chained together in :func:`pipelines <dnikit.base.pipeline>`,
acting as :class:`PipelineStages <dnikit.base.PipelineStage>`. Note that
:class:`Processors <dnikit.processors.Processor>`
always come after a data :ref:`Producer <data_producers>`,
which is what generates batches to begin with.

Here are most of the available batch processors and data loaders,
and links to their API for more information.

Batch filtering and concatenation
---------------------------------

:class:`Composer <dnikit.processors.Composer>` for filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.Composer
    :noindex:

:class:`Concatenator <dnikit.processors.Concatenator>` for merging fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.Concatenator
    :noindex:

Renaming fields and metadata
----------------------------

:class:`FieldRenamer <dnikit.processors.FieldRenamer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.FieldRenamer
    :noindex:

:class:`MetadataRenamer <dnikit.processors.MetadataRenamer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.MetadataRenamer
    :noindex:

Removing fields and metadata
----------------------------

:class:`FieldRemover <dnikit.processors.FieldRemover>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.FieldRemover
    :noindex:

:class:`MetadataRemover <dnikit.processors.MetadataRemover>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.MetadataRemover
    :noindex:

:class:`SnapshotRemover <dnikit.processors.SnapshotRemover>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.SnapshotRemover
    :noindex:

General data transforms
-----------------------

:class:`MeanStdNormalizer <dnikit.processors.MeanStdNormalizer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.MeanStdNormalizer
    :noindex:

:class:`Pooler <dnikit.processors.Pooler>` (Max Pooling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.Pooler
    :noindex:

:class:`Transposer <dnikit.processors.Transposer>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.Transposer
    :noindex:

Image operations
----------------

:class:`ImageResizer <dnikit.processors.ImageResizer>` to resize images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.ImageResizer
    :noindex:

:class:`ImageRotationProcessor <dnikit.processors.ImageRotationProcessor>` to rotate images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.ImageRotationProcessor
    :noindex:

Augmentations
^^^^^^^^^^^^^

:class:`ImageGaussianBlurProcessor <dnikit.processors.ImageGaussianBlurProcessor>`
**********************************************************************************

.. autoclass:: dnikit.processors.ImageGaussianBlurProcessor
    :noindex:

:class:`ImageGammaContrastProcessor <dnikit.processors.ImageGammaContrastProcessor>`
************************************************************************************

.. autoclass:: dnikit.processors.ImageGammaContrastProcessor
    :noindex:

Utility processors
------------------

:class:`Cacher <dnikit.processors.Cacher>` to cache responses from pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dnikit.processors.Cacher
    :noindex:

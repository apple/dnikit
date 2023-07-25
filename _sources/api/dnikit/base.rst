========
Base API
========

.. contents:: Contents
    :local:

Data Management – Producer
--------------------------

.. autoclass:: dnikit.base.Producer
    :members:

Data Management – Batch
-----------------------

.. autoclass:: dnikit.base.Batch
    :members: fields, snapshots, metadata, batch_size, elements

.. autoclass:: dnikit.base.Batch.ElementsView
    :special-members: __getitem__, __iter__

.. autoclass:: dnikit.base.Batch.ElementType
    :members: fields, snapshots, metadata

.. autoclass:: dnikit.base.Batch.MetaKey
    :members: name

.. autoclass:: dnikit.base.Batch.DictMetaKey
    :members: name

.. autoclass:: dnikit.base.Batch.MetadataType
    :members: __getitem__, __contains__, __bool__, keys

.. autoclass:: dnikit.base.Batch.MetadataType.ElementType
    :members: __getitem__, __contains__, __bool__

.. autoclass:: dnikit.base.Batch.Builder
    :members: fields, snapshots, metadata, make_batch

.. autoclass:: dnikit.base.Batch.StdKeys
    :members: IDENTIFIER, LABELS, PATH
    :undoc-members:

.. autoclass:: dnikit.base.Batch.Builder.MutableMetadataType
    :members: __getitem__, __setitem__, __delitem__, __contains__, __bool__


Pipelines
---------

.. autofunction:: dnikit.base.pipeline

.. autoclass:: dnikit.base.PipelineStage
    :members:
    :private-members: _pipeline, _get_batch_processor

.. autoclass:: dnikit.base.Model
    :members:
    :special-members: __call__

.. autoclass:: dnikit.base.ResponseInfo
    :members:
    :undoc-members:

.. autoclass:: dnikit.base._model._ModelDetails
    :members:

Introspectors
-------------

.. autoclass:: dnikit.base.Introspector
    :members:

.. autofunction:: dnikit.base.multi_introspect


Utilities
---------

.. autoclass:: dnikit.base.TrainTestSplitProducer
    :members:
    :show-inheritance:
    :special-members: __call__

.. autoclass:: dnikit.base.CachedProducer
    :members:
    :show-inheritance:
    :special-members: __call__

.. autoclass:: dnikit.base.ImageProducer
    :members:
    :show-inheritance:
    :special-members: __call__

.. autoclass:: dnikit.base.ImageFormat
    :members:
    :show-inheritance:

.. autofunction:: dnikit.base.peek_first_batch

.. _connect_your_model:

============
Load a model
============

DNIKit supports loading models from frameworks using built-ins for TensorFlow (v1 + v2) and Keras,
or from other model types using custom loading
(:ref:`see below in "Other Scenarios" <Other scenarios>`).


TensorFlow and Keras
^^^^^^^^^^^^^^^^^^^^

To connect TensorFlow or Keras models with DNIKit, there are two built-in
functions available in the ``dnikit_tensorflow`` package:
:func:`load_tf_model_from_path <dnikit_tensorflow.load_tf_model_from_path>`
or :func:`load_tf_model_from_memory <dnikit_tensorflow.load_tf_model_from_memory>`.

Models from these frameworks are loaded into DNIKit as :class:`dnikit.base.Model` objects.

To load from a file path:

.. code-block:: python

   from dnikit_tensorflow import load_tf_model_from_path

   dni_model = load_tf_model_from_path("/path/to/model")

Models can also be loaded if the model or graph is currently in memory.
For TF2 models, load the model and use parameter ``model``:

.. code-block:: python

   from dnikit_tensorflow import load_tf_model_from_memory

   tf2_model = ... # grab the TF2 model
   dni_model = load_tf_model_from_memory(model=tf2_model)


For TF1, grab the current session and pass in with parameter ``session``:

.. code-block:: python

   from dnikit_tensorflow import load_tf_model_from_memory

   tf1_session = ... # get current Session here
   dni_model = load_tf_model_from_memory(session=tf1_session)

.. _producer_model_responses:

Other scenarios
^^^^^^^^^^^^^^^
For other frameworks, data can be fed through a model outside of DNIKit, where model responses
are extracted and then :doc:`set up in a Producer </how_to/connect_data>` (explained in the next
section) to produce :class:`Batches <dnikit.base.Batch>` of model responses. This
:ref:`model response producer <creating_response_producer>`
will be passed into a DNIKit :doc:`introspector </how_to/introspect>`'s :func:`introspect` method.
When using this setup, inference can also be run only once by caching model responses in advance,
say in a Pandas DataFrame or pickle, and then by creating a :class:`Producer <dnikit.base.Producer>`
that pulls batches of data from the cache. Alternatively, responses from a model can also be
:ref:`cached in DNIKit <response_caching>` directly.

.. note::
  DNIKit :doc:`introspectors </how_to/introspect>` only need model *responses*
  —i.e., outputs from performing inference on data,
  from intermediate or final layers —to work;
  they do not need to access the model directly.

See below for an illustrative comparison between typical pipelines and producing responses outside
DNIKit and feeding directly to an :class:`Introspector <dnikit.base.Introspector>`.

**TYPICAL**

.. image:: ../img/generic_pipeline.png
    :alt: A picture of a generic DNIKit pipeline. Starting with a Producer that yields
          batches (one batch at a time). The Batch then goes through various optional
          Pipeline Stages, including two Processors (pre and post) and one Model inference.
          The transformed Batch is then fed into the Introspector.

**PRODUCE RESPONSES**

.. image:: ../img/response_producer.png
    :alt: A picture of a DNIKit pipeline where the Producer does not produce data, but rather
          model responses. Inference is run outside of DNIKit, and then via the Producer,
          the model responses are consumed directly by the Introspector.

Next Steps
^^^^^^^^^^

After loading a model into DNIKit, the next step is
to :ref:`load data <connect_your_data>`
so that a :func:`pipeline <dnikit.base.pipeline>` can be set up, which feeds data
into the DNIKit model. Learn more :ref:`in the next section <connect_your_data>`.

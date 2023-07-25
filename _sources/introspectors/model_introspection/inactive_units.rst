.. _inactive_units:

================================
Finding a Model's Inactive Units
================================

Compute statistics about inactive units in a network as data is passed through
(mean, standard deviation, counts, and unit frequency).
IUA can be used to detect bad practices in training to avoid adverse effects, such as dying ReLU.

For more information about how the algorithm works and the motivation behind using it,
please see the :ref:`Description and Algorithm sections below <description-iua>`.

General Usage
-------------

Assuming a :func:`pipeline <dnikit.base.pipeline>` has been
set up to produce responses from a model,
:class:`IUA <dnikit.introspectors.IUA>` can be run as so:

.. code-block:: python

   from dnikit.introspectors import IUA

   producer = ...  # pipeline setup here

   # Run IUA analysis on responses from a producer
   iua = IUA.introspect(producer, batch_size=128)


For IUA, like :ref:`PFA <network_compression>`,
inputs to introspection should be prepared by selecting multiple
**layer responses** to analyze. Specifically, units of these layers will be
introspected. For instance, for a model
that uses Conv2D layers, one option could be selecting all the responses
for those layers by reviewing the DNIKit :class:`Model <dnikit.base.Model>`'s
:func:`.response_infos() <dnikit.base.Model.response_infos>` and passing them where the model is
used in the pipeline, e.g.:

.. code-block:: python

   dnikit_model = ... # load model here

   # Find only conv2d layer responses
   response_infos = dnikit_model.response_infos()
   conv_response_names = [
        info.name
        for info in response_infos.values()
        if info.layer.kind == ResponseInfo.LayerKind.CONV_2D
   ]

   producer = pipeline(
        dataset,
        ...
        # Tell the model which responses to look at
        dnikit_model(conv_response_names),
        ...
   )

Visualization
-------------

The result of :func:`IUA.introspect <dnikit.introspectors.IUA.introspect>`
can be shown with :code:`IUA.show()` directly, e.g.:

.. code-block:: python

    iua = IUA.introspect(producer, batch_size=64)
    IUA.show(iua)

.. image:: /img/iua-show-table.png
    :class: bot-margin
    :alt: IUA results shown in a Panda's DataFrame with columns: response name, mean inactive units,
          and standard inactive units.

The :code:`IUA.VisType.CHART` vis option can also be used to display
a heatmap of activations:

.. code-block:: python

    IUA.show(iua, vis_type=IUA.VisType.CHART)

.. image:: /img/iua-show-heatmap.png
    :class: bot-margin
    :alt: IUA results shown in a heatmap for one specific layer, with units that have been
          activated more frequently shown in a lighter color.

Config options
--------------

Like all introspectors, :class:`IUA <dnikit.introspectors.IUA>` takes
a :class:`Producer <dnikit.base.Producer>`
as its first argument and an optional :code:`batch_size` keyword argument
to set the size of batches pulled from the producer.

IUA also has two optional arguments, :attr:`rtol` and :attr:`atol`,
which are passed as parameters to the
`numpy.isclose <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>`_
method. This method determines what units count as "inactive" —which are effectively zero.
It's recommended that only experienced users change these parameters.
For more information on their meaning, see `the Notes section of the numpy.isclose
documentation <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>`_.

.. _description-iua:

Description
-----------
ML researchers use a variety of information to make informed decisions about their
model training and deployment. One such factor is the model capacity.
However, when there are dead units in the model, the *effective* model capacity
can be much smaller than first seems, and not an accurate
depiction of the model's state.

Inactive Unit Analysis (IUA) computes
aggregate statistics about units' responses to input
probes, which can be used to evaluate how active the units are
in response to the specified dataset. More inactive units indicates that
model capacity is actually much smaller than intended. It may also
indicate that there was a problem during training,
such as dying ReLU. This could be caused a learning rate that is
too high, so careful re-training might be necessary to
achieve the desired accuracy.

Algorithm
~~~~~~~~~
As data is passed through a model, IUA counts how many times each unit in the model is activated.
After every data sample has passed through, IUA has a total number of times each unit has activated,
for the number of input data samples. IUA then uses this information to compute aggregate statistics
about how the units responded to the input data, including mean, std, proportion of inactive units,
and raw count.

This information can be useful in assessing model health, as it can help
identify when all (or nearly all) units in a layer are dead, e.g. after a ReLU.

.. image:: /img/dni-health.jpeg
   :scale: 30 %
   :alt: Inactive Unit Analysis showing that different layers can have different percentages of
         dead neurons.
   :align: center

Usage in DNIKit, see :class:`IUA API <dnikit.introspectors.IUA>` for more information.

.. code-block:: python

    from dnikit.introspectors import IUA

    iua_results = IUA.introspect(
        producer=response_producer, # only required arg
        batch_size=32,
        rtol: float = 1e-05,
        atol: float = 1e-08
    )

.. _iua_example:

Example
-------
.. toctree::
   :maxdepth: 1

   Jupyter Notebook: Inactive Unit Analysis (IUA) <../../notebooks/model_introspection/inactive_unit_analysis.ipynb>

Relevant API
------------

- :class:`IUA introspector <dnikit.introspectors.IUA>`

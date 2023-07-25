.. _dataset_report:

==============
Dataset Report
==============

Explore a dataset to find rare data samples, duplicate data, annotation errors,
or dataset bias. The :class:`DatasetReport <dnikit.introspectors.DatasetReport>` is a combination of
three DNIKit dataset introspection algorithms:

- :ref:`Familiarity <familiarity>`
- :ref:`Duplicates <duplicates>`
- :ref:`DimensionReduction (data projection) <dimension_reduction>`

To explore the dataset in an interactive UI, the Dataset Report results can be fed directly
into `Symphony <https://github.com/apple/ml-symphony>`_, a research platform for creating
interactive data science components that allows for filtering, sorting, and exporting data samples.

For motivation behind the Dataset Report, see :ref:`Description` below.

General Usage
-------------

For getting started with DNIKit code, please see the :ref:`how-to pages <connect_your_model>`.

Assuming a :func:`pipeline <dnikit.base.pipeline>` is
set up to produce responses from a model, the `DatasetReport` can be run as so:

.. code-block:: python

   from dnikit.introspectors import DatasetReport

   producer = ...  # pipeline setup here

   # Run DatasetReport on responses from a producer
   report = DatasetReport.introspect(producer, batch_size=128)

Introspection is typically performed on **intermediate model responses**
(rather than the final outputs of a network).
Here's a full example using the CIFAR10 dataset, which uses
the outputs of the last convolution layer :code:`conv_pw_13` from a
MobileNet model to run the analysis:

.. code-block:: python

   from dnikit.introspectors import DatasetReport
   from dnikit_tensorflow import TFDatasetExamples, TFModelExamples
   from dnikit.processors import Cacher, ImageResizer
   from dnikit.base import pipeline

   # Load CIFAR10 dataset and feed into MobileNet,
   # observing responses from layer conv_pw_13
   cifar10 = TFDatasetExamples.CIFAR10(attach_metadata=True)
   mobilenet = TFModelExamples.MobileNet()
   producer = pipeline(
      cifar10,
      ImageResizer(pixel_format=ImageResizer.Format.HWC, size=(224, 224)),
      mobilenet(requested_responses=['conv_pw_13']),
      Pooler(dim=(1, 2), method=Pooler.Method.MAX),
      Cacher()
   )

   # Run DatasetReport on intermediate layer conv_pw_13's responses to the data:
   report = DatasetReport.introspect(producer)

Visualization
-------------

Exploring with Symphony
^^^^^^^^^^^^^^^^^^^^^^^

DNIKit's DatasetReport can also connect with the Symphony UI framework
to explore a dataset in a web browser or in a jupyter notebook. Please see Symphony's
`documentation <https://github.com/apple/ml-symphony>`_ for an example of how to feed the
output of ``DatasetReport.introspect`` directly into Symphony.
These reports created with Symphony are interactive and shareable.

.. warning::
    The current release of `Symphony <https://github.com/apple/ml-symphony>`_ operates only on images, audio, and tabular data.
    To visualize other data types, it's possible to run the DNIKit side of the
    `DatasetReport` on any dataset type and visualize in a custom manner.

    ``pip install "dnikit[dataset-report]"``


Exploring as Pandas DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The resulting :class:`Dataset Report <dnikit.introspectors.DatasetReport>` object has a property,
:attr:`data <dnikit.introspectors.DatasetReport.data>`, that is a
`Pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
of all `DatasetReport` results. Each row represents a data sample, and each column is report data,
e.g., duplicate set.

.. code-block:: python

   report.data

These results can be visualized in a custom manner, but it's recommended to try
:ref:`Symphony <Exploring with Symphony>` for image, audio, or tabular data.

Saving and Loading
------------------
To save the report, call :func:`to_disk() <dnikit.introspectors.DatasetReport.to_disk>`
on the report object.
To load a saved report, use
:func:`DatasetReport.from_disk(filepath) <dnikit.introspectors.DatasetReport.from_disk>`.

Config Options
--------------

Dataset Report's :func:`introspect <dnikit.introspectors.DatasetReport.introspect>`
method has a parameter :code:`config` that accepts a
:class:`ReportConfig <dnikit.introspectors.ReportConfig>` object. The config can be
used to run only a subset of introspectors. For instance,
to run only duplicates analysis:

.. code-block:: python

    from dnikit.introspectors import DatasetReport, ReportConfig

    config = ReportConfig(
        projection=None,
        familiarity=None
    )

The strategies used in the underlying algorithms can also be modified via the config.
See :class:`ReportConfig <dnikit.introspectors.ReportConfig>` in the API docs for more details.


Description
-----------
Automated dataset diversity analysis often looks *inter-class* diversity,
i.e. diversity across classes, as defined by metadata labels. Known methods include
grouping data by label and performing various statistical analyses to see how well the
number of data samples or model accuracy is distributed across these different labels.

*Intra-class* diversity, like fairness *within* a particular class label, is also important,
yet harder to evaluate in an automated fashion. Intra-class diversity analysis is often
manual, which doesn't scale to large datasets. Manual analysis can also make it harder to communicate
findings with team members or partners. Because of these problems, sometimes intra-class
diversity analysis is skipped altogether.

The *Dataset Report* aims to automate and simplify the process of analyzing datasets for both inter
and intra-class diversity, in a manner that enables sharing and exploration. With the
`Symphony <https://github.com/apple/ml-symphony>`_ framework, it's possible to build a standalone
static report or explore results live in a Jupyter notebook.
Symphony also contains a centralized filtering, grouping, highlighting, and selection
across all widgets, to form a cohesive workspace for dataset exploration.


.. _datasetreport_example:

Example
-------
A Jupyter notebook that demonstrates how to run the Dataset Report on the CIFAR-10 dataset:

.. toctree::
   :maxdepth: 1

   Jupyter Notebook: Dataset Report <../../notebooks/data_introspection/dataset_report.ipynb>

Relevant API
------------
* :class:`Dataset Report  <dnikit.introspectors.DatasetReport>`
* :class:`Dataset Report Config  <dnikit.introspectors.ReportConfig>`
* :class:`Familiarity <dnikit.introspectors.Familiarity>`
* :class:`DimensionReduction <dnikit.introspectors.DimensionReduction>`
* :class:`Duplicates <dnikit.introspectors.Duplicates>`

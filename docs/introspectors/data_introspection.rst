.. _data_introspection:

==================
Data Introspection
==================

Data introspectors observe intermediate model responses,
and process data in batches when calling
:attr:`.introspect() <dnikit.base.Introspector.introspect>`.

:ref:`Dataset Report <dataset_report>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :ref:`DatasetReport <dataset_report>` bundles :ref:`Familiarity <familiarity>`,
:ref:`Duplicates <duplicates>` and :ref:`Dimension Reduction <dimension_reduction>`
introspectors (below) in an interactive interface with various visualization options.

:ref:`Familiarity <familiarity>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:ref:`Familiarity <familiarity>` quantifies how *familiar* a data point is to a specific dataset
or subset, by fitting a probability distribution to the activations of the specified layer(s),
and then evaluating the probability of any data sample according to the distribution.

:ref:`Duplicates <duplicates>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Find near-duplicate data. Uses an approximate nearest neighbor to build a distance matrix for all
samples and clusters the closest samples.

:ref:`Dimension Reduction <dimension_reduction>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Projects high dimensional activation data to a lower dimension, usually for consumption by a
different introspector or for 2D or 3D visualization.

.. toctree::
   :hidden:
   :maxdepth: 1

   Dataset Report <data_introspection/dataset_report.rst>
   Familiarity <data_introspection/familiarity.rst>
   Duplicates <data_introspection/duplicates.rst>
   Dimension Reduction <data_introspection/dimension_reduction.rst>

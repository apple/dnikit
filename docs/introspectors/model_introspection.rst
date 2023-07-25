.. _model_introspection:

=====================
Network Introspection
=====================

Network introspectors help analyze model performance, offering suggestions
for how to improve model efficiency and size.

:ref:`Principal Filter Analysis <network_compression>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Guide neural network compression by removing redundant filters detected by
correlated filter responses, to target a minimum loss in accuracy, or a specific model footprint.

:ref:`Inactive Unit Analysis <inactive_units>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates the extent to which units are inactive for a given set of probe data points by
computing aggregate statistics using the unit responses to the input probes.


.. toctree::
   :hidden:
   :maxdepth: 1

   Network Compression with PFA <model_introspection/network_compression.rst>
   Find Inactive Units with IUA <model_introspection/inactive_units.rst>

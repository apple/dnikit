.. _dimension_reduction:

Dimension Reduction
===================

DNIKit provides a :class:`DimensionReduction <dnikit.introspectors.DimensionReduction>`
introspector with a variety of strategies (algorithms).
`DimensionReduction` has two primary uses:

- reduce high-dimensional data to something lower for consumption by a different :class:`Introspector <dnikit.base.Introspector>`
- reduce data to 2D or 3D for visualization (e.g. :ref:`Dataset Report <dataset_report>`).

Often, model responses are very large in the number of dimensions. However,
some algorithms work better on lower dimensional data.
For example :class:`Familiarity <dnikit.introspectors.Familiarity>` and
even the
:class:`DimensionReduction Strategies <dnikit.introspectors.DimensionReduction.Strategy>`
other than `PCA` work better on e.g. 40 dimensional data.
Some of the algorithms state this is useful for reducing the noise in
very high dimensional data.
`PCA` (Principal Component Analysis) is a great strategy to perform this reduction.

The notebook :ref:`Dimension Reduction Example Notebook <dimensionreduction_example>` below gives an example of
reducing high dimension data for use with various `DimensionReduction` strategies.

`DimensionReduction` to 2D is also a nice way to visualize the clusters and relationships
in the data.  `UMAP`, `PaCMAP` and `t-SNE` are all algorithms that are well suited to this task.
The notebook :ref:`below <dimensionreduction_example>` also shows examples of doing this.

General Usage
-------------

For getting started with DNIKit code, please see the :ref:`how-to pages <connect_your_model>`.

.. code-block:: python

	# a source of embeddings (typically high dimensional data)
	response_producer = pipeline(...)

	# first, create a dimension reduction `PipelineStage` object (`reducer`, here) that is fit
	#    to the input data and will be able to project any data to a lower number of dimensions
	reducer = DimensionReduction.introspect(
        response_producer,
        strategies=DimensionReduction.Strategy.PCA(40)
    )

	# Next, chain the reducer PipelineStage into a new pipeline that will reduce all output data
	#    from `response_producer` into 40 dimensions
	reduced_producer = pipeline(response_producer, reducer)

See the :ref:`example notebook <dimensionreduction_example>` below for more detailed usage.

Config Options
--------------

DNIKit comes with four :class:`Strategies <dnikit.introspectors.DimensionReduction.Strategy>`
for performing dimension reduction, each with their own advantages and disadvantages:

- :class:`PCA <dnikit.introspectors.DimensionReduction.Strategy.PCA>`
	- very fast and good for reducing e.g. 1024 -> 40 dimensions
	- memory efficient
	- not suitable for 2D projection
- :class:`UMAP <dnikit.introspectors.DimensionReduction.Strategy.UMAP>`
	- excellent 2D projections
	- preserves local but not global structure
- :class:`PaCMAP <dnikit.introspectors.DimensionReduction.Strategy.PaCMAP>`
	- excellent 2D projections
	- preserves local and global structure
- :class:`TSNE (t-SNE) <dnikit.introspectors.DimensionReduction.Strategy.TSNE>`
	- largely replaced by newer strategies

For a more in-depth comparison, please see
:ref:`the example notebook <dimensionreduction_example>` below.

Relevant API
------------

- :class:`DimensionReduction <dnikit.introspectors.DimensionReduction>`: introspector for Dimension Reduction


.. _dimensionreduction_example:

Example
-------

.. toctree::
   :maxdepth: 1

   Jupyter Notebook: Dimension Reduction Strategies <../../notebooks/data_introspection/dimension_reduction.ipynb>

References
----------

- `UMAP documentation <https://umap-learn.readthedocs.io/en/0.5dev/index.html>`_
- `UMAP paper <https://arxiv.org/abs/1802.03426>`_
- `Understanding UMAP <https://pair-code.github.io/understanding-umap/>`_

- `PaCMAP <https://github.com/YingfanWang/PaCMAP>`_
- `PaCMAP paper <https://jmlr.org/papers/v22/20-1061.html>`_

- `t-SNE documentation <https://scikit-learn.org/stable/modules/manifold.html#t-sne>`_
- `Understanding t-SNE <https://distill.pub/2016/misread-tsne/>`_


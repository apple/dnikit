.. _duplicates:

==========
Duplicates
==========
Find near duplicate data samples within a given dataset or across multiple datasets, by using
an approximate nearest neighbor to build a distance matrix for all
samples and clusters the closest samples.
This algorithm, for example, can be used to unmask a class imbalance issue or
find duplicate data that exist in both the train and test splits.

For a more thorough discussion of this algorithm and its use, see the
:ref:`Description <duplicates_description>` below.

General Usage
-------------

For getting started with DNIKit code, please see the :ref:`how-to pages <connect_your_model>`.

Assuming a :func:`pipeline <dnikit.base.pipeline>`
is set up that produces responses from a model,
`Duplicates` analysis can be run as so:

.. code-block:: python

   from dnikit.introspectors import Duplicates

   response_producer = ...  # pipeline setup here

   # Run `Duplicates` analysis on responses from a `Producer`
   duplicates = Duplicates.introspect(response_producer, batch_size=128)

Preparing input to introspection usually requires two things:
 - introspect on **intermediate layer responses** (rather than the final outputs of a network).
   For `Duplicates`, layer(s) towards the end of the network could be most appropriate.
 - **reduce the dimensions** of outputs
   with a :ref:`dimension reduction algorithm <dimension_reduction>`.
   While the `Duplicates` algorithm can run on any dimension of data,
   dimension reduction helps with performance.

A full example pipeline for the CIFAR10 dataset and model
(note: this will likely take some time to run!):

.. code-block:: python

   from dnikit.base import pipeline
   from dnikit.processors import Cacher, ImageResizer
   from dnikit.introspectors import Duplicates, DimensionReduction
   from dnikit_tensorflow import TFDatasetExamples, TFModelExamples

   # Load CIFAR10 dataset and feed into MobileNet,
   # observing responses from layer "conv_pw_13/convolution:0'"
   cifar10 = TFDatasetExamples.CIFAR10()
   mobilenet = TFModelExamples.MobileNet()
   producer = pipeline(
      cifar10,
      ImageResizer(pixel_format=ImageResizer.Format.HWC, size=(224, 224)),
      mobilenet(requested_responses=['conv_pw_13/convolution:0']),
      Cacher()
   )

   # Create a processor that reduces dimensions of
   # model responses down to 40, using PCA
   pca = DimensionReduction.introspect(
       producer,
       strategies=DimensionReduction.Strategy.PCA(40)
   )

   # Create a new producer that outputs the reduced data:
   reduced_producer = pipeline(producer, pca)

   # Run `Duplicates` introspector
   duplicates = Duplicates.introspect(reduced_producer, batch_size=128)

Visualization
-------------
It's recommended to visualize the Duplicates results using
`Symphony <https://github.com/apple/ml-symphony>`_, as noted in
the :ref:`DatasetReport <dataset_report>` page. There is also an example visualization
that does not use Symphony in the :ref:`example notebook below <duplicates_example>`.

Config Options
--------------

The :func:`introspect` method of :class:`Duplicates <dnikit.introspectors.Duplicates>`
takes an optional :code:`batch_size` keyword argument for the size of batches
to pull from and an optional :code:`threshold` keyword argument of type
:class:`dnikit.introspectors.Duplicates.DuplicatesThresholdStrategyType`. Please see the
API docs for :class:`Duplicates <dnikit.introspectors.Duplicates>` for more information.

Exploring Duplicates Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The return object of Duplicates is a :code:`dict` mapping response names to a list of
:class:`DuplicateSetCandidates <dnikit.introspectors.Duplicates.DuplicateSetCandidate>`,
which represent clusters of nearby data samples. Assuming
a return object :code:`duplicates`, the results may be traversed as so:

.. code-block:: python

   for response_name, clusters in duplicates.results.items():
      # sort by the mean distance to the centroid
      clusters = sorted(clusters, key=lambda x: x.mean)
      ...


.. _duplicates_description:

Description
-----------

The Harm of Duplicates
^^^^^^^^^^^^^^^^^^^^^^
The presence of near duplicates in a dataset can indicate that there might not be enough variation
in the dataset. When duplicates exist across training and test datasets, the test accuracy is not
an accurate reflection of the model's performance, and doesn't demonstrate that the model will
generalize well to new, unseen, data samples. Near duplicates within a train or test set alone
might not necessarily be harmful, however, they could slow down training and mask a class imbalance
problem. A first glance at the distribution of samples across classes might indicate a
balanced split, the effective distribution might change depending on the presence of duplicates.

In a dataset, there could be variation in the form of the same data sample, but with
various transformations applied. E.g. for an image, maybe translations, crops, scales, etc.
Although this kind of variation improves performance and allows the model to generalize better,
an alternative way to introduce this variation uniformly across all kinds of data samples is to
use data augmentation during training. Data augmentation is a popular way to artificially increase
the size of a dataset, via flipping, rotating, scaling, cropping, translating, etc. the images.
If the plan is to introduce different data augmentation methods during training, then the presence
of these kinds of similar images in the dataset is unnecessary and might, again, mask a class
imbalance problem. In addition, if the dataset has not yet been annotated, then reducing duplicates
also reduces the annotation effort.

Algorithm
^^^^^^^^^

The duplicates are found by searching for the k-nearest neighbors using
`ANNOY - Approximate Nearest Neighbor Oh My! <https://github.com/spotify/annoy>`_, filtering
the results by a threshold distance and building clusters using the transitive closure of the
union of the result sets.

Approximate nearest neighbors uses a hashing technique
to limit the possible search area (this is how the linear performance is obtained).  The
"approximate" means that the hashing may prevent some duplicates from being found, though in
practice multiple hash functions are used to mitigate this effect.  In practice, the algorithm
found the same results as an "exact" algorithm on CIFAR-10. The distances computed are exact.

The run-time of `Duplicates` scales linearly with the number of samples and the
number of dimensions in the response data. If the responses have a high number of dimensions,
consider using :class:`DimensionReduction <dnikit.introspectors.DimensionReduction>` to reduce the
dimensions (e.g. to 40).

Relevant API
------------

:class:`Duplicates <dnikit.introspectors.Duplicates>` -- introspector to find duplicates and near duplicates


.. _duplicates_example:

Example
-------

.. toctree::
   :maxdepth: 1

   Jupyter Notebook: Find Near-Duplicate Data <../../notebooks/data_introspection/duplicates.ipynb>

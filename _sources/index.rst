DNIKit – Data and Network Introspection Kit
===========================================

A Python toolkit for analyzing machine learning models and datasets. DNIKit can:

- :ref:`create a comprehensive dataset analysis report <dataset_report>`
- :ref:`identify duplicate data samples <duplicates>`
- :ref:`find rare data samples, annotation errors, or model biases <familiarity>`
- :ref:`compress networks <network_compression>`
- :ref:`detect inactive units in a model (e.g., due to dying ReLU) <inactive_units>`
- :ref:`...and more <how_to_introspect>`.

DNIKit algorithms (also known as **introspectors**) view data
*through the eyes of a neural network*. They operate using intermediate responses
of the networks to provide a unique glimpse of how the network perceives
data throughout the different stages of computation.

.. image:: img/responses.gif
   :scale: 50 %
   :alt: Intermediate responses are extracted from models and used in DNIKit algorithms.
   :align: center
   :class: bot-margin

Getting Started
---------------
Explore the links in the sidebar on the left. Here are some good places to get started:

- :ref:`DNIKit Installation Page <python_support>`
- :ref:`How To Use DNIKit, starting with loading a model <connect_your_model>`
- :ref:`Jupyter notebook examples <example_notebooks>`
- :ref:`How to Cite DNIKit <how_to_cite>`

and see the following DNIKit use case examples.

Examples
--------

Identify model biases and annotation errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`Familiarity introspector <familiarity>` can be used to plot the most and least
representative data samples for a given set of data samples, which can assist in
identifying model biases and annotation errors. For instance, introspecting with
`a simple CNN model trained on MNIST <https://keras.io/examples/vision/mnist_convnet/>`_,
and the MNIST dataset, there is a clear predilection for slanted 5's (top, "most familiar"):

.. image:: img/fam_mnist_example.png
    :class: bot-margin
    :alt: 20 most familiar and 20 least familiar 5's. The most familiar 5's all look very crisp
          and clean and the least familiar look atypical.

Fives that aren't slanted, written with a fine-tipped pen, or look like capital S's
are the least "five-like" when compared to the overall dataset (bottom, "least familiar").
In the "least familiar" plot, at the far right of the first row,
there is a "J"-like symbol that is not very interpretable as a 5, and we
might want to remove it from the dataset. [#f0]_

Familiarity analyzes intermediate network embeddings, rather than
final model output. This means it may also be run
on models that will be fine-tuned or adapted and were not trained on the target task, like
`MobileNet <https://keras.io/api/applications/mobilenet/>`_, trained on ImageNet.
For instance, running Familiarity on data samples with the "deer" label
in the `CIFAR10 dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_, using responses
extracted with this MobileNet model, the results are as follows:

.. image:: img/fam_deer_example.png
    :class: bot-margin
    :alt: 20 most familiar and 20 least familiar deer discovered with Familiarity.

The most representative (most *familiar*) deer images include scenes with green foliage and
coarse textures, while the least representative (least *familiar*) samples show deer
on single-color backgrounds or within circles. To improve diversity within the "deer"
data samples, a user might consider collecting more data samples like what are shown in the
"least familiar" rows.


Compress a network
~~~~~~~~~~~~~~~~~~

:ref:`Principal Filter Analysis (PFA) <network_compression>` can be used
to compress networks :ref:`without compromising accuracy <pfa_evidence>`; e.g.,
tests on `VGG-16 <https://arxiv.org/abs/1409.1556>`_
on CIFAR-10, CIFAR-100 and ImageNet actually show that PFA achieves a compression rate of 8x, 3x,
and 1.4x with an accuracy gain of 0.4%, 1.4% points, and 2.4% respectively.

For a quick example, let's use PFA to compress the
`Keras library's example ConvNet <https://keras.io/examples/vision/mnist_convnet/>`_
for the `MNIST dataset <https://en.wikipedia.org/wiki/MNIST_database>`_.
The model is trained exactly as specified in a Jupyter notebook,
save the model to disk with :code:`model.save("mnist.h5")`, then run:

.. code-block:: python

    from dnikit.base import pipeline, TrainTestSplitProducer
    from dnikit_tensorflow import load_tf_model_from_path

    # Load model into DNIKit
    dni_model = load_tf_model_from_path('mnist.h5')

    # Get Conv2D layer names to request responses from
    req_layers = [
        name for name in dni_model.response_infos.keys()
        if 'conv' in name
    ]

    # Set up pipeline to feed batches of data into model
    # :: Note: use the same x_train, y_train, etc.
    # :: from the Keras MNIST CNN example code.
    mnist = TrainTestSplitProducer(((x_train, y_train), (x_test, y_test)))
    responses = pipeline(
        mnist,
        dni_model(requested_responses=req_layers),
        Pooler(dim=(1, 2), method=Pooler.Method.MAX)
    )

:ref:`PFA.introspect <network_compression>` can be used to compress the model,
estimating new layer sizes:

.. code-block:: python

    pfa = PFA.introspect(responses)  # introspect!

    # Show suggestions for new layer sizes with 'KL' strategy
    PFA.show(pfa.get_recipe())

.. image:: img/pfa_show_example.png
    :class: bot-margin
    :alt: PFA recipe example showing columns of layer names, original unit count, and
          and recommended unit count according to PFA.

To act on the suggestions, the model is then re-trained with the suggested layer sizes.
Note, it's possible in this step to reuse weights and train from there to get better results,
rather than training from scratch.
Looking at the suggestions and setting the two Conv2D layers
at 21 and 45 filters, respectively, PFA helps achieve a **40% reduction in model size**
(271 KB vs 450 KB) with **no significant cost to accuracy**. [#f1]_ [#f2]_ Please see the
`PFA paper <https://arxiv.org/abs/1807.10585>`_ or :ref:`doc page <network_compression>`
for more information about the experiments.

.. rubric:: Footnotes

.. [#f0] The predilection for slanted 5's suggests that a trained model may
        exhibit a bias towards those types of fives.
        To improve the robustness of this model, one solution could be changing the
        pipeline to include data augmentation that rotates and skews images, to reduce the
        model's bias towards 5's with a rightward skew.
.. [#f1] All estimates were made by comparing 8 training runs of the original model
        vs the compressed model, and running an unpaired t-test
        on the accuracy scores for the test set.
.. [#f2] Moreover, the PFA energy strategy with :code:`energy=0.8`
        will compress the model by over 80%, while only sacrificing a tiny bit of accuracy
        —only about 0.5%. The number of Conv2D filters with this energy threshold strategy is
        reduced from 32 to 7 and 64 to 14, resp.:

        .. code-block:: python

            # Get PFA recipe for energy level 0.8
            recipe_80_perc = pfa.get_recipe(
                strategy=PFA.Strategy.Energy(energy_threshold=0.8, min_kept_count=3)
            )

            # Display PFA recipe for energy levels 0.8 and 0.95 (out of 1.0)
            PFA.show(recipe_80_perc)

        While for a toy MNIST example, these differences in model size and training
        time might seem negligible, for large models, 40% or 80% compression may mean the
        difference between shipping a model on device or not. Instead of tweaking
        hyperparameters seemingly at random, PFA offers valuable starting points
        for fine-tuning models based on solid evidence.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   general/installation
   general/example_notebooks

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: How to Use

   1. DNIKit overview <how_to/dnikit_concepts>
   2. Load a model <how_to/connect_model>
   3. Load data <how_to/connect_data>
   4. Introspect <how_to/introspect>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Algorithms

   Data Introspectors <introspectors/data_introspection>
   Network Introspectors <introspectors/model_introspection>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Utilities

   Data Producers <utils/data_producers>
   Batch Processors <utils/pipeline_stages>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   api/index
   reference/how_to_cite
   Support / Debugging <general/support>
   dev/contributing
   Changelog <reference/changelog>

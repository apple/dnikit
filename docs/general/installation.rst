.. _installation:

============
Installation
============

.. admonition:: Quick install

    In :ref:`Python3.9 <Python Support>` environment, in a :ref:`virtualenv <Virtualenv Creation>`:

    .. code-block:: shell

       pip install -U pip wheel
       pip install "dnikit[complete]"

    The preceding command will install the main DNIKit package, TensorFlow2 and PyTorch
    compatibility, and requirements to run the notebook examples.

    **Note:** For dev installation or to install from a specific branch, refer to the
    :ref:`Contributor's Guide <contributing>`.


.. _python_support:

Python Support
--------------
DNIKit currently supports Python version 3.7 or greater for macOS or Linux.
Python 3.9 is recommended. Note: to run TensorFlow 1, install Python 3.7.

To install Python version 3.9 (recommended):

**MacOS:** The Python installer package can be
`downloaded <https://www.python.org/ftp/python/3.9.13/python-3.9.13-macosx10.9.pkg>`_ from the
`Python.org <https://www.python.org/>`_ website. During installation, deselecting
GUI applications, UNIX command-line tools, and Python documentation will reduce the size of what
is installed.

**Ubuntu:**

.. code-block:: shell

    sudo apt install -y python3.9-dev python3.9-venv python3.9-tk
    sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

Virtualenv Creation
-------------------
It's recommended to use a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ to
manage all dependencies:

.. code-block:: shell

    python3.9 -m venv .venv39
    source .venv37/bin/activate

And update pip and wheel::

    pip install --upgrade pip wheel

Installation with pip
----------------------
The base DNIKit is installed with pip as follows::

    pip install dnikit

DNIKit has additional installation options, which are installed in square brackets, using quotes::

    pip install "dnikit[dataset-report,tensorflow,...]"

Here are the options currently available:

+---------------------------+----------------------------------------------------------------------+
| Module                    | Description                                                          |
+===========================+======================================================================+
| dnikit                    | Always installed, base DNIKit, with                                  |
|                           | :ref:`Familiarity <familiarity>`, :ref:`PFA <network_compression>`,  |
|                           | :ref:`INA <inactive_units>`, and                                     |
|                           | :ref:`DimensionReduction <dimension_reduction>`.                     |
+---------------------------+----------------------------------------------------------------------+
| -> [notebook]             | Installs dependencies to run and visualize the jupyter notebook      |
|                           | tutorials, including jupyter_, matplotlib_, pandas_, ...             |
+---------------------------+----------------------------------------------------------------------+
| -> [image]                | Installs opencv_ (headless) and Pillow to enable image processing    |
|                           | capabilities.                                                        |
+---------------------------+----------------------------------------------------------------------+
| -> [dimreduction]         | Installs umap_learn_ and pacmap for dimensionality reduction.        |
+---------------------------+----------------------------------------------------------------------+
| -> [dataset-report]       | Installs all requirements to run the Dataset Report.                 |
+---------------------------+----------------------------------------------------------------------+
| -> [tensorflow]           | Installs :ref:`dnikit_tensorflow <tensorflow_api>` and TF2 to load & |
|                           | run TF_ models within DNIKit.                                        |
+---------------------------+----------------------------------------------------------------------+
| -> [tensorflow1]          | Installs :ref:`dnikit_tensorflow <tensorflow_api>` and TF1 to load & |
|                           | run TF_ models within DNIKit. Must have Python <=3.7 due to TF 1.    |
+---------------------------+----------------------------------------------------------------------+
| -> [tensorflow1-gpu]      | Same as preceding row, but with TensorFlow GPU. Must have            |
|                           | Python <=3.7 due to TF 1 constraints.                                |
+---------------------------+----------------------------------------------------------------------+
| -> [torch]                | Installs ``dnikit_pytorch``: convert between PyTorch Dataset and     |
|                           | DNIKit Producer.                                                     |
+---------------------------+----------------------------------------------------------------------+
| -> [complete]             | Installs ``notebook``, ``image``, ``dimreduction``,                  |
|                           | ``dataset-report``, ``tensorflow``, & ``torch`` options.             |
+---------------------------+----------------------------------------------------------------------+

.. _TF: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf
.. _jupyter: https://jupyter.readthedocs.io/en/latest/running.html#running
.. _matplotlib: https://matplotlib.org
.. _pandas: https://pandas.pydata.org/docs/
.. _opencv: https://docs.opencv.org/master/
.. _umap_learn: https://umap-learn.readthedocs.io

Running the Jupyter Notebooks Examples
--------------------------------------

First, install the notebook dependencies::

    pip install "dnikit[notebook]"

Next, download the
`DNIKit notebooks directly <https://github.com/apple/dnikit/tree/main/docs/notebooks>`_
or use them via :ref:`cloning the dnikit repository <Clone the Code>`.


Finally, launch jupyter to open the notebooks::

    jupyter notebook

Installation for developers
============================

Check out the :ref:`Development Installation` page to install DNIKit for development.

Issues with installation?
=========================
Please file an issue in the GitHub repository.

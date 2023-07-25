=================
Introspectors API
=================

.. contents:: Contents
    :local:


Data Introspectors
-------------------
Familiarity
~~~~~~~~~~~
.. autoclass:: dnikit.introspectors.Familiarity
    :members:

.. autoclass:: dnikit.introspectors.FamiliarityStrategyType
    :members:
    :special-members: __call__

.. autoclass:: dnikit.introspectors.FamiliarityResult
    :members:

.. autoclass:: dnikit.introspectors.GMMCovarianceType
    :members:

.. autoclass:: dnikit.introspectors.FamiliarityDistribution
    :members: compute_familiarity_score

Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dnikit.introspectors.DimensionReduction
    :members:
    :exclude-members: check_batch_size, default_batch_size, fit_complete, fit_incremental,
        is_one_shot, transform, transform_one_shot

.. py:data:: OneOrManyDimStrategies

alias of Union[DimensionReductionStrategyType, Mapping[str, DimensionReductionStrategyType]]

.. autoclass:: dnikit.introspectors.DimensionReductionStrategyType
    :members:

Duplicates
~~~~~~~~~~

.. autoclass:: dnikit.introspectors.Duplicates
    :members: ThresholdStrategy, DuplicateSetCandidate, introspect, results, count
    :undoc-members:

.. autoclass:: dnikit.introspectors.DuplicatesThresholdStrategyType
    :special-members: __call__


Dataset Report
~~~~~~~~~~~~~~

.. autoclass:: dnikit.introspectors.DatasetReport
    :members:

.. autoclass:: dnikit.introspectors.ReportConfig
    :members:


Model Introspectors
-------------------

Principal Filter Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: dnikit.introspectors.PFA
    :members:

.. autoclass:: dnikit.introspectors.PFAKLDiagnostics
    :members:

.. autoclass:: dnikit.introspectors.PFAEnergyDiagnostics
    :members:

.. autoclass:: dnikit.introspectors.PFARecipe
    :members:

.. autoclass:: dnikit.introspectors.PFAUnitSelectionStrategyType
    :special-members: __call__

.. autoclass:: dnikit.introspectors.PFAStrategyType
    :special-members: __call__

.. autoclass:: dnikit.introspectors.PFACovariancesResult
    :members:

Inactive Unit Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dnikit.introspectors.IUA
    :members:

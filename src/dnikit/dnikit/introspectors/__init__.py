#
# Copyright 2020 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# DimReduction imports
from ._dim_reduction._dimension_reduction import DimensionReduction, OneOrManyDimStrategies
from ._dim_reduction._protocols import DimensionReductionStrategyType

# Duplicates imports
from ._duplicates import (
    DuplicatesThresholdStrategyType,
    Duplicates,
)

# Familiarity imports
from ._familiarity._protocols import (
    FamiliarityStrategyType,
    FamiliarityResult,
    FamiliarityDistribution
)
from ._familiarity._familiarity import Familiarity
from ._familiarity._gmm_familiarity import GMMCovarianceType

# IUA imports
from ._iua._iua import IUA

# PFA imports
from ._pfa._recommendation import (
    PFAKLDiagnostics,
    PFAEnergyDiagnostics,
    PFARecipe,
)
from ._pfa._pfa_units import PFAUnitSelectionStrategyType
from ._pfa._pfa_algorithms import PFAStrategyType
from ._pfa._covariances_calculator import PFACovariancesResult
from ._pfa._pfa import PFA

# Dataset Report imports
from ._report._dataset_report import DatasetReport
from ._report._dataset_report_stages import ReportConfig


# Public element lists
__all__ = [
    DimensionReduction.__name__,
    DimensionReductionStrategyType.__name__,
    'OneOrManyDimStrategies',
    Duplicates.__name__,
    DuplicatesThresholdStrategyType.__name__,
    IUA.__name__,
    FamiliarityDistribution.__name__,
    FamiliarityStrategyType.__name__,
    FamiliarityResult.__name__,
    Familiarity.__name__,
    GMMCovarianceType.__name__,
    PFA.__name__,
    PFAKLDiagnostics.__name__,
    PFAEnergyDiagnostics.__name__,
    PFARecipe.__name__,
    PFAUnitSelectionStrategyType.__name__,
    PFAStrategyType.__name__,
    PFACovariancesResult.__name__,
    DatasetReport.__name__,
    ReportConfig.__name__,
]

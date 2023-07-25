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

import dnikit.typing._types as t


class Introspector(t.Protocol):
    """
    Define the ``Introspector`` protocol.

    All DNIKit algorithms implement the ``Introspector`` protocol, which specifies
    that algorithms **must** implement a static factory method :func:`introspect`.

    When ``<Algorithm>.introspect(...)`` is called, the producers will be triggered
    so that the algorithms can consume data.

    Note:
        The arguments and return type of :func:`introspect` are algorithm dependent.

    The current list of DNIKit algorithms includes:

      * :class:`PFA <dnikit.introspectors.PFA>` – Principal Filtering Analysis
      * :class:`IUA <dnikit.introspectors.IUA>` – Inactive Unit Analysis
      * :class:`DimensionReduction <dnikit.introspectors.DimensionReduction>`
        – Dimension Reduction, e.g. PCA
      * :class:`Familiarity <dnikit.introspectors.Familiarity>`
      * :class:`Duplicates <dnikit.introspectors.Duplicates>`
      * :class:`DatasetReport <dnikit.introspectors.DatasetReport>`
        - Runs Familiarity, Duplicates, DimensionReduction
    """

    introspect: t.Callable[..., t.Any]
    """
    Static factory method shared by all implementations of ``Introspector``.

    Note:
        Calling this method will trigger any :class:`Producer` or :func:`pipeline()`
        that is passed as an argument.

    Note:
        **[For DNIKit contributors]** All new introspectors must, obviously, implement
        this protocol. Moreover an introspector is supposed to behave like a
        factory method (ie return a new instance of the introspector).
    """

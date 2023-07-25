======
Typing
======

This module contains all DNIKit custom types.

.. automodule:: dnikit.typing
    :members:
    :undoc-members:
    :show-inheritance:

.. py:data:: dnikit.typing.OneOrMany

alias of Union[_T, Collection[_T]]

.. py:data:: dnikit.typing.OneManyOrNone

alias of Union[None, _T, Collection[_T]]

.. py:data:: dnikit.typing.PathOrStr

alias of Union[str, pathlib.Path]

.. py:data:: dnikit.typing.StringLike

alias of Any.
Similar to "array like" -- these are types that can be losslessly converted
to/from string and they might be used as Identifiers in a Batch.

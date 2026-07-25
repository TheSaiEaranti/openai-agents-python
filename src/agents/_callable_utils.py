from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def get_callable_call_descriptor(func: Callable[..., Any]) -> tuple[type[Any], Any]:
    """Return the class that defines ``__call__`` and its raw descriptor."""
    for owner in type(func).__mro__:
        if "__call__" in owner.__dict__:
            return owner, owner.__dict__["__call__"]
    raise TypeError(f"{func!r} has no __call__ descriptor")


def unwrap_callable_descriptor(descriptor: Any) -> Any:
    """Return the callable behind method and partialmethod descriptors."""
    while isinstance(descriptor, classmethod | staticmethod):
        descriptor = descriptor.__func__
    if isinstance(descriptor, functools.partialmethod):
        return unwrap_callable_descriptor(descriptor.func)
    return descriptor

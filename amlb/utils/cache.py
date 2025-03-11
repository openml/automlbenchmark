from __future__ import annotations
import logging
import functools
from functools import cached_property
from typing import Any, Sequence

log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


def clear_cache(obj: Any, functions: Sequence[str] | None = None) -> None:
    attributes_to_check = functions or dir(type(obj))
    # Must be careful to check the definitions on the class, checking on the instance
    # will trigger invocations of the cached properties through `getattr`.
    cached_properties = [
        name
        for name in attributes_to_check
        if isinstance(getattr(type(obj), name), cached_property)
    ]
    _functions = [fn for fn in attributes_to_check if callable(getattr(type(obj), fn))]

    cleared_properties = []
    for property_ in cached_properties:
        # You need to delete the attribute to evict the cache of a cached property,
        # but you cannot check for it with hasattr as that would invoke it.
        try:
            delattr(obj, property_)
            cleared_properties.append(property_)
        except AttributeError:
            pass

    for cached_function in [getattr(obj, fn) for fn in _functions]:
        if hasattr(cached_function, "cache_clear"):
            cached_function.cache_clear()
            cleared_properties.append(cached_function.__name__)

    log.debug("Cleared cached properties: %s.", cleared_properties)


def cached(fn):
    return functools.cache(fn)


def memoize(fn):
    return functools.cache(fn)


def lazy_property(prop_fn):
    return cached_property(functools.cache(prop_fn))


__all__ = [s for s in dir() if not s.startswith("_") and s not in __no_export]

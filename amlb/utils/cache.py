from __future__ import annotations
import logging
import functools
from functools import cached_property
from typing import Any, Sequence

log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported


def clear_cache(obj: Any, functions: Sequence[str] | None = None) -> None:
    attributes_to_check = functions or dir(type(obj))
    cached_properties = [
        name
        for name in attributes_to_check
        if isinstance(getattr(type(obj), name), cached_property)
    ]
    for property_ in cached_properties:
        delattr(obj, property_)

    log.debug("Cleared cached properties: %s.", cached_properties)


def cached(fn):
    return functools.cache(fn)


def memoize(fn):
    return functools.cache(fn)


def lazy_property(prop_fn):
    return cached_property(functools.cache(prop_fn))


__all__ = [s for s in dir() if not s.startswith("_") and s not in __no_export]

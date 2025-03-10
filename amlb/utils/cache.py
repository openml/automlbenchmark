import logging
import functools


log = logging.getLogger(__name__)

__no_export = set(dir())  # all variables defined above this are not exported

_CACHE_PROP_PREFIX_ = "__cached__"


def _cached_property_name(fn):
    return _CACHE_PROP_PREFIX_ + (fn.__name__ if hasattr(fn, "__name__") else str(fn))


def clear_cache(obj, functions=None):
    attributes = {att: getattr(obj, att) for att in (functions or dir(obj.__class__))}
    functions = {
        name: fn for name, fn in attributes.items() if callable(getattr(obj, name))
    }
    properties = {
        name: getattr(obj.__class__, name).fget
        for name, prop in attributes.items()
        if isinstance(getattr(obj.__class__, name), property)
    }
    # Note that it is not possible to evict specific entries from the lru cache,
    # this means that the property and methods will be cleared for *all* objects
    # obj's class. In practice, doesn't really come into play since clear_cached
    # is only called on Dataset and Datasplit objects.
    cleared_properties = []
    for name, fn in (functions | properties).items():
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
            cleared_properties.append(name)
    log.debug("Cleared cached properties: %s.", cleared_properties)


def cached(fn):
    return functools.cache(fn)


def memoize(fn):
    return functools.cache(fn)


def lazy_property(prop_fn):
    return property(functools.cache(prop_fn))


__all__ = [s for s in dir() if not s.startswith("_") and s not in __no_export]

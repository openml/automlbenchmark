import logging

log = logging.getLogger(__name__)

_CACHE_PROP_PREFIX_ = '__cached__'


def _cached_property_name(fn):
    return _CACHE_PROP_PREFIX_ + (fn.__name__ if hasattr(fn, '__name__') else str(fn))


def clear_cache(self, functions=None):
    cached_properties = [prop for prop in dir(self) if prop.startswith(_CACHE_PROP_PREFIX_)]
    properties_to_clear = cached_properties if functions is None \
        else [prop for prop in [_cached_property_name(fn) for fn in functions] if prop in cached_properties]
    for prop in properties_to_clear:
        delattr(self, prop)
    log.debug("Cleared cached properties: %s.", properties_to_clear)


def cache(self, key, fn):
    """

    :param self: the object that will hold the cached value
    :param key: the key/attribute for the cached value
    :param fn: the function returning the value to be cached
    :return: the value returned by fn on first call
    """
    if not hasattr(self, key):
        value = fn(self)
        setattr(self, key, value)
    return getattr(self, key)


def cached(fn):
    """

    :param fn:
    :return:
    """
    result = _cached_property_name(fn)

    def decorator(self):
        return cache(self, result, fn)

    return decorator


def memoize(fn):
    prop_name = _cached_property_name(fn)

    def decorator(self, key=None):  # TODO: could support unlimited args by making a tuple out of *args + **kwargs: not needed for now
        memo = cache(self, prop_name, lambda _: {})
        if not isinstance(key, str) and hasattr(key, '__iter__'):
            key = tuple(key)
        if key not in memo:
            memo[key] = fn(self) if key is None else fn(self, key)
        return memo[key]

    return decorator


def lazy_property(prop_fn):
    """

    :param prop_fn:
    :return:
    """
    prop_name = _cached_property_name(prop_fn)

    @property
    def decorator(self):
        return cache(self, prop_name, prop_fn)

    return decorator

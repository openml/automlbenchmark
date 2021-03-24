from ast import literal_eval
import base64
from collections import defaultdict
from collections.abc import Iterable, Sized
from functools import reduce, wraps
import hashlib
import json
import logging
import pprint
import re
import threading

log = logging.getLogger(__name__)


class Namespace:

    printer = pprint.PrettyPrinter(indent=2, compact=True)

    @staticmethod
    def parse(*args, **kwargs):
        raw = dict(*args, **kwargs)
        parsed = Namespace()
        dots, nodots = partition(raw.keys(), lambda s: '.' in s)
        for k in nodots:
            v = raw[k]
            try:
                if isinstance(v, str):
                    v = literal_eval(v)
            except Exception:
                pass
            parsed[k] = v
        sublevel = {}
        for k in dots:
            k1, k2 = k.split('.', 1)
            entry = [(k2, raw[k])]
            if k1 in sublevel:
                sublevel[k1].update(entry)
            else:
                sublevel[k1] = dict(entry)
        for k, v in sublevel.items():
            parsed[k] = Namespace.parse(v)
        return parsed

    @staticmethod
    def merge(*namespaces, deep=False):
        merged = Namespace()
        for ns in namespaces:
            if ns is None:
                continue
            if not deep:
                merged + ns
            else:
                for k, v in ns:
                    if isinstance(v, Namespace):
                        merged[k] = Namespace.merge(merged[k], v, deep=True)
                    else:
                        merged[k] = v
        return merged

    @staticmethod
    def dict(namespace, deep=True):
        dic = dict(namespace)
        if not deep:
            return dic
        for k, v in dic.items():
            if isinstance(v, Namespace):
                dic[k] = Namespace.dict(v)
        return dic

    @staticmethod
    def from_dict(dic, deep=True):
        ns = Namespace(dic)
        if not deep:
            return ns
        for k, v in ns:
            if isinstance(v, dict):
                ns[k] = Namespace.from_dict(v)
        return ns

    @staticmethod
    def walk(namespace, fn, inplace=False):
        def _walk(namespace, fn, parents=None, inplace=inplace):
            parents = [] if parents is None else parents
            ns = namespace if inplace else Namespace()
            for k, v in namespace:
                nk, nv = fn(k, v, parents=parents)
                if nk is not None:
                    if v is nv and isinstance(v, Namespace):
                        nv = _walk(nv, fn, parents=parents+[k], inplace=inplace)
                    ns[nk] = nv
            return ns

        return _walk(namespace, fn, inplace=inplace)

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            self.__dict__ = defaultdict(args[0])
            args = args[1:]
        self.__dict__.update(dict(*args, **kwargs))

    def __add__(self, other):
        """extends self with other (always overrides)"""
        if other is not None:
            self.__dict__.update(other)
        return self

    def __mod__(self, other):
        """extends self with other (adds only missing keys)"""
        if other is not None:
            for k, v in other:
                self.__dict__.setdefault(k, v)
        return self

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getattr__(self, item):
        if isinstance(self.__dict__, defaultdict):
            return self.__dict__[item]
        raise AttributeError(f"'Namespace' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        self.__dict__.pop(key, None)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __copy__(self):
        return Namespace(self.__dict__.copy())

    def __dir__(self):
        return list(self.__dict__.keys())

    def __eq__(self, other):
        return isinstance(other, Namespace) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.__dict__)

    def __str__(self):
        return Namespace.printer.pformat(Namespace.dict(self))

    def __repr__(self):
        return repr(self.__dict__)

    def __json__(self):
        return Namespace.dict(self)


def repr_def(obj):
    return "{clazz}({attributes})".format(clazz=type(obj).__name__, attributes=', '.join(("{}={}".format(k, repr(v)) for k, v in obj.__dict__.items())))


def noop(*args, **kwargs):
    pass


def identity(x, *args):
    return (x,) + args if args else x


def as_list(*args):
    if len(args) == 0:
        return list()
    elif len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], str):
        return list(args[0])
    return list(args)


def flatten(iterable, flatten_tuple=False, flatten_dict=False):
    return reduce(lambda l, r: (l.extend(r) if isinstance(r, (list, tuple) if flatten_tuple else list)
                                else l.extend(r.items()) if flatten_dict and isinstance(r, dict)
                                else l.append(r)) or l, iterable, [])


def partition(iterable, predicate=id):
    truthy, falsy = [], []
    for i in iterable:
        if predicate(i):
            truthy.append(i)
        else:
            falsy.append(i)
    return truthy, falsy


def translate_dict(dic, translation_dict):
    tr = dict()
    for k, v in dic.items():
        if k in translation_dict:
            tr[translation_dict[k]] = v
        else:
            tr[k] = v
    return tr


def str2bool(s):
    if s.lower() in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', 'off', '0'):
        return False
    else:
        raise ValueError(s+" can't be interpreted as a boolean.")


_empty_ = "__empty__"


def str_def(o, if_none='', if_empty=_empty_):
    if o is None:
        return if_none
    if if_empty != _empty_ and isinstance(o, Sized) and len(o) == 0:
        return if_empty
    return str(o)


def str_sanitize(s):
    return re.sub(r"[^\w-]", "_", s)


def str_digest(s):
    return base64.b64encode(hashlib.md5(s.encode()).digest()).decode()


def head(s, lines=10):
    s_lines = s.splitlines() if s else []
    return '\n'.join(s_lines[:lines])


def tail(s, lines=10, from_line=None, include_line=True):
    if s is None:
        return None if from_line is None else None, None

    s_lines = s.splitlines()
    start = -lines
    if isinstance(from_line, int):
        start = from_line
        if not include_line:
            start += 1
    elif isinstance(from_line, str):
        try:
            start = s_lines.index(from_line)
            if not include_line:
                start += 1
        except ValueError:
            start = 0
    last_line = dict(index=len(s_lines) - 1,
                     line=s_lines[-1] if len(s_lines) > 0 else None)
    t = '\n'.join(s_lines[start:])
    return t if from_line is None else (t, last_line)


def fn_name(fn):
    return ".".join([fn.__module__, fn.__qualname__])


def json_load(file, as_namespace=False):
    with open(file, 'r') as f:
        return json_loads(f.read(), as_namespace=as_namespace)


def json_loads(s, as_namespace=False):
    if as_namespace:
        return json.loads(s, object_hook=lambda dic: Namespace(**dic))
    else:
        return json.loads(s)


def json_dump(o, file, style='default'):
    with open(file, 'w') as f:
        f.write(json_dumps(o, style=style))


def json_dumps(o, style='default'):
    """

    :param o:
    :param style: str among ('compact', 'default', 'pretty').
                - `compact` removes all blanks (no space, no newline).
                - `default` adds a space after each separator but prints on one line
                - `pretty` adds a space after each separator and indents after opening brackets.
    :return:
    """
    separators = (',', ':') if style == 'compact' else None
    indent = 4 if style == 'pretty' else None

    def default_encode(o):
        if hasattr(o, '__json__') and callable(o.__json__):
            return o.__json__()
        return json.encoder.JSONEncoder.default(None, o)

    return json.dumps(o, indent=indent, separators=separators, default=default_encode)


#################################
# Thread-safe utility functions #
#################################

class ThreadSafeCounter:

    def __init__(self, value=0):
        self.value = value
        self._lock = threading.Lock()

    def inc(self):
        with self._lock:
            self.value += 1

    def dec(self):
        with self._lock:
            self.value -= 1


def threadsafe_iterator(it):
    """
    Wrapper making an iterator thread-safe.
    """
    it = iter(it)
    lock = threading.Lock()
    while True:
        try:
            with lock:
                yield next(it)
        except StopIteration:
            return


def threadsafe_generator(fn):
    """
    Decorator making a generator thread-safe.
    """
    def gen(*args, **kwargs):
        return threadsafe_iterator(fn(*args, **kwargs))
    return gen


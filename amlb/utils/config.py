from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from typing import Callable, List, Union

from ruamel.yaml.constructor import SafeConstructor
from ruamel.yaml.main import YAML

from .core import Namespace, identity, json_load
from .os import normalize_path

log = logging.getLogger(__name__)


class YAMLNamespaceConstructor(SafeConstructor):

    @classmethod
    def init(cls):
        cls.add_constructor(u'tag:yaml.org,2002:map', cls.construct_yaml_map)

    def construct_yaml_map(self, node):
        data = Namespace()
        yield data
        value = self.construct_mapping(node)
        data += value


YAMLNamespaceConstructor.init()


def yaml_load(file, as_namespace=False):
    if as_namespace:
        yaml = YAML(typ='safe', pure=True)
        yaml.Constructor = YAMLNamespaceConstructor
    else:
        yaml = YAML(typ='safe')
    return yaml.load(file)


def config_load(path, verbose=False):
    path = normalize_path(path)
    if not os.path.isfile(path):
        log.log(logging.WARNING if verbose else logging.DEBUG, "No config file at `%s`, ignoring it.", path)
        return Namespace()

    _, ext = os.path.splitext(path.lower())
    loader = json_load if ext == 'json' else yaml_load
    log.log(logging.INFO if verbose else logging.DEBUG, "Loading config file `%s`.", path)
    with open(path, 'r') as file:
        return loader(file, as_namespace=True)


# TransformRule = namedtuple('TransformRule',
#                            ['from_key', 'to_key', 'fn', 'keep_from'],
#                            defaults=[None, identity, False],
#                            module=__name__)
@dataclass
class TransformRule:
    from_key: Union[str, List[str]]
    to_key: str = None
    fn: Callable = identity
    keep_from: bool = False


def transform_config(config: Namespace, transform_rules: [TransformRule], inplace=True) -> Namespace:
    """
    Allows to modify a configuration namespace (for example if the configuration format is modified)
    by applying a list of transformation rules.
    :param config: the config to be transformed.
    :param transform_rules: a list of transformation rules.
    :param inplace: if True, the config is modified inplace.
    :return: the transformed config namespace.
    """
    if not inplace:
        config = deepcopy(config)
    for rule in transform_rules:
        from_keys = [rule.from_key] if isinstance(rule.from_key, str) else rule.from_key
        from_vals = [Namespace.get(config, k) for k in from_keys]
        if all(v is not None for v in from_vals):
            to_val = rule.fn(*from_vals)
            to_key = rule.to_key or rule.from_key
            Namespace.set(config, to_key, to_val)
        if not rule.keep_from:
            for k in from_keys:
                Namespace.delete(config, k)
    return config


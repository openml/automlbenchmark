import logging
import os

from ruamel import yaml

from amlb.utils.core import Namespace, json_load
from amlb.utils.os import normalize_path

log = logging.getLogger(__name__)


class YAMLNamespaceLoader(yaml.loader.SafeLoader):

    @classmethod
    def init(cls):
        cls.add_constructor(u'tag:yaml.org,2002:map', cls.construct_yaml_map)

    def construct_yaml_map(self, node):
        data = Namespace()
        yield data
        value = self.construct_mapping(node)
        data + value


YAMLNamespaceLoader.init()


def yaml_load(file, as_namespace=False):
    if as_namespace:
        return yaml.load(file, Loader=YAMLNamespaceLoader)
    else:
        return yaml.safe_load(file)


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

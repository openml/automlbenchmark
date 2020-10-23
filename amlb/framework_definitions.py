import copy
import itertools
import logging
from typing import Union, List

from amlb.utils import Namespace, config_load

log = logging.getLogger(__name__)


def load_framework_definitions(frameworks_file: Union[str, List[str]], resource) -> Namespace:
    """ Load the framework definition listed in the framework file(s).

    Loads the definition(s) from the file(s),
    :param frameworks_file:
    :return: Namespace containing each framework definition,
    """
    frameworks = _load_and_merge_framework_definitions(frameworks_file)
    _sanitize_and_add_defaults(frameworks, resource)
    log.debug("Available framework definitions:\n%s", frameworks)
    return frameworks


def _load_and_merge_framework_definitions(frameworks_file: Union[str, List[str]]) -> Namespace:
    """ Load and merge the framework file(s), does not allow duplicate definitions. """
    log.info("Loading frameworks definitions from %s.", frameworks_file)
    if not isinstance(frameworks_file, list):
        frameworks_file = [frameworks_file]

    definitions_by_file = [config_load(file) for file in frameworks_file]
    for d1, d2 in itertools.combinations([set(dir(d)) for d in definitions_by_file], 2):
        if d1.intersection(d2) != set():
            raise ValueError(f"Duplicate entry '{d1.intersection(d2).pop()}' found.")
    return Namespace.merge(*definitions_by_file)


def _sanitize_definitions(frameworks: Namespace):
    """ Normalize names, add name field, remove invalid extensions. """
    _add_framework_name(frameworks)
    _remove_frameworks_with_unknown_parent(frameworks)
    _remove_self_reference_extensions(frameworks)


def _sanitize_and_add_defaults(frameworks, resource):
    _sanitize_definitions(frameworks)

    # `module` is the only field that should have a default
    # based on the parent. For that reason we add it before
    # we update children with their parent fields.
    for _, framework in frameworks:
        if "extends" not in framework:
            _add_default_module(framework, resource.config)
    _update_frameworks_with_parent_definitions(frameworks)

    _add_defaults_to_frameworks(frameworks, resource)


def _add_framework_name(frameworks: Namespace):
    """ Adds a 'name' attribute to each framework. """
    for name, framework in frameworks:
        framework.name = name


def _add_default_module(framework, config):
    if "module" not in framework:
        framework.module = f"{config.frameworks.root_module}.{framework.name}"


def _add_default_version(framework):
    if "version" not in framework:
        framework.version = "latest"


def _add_default_setup_args(framework):
    if "setup_args" in framework and isinstance(framework.setup_args, str):
        framework.setup_args = [framework.setup_args]
    else:
        framework.setup_args = [framework.version]
        if "repo" in framework:
            framework.setup_args.append(framework.repo)


def _add_default_setup_script(framework, resource):
    if "setup_script" not in framework:
        framework.setup_script = None
    else:
        framework.setup_script = framework.setup_script.format(
            module=framework.module,
            **resource._common_dirs,
        )


def _add_default_setup_cmd(framework, resource):
    """ Defines default setup_cmd and _setup_cmd, interpolate commands if necessary.

    The default values are `None`.
    In case a setup_cmd is defined, the original definition is saved to `_setup_cmd`.
    The new `setup_cmd` will be a list of commands, where each command has the
    directories, package manager and python binary interpolated.
    `_setup_cmd` will be used for setup inside containers.
    `setup_cmd` will be used when running locally (or on an Amazon image).
    """
    if "setup_cmd" not in framework:
        framework._setup_cmd = None
        framework.setup_cmd = None
    else:
        framework._setup_cmd = framework.setup_cmd
        if isinstance(framework.setup_cmd, str):
            framework.setup_cmd = [framework.setup_cmd]
        framework.setup_cmd = [
            cmd.format(pip="{pip}", py="{py}", **resource._common_dirs)
            for cmd in framework.setup_cmd
        ]


def _add_default_params(framework):
    if "params" not in framework:
        framework.params = dict()
    else:
        framework.params = Namespace.dict(framework.params)


def _add_default_image(framework: Namespace, config: Namespace):
    if "image" not in framework:
        framework.image = copy.deepcopy(config.docker.image_defaults)
    else:
        framework.image = Namespace.merge(config.docker.image_defaults, framework.image)

    if framework.image.tag is None:
        framework.image.tag = framework.version.lower()

    if framework.image.image is None:
        framework.image.image = framework.name

    if framework.image.author is None:
        framework.image.author = ""


def _find_all_parents(framework, frameworks):
    """ Return all definitions framework extends, from direct parent to furthest. """
    parents = []
    while "extends" in framework and framework.extends is not None:
        framework = frameworks[framework.extends]
        parents.append(framework)
    return parents


def _update_frameworks_with_parent_definitions(frameworks: Namespace):
    """ Add fields defined by ancestors

    Extensions do not overwrite fields defined on the framework itself.
    If multiple parents define the same field, the parent that is 'closer'
    to the child framework defines the field value.
    """
    for name, framework in frameworks:
        parents = _find_all_parents(framework, frameworks)
        for parent in parents:
            framework % copy.deepcopy(parent)


def _add_defaults_to_frameworks(frameworks: Namespace, resource):
    for _, framework in frameworks:
        _add_default_version(framework)
        _add_default_setup_args(framework)
        _add_default_params(framework)
        _add_default_image(framework, resource.config)
        _add_default_setup_cmd(framework, resource)
        _add_default_setup_script(framework, resource)


def _remove_self_reference_extensions(frameworks: Namespace):
    for name, framework in frameworks:
        if "extends" in framework and framework.extends == framework.name:
            log.warning("Framework %s extends itself: removing extension.",
                        framework.name)
            framework.extends = None


def _remove_frameworks_with_unknown_parent(frameworks: Namespace):
    frameworks_with_unknown_parent = [
        (name, framework.extends) for name, framework in frameworks
        if "extends" in framework and framework.extends not in frameworks
    ]
    for framework, parent in frameworks_with_unknown_parent:
        log.warning("Removing framework %s as parent %s doesn't exist.", framework, parent)
        del frameworks[framework]

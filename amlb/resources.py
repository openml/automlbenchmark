"""
**resources** modules exposes a singleton ``Resources`` instance providing easy access to app configuration properties,
as well as handy methods to access other resources like *automl frameworks* and *benchmark definitions*
"""
import copy
import itertools
import logging
import os
from queue import Queue
import random
import re
import sys
from typing import List, Union, Optional

from amlb.benchmarks.parser import benchmark_load
from .utils import Namespace, config_load, lazy_property, memoize, normalize_path, touch


log = logging.getLogger(__name__)


class Resources:

    @staticmethod
    def _normalize(config: Namespace, replace=None):
        def nz_path(path):
            if replace is not None:
                path = path.format(**replace)
            return normalize_path(path)

        normalized = copy.copy(config)
        for k, v in config:
            if isinstance(v, Namespace):
                normalized[k] = Resources._normalize(v, replace=replace)
            elif re.search(r'_(dir|file|cmd)s?$', k):
                normalized[k] = [nz_path(p) for p in v] if isinstance(v, list) else nz_path(v)
        return normalized

    def __init__(self, config: Namespace):
        self._config = config
        self._common_dirs = dict(
            input=normalize_path(config.input_dir),
            output=normalize_path(config.output_dir),
            user=normalize_path(config.user_dir),
            root=normalize_path(config.root_dir),
        )
        self.config = Resources._normalize(config, replace=self._common_dirs)
        log.debug("Using config:\n%s", self.config)

        # allowing to load custom modules from user directory
        sys.path.append(self._common_dirs['user'])
        log.debug("Extended Python sys.path to user directory: %s.", sys.path)

    @lazy_property
    def project_info(self):
        split_url = self.config.project_repository.split('#', 2)
        repo = split_url[0]
        tag = None if len(split_url) == 1 else split_url[1]
        branch = tag or 'master'
        return Namespace(
            repo=repo,
            tag=tag,
            branch=branch
        )

    def seed(self, fold=None):
        if isinstance(fold, int) and str(self.config.seed).lower() in ['auto']:
            return fold+self._seed
        else:
            return self._seed

    @lazy_property
    def _seed(self):
        if str(self.config.seed).lower() in ['none', '']:
            return None
        elif str(self.config.seed).lower() in ['auto']:
            return random.randint(1, (1 << 32) - 1)  # limiting seed to int32
        else:
            return self.config.seed

    def framework_definition(self, name):
        """
        :param name:
        :return: name of the framework as defined in the frameworks definition file
        """
        framework = self._frameworks[name.lower()]
        if not framework:
            raise ValueError("Incorrect framework `{}`: not listed in {}.".format(name, self.config.frameworks.definition_file))
        return framework, framework.name

    @lazy_property
    def _frameworks(self):
        frameworks_file = self.config.frameworks.definition_file
        return load_framework_definitions(frameworks_file)

    @memoize
    def constraint_definition(self, name):
        """
        :param name: name of the benchmark constraint definition as defined in the constraints file
        :return: a Namespace object with the constraint config (folds, cores, max_runtime_seconds, ...) for the current benchmamk run.
        """
        constraint_config = self._constraints[name.lower()]
        if not constraint_config:
            raise ValueError("Incorrect constraint definition `{}`: not listed in {}.".format(name, self.config.benchmarks.constraints_file))
        return constraint_config, constraint_config.name

    @lazy_property
    def _constraints(self):
        constraints_file = self.config.benchmarks.constraints_file
        log.info("Loading benchmark constraint definitions from %s.", constraints_file)
        if not isinstance(constraints_file, list):
            constraints_file = [constraints_file]

        constraints = Namespace()
        for ef in constraints_file:
            constraints + config_load(ef)

        for name, c in constraints:
            c.name = name

        log.debug("Available benchmark constraints:\n%s", constraints)
        constraints_lookup = Namespace()
        for name, c in constraints:
            constraints_lookup[name.lower()] = c
        return constraints_lookup

    # @memoize
    def benchmark_definition(self, name, defaults=None):
        """
        :param name: name of the benchmark as defined by resources/benchmarks/{name}.yaml, the path to a user-defined benchmark description file or a study id.
        :param defaults: defaults used as a base config for each task in the benchmark definition
        :return:
        """
        hard_defaults, tasks, benchmark_path, benchmark_name = benchmark_load(name, self.config.benchmarks.definition_dir)

        defaults = Namespace.merge(defaults, hard_defaults, Namespace(name='__defaults__'))
        for task in tasks:
            task % defaults   # add missing keys from hard defaults + defaults
            self._validate_task(task)

        self._validate_task(defaults, lenient=True)
        defaults.enabled = False
        tasks.append(defaults)
        log.debug("Available task definitions:\n%s", tasks)
        return tasks, benchmark_name, benchmark_path

    def _validate_task(self, task, lenient=False):
        missing = []
        for conf in ['name']:
            if task[conf] is None:
                missing.append(conf)
        if not lenient and len(missing) > 0:
            raise ValueError("{missing} mandatory properties as missing in task definition {taskdef}.".format(missing=missing, taskdef=task))

        for conf in ['max_runtime_seconds', 'cores', 'folds', 'max_mem_size_mb', 'min_vol_size_mb']:
            if task[conf] is None:
                task[conf] = self.config.benchmarks.defaults[conf]
                log.debug("Config `{config}` not set for task {name}, using default `{value}`.".format(config=conf, name=task.name, value=task[conf]))

        conf = 'id'
        if task[conf] is None:
            task[conf] = ("openml.org/t/{}".format(task.openml_task_id) if task['openml_task_id'] is not None
                          else "openml.org/d/{}".format(task.openml_dataset_id) if task['openml_dataset_id'] is not None
                          else ((task.dataset['id'] if isinstance(task.dataset, (dict, Namespace))
                                 else task.dataset if isinstance(task.dataset, str)
                                 else None) or task.name) if task['dataset'] is not None
                          else None)
            if not lenient and task[conf] is None:
                raise ValueError("task definition must contain one property among ['openml_task_id', 'dataset']")

        conf = 'metric'
        if task[conf] is None:
            task[conf] = None

        conf = 'ec2_instance_type'
        if task[conf] is None:
            i_series = self.config.aws.ec2.instance_type.series
            i_map = self.config.aws.ec2.instance_type.map
            if str(task.cores) in i_map:
                i_size = i_map[str(task.cores)]
            elif task.cores > 0:
                supported_cores = list(map(int, Namespace.dict(i_map).keys() - {'default'}))
                supported_cores.sort()
                cores = next((c for c in supported_cores if c >= task.cores), 'default')
                i_size = i_map[str(cores)]
            else:
                i_size = i_map.default
            task[conf] = '.'.join([i_series, i_size])
            log.debug("Config `{config}` not set for task {name}, using default selection `{value}`.".format(config=conf, name=task.name, value=task[conf]))

        conf = 'ec2_volume_type'
        if task[conf] is None:
            task[conf] = self.config.aws.ec2.volume_type
            log.debug("Config `{config}` not set for task {name}, using default `{value}`.".format(config=conf, name=task.name, value=task[conf]))


def load_and_merge_framework_definitions(frameworks_file: Union[str, List[str]]) -> Namespace:
    """ Load and merge the framework file(s), does not allow duplicate definitions. """
    log.info("Loading frameworks definitions from %s.", frameworks_file)
    if not isinstance(frameworks_file, list):
        frameworks_file = [frameworks_file]

    definitions_by_file = [config_load(file) for file in frameworks_file]
    for d1, d2 in itertools.combinations([set(dir(d)) for d in definitions_by_file], 2):
        if d1.intersection(d2) != set():
            raise ValueError(f"Duplicate entry '{d1.intersection(d2).pop()}' found.")
    return Namespace.merge(*definitions_by_file)


def sanitize_definitions(frameworks: Namespace):
    """ Normalize names, add name field, remove invalid extensions. """
    add_and_normalize_names(frameworks)
    remove_frameworks_with_unknown_parent(frameworks)
    remove_self_reference_extensions(frameworks)


def sanitize_and_add_defaults(frameworks, resource: Resources):
    sanitize_definitions(frameworks)

    # `module` is the only field that should have a default
    # based on the parent. For that reason we add it before
    # we update children with their parent fields.
    for _, framework in frameworks:
        if "extends" not in framework:
            autocomplete_framework_module(framework, resource.config)
    update_frameworks_with_parent_definitions(frameworks)

    add_defaults_to_frameworks(frameworks, resource)


def load_framework_definitions(frameworks_file: Union[str, List[str]], resource: Resources) -> Namespace:
    """ Load the framework definition listed in the framework file(s).

    Loads the definition(s) from the file(s),
    :param frameworks_file:
    :return: Namespace containing each framework definition,
    """
    frameworks = load_and_merge_framework_definitions(frameworks_file)
    sanitize_and_add_defaults(frameworks, resource)
    log.debug("Available framework definitions:\n%s", frameworks)
    return frameworks


def add_and_normalize_names(frameworks: Namespace):
    """ Converts each framework definition to lowercase and adds a 'name' field. """
    framework_names = dir(frameworks)
    for name in framework_names:
        framework = frameworks[name]
        framework.name = name.lower()
        if name.lower() != name:
            del frameworks[name]
            frameworks[name.lower()] = framework
        if "extends" in framework:
            framework.extends = framework.extends.lower()


def autocomplete_framework_module(framework, config):
    if "module" not in framework:
        framework.module = f"{config.frameworks.root_module}.{framework.name}"


def autocomplete_framework_version(framework):
    if "version" not in framework:
        framework.version = "latest"


def autocomplete_framework_setup_args(framework):
    if "setup_args" in framework:
        framework.setup_args = [framework.setup_args]
    else:
        framework.setup_args = [framework.version]
        if "repo" in framework:
            framework.setup_args.append(framework.repo)


def autocomplete_setup_script(framework, resource):
    if "setup_script" not in framework:
        framework.setup_script = None
    else:
        framework.setup_script = framework.setup_script.format(
            module=framework.module,
            **resource._common_dirs,
        )


def autocomplete_setup_cmd(framework, resource):
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


def autocomplete_params(framework):
    if "params" not in framework:
        framework.params = dict()
    else:
        framework.params = Namespace.dict(framework.params)


def autocomplete_image(framework: Namespace, config_: Namespace):
    if "image" not in framework:
        framework.image = copy.deepcopy(config_.docker.image_defaults)
    else:
        framework.image = Namespace.merge(config_.docker.image_defaults, framework.image)

    if framework.image.tag is None:
        framework.image.tag = framework.version.lower()

    if framework.image.image is None:
        framework.image.image = framework.name


def find_all_parents(framework, frameworks):
    """ Return all definitions framework extends, from direct parent to furthest. """
    parents = []
    while "extends" in framework:
        framework = frameworks[framework.extends]
        parents.append(framework)
    return parents


def update_frameworks_with_parent_definitions(frameworks: Namespace):
    """ Add fields defined by ancestors

    Extensions do not overwrite fields defined on the framework itself.
    If multiple parents define the same field, the parent that is 'closer'
    to the child framework defines the field value.
    """
    for name, framework in frameworks:
        parents = find_all_parents(framework, frameworks)
        for parent in parents:
            framework % copy.deepcopy(parent)


def add_defaults_to_frameworks(frameworks: Namespace, resource: Resources):
    for _, framework in frameworks:
        autocomplete_framework_version(framework)
        autocomplete_framework_setup_args(framework)
        autocomplete_params(framework)
        autocomplete_image(framework, resource.config)
        autocomplete_setup_cmd(framework, resource)
        autocomplete_setup_script(framework, resource)


def autocomplete_definition(framework: Namespace, parent: Optional[Namespace], resource):
    if parent is not None:
        framework % copy.deepcopy(parent)  # adds framework's missing keys from parent

    if framework['module'] is None:
        framework.module = '.'.join(
            [resource.config.frameworks.root_module, framework.name])

    if framework['version'] is None:
        framework.version = 'latest'

    if framework['setup_args'] is None:
        framework.setup_args = [framework.version] if framework['repo'] is None else [
            framework.version, framework.repo]
    elif isinstance(framework.setup_args, str):
        framework.setup_args = [framework.setup_args]

    if framework['setup_script'] is None:
        framework.setup_script = None
    else:
        framework.setup_script = framework.setup_script.format(**resource._common_dirs,
                                                               **dict(
                                                                   module=framework.module))
    if framework['setup_cmd'] is None:
        framework._setup_cmd = None
        framework.setup_cmd = None
    else:
        framework._setup_cmd = framework.setup_cmd
        if isinstance(framework.setup_cmd, str):
            framework.setup_cmd = [framework.setup_cmd]
        framework.setup_cmd = [cmd.format(**resource._common_dirs,
                                          **dict(pip="{pip}",
                                                 py="{py}"))
                               for cmd in framework.setup_cmd]

    if framework['params'] is None:
        framework.params = dict()
    else:
        framework.params = Namespace.dict(framework.params)

    did = copy.copy(resource.config.docker.image_defaults)
    if framework['image'] is None:
        framework['image'] = did
    for conf in ['author', 'image', 'tag']:
        if framework.image[conf] is None:
            framework.image[conf] = did[conf]
    if framework.image.image is None:
        framework.image.image = framework.name.lower()
    if framework.image.tag is None:
        framework.image.tag = framework.version.lower()


def remove_self_reference_extensions(frameworks: Namespace):
    for name, framework in frameworks:
        if "extends" in framework and framework.extends == framework.name:
            log.warning("Framework %s extends itself: removing extension.",
                        framework.name)
            framework.extends = None


def remove_frameworks_with_unknown_parent(frameworks: Namespace):
    frameworks_with_unknown_parent = [
        (name, framework.extends) for name, framework in frameworks
        if "extends" in framework and framework.extends not in frameworks
    ]
    for framework, parent in frameworks_with_unknown_parent:
        log.warning("Removing framework %s as parent %s doesn't exist.", framework, parent)
        del frameworks[framework]


__INSTANCE__: Resources = None


def from_config(config: Namespace):
    global __INSTANCE__
    __INSTANCE__ = Resources(config)


def from_configs(*configs: Namespace):
    global __INSTANCE__
    __INSTANCE__ = Resources(Namespace.merge(*configs, deep=True))


def get() -> Resources:
    return __INSTANCE__


def config():
    return __INSTANCE__.config


def output_dirs(root, session=None, subdirs=None, create=False):
    root = root if root is not None else '.'
    dirs = Namespace(
        root=root,
        session=os.path.join(root, session) if session is not None else root
    )

    subdirs = [] if subdirs is None \
        else [subdirs] if isinstance(subdirs, str) \
        else subdirs

    for d in subdirs:
        dirs[d] = os.path.join(dirs.session, d)
        if create and not os.path.exists(dirs[d]):
            touch(dirs[d], as_dir=True)
    return dirs


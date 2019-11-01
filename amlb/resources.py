"""
**resources** modules exposes a singleton ``Resources`` instance providing easy access to app configuration properties,
as well as handy methods to access other resources like *automl frameworks* and *benchmark definitions*
"""
import copy
import logging
import os
import random
import re
import sys

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
        log.info("Loading frameworks definitions from %s.", frameworks_file)
        if not isinstance(frameworks_file, list):
            frameworks_file = [frameworks_file]

        frameworks = Namespace()
        for ff in frameworks_file:
            frameworks + config_load(ff)

        to_validate = []
        for name, framework in frameworks:
            framework.name = name
            to_validate.append(framework)

        # support for frameworks definition extending other definitions:
        # useful when having multiple definitions with different params
        validated = []
        while len(to_validate) > 0:
            later = []
            for framework in to_validate:
                if framework['extends'] is not None:
                    parent = frameworks[framework.extends]
                    if parent is None:
                        log.warning("Removing framework %s as parent %s doesn't exist.", framework.name, framework.extends)
                        continue
                    elif parent == framework:
                        log.warning("Framework %s extends itself: removing extension.", framework.name)
                        framework.extends = None
                    elif parent not in validated:
                        later.append(framework)
                        continue
                    else:
                        framework % parent  # adds framework's missing keys from parent
                self._validate_framework(framework)
                validated.append(framework)
            to_validate = later

        log.debug("Available framework definitions:\n%s", frameworks)

        frameworks_lookup = Namespace()
        for framework in validated:
            frameworks_lookup[framework.name.lower()] = framework
        return frameworks_lookup

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
        :param name: name of the benchmark as defined by resources/benchmarks/{name}.yaml or the path to a user-defined benchmark description file.
        :param defaults: defaults used as a base config for each task in the benchmark definition
        :return:
        """
        benchmark_name = name
        benchmark_dir = self.config.benchmarks.definition_dir
        if not isinstance(benchmark_dir, list):
            benchmark_dir = [benchmark_dir]

        benchmark_file = None
        for bd in benchmark_dir:
            bf = os.path.join(bd, "{}.yaml".format(benchmark_name))
            if os.path.exists(bf):
                benchmark_file = bf
                break

        if benchmark_file is None:
            benchmark_file = name
            benchmark_name, _ = os.path.splitext(os.path.basename(name))

        if not os.path.exists(benchmark_file):
            # should we support s3 and check for s3 path before raising error?
            raise ValueError("Incorrect benchmark name or path `{}`, name not available in {}.".format(name, self.config.benchmarks.definition_dir))

        log.info("Loading benchmark definitions from %s.", benchmark_file)
        tasks = config_load(benchmark_file)
        defaults = next((task for task in tasks if task.name == '__defaults__'), defaults)
        tasks = [task for task in tasks if task is not defaults]

        for task in tasks:
            task % defaults   # add missing keys from local defaults
            self._validate_task(task)

        self._validate_task(defaults, lenient=True)
        defaults.enabled = False
        tasks.append(defaults)
        log.debug("Available task definitions:\n%s", tasks)
        return tasks, benchmark_name, benchmark_file

    def _validate_framework(self, framework):
        if framework['module'] is None:
            framework.module = '.'.join([self.config.frameworks.root_module, framework.name])

        if framework['setup_args'] is None:
            framework.setup_args = None

        if framework['setup_script'] is None:
            framework.setup_script = None
        else:
            framework.setup_script = framework.setup_script.format(**self._common_dirs,
                                                                   **dict(module=framework.module))
        if framework['setup_cmd'] is None:
            framework._setup_cmd = None
            framework.setup_cmd = None
        else:
            framework._setup_cmd = framework.setup_cmd
            if isinstance(framework.setup_cmd, str):
                framework.setup_cmd = [framework.setup_cmd]
            framework.setup_cmd = [cmd.format(**self._common_dirs,
                                              **dict(setup=os.path.join(framework.module, "setup"),
                                                     pip="PIP",
                                                     py="PY"))
                                   for cmd in framework.setup_cmd]

        if framework['params'] is None:
            framework.params = dict()
        else:
            framework.params = Namespace.dict(framework.params)

        if framework['version'] is None:
            framework.version = 'latest'

        did = copy.copy(self.config.docker.image_defaults)
        if framework['docker_image'] is None:
            framework['docker_image'] = did
        for conf in ['author', 'image', 'tag']:
            if framework.docker_image[conf] is None:
                framework.docker_image[conf] = did[conf]
        if framework.docker_image.image is None:
            framework.docker_image.image = framework.name.lower()
        if framework.docker_image.tag is None:
            framework.docker_image.tag = framework.version.lower()

    def _validate_task(self, task, lenient=False):
        missing = []
        for conf in ['name', 'openml_task_id']:
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
            task[conf] = "openml.org/t/{}".format(task.openml_task_id) if task['openml_task_id'] is not None \
                else "openml.org/d/{}".format(task.openml_dataset_id) if task['openml_dataset_id'] is not None \
                else task.dataset if task['dataset'] is not None \
                else None

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

